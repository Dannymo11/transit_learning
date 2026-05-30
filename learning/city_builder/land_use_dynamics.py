"""LandUseDynamics --- per-zone activity update driven by accessibility.

Implements the M2 / TOP-9 spec from docs/M2_mdp_formalization.md sec 4:

    x_{t+1,i} = clip(x_{t,i} * (base_rate + alpha * A_tilde_{t,i}) + eps_{t,i},
                     0, cap_i)

where A_tilde is the normalized Hansen accessibility under the year-(t+1)
network. alpha is the induced-demand strength (calibrated in M4, ablated in M5
Exp 5). base_rate defaults to 1.0 so that alpha=0 + sigma_eps=0 reproduces
exactly Holliday's static-demand setting.

The module is stateless w.r.t. CityGraphData --- it consumes/returns tensors.
The actual graph-mutation lives in demand_hook.recompute_demand_in_place to
keep unit-testing tight and to make the M4 calibration harness easy to write.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import torch
from torch import Tensor

from .accessibility import hansen_accessibility, normalize_accessibility


AccessibilityFn = Callable[[Tensor, Tensor], Tensor]
"""Signature: (activity[N], drive_times[N,N]) -> raw_accessibility[N]."""


def _default_accessibility(activity: Tensor, drive_times: Tensor) -> Tensor:
    return hansen_accessibility(activity, drive_times, beta=2.0)


# Chosen M2 working value for the induced-demand strength alpha, selected by the
# TOP-10 sensitivity sweep (Mandl, T=10, replan-every-2 years). At alpha=0.5 the
# city shows clear evolution (~2.1x total-activity growth over 10 years, demand
# spatially concentrating) without runaway (<half the zones reach the activity
# cap), whereas alpha>=1.0 becomes runaway-prone (>=2/3 of zones capped). This is
# a PLACEHOLDER default for M3 dynamic training (TOP-12); M4 calibration against
# Bogota data replaces it with the project's headline alpha_c. Note alpha=0
# remains the LandUseConfig default below -> static-Holliday reproduction.
M2_WORKING_ALPHA = 0.5


@dataclass
class LandUseConfig:
    """Calibratable parameters for the per-zone activity update.

    All fields are scalars; cap is per-zone via `cap_multiplier * x_0`.
    """
    alpha: float = 0.0
    """Induced-demand strength. alpha=0 -> static demand (Holliday). M4
    calibration produces the project's headline value; M5 Exp 5 ablates it."""

    base_rate: float = 1.0
    """No-accessibility growth multiplier per step. 1.0 -> no drift at A=0."""

    sigma_eps: float = 0.0
    """Per-zone Gaussian noise stddev (absolute, in activity units)."""

    cap_multiplier: float = 3.0
    """Per-zone activity cap as a multiple of x_0. cap_i = cap_multiplier * x_0,i."""

    beta_accessibility: float = 2.0
    """Distance-decay exponent inside Hansen accessibility (only used by the
    default accessibility_fn). Kept in config for completeness; a custom
    accessibility_fn passed to LandUseDynamics overrides this."""

    def assert_valid(self) -> None:
        if self.alpha < 0:
            raise ValueError(f"alpha must be non-negative, got {self.alpha}")
        if self.base_rate < 0:
            raise ValueError(
                f"base_rate must be non-negative, got {self.base_rate}"
            )
        if self.sigma_eps < 0:
            raise ValueError(
                f"sigma_eps must be non-negative, got {self.sigma_eps}"
            )
        if self.cap_multiplier <= 0:
            raise ValueError(
                f"cap_multiplier must be positive, got {self.cap_multiplier}"
            )


class LandUseDynamics:
    """Stateful per-episode dynamics.

    Carries the initial activity vector (for capacity caps) and a per-episode
    `torch.Generator` so that two episodes with the same seed produce identical
    noise trajectories. This is load-bearing for the TOP-10 alpha-sweep paired
    comparisons.

    Typical usage::

        dyn = LandUseDynamics(
            initial_activity=x0,
            config=LandUseConfig(alpha=0.3, sigma_eps=0.05),
            seed=42,
        )
        for t in range(T):
            # ... agent picks action, network updates, drive_times recomputed
            x_next, A_tilde = dyn.step(x, drive_times)
            recompute_demand_in_place(data, x_next)
            x = x_next
    """

    def __init__(
        self,
        initial_activity: Tensor,
        config: LandUseConfig,
        seed: Optional[int] = None,
        accessibility_fn: Optional[AccessibilityFn] = None,
    ) -> None:
        config.assert_valid()
        if initial_activity.ndim != 1:
            raise ValueError(
                f"initial_activity must be 1-D, got shape "
                f"{tuple(initial_activity.shape)}"
            )
        if (initial_activity < 0).any():
            raise ValueError("initial_activity must be non-negative.")

        self.config = config
        self.initial_activity = initial_activity.detach().clone()
        # cap_i = cap_multiplier * x_0,i. Zones initialized at 0 are pinned at
        # 0 (cap = 0), which matches "this zone has no developable land".
        self.cap = config.cap_multiplier * self.initial_activity

        # Per-episode RNG. We use a torch.Generator on the same device as the
        # activity vector so step() does not have to ship noise across devices.
        self.generator = torch.Generator(device=initial_activity.device)
        if seed is not None:
            self.generator.manual_seed(int(seed))

        if accessibility_fn is None:
            beta = config.beta_accessibility

            def _fn(a: Tensor, d: Tensor) -> Tensor:
                return hansen_accessibility(a, d, beta=beta)

            self.accessibility_fn = _fn
        else:
            self.accessibility_fn = accessibility_fn

    # ----- core update ---------------------------------------------------

    def step(
        self,
        activity: Tensor,
        drive_times: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """One year of the dynamics.

        Args:
            activity: shape (N,), current x_t.
            drive_times: shape (N, N), zone-to-zone times under the post-action
                network G_{t+1} (per MDP doc sec 4: action applied first, then
                dynamics observed against the updated network).

        Returns:
            (activity_next, accessibility_tilde) where both have shape (N,).
            accessibility_tilde is returned for downstream consumption as a
            per-node feature in the policy's observation (per MDP doc sec 6).
        """
        a_raw = self.accessibility_fn(activity, drive_times)
        a_tilde = normalize_accessibility(a_raw)

        growth = self.config.base_rate + self.config.alpha * a_tilde
        x_next = activity * growth

        if self.config.sigma_eps > 0:
            noise = torch.randn(
                activity.shape,
                generator=self.generator,
                device=activity.device,
                dtype=activity.dtype,
            ) * self.config.sigma_eps
            x_next = x_next + noise

        # Clip to [0, cap]. Cap may have zeros where the zone started empty;
        # `torch.minimum` handles that without spurious NaNs.
        x_next = torch.clamp(x_next, min=0.0)
        x_next = torch.minimum(x_next, self.cap)
        return x_next, a_tilde

    # ----- bookkeeping helpers ------------------------------------------

    def reset_rng(self, seed: int) -> None:
        """Reseed mid-episode. Useful for fixture tests; production code should
        prefer constructing a new instance per episode."""
        self.generator.manual_seed(int(seed))
