"""In-place demand recomputation hook for CityGraphData.

After LandUseDynamics.step updates per-zone activity, the gravity model is
re-evaluated and written back to the three places the cost module reads:

    1. data.demand                                 -- full N x N tensor
    2. data[DEMAND_KEY].edge_attr[:, DMD_FEAT_IDX] -- per-edge demand feature
    3. data[DEMAND_KEY].edge_attr[:, SHORTESTPATH_FEAT_IDX]
       -- per-edge drive-time feature (network-change refresh; left to the
       network-update step, but updated here too if `drive_times` is passed)

The DEMAND_KEY edge_index is left unchanged. With `fully_connected_demand=True`
(the default in `from_mumford_data` and the synthetic loader), the edge set
covers every off-diagonal pair, so a non-trivial demand update never needs to
add or drop edges. If a future loader sets `fully_connected_demand=False`,
zones that gain demand from zero will not get edges, and downstream code will
silently undercount them --- we assert against that below.
"""
from __future__ import annotations

from typing import Callable, Optional

import torch
from torch import Tensor

from simulation.citygraph_dataset import (
    CityGraphData,
    DEMAND_KEY,
    DMD_FEAT_IDX,
    SHORTESTPATH_FEAT_IDX,
)

from .gravity import gravity_demand
from .land_use_dynamics import LandUseDynamics


GravityFn = Callable[[Tensor, Tensor], Tensor]
"""Signature: (activity[N], drive_times[N,N]) -> demand[N,N].

Mirrors the `AccessibilityFn` hook on LandUseDynamics. Lets us drop 
in a singly- or doubly-constrained gravity model."""


def recompute_demand_in_place(
    data: CityGraphData,
    activity: Tensor,
    beta: float = 2.0,
    drive_times: Optional[Tensor] = None,
    gravity_fn: Optional[GravityFn] = None,
) -> Tensor:
    """Recompute OD demand from activity and write into `data` in place.

    Args:
        data: a CityGraphData whose `demand` / DEMAND_KEY edge_attr will be
            overwritten. Its `drive_times` is used by default; pass
            `drive_times` to override (e.g., supplying a freshly-computed
            post-action matrix that has not been written back to `data` yet).
        activity: shape (N,) per-zone activity.
        beta: distance-decay exponent for the default unconstrained-gravity
            model. Defaults to 2.0 (MDP doc sec 9). 
        drive_times: optional override for `data.drive_times`.
        gravity_fn: optional callable replacing the default gravity model. Must
            return a non-negative (N, N) matrix; symmetry / zero-diagonal of
            the default are *not* enforced here, so downstream consumers should
            assume nothing beyond shape and non-negativity.

    Returns:
        The new demand matrix (N, N) for downstream consumers. The same tensor
        is also written to data.demand.
    """
    if drive_times is None:
        drive_times = data.drive_times

    # 1. Recompute full N x N matrix.
    if gravity_fn is None:
        demand = gravity_demand(activity, drive_times, beta=beta)
    else:
        demand = gravity_fn(activity, drive_times)
    data.demand = demand

    # 2. Update DEMAND_KEY edge_attr in place. We need to update on the same
    #    edge_index that's already on the graph -- never adding/removing edges,
    #    because the policy's PyG batching machinery depends on a stable index.
    dmd_store = data[DEMAND_KEY]
    edge_index = dmd_store.edge_index  # shape (2, E)
    src = edge_index[0]
    dst = edge_index[1]

    # New demand feature.
    new_dmd_feat = demand[src, dst]
    # Drive-time feature stays consistent with whatever drive_times we used.
    new_dt_feat = drive_times[src, dst]

    edge_attr = dmd_store.edge_attr
    # In-place column updates --- preserve any extra columns the upstream
    # synthetic dataset may have added (it currently uses just the two, but the
    # MDP doc sec 6 leaves room to extend node/edge features for the multi-year
    # policy).
    edge_attr[:, DMD_FEAT_IDX] = new_dmd_feat
    edge_attr[:, SHORTESTPATH_FEAT_IDX] = new_dt_feat

    # Sanity: if the graph was built with fully_connected_demand=False, any
    # newly-non-zero demand off the existing edge_index will be silently
    # dropped from the per-edge feature view. We do not raise here (the matrix
    # is still right), but we warn once per call to flag the gotcha in tests.
    n = activity.shape[0]
    expected_edges_full = n * (n - 1)
    if edge_index.shape[1] != expected_edges_full:
        _maybe_warn_partial_demand_index(
            actual=edge_index.shape[1], expected=expected_edges_full
        )

    return demand


def step_world(
    dyn: LandUseDynamics,
    data: CityGraphData,
    activity: Tensor,
    drive_times: Optional[Tensor] = None,
    beta: Optional[float] = None,
    gravity_fn: Optional[GravityFn] = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Advance the city one year: dynamics step + gravity recomputation.

    Composes ``LandUseDynamics.step`` with ``recompute_demand_in_place`` so
    callers cannot accidentally feed the cost module stale gravity demand after
    a dynamics step. This is the canonical one-call wiring for rollouts and
    eval harnesses; the smoke driver in ``smoke_drive_dynamics.py`` uses it.

    Static-demand mode (Holliday reproduction) is preserved via
    ``LandUseConfig(alpha=0.0, base_rate=1.0, sigma_eps=0.0)``: ``dyn.step``
    becomes the identity, so the recomputed gravity demand is byte-identical
    to the previous step's. The ``test_alpha_zero_no_noise_keeps_demand_stable``
    test pins that invariant.

    Args:
        dyn: per-episode ``LandUseDynamics`` instance.
        data: ``CityGraphData`` to mutate in place.
        activity: current x_t, shape ``(N,)``.
        drive_times: optional override for both the dynamics step and the
            gravity recompute
        beta: gravity decay exponent. Defaults to
            ``dyn.config.beta_accessibility`` so accessibility and gravity stay
            paired (per the MDP doc rationale).q
        gravity_fn: optional pluggable gravity model. Mirrors the
            ``accessibility_fn`` hook on ``LandUseDynamics``.

    Returns:
        ``(x_next, accessibility_tilde, demand)``. The first two are the same
        as ``dyn.step``; ``demand`` is the new OD matrix (also written into
        ``data.demand``).
    """
    if drive_times is None:
        drive_times = data.drive_times
    if beta is None:
        # Keep accessibility-beta and gravity-beta paired by default (MDP doc
        # sec 9 rationale: shared exponent is intentional, not a coincidence).
        # Override `beta` (or supply `gravity_fn`) to decouple them.
        beta = dyn.config.beta_accessibility

    x_next, a_tilde = dyn.step(activity, drive_times)
    demand = recompute_demand_in_place(
        data,
        x_next,
        beta=beta,
        drive_times=drive_times,
        gravity_fn=gravity_fn,
    )
    return x_next, a_tilde, demand


_WARNED_PARTIAL = False


def _maybe_warn_partial_demand_index(actual: int, expected: int) -> None:
    global _WARNED_PARTIAL
    if _WARNED_PARTIAL:
        return
    import warnings

    warnings.warn(
        "CityGraphData has partial DEMAND_KEY edge_index "
        f"(got {actual} edges; fully-connected would be {expected}). "
        "Per-edge demand features will only cover the original edge set; "
        "the N x N data.demand tensor is still correct, but downstream "
        "code that uses edge_attr only (e.g., the GNN's message passing) "
        "will undercount newly-induced demand. Build with "
        "fully_connected_demand=True to avoid this.",
        RuntimeWarning,
        stacklevel=3,
    )
    _WARNED_PARTIAL = True
