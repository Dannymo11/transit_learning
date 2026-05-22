"""Hansen-style accessibility for the LandUseDynamics module.

For zone i under network with drive-time matrix d_{ij}:

    A_i = sum_{j != i} x_j * d_{ij}^{-beta}

with beta = 2 by default (matches the gravity-model exponent in
docs/M2_mdp_formalization.md sec 9). Disconnected pairs (d_{ij} = inf) contribute
zero, which is the desired behavior. The diagonal is masked out so a zone does
not "access itself" with d = 0.

`normalize_accessibility` rescales A to [0, 1] via min-max (per call), which is
what the dynamics update `x_{t+1} = x_t * (1 + alpha * A_tilde) + eps` expects:
A_tilde in [0, 1] keeps growth non-negative and bounds it by alpha.

User decision (2026-05-22): Hansen, beta=2, min-max normalized. Kept pluggable
via `accessibility_fn` in LandUseDynamics for M6 ablations.
"""
from __future__ import annotations

import torch
from torch import Tensor

EPS = 1e-12


def hansen_accessibility(
    activity: Tensor,
    drive_times: Tensor,
    beta: float = 2.0,
) -> Tensor:
    """Hansen accessibility A_i = sum_{j != i} x_j * d_{ij}^{-beta}.

    Args:
        activity: shape (N,) per-zone activity x.
        drive_times: shape (N, N) zone-to-zone drive times. Inf entries
            (disconnected pairs) contribute zero. Diagonal is ignored.
        beta: decay exponent. Default 2.0 matches the placeholder gravity
            model in MDP doc sec 9.

    Returns:
        Tensor of shape (N,), non-negative.
    """
    if activity.ndim != 1:
        raise ValueError(
            f"activity must be 1-D (per-zone), got shape {tuple(activity.shape)}"
        )
    if drive_times.ndim != 2 or drive_times.shape[0] != drive_times.shape[1]:
        raise ValueError(
            f"drive_times must be square (N, N), got shape "
            f"{tuple(drive_times.shape)}"
        )
    if activity.shape[0] != drive_times.shape[0]:
        raise ValueError(
            f"activity length {activity.shape[0]} != drive_times "
            f"side {drive_times.shape[0]}"
        )

    n = activity.shape[0]
    # Mask diagonal: a zone does not contribute to its own accessibility via
    # d_ii = 0 (would be infinite).
    diag_mask = torch.eye(n, dtype=torch.bool, device=drive_times.device)
    # d^{-beta}; inf -> 0, zero -> inf (masked out below).
    decay = drive_times.pow(-beta)
    decay = torch.where(diag_mask, torch.zeros_like(decay), decay)
    # Treat any remaining non-finite entries (e.g. inf from d=0 off-diagonal,
    # though that shouldn't happen on a well-formed graph) as zero contribution.
    decay = torch.where(decay.isfinite(), decay, torch.zeros_like(decay))
    # A_i = sum_j x_j * decay_ij  -> matrix-vector product on rows.
    return decay @ activity


def normalize_accessibility(accessibility: Tensor) -> Tensor:
    """Min-max scale to [0, 1]. Returns zeros if the input is constant.

    Per-step normalization (not per-episode): keeps alpha's role as a scalar
    multiplier on a [0,1] signal, regardless of the absolute scale of A.
    """
    a_min = accessibility.min()
    a_max = accessibility.max()
    spread = a_max - a_min
    if spread.item() < EPS:
        return torch.zeros_like(accessibility)
    return (accessibility - a_min) / spread
