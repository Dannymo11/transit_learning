"""Placeholder gravity model for OD demand.

D_{ij} = x_i * x_j * d_{ij}^{-beta},  D_{ii} = 0

This is the unconstrained-gravity placeholder committed to in
docs/M2_mdp_formalization.md sec 9. Calibration against Bogota's Encuesta de
Movilidad (M4 / TOP-15) may replace it with a singly- or doubly-constrained
variant; the LandUseDynamics interface accepts any callable with the same
signature so the swap is local.
"""
from __future__ import annotations

import torch
from torch import Tensor


def gravity_demand(
    activity: Tensor,
    drive_times: Tensor,
    beta: float = 2.0,
) -> Tensor:
    """Compute an N x N OD demand matrix from per-zone activity + drive times.

    Args:
        activity: shape (N,).
        drive_times: shape (N, N). Inf -> 0 trips (disconnected pair).
        beta: distance-decay exponent.

    Returns:
        Tensor of shape (N, N), non-negative, diagonal zero.
    """
    if activity.ndim != 1:
        raise ValueError(
            f"activity must be 1-D, got shape {tuple(activity.shape)}"
        )
    if drive_times.shape != (activity.shape[0], activity.shape[0]):
        raise ValueError(
            f"drive_times shape {tuple(drive_times.shape)} incompatible "
            f"with activity length {activity.shape[0]}"
        )

    n = activity.shape[0]
    diag_mask = torch.eye(n, dtype=torch.bool, device=drive_times.device)
    decay = drive_times.pow(-beta)
    decay = torch.where(diag_mask, torch.zeros_like(decay), decay)
    decay = torch.where(decay.isfinite(), decay, torch.zeros_like(decay))
    # Outer product x x^T, scaled by decay.
    mass = activity.unsqueeze(1) * activity.unsqueeze(0)
    return mass * decay
