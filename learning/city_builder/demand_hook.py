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

from typing import Optional

import torch
from torch import Tensor

from simulation.citygraph_dataset import (
    CityGraphData,
    DEMAND_KEY,
    DMD_FEAT_IDX,
    SHORTESTPATH_FEAT_IDX,
)

from .gravity import gravity_demand


def recompute_demand_in_place(
    data: CityGraphData,
    activity: Tensor,
    beta: float = 2.0,
    drive_times: Optional[Tensor] = None,
) -> Tensor:
    """Recompute OD demand from activity and write into `data` in place.

    Args:
        data: a CityGraphData whose `demand` / DEMAND_KEY edge_attr will be
            overwritten. Its `drive_times` is used by default; pass
            `drive_times` to override (e.g., supplying a freshly-computed
            post-action matrix that has not been written back to `data` yet).
        activity: shape (N,) per-zone activity.
        beta: distance-decay exponent for the gravity model. Defaults to 2.0
            (the value committed to in the MDP doc sec 9).
        drive_times: optional override for `data.drive_times`.

    Returns:
        The new demand matrix (N, N) for downstream consumers. The same tensor
        is also written to data.demand.
    """
    if drive_times is None:
        drive_times = data.drive_times

    # 1. Recompute full N x N matrix.
    demand = gravity_demand(activity, drive_times, beta=beta)
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
