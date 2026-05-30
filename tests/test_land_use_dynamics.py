"""Tests for learning/city_builder/ -- M2 / TOP-9.

The single most important invariant: alpha=0 + sigma_eps=0 -> identity.
This is the gate for "did we break Holliday's static-demand reproduction?"
(MDP doc sec 4; M1 architecture mapping sec "Critical seam properties").
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

# Allow `python -m pytest tests/` from the repo root without installing.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from learning.city_builder import (  # noqa: E402
    LandUseConfig,
    LandUseDynamics,
    gravity_demand,
    hansen_accessibility,
    normalize_accessibility,
    recompute_demand_in_place,
    step_world,
)
from simulation.citygraph_dataset import (  # noqa: E402
    DEMAND_KEY,
    DMD_FEAT_IDX,
    SHORTESTPATH_FEAT_IDX,
    CityGraphData,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_city():
    """4-zone fully-connected graph with finite drive times."""
    activity = torch.tensor([100.0, 50.0, 30.0, 10.0])
    drive_times = torch.tensor(
        [
            [0.0, 10.0, 20.0, 30.0],
            [10.0, 0.0, 15.0, 25.0],
            [20.0, 15.0, 0.0, 12.0],
            [30.0, 25.0, 12.0, 0.0],
        ]
    )
    return activity, drive_times


@pytest.fixture
def fully_connected_data(small_city):
    """Hand-rolled CityGraphData with a fully-connected DEMAND_KEY edge_index.

    Mirrors `from_mumford_data`'s shape: data.demand is N x N, edge_index
    covers all off-diagonal pairs, edge_attr is (E, 2) with columns
    [demand, drive_time].
    """
    from itertools import permutations

    activity, drive_times = small_city
    n = activity.shape[0]
    od = gravity_demand(activity, drive_times, beta=2.0)

    data = CityGraphData()
    data.drive_times = drive_times.clone()
    data.demand = od.clone()
    dmd_idx = torch.tensor(list(permutations(range(n), 2))).T  # (2, n*(n-1))
    data[DEMAND_KEY].edge_index = dmd_idx
    edge_attr = torch.zeros((dmd_idx.shape[1], 2))
    edge_attr[:, DMD_FEAT_IDX] = od[dmd_idx[0], dmd_idx[1]]
    edge_attr[:, SHORTESTPATH_FEAT_IDX] = drive_times[dmd_idx[0], dmd_idx[1]]
    data[DEMAND_KEY].edge_attr = edge_attr
    return data, activity, drive_times


# ---------------------------------------------------------------------------
# Accessibility primitives
# ---------------------------------------------------------------------------


def test_hansen_accessibility_basic(small_city):
    activity, drive_times = small_city
    a = hansen_accessibility(activity, drive_times, beta=2.0)
    # Zone 0: 50/100 + 30/400 + 10/900 = 0.5 + 0.075 + 0.0111... = 0.586111...
    expected_0 = 50 / 10**2 + 30 / 20**2 + 10 / 30**2
    assert math.isclose(a[0].item(), expected_0, rel_tol=1e-6)
    # All non-negative.
    assert (a >= 0).all()


def test_hansen_accessibility_ignores_diagonal(small_city):
    activity, drive_times = small_city
    # If we naively did d^{-2} with d_ii=0 included, we'd get inf.
    a = hansen_accessibility(activity, drive_times, beta=2.0)
    assert a.isfinite().all()


def test_hansen_accessibility_disconnected(small_city):
    activity, drive_times = small_city
    # Disconnect zone 3 by setting all its edges to inf.
    dt = drive_times.clone()
    dt[3, :] = float("inf")
    dt[:, 3] = float("inf")
    dt[3, 3] = 0.0  # diagonal is masked anyway
    a = hansen_accessibility(activity, dt, beta=2.0)
    # Zone 3 cannot reach anyone -> A = 0.
    assert a[3].item() == 0.0
    # And nobody contributes from zone 3 (x_3 = 10) to others.
    a_ref = hansen_accessibility(
        activity * torch.tensor([1.0, 1.0, 1.0, 0.0]), drive_times, beta=2.0
    )
    for i in range(3):
        assert math.isclose(a[i].item(), a_ref[i].item(), rel_tol=1e-6)


def test_normalize_accessibility_constant():
    # Constant input -> all zeros (avoids div-by-zero).
    a = torch.ones(5)
    assert torch.equal(normalize_accessibility(a), torch.zeros(5))


def test_normalize_accessibility_minmax():
    a = torch.tensor([1.0, 2.0, 3.0, 4.0])
    out = normalize_accessibility(a)
    assert torch.allclose(out, torch.tensor([0.0, 1 / 3, 2 / 3, 1.0]))


# ---------------------------------------------------------------------------
# Gravity model
# ---------------------------------------------------------------------------


def test_gravity_symmetric(small_city):
    activity, drive_times = small_city
    od = gravity_demand(activity, drive_times, beta=2.0)
    assert torch.allclose(od, od.T)


def test_gravity_zero_diagonal(small_city):
    activity, drive_times = small_city
    od = gravity_demand(activity, drive_times, beta=2.0)
    assert torch.equal(od.diagonal(), torch.zeros(activity.shape[0]))


# ---------------------------------------------------------------------------
# LandUseDynamics core update
# ---------------------------------------------------------------------------


def test_alpha_zero_no_noise_is_identity(small_city):
    """The decision gate from MDP doc sec 4: alpha=0, sigma=0 -> static demand.

    No other property of the module matters if this one fails."""
    activity, drive_times = small_city
    dyn = LandUseDynamics(
        initial_activity=activity,
        config=LandUseConfig(alpha=0.0, base_rate=1.0, sigma_eps=0.0),
        seed=0,
    )
    x_next, _ = dyn.step(activity.clone(), drive_times)
    assert torch.equal(x_next, activity)


def test_alpha_positive_grows_well_connected_zones_faster(small_city):
    """Well-connected zones (high accessibility) grow faster than peripheral
    zones under alpha > 0."""
    activity, drive_times = small_city
    dyn = LandUseDynamics(
        initial_activity=activity,
        config=LandUseConfig(alpha=0.5, base_rate=1.0, sigma_eps=0.0),
        seed=0,
    )
    x_next, a_tilde = dyn.step(activity.clone(), drive_times)
    growth_rates = x_next / activity
    # Highest-accessibility zone grew strictly more than the lowest.
    high_idx = int(a_tilde.argmax())
    low_idx = int(a_tilde.argmin())
    assert growth_rates[high_idx] > growth_rates[low_idx]
    # Lowest-accessibility zone is exactly base_rate (A_tilde=0 there).
    assert math.isclose(growth_rates[low_idx].item(), 1.0, rel_tol=1e-6)


def test_capacity_cap_saturates(small_city):
    """Activity cannot exceed cap_multiplier * x_0."""
    activity, drive_times = small_city
    dyn = LandUseDynamics(
        initial_activity=activity,
        config=LandUseConfig(
            alpha=10.0, base_rate=1.0, sigma_eps=0.0, cap_multiplier=2.0
        ),
        seed=0,
    )
    # alpha=10 with A_tilde=1 means highest-A zone wants to grow 11x -> capped.
    x_next, _ = dyn.step(activity, drive_times)
    assert (x_next <= 2.0 * activity + 1e-6).all()


def test_seed_determinism(small_city):
    """Same seed -> identical noise trajectory. Load-bearing for TOP-10
    paired alpha-sweep comparisons."""
    activity, drive_times = small_city
    cfg = LandUseConfig(alpha=0.3, sigma_eps=5.0)
    dyn_a = LandUseDynamics(initial_activity=activity, config=cfg, seed=123)
    dyn_b = LandUseDynamics(initial_activity=activity, config=cfg, seed=123)
    x = activity.clone()
    y = activity.clone()
    for _ in range(5):
        x, _ = dyn_a.step(x, drive_times)
        y, _ = dyn_b.step(y, drive_times)
        assert torch.equal(x, y)


def test_different_seeds_diverge(small_city):
    activity, drive_times = small_city
    cfg = LandUseConfig(alpha=0.3, sigma_eps=5.0)
    dyn_a = LandUseDynamics(initial_activity=activity, config=cfg, seed=1)
    dyn_b = LandUseDynamics(initial_activity=activity, config=cfg, seed=2)
    x, _ = dyn_a.step(activity.clone(), drive_times)
    y, _ = dyn_b.step(activity.clone(), drive_times)
    assert not torch.equal(x, y)


def test_activity_non_negative(small_city):
    """Even with large negative noise, clip ensures x >= 0."""
    activity, drive_times = small_city
    dyn = LandUseDynamics(
        initial_activity=activity,
        # Massive noise to test the clip kicks in.
        config=LandUseConfig(alpha=0.0, base_rate=1.0, sigma_eps=1e6),
        seed=42,
    )
    x_next, _ = dyn.step(activity, drive_times)
    assert (x_next >= 0).all()


def test_zero_initial_zones_stay_zero(small_city):
    """If a zone starts at x=0 then cap_i=0 -> it cannot grow.

    Matches 'no developable land' semantics, the only behavior we can specify
    without an exogenous capacity table."""
    activity, drive_times = small_city
    activity = activity.clone()
    activity[2] = 0.0
    dyn = LandUseDynamics(
        initial_activity=activity,
        config=LandUseConfig(alpha=0.5, base_rate=1.0, sigma_eps=0.0),
        seed=0,
    )
    x_next, _ = dyn.step(activity, drive_times)
    assert x_next[2].item() == 0.0


def test_base_rate_below_one_shrinks(small_city):
    """base_rate < 1 with alpha=0 -> uniform decay. Useful for stress-testing
    the gravity-recomputation hook against shrinking cities."""
    activity, drive_times = small_city
    dyn = LandUseDynamics(
        initial_activity=activity,
        config=LandUseConfig(alpha=0.0, base_rate=0.9, sigma_eps=0.0),
        seed=0,
    )
    x_next, _ = dyn.step(activity, drive_times)
    assert torch.allclose(x_next, 0.9 * activity)


def test_invalid_config_rejected():
    with pytest.raises(ValueError):
        LandUseConfig(alpha=-1.0).assert_valid()
    with pytest.raises(ValueError):
        LandUseConfig(sigma_eps=-1.0).assert_valid()
    with pytest.raises(ValueError):
        LandUseConfig(cap_multiplier=0.0).assert_valid()


# ---------------------------------------------------------------------------
# Demand recomputation hook
# ---------------------------------------------------------------------------


def test_recompute_demand_writes_back_full_matrix(fully_connected_data):
    """data.demand reflects the new activity after the hook."""
    data, activity, drive_times = fully_connected_data
    new_activity = activity * 2.0  # double-size city
    new_demand = recompute_demand_in_place(data, new_activity, beta=2.0)
    expected = gravity_demand(new_activity, drive_times, beta=2.0)
    assert torch.allclose(data.demand, expected)
    assert torch.allclose(new_demand, expected)


def test_recompute_demand_writes_back_edge_attr(fully_connected_data):
    """Per-edge demand feature mirrors the matrix on the SAME edge index."""
    data, activity, drive_times = fully_connected_data
    new_activity = activity * 2.0
    recompute_demand_in_place(data, new_activity, beta=2.0)
    edge_index = data[DEMAND_KEY].edge_index
    edge_attr = data[DEMAND_KEY].edge_attr
    for k in range(edge_index.shape[1]):
        i, j = int(edge_index[0, k]), int(edge_index[1, k])
        assert math.isclose(
            edge_attr[k, DMD_FEAT_IDX].item(),
            data.demand[i, j].item(),
            rel_tol=1e-6,
        )
        assert math.isclose(
            edge_attr[k, SHORTESTPATH_FEAT_IDX].item(),
            drive_times[i, j].item(),
            rel_tol=1e-6,
        )


def test_recompute_demand_with_alpha_zero_is_identity(fully_connected_data):
    """alpha=0 + sigma=0 keeps activity the same; the hook with the same
    activity must keep `data.demand` byte-identical."""
    data, activity, drive_times = fully_connected_data
    demand_before = data.demand.clone()
    edge_attr_before = data[DEMAND_KEY].edge_attr.clone()

    dyn = LandUseDynamics(
        initial_activity=activity,
        config=LandUseConfig(alpha=0.0, base_rate=1.0, sigma_eps=0.0),
        seed=0,
    )
    x_next, _ = dyn.step(activity, drive_times)
    recompute_demand_in_place(data, x_next, beta=2.0)

    assert torch.allclose(data.demand, demand_before)
    assert torch.allclose(data[DEMAND_KEY].edge_attr, edge_attr_before)


def test_recompute_demand_warns_on_partial_index(small_city):
    """If a graph was built with fully_connected_demand=False, we warn loudly."""
    import warnings

    # Reset the module-level warned flag so this test is independent of order.
    import learning.city_builder.demand_hook as hook_mod

    hook_mod._WARNED_PARTIAL = False

    activity, drive_times = small_city
    data = CityGraphData()
    data.drive_times = drive_times.clone()
    data.demand = gravity_demand(activity, drive_times, beta=2.0)
    # Partial edge index: only 2 demand edges.
    data[DEMAND_KEY].edge_index = torch.tensor([[0, 1], [1, 0]])
    data[DEMAND_KEY].edge_attr = torch.zeros(2, 2)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        recompute_demand_in_place(data, activity * 2.0, beta=2.0)
    assert any("partial" in str(w.message).lower() for w in caught), (
        f"expected partial-demand warning; got {[str(w.message) for w in caught]}"
    )


# ---------------------------------------------------------------------------
# Caching-trap regression guards (from the demand-caching audit)
#
# These two tests do not exercise new behavior; they lock in invariants the
# multi-year experiment silently depends on. See docs/M1_architecture_mapping.md
# "Critical seam properties".
# ---------------------------------------------------------------------------


@pytest.fixture
def state_ready_data(small_city):
    """A CityGraphData complete enough to build a RouteGenBatchState.

    Mirrors `from_mumford_data`'s schema: pos+degree node features, a finite
    street network, precomputed drive_times/nexts, fully-connected demand with
    an (E, 2) edge_attr. Built from `small_city` so no data files are needed.
    """
    from itertools import permutations

    import torch_utils as tu
    from simulation.citygraph_dataset import (
        STOP_KEY,
        STREET_KEY,
        get_node_features,
    )

    activity, drive_times = small_city
    n = activity.shape[0]
    od = gravity_demand(activity, drive_times, beta=2.0)

    # The (finite, complete) drive-time matrix doubles as the street adjacency.
    street_adj = drive_times.clone()
    nexts, dts = tu.floyd_warshall(street_adj)
    street_idx = torch.stack(
        torch.where((street_adj > 0) & street_adj.isfinite())
    )

    data = CityGraphData()
    data.fixed_routes = torch.zeros((0, n))
    data[STOP_KEY].pos = torch.zeros((n, 2))
    data[STOP_KEY].x = torch.cat(
        (data[STOP_KEY].pos, get_node_features(street_idx, od)), dim=1
    )
    data[STREET_KEY].edge_index = street_idx
    data[STREET_KEY].edge_attr = street_adj[street_idx[0], street_idx[1]]
    data.street_adj = street_adj
    data.drive_times = dts.squeeze(0)
    data.nexts = nexts.squeeze(0)
    data.demand = od.clone()

    dmd_idx = torch.tensor(list(permutations(range(n), 2))).T
    data[DEMAND_KEY].edge_index = dmd_idx
    edge_attr = torch.zeros((dmd_idx.shape[1], 2))
    edge_attr[:, DMD_FEAT_IDX] = od[dmd_idx[0], dmd_idx[1]]
    edge_attr[:, SHORTESTPATH_FEAT_IDX] = data.drive_times[dmd_idx[0], dmd_idx[1]]
    data[DEMAND_KEY].edge_attr = edge_attr
    return data, activity, drive_times


def test_state_does_not_alias_source_demand(state_ready_data):
    """Risk 1: RouteGenBatchState wraps graph_data via Batch.from_data_list,
    which COPIES tensors into new storage. A state built BEFORE an in-place
    demand mutation therefore does NOT see the new demand.

    Consequence for the experiment: the multi-year harness must REBUILD the
    state after every recompute_demand_in_place (as smoke_drive_dynamics does);
    a reused state would feed the cost oracle stale demand and make alpha>0
    look like alpha=0. This test pins that contract -- if PyG ever starts
    aliasing, it fails and we can drop the rebuild requirement deliberately
    rather than discovering a silent bug in the results.
    """
    from simulation.transit_time_estimator import (
        MyCostModule,
        RouteGenBatchState,
    )

    data, activity, _ = state_ready_data
    cost = MyCostModule(symmetric_routes=True)
    state = RouteGenBatchState(
        data, cost, n_routes_to_plan=1, min_route_len=2
    )

    # The state owns a distinct graph object, not the source.
    assert state.graph_data is not data
    sum_before = state.demand.sum().item()
    assert sum_before > 0

    # Mutate the SOURCE demand storage in place (strictest aliasing probe).
    data.demand.mul_(5.0)

    # The source changed...
    assert math.isclose(
        data.demand.sum().item(), 5.0 * sum_before, rel_tol=1e-6
    )
    # ...but the prebuilt state is decoupled and still holds the old demand.
    assert math.isclose(
        state.demand.sum().item(), sum_before, rel_tol=1e-9
    ), (
        "RouteGenBatchState.demand now tracks the source graph; the "
        "rebuild-per-year contract in the multi-year harness can be revisited."
    )


def test_node_features_are_demand_free(small_city):
    """Risk 2: the policy caches normalized node features at setup_planning
    (models.set_normalized_features -> state.norm_node_features) and the demand
    hook does NOT refresh STOP_KEY.x. That is only safe while node features
    carry no demand.

    get_node_features must stay demand-independent. If it ever re-encodes demand
    (cf. the dormant OUT_DEMAND_FEAT_IDX / IN_DEMAND_FEAT_IDX constants and
    DemandScaleTransform), this fails -- a signal that recompute_demand_in_place
    must also rewrite STOP_KEY.x and bust the cached norm_node_features.
    """
    from simulation.citygraph_dataset import get_node_features

    _, drive_times = small_city
    street_idx = torch.stack(
        torch.where((drive_times > 0) & drive_times.isfinite())
    )
    # Two very different demand matrices over the SAME street network.
    base = torch.tensor([100.0, 50.0, 30.0, 10.0])
    demand_a = gravity_demand(base, drive_times, beta=2.0)
    demand_b = gravity_demand(base * 7.0 + 3.0, drive_times, beta=2.0)

    feats_a = get_node_features(street_idx, demand_a)
    feats_b = get_node_features(street_idx, demand_b)
    assert torch.equal(feats_a, feats_b), (
        "get_node_features now depends on demand values; extend "
        "recompute_demand_in_place to refresh STOP_KEY.x and invalidate "
        "norm_node_features, or the multi-year policy sees stale demand."
    )


# ---------------------------------------------------------------------------
# step_world: composed dynamics + gravity recompute
#
# The contract is "rollouts can't skip the recompute". These tests pin:
#   1. alpha=0 + sigma=0 keeps data.demand byte-identical across many steps
#      (the Holliday static-demand reproduction gate, multi-step variant).
#   2. alpha>0 actually drives data.demand to change (the dynamic-demand
#      smoke test --- if this fails silently, alpha-sweeps are meaningless).
#   3. step_world threads `drive_times` and `gravity_fn` through correctly,
#      and `beta` defaults to dyn.config.beta_accessibility.
# ---------------------------------------------------------------------------


def test_step_world_alpha_zero_no_noise_keeps_demand_stable(fully_connected_data):
    """alpha=0 + sigma=0 -> dynamics is identity -> demand is byte-identical
    across an arbitrary number of step_world calls. This is the multi-step
    extension of test_recompute_demand_with_alpha_zero_is_identity, and the
    primary gate for the Holliday static-demand reproduction in TOP-11.
    """
    data, activity, drive_times = fully_connected_data
    dyn = LandUseDynamics(
        initial_activity=activity,
        config=LandUseConfig(alpha=0.0, base_rate=1.0, sigma_eps=0.0),
        seed=0,
    )
    # Seed demand from the gravity model so the comparison starts at the same
    # surface the seam will recompute back to.
    recompute_demand_in_place(data, activity, beta=2.0)
    demand_t0 = data.demand.clone()
    edge_attr_t0 = data[DEMAND_KEY].edge_attr.clone()

    x = activity.clone()
    for _ in range(5):
        x, _, demand_t = step_world(dyn, data, x)
        assert torch.equal(x, activity), "alpha=0 + sigma=0 must be identity"
        assert torch.allclose(demand_t, demand_t0), (
            "alpha=0 path drifted demand --- the static-demand mode is broken"
        )
        assert torch.allclose(data.demand, demand_t0)
        assert torch.allclose(data[DEMAND_KEY].edge_attr, edge_attr_t0)


def test_step_world_alpha_positive_changes_demand(fully_connected_data):
    """alpha>0 makes well-connected zones grow, which the gravity model picks
    up. data.demand must strictly differ from the previous year's; otherwise
    we have a silent stale-demand bug."""
    data, activity, drive_times = fully_connected_data
    dyn = LandUseDynamics(
        initial_activity=activity,
        config=LandUseConfig(alpha=0.5, base_rate=1.0, sigma_eps=0.0),
        seed=0,
    )
    recompute_demand_in_place(data, activity, beta=2.0)
    demand_before = data.demand.clone()

    x_next, _, demand_after = step_world(dyn, data, activity.clone())

    # Activity moved -> demand must move.
    assert not torch.equal(x_next, activity)
    assert not torch.allclose(demand_after, demand_before)
    # And the in-place write actually landed.
    assert torch.allclose(data.demand, demand_after)
    # Edge-attr feature mirrors the matrix on the same edge_index.
    edge_index = data[DEMAND_KEY].edge_index
    assert torch.allclose(
        data[DEMAND_KEY].edge_attr[:, DMD_FEAT_IDX],
        demand_after[edge_index[0], edge_index[1]],
    )


def test_step_world_matches_manual_two_step(fully_connected_data):
    """step_world must produce byte-identical results to the explicit
    dyn.step + recompute_demand_in_place pair --- it is sugar, not new
    semantics. Locks in the contract so future refactors of either piece
    cannot quietly diverge the composed path."""
    data, activity, drive_times = fully_connected_data

    # Manual path on a copy of `data`.
    import copy
    data_manual = copy.deepcopy(data)
    dyn_manual = LandUseDynamics(
        initial_activity=activity,
        config=LandUseConfig(alpha=0.4, base_rate=1.0, sigma_eps=0.1),
        seed=7,
    )
    x_manual, a_manual = dyn_manual.step(activity, drive_times)
    demand_manual = recompute_demand_in_place(data_manual, x_manual, beta=2.0)

    # Composed path on the original `data`. Same seed -> same noise.
    dyn_world = LandUseDynamics(
        initial_activity=activity,
        config=LandUseConfig(alpha=0.4, base_rate=1.0, sigma_eps=0.1),
        seed=7,
    )
    x_world, a_world, demand_world = step_world(dyn_world, data, activity)

    assert torch.equal(x_manual, x_world)
    assert torch.equal(a_manual, a_world)
    assert torch.allclose(demand_manual, demand_world)


def test_step_world_uses_drive_times_override(fully_connected_data):
    """drive_times override must flow into BOTH the dynamics step (changes
    A_tilde) and the gravity recompute (changes demand). The whole point of
    the override is to evaluate dynamics against a post-action network that
    hasn't been written into `data` yet."""
    data, activity, drive_times = fully_connected_data
    dyn = LandUseDynamics(
        initial_activity=activity,
        config=LandUseConfig(alpha=0.5, base_rate=1.0, sigma_eps=0.0),
        seed=0,
    )
    # A perturbed drive-time matrix: halve every off-diagonal cost.
    dt_fast = drive_times.clone()
    mask = ~torch.eye(dt_fast.shape[0], dtype=torch.bool)
    dt_fast[mask] = dt_fast[mask] * 0.5

    _, _, demand_default = step_world(dyn, data, activity.clone())
    # Reset state for a fair second call.
    recompute_demand_in_place(data, activity, beta=2.0)
    dyn2 = LandUseDynamics(
        initial_activity=activity,
        config=LandUseConfig(alpha=0.5, base_rate=1.0, sigma_eps=0.0),
        seed=0,
    )
    _, _, demand_fast = step_world(dyn2, data, activity.clone(), drive_times=dt_fast)

    # Faster network -> different accessibility -> different x_next ->
    # different demand. Asserts the override was actually threaded through.
    assert not torch.allclose(demand_default, demand_fast)


def test_step_world_uses_gravity_fn_hook(fully_connected_data):
    """gravity_fn override must replace the default gravity model. Tests the
    M4-calibration seam without committing to a particular variant."""
    data, activity, drive_times = fully_connected_data
    dyn = LandUseDynamics(
        initial_activity=activity,
        config=LandUseConfig(alpha=0.3, base_rate=1.0, sigma_eps=0.0),
        seed=0,
    )

    sentinel = torch.full(
        (activity.shape[0], activity.shape[0]), 42.0
    )
    sentinel.fill_diagonal_(0.0)

    def constant_gravity(x: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        # Ignores both inputs --- proves the hook is what's being called.
        return sentinel.clone()

    _, _, demand = step_world(
        dyn, data, activity.clone(), gravity_fn=constant_gravity
    )
    assert torch.allclose(demand, sentinel)
    assert torch.allclose(data.demand, sentinel)


def test_step_world_default_beta_pairs_with_accessibility(fully_connected_data):
    """When `beta` is not supplied, step_world uses dyn.config.beta_accessibility
    so the accessibility and gravity exponents stay paired (MDP doc sec 9
    rationale). Sanity: changing beta_accessibility changes the demand."""
    data, activity, drive_times = fully_connected_data

    dyn_b2 = LandUseDynamics(
        initial_activity=activity,
        config=LandUseConfig(
            alpha=0.0, base_rate=1.0, sigma_eps=0.0, beta_accessibility=2.0,
        ),
        seed=0,
    )
    dyn_b1 = LandUseDynamics(
        initial_activity=activity,
        config=LandUseConfig(
            alpha=0.0, base_rate=1.0, sigma_eps=0.0, beta_accessibility=1.0,
        ),
        seed=0,
    )

    import copy
    data_b2 = copy.deepcopy(data)
    data_b1 = copy.deepcopy(data)
    _, _, demand_b2 = step_world(dyn_b2, data_b2, activity.clone())
    _, _, demand_b1 = step_world(dyn_b1, data_b1, activity.clone())

    # alpha=0 means activity is unchanged; the only difference is gravity beta.
    expected_b2 = gravity_demand(activity, drive_times, beta=2.0)
    expected_b1 = gravity_demand(activity, drive_times, beta=1.0)
    assert torch.allclose(demand_b2, expected_b2)
    assert torch.allclose(demand_b1, expected_b1)
    assert not torch.allclose(demand_b2, demand_b1)
