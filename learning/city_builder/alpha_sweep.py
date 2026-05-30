"""M2 / TOP-10 --- alpha-sensitivity sweep + decision gate.

Runs the greedy and random baselines under a multi-year land-use rollout for
a grid of induced-demand strengths alpha, and measures how the greedy-vs-random
welfare gap responds to alpha. This is the M2 decision gate:

  * alpha=0 must reproduce Holliday's static-demand setting --- with no induced
    demand and no noise, per-zone activity is frozen, so the gravity demand and
    therefore the per-year cost are constant year-over-year (greedy and random
    each flat). This is the milestone's most important sanity check.
  * The chosen alpha should open a clear greedy-vs-random gap (induced demand is
    doing real work) without driving zones into the per-zone activity cap
    (runaway).

Protocol (locked 2026-05-30):
  * Instance: Mandl (15 nodes, 6 routes, route len 2..8) --- Holliday's canonical
    sanity instance. Holliday's default modal params (bus-on-roads); metro params
    are an M4 concern (see M2 pivot note).
  * Horizon T = 10 years; baselines REPLAN every 2 years (years 0/2/4/6/8) against
    the then-current (evolved) demand. Greedy reacts to induced demand; random
    does not --- so the gap should widen with alpha if induced demand matters.
  * Greedy = john_init (demand-aware constructive heuristic, repo-native), its
    internal time/demand tradeoff (w_p) held fixed; only the induced-demand alpha
    is swept. Random = networks of random valid street-network walks, same route
    count.
  * 5 seeds per (alpha, baseline). Welfare W_t = -cost_t (MyCostModule returns a
    cost; lower = better). We report cumulative welfare over the horizon.

NOTE on naming: john_init's `alpha` argument is Holliday's time/demand tradeoff
(== w_p in our writeup), NOT our induced-demand alpha. We pass it as `w_p` here
to avoid the collision flagged in docs/M2_mdp_formalization.md.

Run from the repo root::

    python -m learning.city_builder.alpha_sweep --out results/top10_alpha_sweep
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from simulation.citygraph_dataset import CityGraphData
from simulation.transit_time_estimator import MyCostModule, RouteGenBatchState
from learning.initialization import john_init
from torch_utils import get_batch_tensor_from_routes

from learning.city_builder import (
    LandUseConfig,
    LandUseDynamics,
    recompute_demand_in_place,
    step_world,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INSTANCES_DIR = REPO_ROOT / "datasets" / "mumford_dataset" / "Instances"

# --- Mandl benchmark constants (cfg/eval/mandl.yaml) ----------------------
N_ROUTES = 6
MIN_ROUTE_LEN = 2
MAX_ROUTE_LEN = 8

# --- sweep defaults -------------------------------------------------------
ALPHA_GRID = [0.0, 0.25, 0.5, 1.0, 2.0]
N_SEEDS = 5
HORIZON = 10
REPLAN_EVERY = 2
W_P_FIXED = 0.5          # john_init time/demand tradeoff, held fixed
BETA_GRAVITY = 2.0       # gravity / accessibility distance-decay
CAP_MULTIPLIER = 3.0     # per-zone activity cap = CAP_MULTIPLIER * x_0


def _initial_activity_from_od(od: torch.Tensor) -> torch.Tensor:
    """Per-zone activity proxy from OD row sums (Mandl has no population vector;
    the M4 Bogota adapter replaces this with calibrated x_0)."""
    return od.sum(dim=1)


def _fresh_data(instances_dir: str, instance: str) -> CityGraphData:
    return CityGraphData.from_mumford_data(instances_dir, instance_name=instance)


def _new_state(data: CityGraphData, cost_obj: MyCostModule) -> RouteGenBatchState:
    return RouteGenBatchState(
        data, cost_obj, n_routes_to_plan=N_ROUTES,
        min_route_len=MIN_ROUTE_LEN, max_route_len=MAX_ROUTE_LEN,
    )


def _greedy_network(data: CityGraphData, cost_obj: MyCostModule) -> torch.Tensor:
    """john_init network on the CURRENT demand. Returns (1, n_routes, max_len).

    john_init reads state.demand, so building the state on the freshly-recomputed
    `data` is what makes greedy 'react' to induced demand at each replan."""
    state = _new_state(data, cost_obj)
    nets = john_init(state, alpha=W_P_FIXED, prioritize_direct_connections=True)
    # With a scalar w_p, john_init returns a single network in a length-1 batch.
    return nets


def _random_network(data: CityGraphData, cost_obj: MyCostModule,
                    gen: torch.Generator) -> torch.Tensor:
    """N_ROUTES random valid walks on the street network. Reproducible via `gen`.

    A route is a random walk: start at a random node, repeatedly step to a random
    street-adjacent node not already on the route, until MAX_ROUTE_LEN or stuck;
    kept if it reaches MIN_ROUTE_LEN."""
    state = _new_state(data, cost_obj)
    street_adj = state.street_adj[0]                       # (N, N)
    have_edge = (street_adj > 0) & street_adj.isfinite()
    n_nodes = have_edge.shape[0]

    routes: list[list[int]] = []
    attempts = 0
    while len(routes) < N_ROUTES and attempts < N_ROUTES * 50:
        attempts += 1
        start = int(torch.randint(n_nodes, (1,), generator=gen).item())
        route = [start]
        on_route = torch.zeros(n_nodes, dtype=torch.bool)
        on_route[start] = True
        while len(route) < MAX_ROUTE_LEN:
            cur = route[-1]
            cand = have_edge[cur] & ~on_route
            cand_idx = torch.nonzero(cand).squeeze(-1)
            if cand_idx.numel() == 0:
                break
            pick = int(torch.randint(cand_idx.numel(), (1,), generator=gen).item())
            nxt = int(cand_idx[pick].item())
            route.append(nxt)
            on_route[nxt] = True
        if len(route) >= MIN_ROUTE_LEN:
            routes.append(route)
    if len(routes) < N_ROUTES:
        # Pad with the shortest acceptable random edges so the network is the
        # right size; rare on Mandl (dense enough), but keeps the run robust.
        while len(routes) < N_ROUTES:
            a = int(torch.randint(n_nodes, (1,), generator=gen).item())
            nb = torch.nonzero(have_edge[a]).squeeze(-1)
            if nb.numel() == 0:
                continue
            b = int(nb[torch.randint(nb.numel(), (1,), generator=gen)].item())
            routes.append([a, b])
    return get_batch_tensor_from_routes([routes], device=state.device,
                                        max_route_len=MAX_ROUTE_LEN)


def _evaluate(data: CityGraphData, cost_obj: MyCostModule,
              network: torch.Tensor) -> float:
    """Scalar cost of `network` on the current `data.demand`."""
    state = _new_state(data, cost_obj)
    state.add_new_routes(network)
    with torch.no_grad():
        out = cost_obj(state)
    return float(out.cost.item())


@dataclass
class RolloutResult:
    alpha: float
    seed: int
    baseline: str
    per_year_cost: list[float]
    per_year_sum_activity: list[float]
    per_year_frac_at_cap: list[float]
    cumulative_welfare: float       # sum_t (-cost_t)


def run_rollout(alpha: float, seed: int, baseline: str,
                instances_dir: str, instance: str,
                horizon: int, replan_every: int) -> RolloutResult:
    data = _fresh_data(instances_dir, instance)
    cost_obj = MyCostModule(symmetric_routes=True)

    x_0 = _initial_activity_from_od(data.demand)
    cap = CAP_MULTIPLIER * x_0
    # Set t=0 demand to the gravity demand (so the alpha=0 invariance is honest:
    # every year recomputes the same gravity demand under frozen activity).
    recompute_demand_in_place(data, x_0, beta=BETA_GRAVITY)

    dyn = LandUseDynamics(
        initial_activity=x_0,
        config=LandUseConfig(alpha=alpha, base_rate=1.0, sigma_eps=0.0,
                             cap_multiplier=CAP_MULTIPLIER,
                             beta_accessibility=BETA_GRAVITY),
        seed=seed,
    )
    rand_gen = torch.Generator().manual_seed(10_000 + seed)

    x = x_0.clone()
    network = None
    per_cost, per_sum, per_cap = [], [], []
    for t in range(horizon + 1):
        if t % replan_every == 0:
            if baseline == "greedy":
                network = _greedy_network(data, cost_obj)
            elif baseline == "random":
                network = _random_network(data, cost_obj, rand_gen)
            else:
                raise ValueError(baseline)
        per_cost.append(_evaluate(data, cost_obj, network))
        per_sum.append(float(x.sum().item()))
        per_cap.append(float((x >= cap - 1e-6).float().mean().item()))
        if t == horizon:
            break
        x, _, _ = step_world(dyn, data, x)

    return RolloutResult(
        alpha=alpha, seed=seed, baseline=baseline,
        per_year_cost=per_cost, per_year_sum_activity=per_sum,
        per_year_frac_at_cap=per_cap,
        cumulative_welfare=float(-sum(per_cost)),
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance", default="Mandl")
    parser.add_argument("--instances-dir", default=str(DEFAULT_INSTANCES_DIR))
    parser.add_argument("--alphas", type=float, nargs="+", default=ALPHA_GRID)
    parser.add_argument("--seeds", type=int, default=N_SEEDS)
    parser.add_argument("--horizon", type=int, default=HORIZON)
    parser.add_argument("--replan-every", type=int, default=REPLAN_EVERY)
    parser.add_argument("--out", default="results/top10_alpha_sweep",
                        help="Output path stem (writes <stem>.json).")
    args = parser.parse_args(argv)

    results: list[RolloutResult] = []
    for alpha in args.alphas:
        for seed in range(args.seeds):
            for baseline in ("greedy", "random"):
                res = run_rollout(
                    alpha, seed, baseline,
                    args.instances_dir, args.instance,
                    args.horizon, args.replan_every,
                )
                results.append(res)
                print(f"alpha={alpha:<4} seed={seed} {baseline:<6} "
                      f"cum_welfare={res.cumulative_welfare:12.2f} "
                      f"sum(x): {res.per_year_sum_activity[0]:.0f}->"
                      f"{res.per_year_sum_activity[-1]:.0f} "
                      f"frac_at_cap_end={res.per_year_frac_at_cap[-1]:.2f}")

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = REPO_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_json = out_path.with_suffix(".json")
    payload = {
        "config": {
            "instance": args.instance, "alphas": args.alphas,
            "seeds": args.seeds, "horizon": args.horizon,
            "replan_every": args.replan_every, "n_routes": N_ROUTES,
            "w_p_fixed": W_P_FIXED, "beta_gravity": BETA_GRAVITY,
            "cap_multiplier": CAP_MULTIPLIER,
        },
        "results": [asdict(r) for r in results],
    }
    out_json.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {out_json}")


if __name__ == "__main__":
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    main()
