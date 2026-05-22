"""End-to-end smoke test for the M2 / TOP-9 seam.

Loads a Mumford instance, picks a hand-built fixed bus route, and runs
T years of:

    LandUseDynamics.step  ->  recompute_demand_in_place  ->
    RouteGenBatchState    ->  MyCostModule.forward

then prints per-year cost + per-year total demand. This verifies that the
demand-recomputation hook actually feeds MyCostModule and that the cost
changes year-over-year under positive alpha (and stays constant under
alpha = 0, the static-Holliday baseline).

Run from the repo root::

    python -m learning.city_builder.smoke_drive_dynamics \
        --instance Mandl --alpha 0.0 --horizon 5
    python -m learning.city_builder.smoke_drive_dynamics \
        --instance Mandl --alpha 0.5 --horizon 5

The driver is intentionally minimal: a single fixed route, no policy. The
purpose is to show the seam works, not to do research. The TOP-10
alpha-sweep harness will replace the fixed route with policy rollouts.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from simulation.citygraph_dataset import CityGraphData
from simulation.transit_time_estimator import MyCostModule, RouteGenBatchState
from torch_utils import get_batch_tensor_from_routes

from learning.city_builder import (
    LandUseConfig,
    LandUseDynamics,
    recompute_demand_in_place,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INSTANCES_DIR = REPO_ROOT / "datasets" / "mumford_dataset" / "Instances"


def _initial_activity_from_od(od: torch.Tensor) -> torch.Tensor:
    """Bootstrap a per-zone activity proxy from a Mumford-style OD matrix.

    Mumford data has no separate population vector. Total demand out of zone
    i (row sum of OD) is a reasonable activity proxy for the smoke test --
    the M4 Bogota adapter will replace this with calibrated x_0 from DANE
    estimates."""
    return od.sum(dim=1)


def _build_fixed_route(num_nodes: int, route_len: int = 6) -> torch.Tensor:
    """Hand-built bus route covering the first `route_len` zones."""
    stops = list(range(min(route_len, num_nodes)))
    return get_batch_tensor_from_routes([[stops]])  # (1, 1, route_len)


def _evaluate(data: CityGraphData, cost_obj: MyCostModule,
              fixed_route_batch: torch.Tensor) -> torch.Tensor:
    """Build a RouteGenBatchState with the fixed route and return scalar cost."""
    state = RouteGenBatchState(
        data, cost_obj, n_routes_to_plan=1,
        min_route_len=2, max_route_len=fixed_route_batch.shape[-1],
    )
    # Add the route. add_new_routes accepts a (batch, n_routes, max_len)
    # tensor (batch dim = state.batch_size, here = 1).
    state.add_new_routes(fixed_route_batch)
    with torch.no_grad():
        out = cost_obj(state)
    return out.cost


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--instance", default="Mandl",
        help="Mumford instance name (Mandl, Mumford0, ...)",
    )
    parser.add_argument(
        "--instances-dir", default=str(DEFAULT_INSTANCES_DIR),
        help="Directory containing <instance>{Coords,TravelTimes,Demand}.txt",
    )
    parser.add_argument("--alpha", type=float, default=0.0,
                        help="Induced-demand strength.")
    parser.add_argument("--sigma-eps", type=float, default=0.0,
                        help="Per-zone activity noise stddev.")
    parser.add_argument("--horizon", type=int, default=5,
                        help="Number of years to simulate.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--route-len", type=int, default=6)
    args = parser.parse_args(argv)

    # ----- load city ----------------------------------------------------
    data = CityGraphData.from_mumford_data(
        args.instances_dir, instance_name=args.instance,
    )
    n_nodes = data.demand.shape[0]
    print(f"Loaded {args.instance}: {n_nodes} nodes, "
          f"{int((data.demand > 0).sum())} non-zero OD pairs.")

    # ----- bootstrap activity from OD row sums --------------------------
    x_0 = _initial_activity_from_od(data.demand)
    print(f"Initial activity range: [{x_0.min():.1f}, {x_0.max():.1f}]")

    # ----- dynamics -----------------------------------------------------
    dyn = LandUseDynamics(
        initial_activity=x_0,
        config=LandUseConfig(
            alpha=args.alpha,
            base_rate=1.0,
            sigma_eps=args.sigma_eps,
        ),
        seed=args.seed,
    )

    # ----- cost module + fixed route -----------------------------------
    cost_obj = MyCostModule(symmetric_routes=True)
    fixed_route_batch = _build_fixed_route(n_nodes, route_len=args.route_len)

    # Rewrite demand once at t=0 so all rows below report the gravity-model
    # demand, not the raw Mumford OD. This makes the alpha=0 invariance
    # check honest: with alpha=0 every subsequent year recomputes the same
    # gravity demand and therefore the same cost.
    recompute_demand_in_place(data, x_0, beta=2.0)

    # ----- per-year loop -----------------------------------------------
    print(f"\n  year |   sum(x) |  sum(demand) |       cost")
    print(f"  -----+----------+--------------+-----------")
    x = x_0.clone()
    for t in range(args.horizon + 1):
        cost = _evaluate(data, cost_obj, fixed_route_batch)
        print(f"   {t:3d} | {x.sum():8.1f} | {data.demand.sum():12.1f} | "
              f"{cost.item():10.4f}")
        if t == args.horizon:
            break
        x, _ = dyn.step(x, data.drive_times)
        recompute_demand_in_place(data, x, beta=2.0)
    print()


if __name__ == "__main__":
    # Make sure the repo root is on sys.path when run as `python -m ...`
    # from an arbitrary cwd.
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    main()
