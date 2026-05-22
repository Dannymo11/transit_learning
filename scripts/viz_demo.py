"""
Quick visual smoke test of the existing matplotlib helpers.

Generates one synthetic city, picks a few random shortest-path "routes"
(no policy needed), and saves PNGs of:
    1. the city: streets (black) + demand overlay (red dashed)
    2. the same city with the random routes coalesced onto it
    3. the same city with each route drawn in a distinct colour

Run from the repo root with the project venv active:

    python scripts/viz_demo.py
    python scripts/viz_demo.py --n-nodes 40 --n-routes 5 --seed 1

Outputs go to ./viz_outputs/.
"""

from __future__ import annotations

import argparse
import random
import sys
from itertools import cycle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe; works over SSH too
import matplotlib.pyplot as plt
import networkx as nx
import torch
from torch_geometric.data import Data
import torch_geometric.utils as pygu

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simulation.citygraph_dataset import (
    DynamicCityGraphDataset,
    MIXED,
    STOP_KEY,
)
from simulation.drawing import draw_coalesced_routes
from simulation.transit_time_estimator import MyCostModule, RouteGenBatchState
from learning.initialization import nikolic_init


def reconstruct_path(nexts: torch.Tensor, src: int, dst: int) -> torch.Tensor:
    """Walk the precomputed next-hop matrix to reconstruct a shortest path."""
    path = [src]
    while src != dst:
        src = int(nexts[src, dst].item())
        path.append(src)
    return torch.tensor(path, dtype=torch.long)


def sample_random_routes(
    graph, n_routes: int, rng: random.Random
) -> list[torch.Tensor]:
    """Pick `n_routes` random terminal pairs and return their shortest paths."""
    n = graph.num_nodes
    nexts = graph.nexts
    routes: list[torch.Tensor] = []
    seen: set[tuple[int, int]] = set()
    attempts = 0
    while len(routes) < n_routes and attempts < n_routes * 50:
        attempts += 1
        s, d = rng.sample(range(n), 2)
        key = tuple(sorted((s, d)))
        if key in seen:
            continue
        seen.add(key)
        path = reconstruct_path(nexts, s, d)
        if path.numel() >= 3:  # skip trivial 2-node "routes"
            routes.append(path)
    return routes


def nikolic_routes(
    graph,
    n_routes: int,
    min_route_len: int = 2,
    max_route_len: int | None = None,
) -> list[torch.Tensor]:
    """Build a transit network with Nikolic & Teodorović (2013) — at each step
    pick the shortest path that satisfies the most still-uncovered demand."""
    cost_obj = MyCostModule(symmetric_routes=True)
    state = RouteGenBatchState(
        graph,
        cost_obj,
        n_routes_to_plan=n_routes,
        min_route_len=min_route_len,
        max_route_len=max_route_len if max_route_len is not None else graph.num_nodes,
    )
    # shape: (batch=1, n_routes, max_n_nodes), padded with -1
    networks = nikolic_init(state)
    routes: list[torch.Tensor] = []
    for row in networks[0]:
        valid = row[row > -1]
        if valid.numel() >= 2:
            routes.append(valid)
    return routes


def draw_routes_coloured(node_locs: torch.Tensor, routes, ax) -> None:
    """Each route in a distinct colour, all overlaid on the same node layout."""
    locs = node_locs.cpu().numpy()
    base = nx.Graph()
    base.add_nodes_from(range(len(locs)))
    nx.draw_networkx_nodes(base, pos=locs, node_size=40, node_color="black", ax=ax)

    colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for route, colour in zip(routes, cycle(colours)):
        rg = base.copy()
        edges = [
            (int(a.item()), int(b.item()))
            for a, b in torch.stack((route[:-1], route[1:]), dim=-1)
        ]
        rg.add_edges_from(edges)
        nx.draw_networkx_edges(rg, pos=locs, edge_color=colour, width=3, ax=ax)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-nodes", type=int, default=20)
    ap.add_argument("--n-routes", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--baseline",
        choices=["nikolic", "random"],
        default="nikolic",
        help="how to construct the routes to visualize "
        "('nikolic' = greedy demand-coverage from learning/initialization.py; "
        "'random' = random shortest paths between terminal pairs)",
    )
    ap.add_argument("--out", type=Path, default=Path("viz_outputs"))
    args = ap.parse_args()

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    args.out.mkdir(exist_ok=True)

    ds = DynamicCityGraphDataset(
        min_nodes=args.n_nodes,
        max_nodes=args.n_nodes,
        data_type=MIXED,
        fully_connected_demand=True,
    )
    graph = next(iter(ds))
    print(f"generated city with {graph.num_nodes} nodes")

    # --- 1. City: streets + demand ------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 8))
    graph.draw(ax=ax, node_size=80, show_demand=True)
    ax.set_title(f"Synthetic city, {graph.num_nodes} nodes — streets + demand")
    fig.tight_layout()
    fig.savefig(args.out / "city.png", dpi=130)
    plt.close(fig)
    print(f"  wrote {args.out / 'city.png'}")

    # --- 2. Routes ----------------------------------------------------------
    if args.baseline == "nikolic":
        routes = nikolic_routes(graph, args.n_routes)
        baseline_label = "Nikolic-greedy"
    else:
        routes = sample_random_routes(graph, args.n_routes, rng)
        baseline_label = "random shortest-path"
    print(f"built {len(routes)} {baseline_label} routes")

    pos = graph[STOP_KEY].pos

    fig, ax = plt.subplots(figsize=(8, 8))
    graph.draw(ax=ax, node_size=40, show_demand=False)  # streets only as backdrop
    draw_coalesced_routes(pos, routes, ax=ax)
    ax.set_title(f"{len(routes)} {baseline_label} routes (coalesced)")
    fig.tight_layout()
    fig.savefig(args.out / "routes_coalesced.png", dpi=130)
    plt.close(fig)
    print(f"  wrote {args.out / 'routes_coalesced.png'}")

    fig, ax = plt.subplots(figsize=(8, 8))
    draw_routes_coloured(pos, routes, ax=ax)
    ax.set_title(f"{len(routes)} {baseline_label} routes (per-route colour)")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(args.out / "routes_coloured.png", dpi=130)
    plt.close(fig)
    print(f"  wrote {args.out / 'routes_coloured.png'}")


if __name__ == "__main__":
    main()
