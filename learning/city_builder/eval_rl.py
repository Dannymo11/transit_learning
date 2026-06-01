"""TOP-12 Stage 2 — evaluate a trained RL policy vs the baselines, apples-to-apples.

The trained GNN policy builds routes by SAMPLING (Holliday's LC-100 = best of N
samples), so we sample N rollouts and keep the best (plus the mean).

Crucially, we do NOT trust the cost read off the live, incrementally-built
RouteGenBatchState (it stays pinned at the empty-network value here). Instead we
EXTRACT the routes the policy builds and replay them through CityBuilderEnv --
the exact `add_new_routes` welfare path used to score the greedy/random
baselines (which correctly goes 12 -> ~1). That makes the RL number directly
comparable to the baselines.

Run:
    python -m learning.city_builder.eval_rl --config-name=ppo_citybuilder_mandl \
        +model.weights=/checkpoints/weights/citybuilder_ppo_citybuilder_mandl_seed0.pt
"""
from __future__ import annotations

import sys
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig
from torch_geometric.loader import DataLoader

from simulation.citygraph_dataset import get_dataset_from_config
import learning.utils as lrnu
import learning.inductive_route_learning as irl

from learning.city_builder.train_rl import _make_episode_state, _advance_year
from learning.city_builder.multi_year_mdp import (
    CityBuilderEnv, greedy_policy, make_random_policy, run_episode,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def rl_build_routes(model, cost_obj, template, cfg, device, greedy):
    """Run one RL rollout and return the list of route node-lists it built
    (one completed route per year). Welfare is NOT measured here -- see replay."""
    T = int(cfg.eval.n_routes)
    horizon = int(cfg.ppo.horizon)
    beta = cfg.city_builder.beta_gravity

    state, dyn, x_0 = _make_episode_state(template, cost_obj, T, cfg, device)
    state = model.setup_planning(state)
    x = x_0.clone()
    n_fin_prev = 0
    with torch.no_grad():
        for _ in range(horizon):
            actions, _, _ = model.step(state, greedy=greedy)
            state.shortest_path_action(actions)
            n_fin = int(state.n_finished_routes.sum().item())
            if n_fin > n_fin_prev:
                x = _advance_year(dyn, state, x, beta)
                n_fin_prev = n_fin
            # Break BEFORE reset_dones: reset_dones() -> _clear_routes_helper()
            # wipes _finished_routes for any is_done() episode, which would
            # erase the very network we're about to read.
            if state.is_done().all():
                break
            state.reset_dones()
    # state.routes[0]: list of finished route tensors (batch=1), already trimmed.
    return [[int(n) for n in r.tolist()] for r in state.routes[0]]


def _edges_between(prev, cur):
    """Ordered new transit edges that turn route `prev` into `cur`. Handles a
    shortest_path_action that extends at the BACK (prev is a prefix of cur), at
    the FRONT (prev is a suffix of cur), or adds several stops at once; falls
    back to all of `cur`'s edges if the route changed wholesale."""
    if cur == prev or len(cur) < 2:
        return []
    if not prev:
        return [(cur[i], cur[i + 1]) for i in range(len(cur) - 1)]
    if cur[:len(prev)] == prev:                      # appended at back
        return [(cur[i], cur[i + 1]) for i in range(len(prev) - 1, len(cur) - 1)]
    if cur[-len(prev):] == prev:                     # prepended at front
        add = len(cur) - len(prev)
        return [(cur[i], cur[i + 1]) for i in range(add)]   # added internal + join
    return [(cur[i], cur[i + 1]) for i in range(len(cur) - 1)]


def rl_build_routes_logged(model, cost_obj, template, cfg, device, greedy):
    """Like rl_build_routes, but also returns the TRUE per-step construction log:
    one entry per transit segment the policy adds, in the exact order it adds it.

    Each `model.step` + `shortest_path_action` is one routing action; we snapshot
    state.current_routes / state._finished_routes after every action and diff to
    recover which route grew and which stop(s) it gained. Returns
    (routes, construction) where construction items are
    {"route", "from", "to", "partial", "action"} -- the schema the HTML scrubber
    and render_viz.py consume for the Phase-1 build animation."""
    T = int(cfg.eval.n_routes)
    horizon = int(cfg.ppo.horizon)
    beta = cfg.city_builder.beta_gravity

    state, dyn, x_0 = _make_episode_state(template, cost_obj, T, cfg, device)
    state = model.setup_planning(state)
    x = x_0.clone()
    n_fin_prev = 0

    construction = []
    built = []          # latest node-list per route, in build order
    active = -1
    prev_cur = []

    def cur_route():
        cr = state.current_routes[0]
        return [int(n) for n in cr[cr >= 0].tolist()]

    def finished_routes():
        return [[int(n) for n in r.tolist()] for r in state._finished_routes[0]]

    def emit(route_idx, prev, now):
        for a, b in _edges_between(prev, now):
            construction.append({"route": route_idx, "from": a, "to": b,
                                 "partial": list(now), "action": len(construction)})

    with torch.no_grad():
        for _ in range(horizon):
            actions, _, _ = model.step(state, greedy=greedy)
            state.shortest_path_action(actions)

            cur = cur_route()
            if cur:
                if not prev_cur:                 # a new route just began
                    active += 1
                    built.append([])
                emit(active, built[active], cur)
                built[active] = cur
                prev_cur = cur
            else:
                if prev_cur:                     # active route completed this step
                    fin = finished_routes()
                    final = fin[-1] if fin else prev_cur
                    emit(active, built[active], final)
                    built[active] = final
                prev_cur = []

            n_fin = int(state.n_finished_routes.sum().item())
            if n_fin > n_fin_prev:
                x = _advance_year(dyn, state, x, beta)
                n_fin_prev = n_fin
            if state.is_done().all():
                break
            state.reset_dones()

    routes = [[int(n) for n in r.tolist()] for r in state.routes[0]]
    return routes, construction


def replay_welfare(routes, cfg) -> float:
    """Cumulative welfare gain from replaying `routes` (one per year) through
    CityBuilderEnv -- the same add_new_routes path used for the baselines."""
    env = CityBuilderEnv(instance=cfg.eval.dataset.city,
                         alpha=cfg.city_builder.alpha,
                         horizon=int(cfg.eval.n_routes), seed=0)
    it = iter(routes)
    recs = run_episode(env, lambda e: next(it, None), verbose=False)
    return sum(r["reward"] for r in recs)


@hydra.main(version_base=None, config_path="../../cfg",
            config_name="ppo_citybuilder_mandl")
def main(cfg: DictConfig):
    assert cfg.model.get("weights", None), "set +model.weights=<checkpoint.pt>"
    device, _, _, cost_obj, model = lrnu.process_standard_experiment_cfg(
        cfg, "eval_rl_", weights_required=True)
    irl.DEVICE = device
    model.eval()
    cost_obj.variable_weights = False
    cost_obj.set_weights(demand_time_weight=0.5, route_time_weight=0.5)

    template = next(iter(DataLoader(
        get_dataset_from_config(cfg.eval.dataset), batch_size=1)))

    n_samples = int(cfg.get("n_samples", 50))
    welfares = []
    for s in range(n_samples):
        torch.manual_seed(1000 + s)
        routes = rl_build_routes(model, cost_obj, template, cfg, device,
                                 greedy=False)
        welfares.append(replay_welfare(routes, cfg))
    w = torch.tensor(welfares)
    best, mean, std = w.max().item(), w.mean().item(), w.std().item()

    greedy_routes = rl_build_routes(model, cost_obj, template, cfg, device,
                                    greedy=True)
    greedy_w = replay_welfare(greedy_routes, cfg)

    # Baselines through the SAME CityBuilderEnv path.
    env_g = CityBuilderEnv(instance=cfg.eval.dataset.city,
                           alpha=cfg.city_builder.alpha,
                           horizon=int(cfg.eval.n_routes), seed=0)
    base_greedy = sum(r["reward"] for r in run_episode(env_g, greedy_policy, verbose=False))
    env_r = CityBuilderEnv(instance=cfg.eval.dataset.city,
                           alpha=cfg.city_builder.alpha,
                           horizon=int(cfg.eval.n_routes), seed=0)
    base_random = sum(r["reward"] for r in run_episode(env_r, make_random_policy(0), verbose=False))

    print("\n=== TOP-12 RL eval (Mandl, alpha=%.2f; welfare via CityBuilderEnv) ==="
          % cfg.city_builder.alpha)
    print(f"  RL best-of-{n_samples} : {best:7.4f}")
    print(f"  RL mean-of-{n_samples} : {mean:7.4f} +/- {std:.4f}")
    print(f"  RL greedy        : {greedy_w:7.4f}")
    print(f"  greedy baseline  : {base_greedy:7.4f}")
    print(f"  random baseline  : {base_random:7.4f}")
    verdict = ("RL (best-of-N) BEATS both baselines"
               if best > max(base_greedy, base_random)
               else "RL (best-of-N) beats greedy only" if best > base_greedy
               else "RL does NOT beat baselines yet")
    print(f"  -> {verdict}")

    # --- optional: one-command viz export (true RL action order) -------------
    # python -m learning.city_builder.eval_rl ... +model.weights=<ckpt.pt> \
    #     +viz_out=results/viz_trajectory.json
    if cfg.get("viz_out", None):
        from learning.city_builder.export_viz_trajectory import build_payload, write_payload
        routes, construction = rl_build_routes_logged(
            model, cost_obj, template, cfg, device, greedy=True)
        alphas = list(cfg.get("viz_alphas", [0.0, 0.5, 1.0, 2.0]))
        horizon = int(cfg.eval.n_routes)
        seed = int(cfg.experiment.get("seed", 0))
        instance = cfg.eval.dataset.city
        payload = build_payload(
            instance, str(REPO_ROOT / "datasets" / "mumford_dataset" / "Instances"),
            routes, construction, alphas, horizon, seed,
            source_label="rl policy (true action order), real torch dynamics")
        write_payload(payload, cfg.viz_out)

    return best


if __name__ == "__main__":
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    main()
