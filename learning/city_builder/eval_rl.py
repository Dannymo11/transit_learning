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

from learning.city_builder.train_rl import (
    _make_episode_state, _advance_year, horizon_T, build_years_cfg,
)
from learning.city_builder.multi_year_mdp import (
    CityBuilderEnv, greedy_policy, make_random_policy, run_episode,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _make_env(cfg) -> CityBuilderEnv:
    """CityBuilderEnv configured for the instance under eval: per-instance route
    geometry (cfg.eval.min/max_route_len) and the configurable horizon
    (city_builder.horizon_years, via horizon_T). Used for both the RL route
    replay and the greedy/random baselines so all welfare numbers are scored
    identically."""
    return CityBuilderEnv(
        instance=cfg.eval.dataset.city,
        alpha=cfg.city_builder.alpha,
        horizon=horizon_T(cfg),
        build_years=build_years_cfg(cfg),
        min_route_len=int(cfg.eval.min_route_len),
        max_route_len=int(cfg.eval.max_route_len),
        cap_mode=str(cfg.city_builder.get("cap_mode", "proportional")),
        add_rate=float(cfg.city_builder.get("add_rate", 0.0)),
        cap_blend=(float(cfg.city_builder["cap_blend"])
                   if cfg.city_builder.get("cap_blend", None) is not None
                   else None),
        seed=0,
    )


def rl_build_routes(model, cost_obj, template, cfg, device, greedy):
    """Run one RL rollout and return the list of route node-lists it built
    (one completed route per year). Welfare is NOT measured here -- see replay."""
    T = build_years_cfg(cfg)   # build_years routes (== n_routes_to_plan); the
                               # watch years add no routes, only frozen dynamics
    horizon = int(cfg.ppo.horizon)
    beta = cfg.city_builder.beta_gravity

    state, dyn, x_0, _ = _make_episode_state(template, cost_obj, T, cfg, device)
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


def _step_record(sc, route_idx, before, cur, fin, action_idx):
    """Turn one stashed model._last_step_scores (batch=1) into a viewer `steps`
    entry: the candidate action distribution, the chosen move (or halt), the halt
    probability, entropy, and the resulting partial route / network-so-far."""
    n = sc["n_nodes"]
    probs = sc["ext_probs"][0].tolist()
    serial = sc["serial_halting"]
    halt_prob = float(sc["halt_prob"][0])
    folded = [int(v) for v in sc["folded_idxs"][0].tolist()]
    chose_halt = bool(sc["chose_halt"][0])
    entropy = float(sc["entropy"][0])

    scale = (1.0 - halt_prob) if serial else 1.0   # serial ext-probs are conditional
    cands = []
    for f in range(n * n):                          # ignore any trailing joint-halt idx
        p = probs[f] * scale
        if p > 1e-3:
            cands.append({"from": f // n, "to": f % n, "prob": float(p)})
    cands.sort(key=lambda d: -d["prob"])
    cands = cands[:12]

    frm, to = (-1, -1) if chose_halt else (folded[0], folded[1])
    partial_after = cur if cur else (fin[-1] if fin else before)
    network = [list(map(int, r)) for r in fin] + ([list(map(int, cur))] if cur else [])
    label = "halt" if chose_halt else ("start" if len(before) == 0 else "extend")
    # route_idx == the year being built (one completed route per year), so the
    # interleaved per-year view keys off `year`.
    return {"action": action_idx, "route": int(route_idx), "year": int(route_idx),
            "phase_label": label, "from": frm, "to": to, "halt": chose_halt,
            "halt_prob": halt_prob, "entropy": round(entropy, 3),
            "partial_after": list(map(int, partial_after)),
            "network_so_far": network, "candidates": cands}


def rl_build_routes_logged(model, cost_obj, template, cfg, device, greedy):
    """Like rl_build_routes, but also returns the TRUE per-step logs.

    Each `model.step` + `shortest_path_action` is one routing action. We (a)
    snapshot state.current_routes / state._finished_routes after each action and
    diff them to recover the build order (`construction`, edge-level), and (b)
    read the policy's stashed action distribution (model._last_step_scores,
    enabled via model.log_scores) to record what the agent SAW and DECIDED at each
    action (`steps`, action-level -- the policy-inspector schema the HTML scrubber
    and render_viz.py consume).

    Returns (routes, construction, steps)."""
    T = build_years_cfg(cfg)   # build_years routes (== n_routes_to_plan); the
                               # watch years add no routes, only frozen dynamics
    horizon = int(cfg.ppo.horizon)
    beta = cfg.city_builder.beta_gravity

    state, dyn, x_0, _ = _make_episode_state(template, cost_obj, T, cfg, device)
    state = model.setup_planning(state)
    x = x_0.clone()
    n_fin_prev = 0

    construction, steps = [], []
    year_activity = [x.detach().cpu().numpy().tolist()]   # x_t the agent observes each year
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

    prev_log_scores = getattr(model, "log_scores", False)
    model.log_scores = True
    try:
        with torch.no_grad():
            for _ in range(horizon):
                before = cur_route()
                route_idx = len(state._finished_routes[0])
                actions, _, _ = model.step(state, greedy=greedy)
                sc = getattr(model, "_last_step_scores", None)
                state.shortest_path_action(actions)

                cur = cur_route()
                fin = finished_routes()
                if cur:
                    if not prev_cur:                 # a new route just began
                        active += 1
                        built.append([])
                    emit(active, built[active], cur)
                    built[active] = cur
                    prev_cur = cur
                else:
                    if prev_cur:                     # active route completed this step
                        final = fin[-1] if fin else prev_cur
                        emit(active, built[active], final)
                        built[active] = final
                    prev_cur = []

                if sc is not None:
                    steps.append(_step_record(sc, route_idx, before, cur, fin, len(steps)))

                n_fin = int(state.n_finished_routes.sum().item())
                if n_fin > n_fin_prev:
                    x = _advance_year(dyn, state, x, beta)
                    year_activity.append(x.detach().cpu().numpy().tolist())
                    n_fin_prev = n_fin
                if state.is_done().all():
                    break
                state.reset_dones()
    finally:
        model.log_scores = prev_log_scores

    routes = [[int(n) for n in r.tolist()] for r in state.routes[0]]
    return routes, construction, steps, year_activity


def replay_welfare(routes, cfg) -> float:
    """Telescoped welfare gain from replaying `routes` (one per year) through
    CityBuilderEnv -- the same add_new_routes path used for the baselines.
    Kept for backward compatibility; equals replay_metrics(...)['delta']."""
    return replay_metrics(routes, cfg)["delta"]


def replay_metrics(routes, cfg) -> dict:
    """Replay `routes` (one per year) through CityBuilderEnv and return BOTH
    welfare functionals from the SAME rollout:

      * 'delta'      : sum_t r_t = C(s_0) - C(s_T), the telescoped endpoint cost
                       reduction (the original metric). Demand magnitude scales
                       both endpoints, so this is nearly alpha-insensitive.
      * 'integrated' : -sum_{t=1..T} C(s_t), the M2-style sum of per-year cost
                       LEVELS (welfare experienced over the operating horizon).
                       Scales with demand, so this is the alpha-sensitive metric.
                       (Year 0 is an alpha-independent constant, omitted; it only
                       shifts every curve equally and cancels in any gap.)
    """
    env = _make_env(cfg)
    it = iter(routes)
    recs = run_episode(env, lambda e: next(it, None), verbose=False)
    return {
        "delta": sum(r["reward"] for r in recs),
        "integrated": -sum(r["welfare_cost"] for r in recs),
    }


def run_eval(model, cost_obj, template, cfg, device, n_samples: int) -> dict:
    """Score the trained policy vs the baselines at the CURRENT
    cfg.city_builder.alpha and return a machine-readable results dict.

    All five welfare numbers go through the SAME CityBuilderEnv add_new_routes
    path so they are directly comparable:
      * rl_best / rl_mean / rl_std : best / mean / std over `n_samples` SAMPLED
        rollouts (Holliday LC-N; headline N=100);
      * rl_greedy : the policy's own argmax (greedy-decode) rollout;
      * base_greedy / base_random : the john_init and random-walk baselines.

    The gaps are RL minus the GREEDY baseline (the strong baseline): gap_best is
    the headline curve, gap_mean is the honest/expected companion whose per-seed
    spread feeds the significance test, gap_greedy is the deterministic
    apples-to-apples reference.
    """
    alpha = float(cfg.city_builder.alpha)
    horizon = horizon_T(cfg)            # T (build-years), configurable

    # Sample N rollouts; score each on BOTH functionals from one replay.
    deltas, ints = [], []
    for s in range(n_samples):
        torch.manual_seed(1000 + s)
        routes = rl_build_routes(model, cost_obj, template, cfg, device,
                                 greedy=False)
        m = replay_metrics(routes, cfg)
        deltas.append(m["delta"]); ints.append(m["integrated"])

    greedy_routes = rl_build_routes(model, cost_obj, template, cfg, device,
                                    greedy=True)
    gm = replay_metrics(greedy_routes, cfg)

    # Baselines through the SAME CityBuilderEnv path, both functionals.
    def _baseline(policy) -> dict:
        env = _make_env(cfg)
        recs = run_episode(env, policy, verbose=False)
        return {"delta": sum(r["reward"] for r in recs),
                "integrated": -sum(r["welfare_cost"] for r in recs)}

    bg = _baseline(greedy_policy)
    br = _baseline(make_random_policy(0))

    out = {
        "alpha": alpha,
        "seed": int(cfg.experiment.get("seed", 0)),
        "instance": cfg.eval.dataset.city,
        "horizon": horizon,
        "n_samples": n_samples,
    }
    # Emit both metric families. "" = telescoped delta (original);
    # "_int" = M2-style integrated level sum (the alpha-sensitive metric).
    for suffix, key in (("", "delta"), ("_int", "integrated")):
        w = torch.tensor([d if key == "delta" else i
                          for d, i in zip(deltas, ints)])
        best, mean, std = w.max().item(), w.mean().item(), w.std().item()
        greedy_w = gm[key]
        base_greedy, base_random = bg[key], br[key]
        out.update({
            f"rl_best{suffix}": best,
            f"rl_mean{suffix}": mean,
            f"rl_std{suffix}": std,
            f"rl_greedy{suffix}": greedy_w,
            f"base_greedy{suffix}": base_greedy,
            f"base_random{suffix}": base_random,
            f"gap_best{suffix}": best - base_greedy,
            f"gap_mean{suffix}": mean - base_greedy,
            f"gap_greedy{suffix}": greedy_w - base_greedy,
        })
    return out


def _print_eval(res: dict) -> None:
    n = res["n_samples"]
    print("\n=== TOP-12 RL eval (%s, alpha=%.2f; welfare via CityBuilderEnv) ==="
          % (res["instance"], res["alpha"]))
    for label, suffix in (("delta C(s0)-C(sT)", ""),
                          ("integrated -sum_t C", "_int")):
        print(f"  -- {label} --")
        print(f"  RL best-of-{n} : {res['rl_best'+suffix]:8.4f}")
        print(f"  RL mean-of-{n} : {res['rl_mean'+suffix]:8.4f} "
              f"+/- {res['rl_std'+suffix]:.4f}")
        print(f"  RL greedy        : {res['rl_greedy'+suffix]:8.4f}")
        print(f"  greedy baseline  : {res['base_greedy'+suffix]:8.4f}")
        print(f"  random baseline  : {res['base_random'+suffix]:8.4f}")
        print(f"  gap best-greedy  : {res['gap_best'+suffix]:+8.4f}   "
              f"gap mean-greedy: {res['gap_mean'+suffix]:+8.4f}")


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
    res = run_eval(model, cost_obj, template, cfg, device, n_samples)
    _print_eval(res)
    best = res["rl_best"]

    # Machine-readable output for the alpha-ablation driver (one JSON per
    # (alpha, seed) run; the driver pools them across the grid).
    if cfg.get("eval_out", None):
        import json
        out_path = Path(cfg.eval_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(res, indent=2))
        print(f"  wrote {out_path}")

    # --- optional: one-command viz export (true RL action order) -------------
    # python -m learning.city_builder.eval_rl ... +model.weights=<ckpt.pt> \
    #     +viz_out=results/viz_trajectory.json
    if cfg.get("viz_out", None):
        from learning.city_builder.export_viz_trajectory import build_payload, write_payload
        routes, construction, steps, year_activity = rl_build_routes_logged(
            model, cost_obj, template, cfg, device, greedy=True)
        alphas = list(cfg.get("viz_alphas", [0.0, 0.5, 1.0, 2.0]))
        horizon = horizon_T(cfg)
        seed = int(cfg.experiment.get("seed", 0))
        instance = cfg.eval.dataset.city
        payload = build_payload(
            instance, str(REPO_ROOT / "datasets" / "mumford_dataset" / "Instances"),
            routes, construction, alphas, horizon, seed,
            source_label="rl policy (true action order + decision inspector), real torch dynamics",
            steps=steps, rollout_activity=year_activity,
            rollout_alpha=float(cfg.city_builder.alpha))
        write_payload(payload, cfg.viz_out)

    return best


if __name__ == "__main__":
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    main()
