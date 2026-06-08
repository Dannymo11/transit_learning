"""TOP-12 Stage 0 — train a GNN policy with PPO under the closed induced-demand
loop (Mandl, batch_size=1).

This is a focused fork of Holliday's `inductive_route_learning.train_ppo`: the
PPO rollout + GAE + clipped update are reused as-is; the ONE addition is that
after each transit route is completed, the city advances one year via
`step_world` (closed loop: growth follows the network's transit accessibility,
gravity demand stays on street times). See docs/M3_TOP-12_rl_training_design.md.

Mapping (design doc sec 3-4):
  * each `model.step` + `state.shortest_path_action` is a route SEGMENT step;
  * a route completes on a halt action -> `state.n_finished_routes` increments;
  * a YEAR = one completed route; the land-use dynamics fire on completion;
  * n_routes_to_plan = T years.

Stage 0 keeps batch_size=1 so the dynamics insertion is single-instance and
matches the validated CityBuilderEnv path exactly. Stage 1 generalizes to
batched dynamics. This file deliberately drops train_ppo's optuna / dataloader /
n-routes-randomization machinery; the PPO math is copied verbatim.

Run (GPU on Modal recommended):
    python -m learning.city_builder.train_rl --config-name=ppo_citybuilder_mandl \
        +run_name=cb_mandl_seed0 experiment.seed=0
"""
from __future__ import annotations

import copy
import logging as log
import os
import sys
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig
from tqdm import tqdm

from torch_geometric.loader import DataLoader
from simulation.citygraph_dataset import (
    get_dataset_from_config, DEMAND_KEY, DMD_FEAT_IDX, SHORTESTPATH_FEAT_IDX,
)
from simulation.transit_time_estimator import RouteGenBatchState
from torch_utils import get_batch_tensor_from_routes
import learning.utils as lrnu
import learning.inductive_route_learning as irl
from learning.inductive_route_learning import NNBaseline

from learning.city_builder import LandUseConfig, LandUseDynamics, gravity_demand

REPO_ROOT = Path(__file__).resolve().parents[2]


def _squeeze_b1(mat: torch.Tensor) -> torch.Tensor:
    """(1,N,N) or (N,N) -> (N,N). RouteGenBatchState always batches graph_data,
    so demand/drive_times/transit_times are (B,N,N); Stage 0 is batch=1."""
    return mat[0] if mat.dim() == 3 else mat


def _initial_activity_b1(demand: torch.Tensor) -> torch.Tensor:
    """Per-zone out-demand (N,) from a (1,N,N) or (N,N) OD matrix."""
    return _squeeze_b1(demand).sum(dim=1)


def _write_gravity_demand_b1(graph_data, x: torch.Tensor, beta: float):
    """Write gravity demand under activity `x` into the batched (B=1) graph_data,
    in place: the (1,N,N) `demand` matrix and the DEMAND_KEY edge features the
    GNN reads. Street times stay the gravity basis (closed loop only changes the
    ACCESSIBILITY basis, handled in _advance_year)."""
    street = _squeeze_b1(graph_data.drive_times)            # (N,N)
    demand = gravity_demand(x, street, beta=beta)           # (N,N)
    graph_data.demand = (demand.unsqueeze(0)
                         if graph_data.demand.dim() == 3 else demand)
    dmd = graph_data[DEMAND_KEY]
    ei = dmd.edge_index                                     # batch=1 -> node idx
    dmd.edge_attr[:, DMD_FEAT_IDX] = demand[ei[0], ei[1]]
    dmd.edge_attr[:, SHORTESTPATH_FEAT_IDX] = street[ei[0], ei[1]]


def _advance_year(dyn: LandUseDynamics, state: RouteGenBatchState,
                  x: torch.Tensor, beta: float) -> torch.Tensor:
    """One year of closed-loop dynamics on the LIVE state (batch=1): grow
    activity under the network's TRANSIT accessibility, then rewrite gravity
    demand (street-based) into state.graph_data so the next model.step sees it."""
    transit = _squeeze_b1(state.transit_times)              # (N,N)
    x_next, _ = dyn.step(x, transit)
    _write_gravity_demand_b1(state.graph_data, x_next, beta)
    return x_next


def horizon_T(cfg) -> int:
    """T = number of build-years (one route per year, == n_routes_to_plan).
    Configurable via city_builder.horizon_years so the horizon is decoupled from
    the instance's benchmark route count (cfg.eval.n_routes); falls back to
    eval.n_routes when horizon_years is absent (Mandl back-compat)."""
    hy = cfg.city_builder.get("horizon_years", None)
    return int(hy) if hy is not None else int(cfg.eval.n_routes)


def build_years_cfg(cfg) -> int:
    """Number of route-building years (== n_routes_to_plan for the policy). In the
    build-then-watch regime this is < horizon_T(cfg): routes are built in years
    [0, build_years), then the network FREEZES for the remaining watch years while
    dynamics keep advancing. Defaults to horizon_T(cfg) => rebuild-every-year
    (no watch phase; original behavior)."""
    by = cfg.city_builder.get("build_years", None)
    return int(by) if by is not None else horizon_T(cfg)


def _make_episode_state(template, cost_obj, T, cfg, device):
    """Fresh episode: reset demand to gravity under x_0, build the state, and
    return (state, dyn, x_0). Mirrors CityBuilderEnv.reset. `template` is a
    batched (num_graphs=1) graph so the GNN's to_data_list works."""
    data = copy.deepcopy(template).to(device)
    beta = cfg.city_builder.beta_gravity
    x_0 = _initial_activity_b1(data.demand)
    _write_gravity_demand_b1(data, x_0, beta)

    cost_weights = cost_obj.sample_variable_weights(data.num_graphs, device)
    state = RouteGenBatchState(
        data, cost_obj, T, cfg.eval.min_route_len, cfg.eval.max_route_len,
        cost_weights=cost_weights)

    dyn = LandUseDynamics(
        initial_activity=x_0,
        config=LandUseConfig(
            alpha=cfg.city_builder.alpha, base_rate=1.0,
            sigma_eps=cfg.city_builder.sigma_eps,
            cap_multiplier=cfg.city_builder.cap_multiplier,
            cap_mode=str(cfg.city_builder.get("cap_mode", "proportional")),
            add_rate=float(cfg.city_builder.get("add_rate", 0.0)),
            cap_blend=(float(cfg.city_builder["cap_blend"])
                       if cfg.city_builder.get("cap_blend", None) is not None
                       else None),
            beta_accessibility=beta),
        seed=int(cfg.experiment.get("seed", 0)),
    )
    return state, dyn, x_0, cost_weights


def _realized_network_cost(state, cost_obj, cfg, cost_weights) -> torch.Tensor:
    """Welfare cost C(s_y) sized to the REALIZED network, mirroring
    eval_rl/_welfare_cost: n_routes_to_plan = number of FINISHED routes (>=1),
    NOT the full horizon T. Built fresh from the live state's current demand
    (state.graph_data, already post-growth at the call site) and finished routes,
    with the episode's cost_weights.

    This is the fix for the train/eval cost divergence the parity test caught:
    the live planning state keeps n_routes_to_plan=T for decoding, but the cost
    module penalizes the (T - n_finished) empty planned-route slots (route-cost
    normalizer + OOB-stops term, transit_time_estimator ~L1056-1100), inflating
    the reward off a surface the eval metric / baselines are never scored on.
    Sizing the reward cost to the realized network removes that artifact and makes
    training-side integrated_welfare equal eval_rl.replay_metrics()['integrated'].
    """
    finished = [[int(n) for n in r] for r in state._finished_routes[0]]  # batch=1
    n_routes = max(len(finished), 1)
    rgs = RouteGenBatchState(
        state.graph_data, cost_obj, n_routes,
        int(cfg.eval.min_route_len), int(cfg.eval.max_route_len),
        cost_weights={k: v for k, v in cost_weights.items()})
    if finished:
        # get_batch_tensor_from_routes builds on CPU; the state/graph live on the
        # training device (cuda). Move the net over before add_new_routes, or the
        # cost module hits a cross-device op. (Eval never saw this -- its env is
        # CPU-only.)
        net = get_batch_tensor_from_routes(
            [finished], max_route_len=int(cfg.eval.max_route_len)).to(state.device)
        rgs.add_new_routes(net)
    return cost_obj(rgs).cost


def _no_build_cost(graph_data, cost_obj, cfg, cost_weights) -> torch.Tensor:
    """C_ref: cost of the EMPTY network under graph_data's CURRENT demand, sized
    as eval scores an empty network (n_routes_to_plan=1, no routes), using the
    episode's SAME cost_weights. No-build counterfactual baseline subtracted in
    the 'integrated_advantage' reward mode."""
    ref = RouteGenBatchState(
        graph_data, cost_obj, 1,
        int(cfg.eval.min_route_len), int(cfg.eval.max_route_len),
        cost_weights={k: v for k, v in cost_weights.items()})
    return cost_obj(ref).cost


class RunningMeanStd:
    """Welford running mean/variance, persisted ACROSS episodes/iterations.
    Used to normalize reward scale (see reward_norm): rewards are divided by the
    running std of the discounted return so the critic regresses on unit-scale
    targets and GAE deltas stay well-conditioned under non-stationary demand.
    Advantages are still batch-normalized downstream, so the policy gradient is
    scale-invariant; this only conditions the value function."""
    def __init__(self, eps: float = 1e-4):
        self.mean = 0.0
        self.var = 1.0
        self.count = eps

    def update(self, xs) -> None:
        import numpy as np
        xs = np.asarray(list(xs), dtype=np.float64)
        if xs.size == 0:
            return
        bm, bv, bc = float(xs.mean()), float(xs.var()), int(xs.size)
        d = bm - self.mean
        tot = self.count + bc
        self.mean += d * bc / tot
        m2 = self.var * self.count + bv * bc + d * d * self.count * bc / tot
        self.var = m2 / tot
        self.count = tot

    @property
    def std(self) -> float:
        return max(self.var, 1e-8) ** 0.5


def _collect_episode(model, cost_obj, val_module, template, cfg, device,
                     ret_rms=None):
    """Run one episode (batch=1) with the validated single-instance dynamics and
    return its per-step buffers + GAE. Stage 1 calls this n_episodes_per_iter
    times and pools the results into one PPO update for lower-variance gradients."""
    T_total = horizon_T(cfg)                  # total welfare horizon (years)
    build_years = build_years_cfg(cfg)        # routes built == n_routes_to_plan
    watch_years = max(T_total - build_years, 0)  # frozen-network watch years (BTW)
    horizon = int(cfg.ppo.horizon)
    gamma = cfg.discount_rate
    return_scale = cfg.get("reward_scale", 1.0)
    beta = cfg.city_builder.beta_gravity
    # Welfare functional the reward encodes:
    #   "delta"      -> r_t = C(s_{t-1}) - C(s_t) per SEGMENT; sums to the
    #                   telescoped endpoint reduction C(s_0) - C(s_T) (original,
    #                   nearly alpha-insensitive).
    #   "integrated" -> r = -C(s_y) paid ONCE at each YEAR completion (after the
    #                   dynamics advance, under the post-growth demand); sums to
    #                   -sum_{y=1..T} C(s_y), the M2-style integrated welfare that
    #                   scales with demand and so is alpha-sensitive. Matches
    #                   eval_rl.replay_metrics()['integrated'].
    #   "integrated_advantage" -> r = -(C(s_y) - C_ref(demand_y)) = the welfare the
    #                   network CAPTURES relative to a no-build counterfactual, under
    #                   year-y demand. C_ref = cost of the EMPTY network on the same
    #                   (post-growth) demand. This subtracts the exogenous demand
    #                   level -- the part the agent cannot move -- from the reward,
    #                   recovering 'delta'-like signal-to-noise while staying
    #                   alpha-sensitive (the captured welfare grows with demand).
    #                   The logged integrated_welfare metric is UNCHANGED (still
    #                   -sum_y C(s_y)); only the policy-gradient reward is reshaped.
    reward_mode = str(cfg.city_builder.get("reward_mode", "delta"))
    assert reward_mode in ("delta", "integrated", "integrated_advantage"), reward_mode
    # How the build-then-watch watch reward is credited to build decisions:
    #   'spread' (default): distributed evenly across the build-year steps --
    #     objective-preserving (same total) but removes the single-step spike that
    #     destabilizes the critic at high alpha.
    #   'lump': all on the last build step (original decision B).
    watch_credit = str(cfg.city_builder.get("watch_credit", "spread"))

    state, dyn, x_0, cost_weights = _make_episode_state(
        template, cost_obj, build_years, cfg, device)
    state = model.setup_planning(state)
    x = x_0.clone()
    n_fin_prev = int(state.n_finished_routes.sum().item())
    integrated_welfare = 0.0      # -sum_y C(s_y), accumulated at year boundaries

    buf_rewards = torch.zeros((horizon, state.batch_size), device=device)
    buf_val_ests = buf_rewards.clone()
    buf_logits = buf_rewards.clone()
    done_at_step_mask = buf_rewards.clone().bool()
    buf_actions = torch.full((horizon, state.batch_size, 2), -1,
                             device=device, dtype=torch.long)
    buf_states = []
    year_boundary_steps = []      # step indices where a route completed (a year ticked)
    n_steps = horizon

    model.eval()
    with torch.no_grad():
        prev_cost = cost_obj(state).cost
        base_cost = prev_cost
        for tt in range(horizon):
            done_at_step_mask[tt] = state.is_done()
            buf_val_ests[tt] = val_module.from_state(state)
            buf_states.append(state.clone().to_device("cpu"))
            actions, logits, _ = model.step(state)
            buf_actions[tt] = actions
            state.shortest_path_action(actions)
            buf_logits[tt] = logits
            cost = cost_obj(state).cost
            delta_r = (prev_cost - cost) * return_scale
            prev_cost = cost
            n_fin = int(state.n_finished_routes.sum().item())
            year_done = n_fin > n_fin_prev
            if year_done:
                # Dynamics advance -> demand reflects this year's induced growth.
                x = _advance_year(dyn, state, x, beta)
                n_fin_prev = n_fin
                year_boundary_steps.append(tt)
                # C(s_y) under new demand, sized to the REALIZED network (NOT the
                # live state's n_routes_to_plan=T) so it equals the eval metric.
                level_cost = _realized_network_cost(state, cost_obj, cfg,
                                                    cost_weights)
                integrated_welfare += float((-level_cost).mean().item())
            # Reward per mode (year-boundary only; segments within a year get 0):
            #   integrated           -> -C(s_y)
            #   integrated_advantage -> -(C(s_y) - C_ref), no-build baseline
            #   delta                -> per-segment cost reduction
            if reward_mode in ("integrated", "integrated_advantage"):
                if year_done:
                    if reward_mode == "integrated_advantage":
                        # C_ref = empty-network cost under THIS year's demand.
                        ref_cost = _no_build_cost(state.graph_data, cost_obj,
                                                  cfg, cost_weights)
                        buf_rewards[tt] = -(level_cost - ref_cost) * return_scale
                    else:
                        buf_rewards[tt] = -level_cost * return_scale
                # else: leave the pre-zeroed 0.0
            else:
                buf_rewards[tt] = delta_r
            if state.is_done().all():
                n_steps = tt + 1
                break
            state.reset_dones()

        # --- watch phase (build-then-watch): network FROZEN, dynamics advance ---
        # The build loop above built `build_years` routes (state is now done, no
        # more model.step). For each remaining watch year, advance induced demand
        # and accrue welfare. The watch welfare is credited back to the build
        # decisions (via GAE) either SPREAD evenly across the build-year steps
        # (default; objective-preserving, removes the single-step spike that
        # destabilizes the critic at high alpha) or LUMPED on the last build step
        # (original decision B). watch_years == 0 => no-op (rebuild-yearly).
        if watch_years > 0:
            watch_reward = torch.zeros_like(buf_rewards[n_steps - 1])
            for _w in range(watch_years):
                x = _advance_year(dyn, state, x, beta)
                level_cost = _realized_network_cost(state, cost_obj, cfg,
                                                    cost_weights)
                integrated_welfare += float((-level_cost).mean().item())
                if reward_mode == "integrated_advantage":
                    ref_cost = _no_build_cost(state.graph_data, cost_obj, cfg,
                                              cost_weights)
                    watch_reward = watch_reward - (level_cost - ref_cost)
                elif reward_mode == "integrated":
                    watch_reward = watch_reward - level_cost
                # delta mode: watch contributes nothing to the telescoped reward
            watch_reward = watch_reward * return_scale
            if watch_credit == "spread" and year_boundary_steps:
                share = watch_reward / len(year_boundary_steps)
                for idx in year_boundary_steps:
                    buf_rewards[idx] = buf_rewards[idx] + share
            else:                                   # 'lump' (decision B) or fallback
                buf_rewards[n_steps - 1] = buf_rewards[n_steps - 1] + watch_reward

        final_val_ests = val_module.from_state(state)

        # --- running return normalization (reward_norm) -----------------------
        # Scale rewards by the running std of the discounted return so the critic
        # fits unit-scale targets and GAE deltas stay well-conditioned. Uses the
        # std estimated from PRIOR episodes (stale) to avoid leakage, then updates
        # the running stats with this episode's raw return-to-go. Off by default.
        if ret_rms is not None:
            with torch.no_grad():
                g, raw_returns = 0.0, []
                for tt in reversed(range(n_steps)):
                    g = float(buf_rewards[tt].mean().item()) + gamma * g
                    raw_returns.append(g)
            std = ret_rms.std                       # from prior episodes (1.0 init)
            ret_rms.update(raw_returns)
            buf_rewards[:n_steps] = buf_rewards[:n_steps] / (std + 1e-8)
        buf_advantages = torch.zeros_like(buf_rewards)
        lastgaelam = 0
        for tt in reversed(range(n_steps)):
            if tt == n_steps - 1:
                nextnonterminal = ~state.is_done()
                nextvalues = final_val_ests
            else:
                nextnonterminal = ~done_at_step_mask[tt + 1]
                nextvalues = buf_val_ests[tt + 1]
            delta = buf_rewards[tt] + \
                gamma * nextvalues * nextnonterminal - buf_val_ests[tt]
            buf_advantages[tt] = lastgaelam = delta + \
                gamma * cfg.ppo.gae_lambda * nextnonterminal * lastgaelam
        buf_returns = buf_advantages + buf_val_ests

    return {
        "states": buf_states,                 # len n_steps
        "actions": buf_actions[:n_steps],      # (n_steps, 1, 2)
        "logits": buf_logits[:n_steps],        # (n_steps, 1)
        "advantages": buf_advantages[:n_steps],
        "returns": buf_returns[:n_steps],
        "welfare_gain": float((base_cost - prev_cost).mean().item()),
        "integrated_welfare": integrated_welfare,
        "sum_x": float(x.sum().item()),
    }


def train_citybuilder(cfg: DictConfig):
    device, run_name, sumwriter, cost_obj, model = \
        lrnu.process_standard_experiment_cfg(cfg, "citybuilder_")
    # NNBaseline (and other helpers in inductive_route_learning) read a
    # module-level DEVICE global that is normally set inside that file's own
    # training entrypoint. We call process_standard_experiment_cfg directly, so
    # set it here before constructing NNBaseline.
    irl.DEVICE = device

    # Native W&B logging: this (training) process owns the run, so the welfare
    # curves land in W&B Charts directly (TensorBoard add_scalar from a
    # subprocess doesn't reliably sync). Optional -- disabled cleanly if wandb
    # or an API key isn't available (e.g. local runs).
    wb = None
    try:
        import wandb
        wb = wandb.init(
            project=os.environ.get("WANDB_PROJECT", "cs224r-city-builder"),
            name=run_name,
            group=os.environ.get("WANDB_RUN_GROUP") or None,
            config={"run_name": run_name,
                    "alpha": float(cfg.city_builder.alpha),
                    "n_episodes_per_iter": int(cfg.ppo.get("n_episodes_per_iter", 1)),
                    "n_iterations": int(cfg.ppo.n_iterations),
                    "milestone": "M3-TOP-12"},
            reinit=True,
        )
    except Exception as e:  # noqa: BLE001
        log.warning("wandb logging disabled: %s", e)

    T = horizon_T(cfg)                         # years == routes-to-plan
    horizon = int(cfg.ppo.horizon)             # generous step cap
    gamma = cfg.discount_rate
    epsilon = cfg.ppo.epsilon
    return_scale = cfg.get("reward_scale", 1.0)

    # Batched (num_graphs=1) template: the GNN's model.step calls
    # graph_data.to_data_list(), which requires a PyG Batch, not a single
    # CityGraphData. Mirror train_ppo's DataLoader path.
    _dataset = get_dataset_from_config(cfg.eval.dataset)
    template = next(iter(DataLoader(_dataset, batch_size=1)))

    # maximize=True: the PPO objective (clip_obj + entropy bonus) is MAXIMIZED,
    # matching Holliday's train_ppo optimizer. (Plain Adam minimizes -> the
    # policy is driven the wrong way and welfare DEGRADES, as observed.)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.lr,
        weight_decay=cfg.get("decay", 0.0), maximize=True)
    # Stabilized critic: gradient clipping + Huber loss prevent the value net
    # from diverging on outlier episodes under non-stationary induced demand
    # (the baseline +/-1000 spikes). Tunable; set baseline_grad_clip=null +
    # baseline_loss=mse to recover the original behavior.
    val_module = NNBaseline(
        learning_rate=cfg.baseline_lr,
        grad_clip=cfg.get("baseline_grad_clip", 10.0),
        loss=str(cfg.get("baseline_loss", "huber")),
        input_clamp=cfg.get("baseline_input_clamp", 5.0))

    out_dir = Path(cfg.outdir) if "outdir" in cfg else Path("output")
    out_dir.mkdir(parents=True, exist_ok=True)

    pbar = tqdm(total=cfg.ppo.n_iterations)
    best_cost = float("inf")
    best_model = copy.deepcopy(model)

    # Stage 1: collect n_episodes_per_iter episodes per PPO iteration and pool
    # them into one update. Default 1 -> original single-episode behavior.
    n_eps = int(cfg.ppo.get("n_episodes_per_iter", 1))

    # The objective we track for best-model selection must match the reward the
    # policy is actually optimizing (delta -> welfare_gain; integrated ->
    # integrated_welfare). Both are always logged for comparison.
    reward_mode = str(cfg.city_builder.get("reward_mode", "delta"))
    obj_key = ("integrated_welfare"
               if reward_mode in ("integrated", "integrated_advantage")
               else "welfare_gain")
    log.info("reward_mode=%s -> selecting best model on '%s'", reward_mode, obj_key)

    # Running return normalization (off by default): one persistent estimator
    # across all iterations so the reward scale the critic sees stays unit-ish.
    reward_norm = bool(cfg.get("reward_norm", False))
    ret_rms = RunningMeanStd() if reward_norm else None
    log.info("reward_norm=%s", reward_norm)

    for iteration in range(cfg.ppo.n_iterations):
        episodes = [_collect_episode(model, cost_obj, val_module, template,
                                     cfg, device, ret_rms=ret_rms)
                    for _ in range(n_eps)]

        # Holliday freezes FeatureNorm at iter 0 because his demand is STATIC.
        # Under induced demand the input distribution is non-stationary AND
        # coupled to policy quality: iter-0 (random policy, low growth) stats
        # under-cover the high-demand states a trained policy produces, so a
        # frozen norm under-normalizes and the GNN's nodepair scores overflow
        # (worst at high alpha). Keep the running stats live so they track the
        # drifting demand distribution. Trade-off: normalization can shift
        # slightly between a rollout and its update, adding minor PPO
        # importance-ratio noise -- bounded (momentum-weighted, slow) and
        # backstopped by the nodepair-score clamp in models.py._encode_graph.
        model.update_feature_norms()

        # ---- pool transitions across the episodes ---------------------
        states = [s for ep in episodes for s in ep["states"]]
        actions = torch.cat([ep["actions"] for ep in episodes], dim=0)
        old_logits = torch.cat([ep["logits"] for ep in episodes], dim=0)
        advantages = torch.cat([ep["advantages"] for ep in episodes], dim=0)
        returns = torch.cat([ep["returns"] for ep in episodes], dim=0)
        # Normalize advantages over the POOLED buffer.
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        mean_gain = sum(ep["welfare_gain"] for ep in episodes) / len(episodes)
        best_gain = max(ep["welfare_gain"] for ep in episodes)
        mean_int = sum(ep["integrated_welfare"] for ep in episodes) / len(episodes)
        # Objective tracked for selection/logging matches the active reward mode.
        objective = mean_int if obj_key == "integrated_welfare" else mean_gain
        sumwriter.add_scalar("episode welfare gain", mean_gain, pbar.n)
        sumwriter.add_scalar("episode integrated welfare", mean_int, pbar.n)
        sumwriter.add_scalar("baseline", returns.mean(), pbar.n)
        if wb is not None:
            wb.log({"episode_welfare_gain": mean_gain,
                    "best_welfare_gain": best_gain,
                    "episode_integrated_welfare": mean_int,
                    "baseline": float(returns.mean().item())}, step=iteration)

        # ---- PPO update: gradient accumulation over pooled single states --
        total = len(states)
        mb_size = max(1, min(int(cfg.ppo.minibatch_size), total))
        train_order = torch.cat([torch.randperm(total, device=device)
                                 for _ in range(cfg.ppo.n_epochs)])
        for idxs in torch.split(train_order, mb_size):
            idx_list = idxs.tolist()
            optimizer.zero_grad()
            for i in idx_list:
                st = states[i].to_device(device)
                val_module.from_state(st)
                val_module.update(returns[i])
                _, logit, entropy = model.step(st, actions=actions[i])
                ratio = (logit - old_logits[i]).exp()
                adv = advantages[i]
                clipped = ratio.clamp(1 - epsilon, 1 + epsilon)
                clip_obj = torch.minimum(ratio * adv, clipped * adv)
                loss = (clip_obj.mean()
                        + entropy.mean() * cfg.entropy_weight) / len(idx_list)
                loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5,
                                           error_if_nonfinite=True)
            optimizer.step()

        # ---- periodic logging / checkpoint ----------------------------
        if iteration % cfg.ppo.val_period == 0:
            log.info("iter %d: %s=%.4f (delta=%.4f, integrated=%.4f, n_eps=%d)",
                     iteration, obj_key, objective, mean_gain, mean_int, n_eps)
            if -objective < best_cost:      # higher objective = better
                best_cost = -objective
                best_model = copy.deepcopy(model)
                # Persist the best model TO DISK as soon as it improves, not only
                # at the end. On preemptible GPUs a run can be killed mid-training
                # (Modal restarts from iter 0); without this, a preempted run
                # leaves NO checkpoint and its eval fails. Now the peak-so-far
                # always survives on the volume.
                torch.save(best_model.state_dict(), out_dir / (run_name + ".pt"))
                log.info("checkpointed best (%s=%.4f) -> %s",
                         obj_key, objective, out_dir / (run_name + ".pt"))
        pbar.update(1)

    pbar.close()
    torch.save(best_model.state_dict(), out_dir / (run_name + ".pt"))
    log.info("saved %s", out_dir / (run_name + ".pt"))
    if wb is not None:
        wb.finish()
    return best_model


@hydra.main(version_base=None, config_path="../../cfg",
            config_name="ppo_citybuilder_mandl")
def main(cfg: DictConfig):
    return train_citybuilder(cfg)


if __name__ == "__main__":
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    main()
