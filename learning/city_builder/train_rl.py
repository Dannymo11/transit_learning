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
            beta_accessibility=beta),
        seed=int(cfg.experiment.get("seed", 0)),
    )
    return state, dyn, x_0


def _collect_episode(model, cost_obj, val_module, template, cfg, device):
    """Run one episode (batch=1) with the validated single-instance dynamics and
    return its per-step buffers + GAE. Stage 1 calls this n_episodes_per_iter
    times and pools the results into one PPO update for lower-variance gradients."""
    T = int(cfg.eval.n_routes)
    horizon = int(cfg.ppo.horizon)
    gamma = cfg.discount_rate
    return_scale = cfg.get("reward_scale", 1.0)
    beta = cfg.city_builder.beta_gravity

    state, dyn, x_0 = _make_episode_state(template, cost_obj, T, cfg, device)
    state = model.setup_planning(state)
    x = x_0.clone()
    n_fin_prev = int(state.n_finished_routes.sum().item())

    buf_rewards = torch.zeros((horizon, state.batch_size), device=device)
    buf_val_ests = buf_rewards.clone()
    buf_logits = buf_rewards.clone()
    done_at_step_mask = buf_rewards.clone().bool()
    buf_actions = torch.full((horizon, state.batch_size, 2), -1,
                             device=device, dtype=torch.long)
    buf_states = []
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
            buf_rewards[tt] = (prev_cost - cost) * return_scale
            prev_cost = cost
            n_fin = int(state.n_finished_routes.sum().item())
            if n_fin > n_fin_prev:
                x = _advance_year(dyn, state, x, beta)
                n_fin_prev = n_fin
            if state.is_done().all():
                n_steps = tt + 1
                break
            state.reset_dones()

        final_val_ests = val_module.from_state(state)
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

    T = int(cfg.eval.n_routes)                 # years == routes-to-plan
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
    val_module = NNBaseline(learning_rate=cfg.baseline_lr)

    out_dir = Path(cfg.outdir) if "outdir" in cfg else Path("output")
    out_dir.mkdir(parents=True, exist_ok=True)

    pbar = tqdm(total=cfg.ppo.n_iterations)
    best_cost = float("inf")
    best_model = copy.deepcopy(model)

    # Stage 1: collect n_episodes_per_iter episodes per PPO iteration and pool
    # them into one update. Default 1 -> original single-episode behavior.
    n_eps = int(cfg.ppo.get("n_episodes_per_iter", 1))

    for iteration in range(cfg.ppo.n_iterations):
        episodes = [_collect_episode(model, cost_obj, val_module, template,
                                     cfg, device)
                    for _ in range(n_eps)]

        # Freeze FeatureNorm after the first iteration's rollouts have
        # accumulated stats (mirrors train_ppo's iter-0 freeze); model stays
        # in eval() throughout so batch=1 updates never var() over one sample.
        if iteration == 0:
            model.update_and_freeze_feature_norms()

        # ---- pool transitions across the episodes ---------------------
        states = [s for ep in episodes for s in ep["states"]]
        actions = torch.cat([ep["actions"] for ep in episodes], dim=0)
        old_logits = torch.cat([ep["logits"] for ep in episodes], dim=0)
        advantages = torch.cat([ep["advantages"] for ep in episodes], dim=0)
        returns = torch.cat([ep["returns"] for ep in episodes], dim=0)
        # Normalize advantages over the POOLED buffer.
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        mean_gain = sum(ep["welfare_gain"] for ep in episodes) / len(episodes)
        sumwriter.add_scalar("episode welfare gain", mean_gain, pbar.n)
        sumwriter.add_scalar("baseline", returns.mean(), pbar.n)

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
            log.info("iter %d: mean welfare gain=%.4f (n_eps=%d)",
                     iteration, mean_gain, n_eps)
            if -mean_gain < best_cost:      # higher gain = better
                best_cost = -mean_gain
                best_model = copy.deepcopy(model)
        pbar.update(1)

    pbar.close()
    torch.save(best_model.state_dict(), out_dir / (run_name + ".pt"))
    log.info("saved %s", out_dir / (run_name + ".pt"))
    return best_model


@hydra.main(version_base=None, config_path="../../cfg",
            config_name="ppo_citybuilder_mandl")
def main(cfg: DictConfig):
    return train_citybuilder(cfg)


if __name__ == "__main__":
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    main()
