"""Multi-year MDP wrapper for City Builder (M2 / TOP-9).

This is the OUTER environment the project's contribution lives in: a year-by-year
sequential decision problem on top of Holliday's (inner) route machinery. Per the
library-not-framework decision, we do NOT modify Holliday's training loop — we
wrap his cost module + route state in a new MDP and will drive it with his GNN
route generator (TOP-12) or with baselines (here / M5).

State    s_t = (G_t, x_t, B_t):
  - G_t : the transit network built so far (a list of routes), over a fixed
          CityGraphData whose street graph never changes.
  - x_t : per-zone activity (LandUseDynamics state).
  - B_t : remaining annual construction budget (HOOK ONLY for now; capex/budget
          enforcement is M4 / TOP-17, so the base env leaves it unconstrained).

Action   a_t : a complete new transit route (a node sequence) added to G_t, or
          None (= no-build). The env is POLICY-AGNOSTIC: a_t is just a route.
          Baselines provide one here; TOP-12 will have Holliday's
          PathCombiningRouteGenerator (n_routes_to_plan=1) provide it.

Reward   r_t = W(s_{t+1}) - W(s_t),  W(s) = -MyCostModule(materialize(s)).cost
          (lower cost = higher welfare), so r_t = C(s_t) - C(s_{t+1}). gamma=0.95
          at training time telescopes (gamma=1) to total welfare gain.

Transition: apply a_t to G_t -> G_{t+1}; observe LandUseDynamics
          against G_{t+1} (closed loop: accessibility uses the TRANSIT OD times
          of G_{t+1}; gravity demand stays on street times); recompute demand.

Horizon  T years (default 10), fixed-length episodes.

Smoke (one episode under a baseline policy)::

    python -m learning.city_builder.multi_year_mdp --alpha 0.5 --policy greedy
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import networkx as nx
import torch

from simulation.transit_time_estimator import MyCostModule, RouteGenBatchState
from torch_utils import get_batch_tensor_from_routes

from learning.city_builder import (
    LandUseConfig, LandUseDynamics, recompute_demand_in_place, step_world,
    M2_WORKING_ALPHA,
)
from learning.city_builder.alpha_sweep import (
    REPO_ROOT, DEFAULT_INSTANCES_DIR,
    N_ROUTES, MIN_ROUTE_LEN, MAX_ROUTE_LEN, HORIZON,
    BETA_GRAVITY, CAP_MULTIPLIER, W_P_FIXED,
    _fresh_data, _new_state, _initial_activity_from_od, transit_drive_times,
)

# An action is a route (list of node indices) or None (no-build).
Action = Optional[list]
# A policy maps the current env to its next action.
Policy = Callable[["CityBuilderEnv"], Action]


# --- route-geometry-aware helpers (instance-parameterized) --------------
# These replace alpha_sweep's Mandl-globals versions so the env works on any
# instance (Mumford0/1/...) just by passing its min/max route length.
def _street_state(data, cost_obj, min_len: int, max_len: int) -> RouteGenBatchState:
    """Minimal state used only to read the street adjacency / build baselines."""
    return RouteGenBatchState(
        data, cost_obj, n_routes_to_plan=1,
        min_route_len=min_len, max_route_len=max_len)


def _transit_times(data, cost_obj, net, min_len: int, max_len: int) -> torch.Tensor:
    """OD transit-time matrix (N, N) experienced ON `net` -- the closed-loop
    accessibility basis. Parameterized copy of alpha_sweep.transit_drive_times;
    n_routes_to_plan is sized to the network so larger instances don't overflow
    the cost module's stop bookkeeping."""
    n_routes = max(int(net.shape[1]), 1)
    state = RouteGenBatchState(
        data, cost_obj, n_routes_to_plan=n_routes,
        min_route_len=min_len, max_route_len=max_len)
    state.add_new_routes(net)
    tt = state.transit_times
    return tt[0] if tt.dim() == 3 else tt


@dataclass
class CityBuilderState:
    """Observable multi-year state. The heavy tensors live on the env; this is
    the lightweight view a policy / logger consumes."""
    year: int
    activity: torch.Tensor          # (N,) x_t
    built_routes: list              # list[list[int]] — G_t
    b_remaining: float              # budget hook (unenforced pre-M4)
    welfare_cost: float             # C(s_t) = MyCostModule cost of (G_t, x_t)


class CityBuilderEnv:
    """Year-by-year induced-demand transit-design MDP.

    Reusable by both baselines and (TOP-12) the GNN policy: call `reset()` then
    `step(action)` where `action` is a route node-list or None.
    """

    def __init__(self, instance: str = "Mandl",
                 instances_dir: str = str(DEFAULT_INSTANCES_DIR),
                 alpha: float = M2_WORKING_ALPHA, horizon: int = HORIZON,
                 sigma_eps: float = 0.0, seed: int = 0,
                 b_annual: float = float("inf"),
                 initial_routes: Optional[list] = None,
                 min_route_len: int = MIN_ROUTE_LEN,
                 max_route_len: int = MAX_ROUTE_LEN,
                 cap_mode: str = "proportional",
                 add_rate: float = 0.0,
                 cap_blend: Optional[float] = None,
                 build_years: Optional[int] = None):
        self.instance = instance
        self.instances_dir = instances_dir
        self.alpha = alpha
        self.horizon = horizon
        # Build-then-watch: routes may only be added in years [0, build_years).
        # After that the network FREEZES while dynamics + cost keep advancing to
        # `horizon`. Default = horizon => rebuild-every-year (original behavior).
        self.build_years = int(build_years) if build_years is not None else int(horizon)
        self.sigma_eps = sigma_eps
        self.seed = seed
        # Induced-demand dynamics shaping; defaults reproduce the original
        # multiplicative, x_0-proportional-cap behavior. See LandUseConfig.
        self.cap_mode = cap_mode
        self.add_rate = add_rate
        self.cap_blend = cap_blend
        self.b_annual = b_annual            # budget hook; inf => unconstrained
        self.initial_routes = initial_routes or []
        # Per-instance route geometry (Mandl 2..8; Mumford0 2..15; Mumford1
        # 10..30). Defaults preserve the Mandl benchmark so existing callers and
        # the M2 sweep are unchanged.
        self.min_route_len = int(min_route_len)
        self.max_route_len = int(max_route_len)
        self.cost_obj = MyCostModule(symmetric_routes=True)
        self.reset()

    # --- core API -------------------------------------------------------
    def reset(self) -> CityBuilderState:
        self.data = _fresh_data(self.instances_dir, self.instance)
        self.x_0 = _initial_activity_from_od(self.data.demand)
        # t=0 demand = gravity demand under x_0 (street-based; the closed loop
        # only changes the ACCESSIBILITY basis, not the gravity demand).
        recompute_demand_in_place(self.data, self.x_0, beta=BETA_GRAVITY)

        self.dyn = LandUseDynamics(
            initial_activity=self.x_0,
            config=LandUseConfig(alpha=self.alpha, base_rate=1.0,
                                 sigma_eps=self.sigma_eps,
                                 cap_multiplier=CAP_MULTIPLIER,
                                 cap_mode=self.cap_mode,
                                 add_rate=self.add_rate,
                                 cap_blend=self.cap_blend,
                                 beta_accessibility=BETA_GRAVITY),
            seed=self.seed,
        )
        # Mirror the dynamics' actual per-zone cap so frac_at_cap reporting is
        # correct under cap_mode='uniform' too.
        self.cap = self.dyn.cap
        self.x = self.x_0.clone()
        self.year = 0
        self.built_routes = [list(r) for r in self.initial_routes]
        self.b_remaining = self.b_annual
        self._prev_cost = self._welfare_cost()   # C(s_0)
        return self._state()

    def step(self, action: Action):
        """Apply one year. Returns (state, reward, done, info)."""
        if self.year >= self.horizon:
            raise RuntimeError("episode is done; call reset()")

        # 1. apply action a_t -> G_{t+1}, but ONLY during the build window.
        # In the watch phase (year >= build_years) the network is frozen: actions
        # are ignored while dynamics + cost keep advancing below.
        in_build_window = self.year < self.build_years
        if in_build_window and action is not None and len(action) >= self.min_route_len:
            self.built_routes.append([int(n) for n in action])
        # (budget hook: in M4, debit capex(action) from self.b_remaining here)

        # 2. transition: dynamics observed against G_{t+1} (closed loop).
        net = self._materialize_network()
        acc_dt = (_transit_times(self.data, self.cost_obj, net,
                                 self.min_route_len, self.max_route_len)
                  if net is not None else self.data.drive_times)
        self.x, _, _ = step_world(self.dyn, self.data, self.x,
                                  accessibility_drive_times=acc_dt)

        # 3. reward = C(s_t) - C(s_{t+1})  (welfare delta; cost down => reward up)
        c_next = self._welfare_cost()
        reward = self._prev_cost - c_next
        self._prev_cost = c_next

        # 4. advance; fixed-horizon termination
        self.year += 1
        self.b_remaining = self.b_annual       # reset annual budget (no rollover)
        done = self.year >= self.horizon
        info = {
            "welfare_cost": c_next,
            "sum_activity": float(self.x.sum().item()),
            "frac_at_cap": float((self.x >= self.cap - 1e-6).float().mean().item()),
            "n_routes_built": len(self.built_routes),
            "in_build_window": in_build_window,
        }
        return self._state(), float(reward), done, info

    # --- helpers --------------------------------------------------------
    def _materialize_network(self) -> Optional[torch.Tensor]:
        """built_routes -> (1, R, L) batch tensor, or None if empty."""
        if not self.built_routes:
            return None
        return get_batch_tensor_from_routes(
            [self.built_routes], max_route_len=self.max_route_len)

    def _welfare_cost(self) -> float:
        """C(s) = MyCostModule cost of the current network under current demand.
        Empty network => Holliday's unserved-demand penalty (high but finite)."""
        net = self._materialize_network()
        # Size n_routes_to_plan to the accumulated route count (grows 1/year).
        # _new_state fixes it at 6, which overflows the cost module's stop
        # bookkeeping once the network exceeds 6 routes.
        n_routes = max(len(self.built_routes), 1)
        state = RouteGenBatchState(
            self.data, self.cost_obj, n_routes_to_plan=n_routes,
            min_route_len=self.min_route_len, max_route_len=self.max_route_len,
        )
        if net is not None:
            state.add_new_routes(net)
        with torch.no_grad():
            return float(self.cost_obj(state).cost.item())

    def _state(self) -> CityBuilderState:
        return CityBuilderState(
            year=self.year, activity=self.x.clone(),
            built_routes=[list(r) for r in self.built_routes],
            b_remaining=self.b_remaining, welfare_cost=self._prev_cost,
        )


# --- baseline policies (one route per year) -----------------------------
def _route_list_from_network(net: torch.Tensor) -> Action:
    """First valid route (>=2 nodes) from a (1, R, L) network tensor."""
    for r in range(net[0].shape[0]):
        seq = net[0][r]
        seq = seq[seq >= 0].tolist()
        if len(seq) >= MIN_ROUTE_LEN:
            return [int(n) for n in seq]
    return None


# How many high-unserved-demand seed pairs to turn into candidate routes/year.
GREEDY_N_CANDIDATES = 12


def _unserved_demand(env: "CityBuilderEnv") -> torch.Tensor:
    """(N, N) demand weighted by how poorly each OD pair is currently served.

    served quality q[i,j] = street_time / transit_time in [0,1] (1 if transit
    matches street, ->0 as transit slows or the pair is disconnected). unserved
    = demand * (1 - q): full demand for disconnected pairs, ~0 for well-served."""
    data = env.data
    n = int(data.demand.shape[0])
    net = env._materialize_network()
    transit = (_transit_times(data, env.cost_obj, net,
                              env.min_route_len, env.max_route_len)
               if net is not None else torch.full((n, n), float("inf")))
    with torch.no_grad():
        quality = torch.nan_to_num(data.drive_times / transit,
                                   nan=0.0, posinf=0.0, neginf=0.0).clamp(0.0, 1.0)
    unserved = data.demand * (1.0 - quality)
    unserved.fill_diagonal_(0.0)
    return unserved


def _street_graph(env: "CityBuilderEnv"):
    street_adj = _street_state(
        env.data, env.cost_obj, env.min_route_len, env.max_route_len).street_adj[0]
    have = (street_adj > 0) & street_adj.isfinite()
    n = have.shape[0]
    g = nx.Graph()
    g.add_nodes_from(range(n))
    ii, jj = torch.nonzero(torch.triu(have, diagonal=1), as_tuple=True)
    for i, j in zip(ii.tolist(), jj.tolist()):
        g.add_edge(i, j, weight=float(street_adj[i, j]))
    return g, have


def _coverage_route(g, have: torch.Tensor, src: int, dst: int,
                    node_unserved: torch.Tensor,
                    max_len: int, min_len: int) -> Optional[list]:
    """Shortest src->dst street path, then greedily extended (at either end) to
    max_len by appending the reachable node with the most unserved demand. This
    turns a 2-node corridor into a coverage route through high-demand zones."""
    try:
        path = nx.shortest_path(g, src, dst, weight="weight")
    except nx.NetworkXException:
        return None
    if len(path) > max_len:
        path = path[:max_len]
    on = set(path)
    while len(path) < max_len:
        best, best_score, best_end = None, -1.0, None
        for end in (path[0], path[-1]):
            nbrs = torch.nonzero(have[end]).squeeze(-1).tolist()
            for k in nbrs:
                if k in on:
                    continue
                s = float(node_unserved[k])
                if s > best_score:
                    best, best_score, best_end = k, s, end
        if best is None:
            break
        if best_end == path[0]:
            path.insert(0, best)
        else:
            path.append(best)
        on.add(best)
    return path if len(path) >= min_len else None


def greedy_policy(env: "CityBuilderEnv") -> Action:
    """Strong myopic greedy: each year add the single route that most reduces
    CURRENT-demand cost (judged by MyCostModule), among coverage routes seeded
    from the highest-unserved-demand OD pairs.

    This is the demand-greedy spirit of Holliday's john_init (John et al. 2014:
    maximize served demand for the given demand matrix) made incremental and
    robust: it respects the same one-route-per-year build-out as the RL agent,
    uses the real cost objective as the judge, and is deliberately MYOPIC — it
    optimizes for today's demand and does not anticipate induced growth. That
    myopia is exactly the gap an anticipating RL policy should beat."""
    data = env.data
    n = int(data.demand.shape[0])
    unserved = _unserved_demand(env)
    if float(unserved.max()) <= 0:        # everything already well served
        return None
    node_unserved = unserved.sum(dim=1)   # per-node poorly-served demand

    g, have = _street_graph(env)

    # Candidate seed pairs = top-K unserved OD pairs.
    k = min(GREEDY_N_CANDIDATES, n * n)
    flat_idx = torch.topk(unserved.flatten(), k).indices.tolist()
    seen, candidates = set(), []
    for flat in flat_idx:
        src, dst = divmod(int(flat), n)
        route = _coverage_route(g, have, src, dst, node_unserved,
                                env.max_route_len, env.min_route_len)
        if route is None:
            continue
        key = tuple(sorted(route))
        if key in seen:
            continue
        seen.add(key)
        candidates.append(route)
    if not candidates:
        return None

    # Score each candidate by the marginal cost reduction it yields on the
    # CURRENT demand, given the already-built network. Pick the best.
    base_cost = env._welfare_cost()
    best_route, best_cost = None, base_cost
    for route in candidates:
        trial = env.built_routes + [route]
        net = get_batch_tensor_from_routes([trial],
                                           max_route_len=env.max_route_len)
        state = RouteGenBatchState(
            env.data, env.cost_obj, n_routes_to_plan=max(len(trial), 1),
            min_route_len=env.min_route_len, max_route_len=env.max_route_len)
        state.add_new_routes(net)
        with torch.no_grad():
            c = float(env.cost_obj(state).cost.item())
        if c < best_cost - 1e-9:
            best_route, best_cost = route, c
    return best_route


def make_random_policy(seed: int = 0) -> Policy:
    gen = torch.Generator().manual_seed(20_000 + seed)

    def _policy(env: "CityBuilderEnv") -> Action:
        state = _street_state(env.data, env.cost_obj,
                              env.min_route_len, env.max_route_len)
        street_adj = state.street_adj[0]
        have = (street_adj > 0) & street_adj.isfinite()
        n = have.shape[0]
        start = int(torch.randint(n, (1,), generator=gen).item())
        route = [start]
        on = torch.zeros(n, dtype=torch.bool); on[start] = True
        while len(route) < env.max_route_len:
            cand = torch.nonzero(have[route[-1]] & ~on).squeeze(-1)
            if cand.numel() == 0:
                break
            nxt = int(cand[torch.randint(cand.numel(), (1,), generator=gen)].item())
            route.append(nxt); on[nxt] = True
        return route if len(route) >= env.min_route_len else None

    return _policy


def run_episode(env: CityBuilderEnv, policy: Policy, verbose: bool = True):
    """Drive a full episode with `policy`; return per-year records."""
    state = env.reset()
    records = []
    if verbose:
        print(f"  year | reward |   welfare C |   sum(x) | %cap | routes")
        print(f"  -----+--------+-------------+----------+------+-------")
        print(f"   {0:3d} |    --- | {state.welfare_cost:11.4f} | "
              f"{float(state.activity.sum()):8.0f} |   -- | "
              f"{len(state.built_routes):3d}")
    done = False
    while not done:
        # Build-then-watch: only consult the policy during the build window. In
        # the watch phase the network is frozen, so skip the (expensive) baseline
        # policy call and pass None. Keeps greedy/random honest to the same
        # build-window constraint as RL (TOP-33) and avoids wasted computation.
        action = policy(env) if env.year < env.build_years else None
        state, reward, done, info = env.step(action)
        records.append({"year": state.year, "reward": reward, **info})
        if verbose:
            print(f"   {state.year:3d} | {reward:6.3f} | "
                  f"{info['welfare_cost']:11.4f} | {info['sum_activity']:8.0f} | "
                  f"{info['frac_at_cap']*100:4.0f} | {info['n_routes_built']:3d}")
    total = sum(r["reward"] for r in records)
    if verbose:
        print(f"\n  cumulative welfare gain (sum r_t) = {total:.4f}")
    return records


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--instance", default="Mandl")
    p.add_argument("--alpha", type=float, default=M2_WORKING_ALPHA)
    p.add_argument("--horizon", type=int, default=HORIZON)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--policy", choices=("greedy", "random"), default="greedy")
    args = p.parse_args(argv)

    env = CityBuilderEnv(instance=args.instance, alpha=args.alpha,
                         horizon=args.horizon, seed=args.seed)
    policy = greedy_policy if args.policy == "greedy" else make_random_policy(args.seed)
    print(f"CityBuilderEnv smoke: {args.instance}, alpha={args.alpha}, "
          f"policy={args.policy}, T={args.horizon}")
    run_episode(env, policy, verbose=True)


if __name__ == "__main__":
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    main()
