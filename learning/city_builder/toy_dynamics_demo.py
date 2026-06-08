"""Toy demo: does connecting an underserved zone create demand greedy would
have missed?  Compares the ORIGINAL induced-demand dynamics (multiplicative
growth, cap = 3*x_0) against the SHAPED dynamics (cap_mode='uniform' + additive
accessibility growth) on a tiny hand-built city.

The point: under the original dynamics a peripheral zone that starts small can
never grow large (its cap is tied to its own tiny x_0), so demand piles onto the
already-busy core -- which greedy already serves, leaving nothing for an
anticipating agent to win. Under the shaped dynamics, connecting that peripheral
zone injects size-independent demand and lets it grow into a real demand center,
which is exactly the future a myopic baseline cannot foresee.

Pure torch, no GPU / PyG needed:
    python -m learning.city_builder.toy_dynamics_demo
"""
from __future__ import annotations

import torch

from learning.city_builder.land_use_dynamics import LandUseConfig, LandUseDynamics


def _city():
    """5 zones: 0,1,2 = busy core (close together), 3,4 = sparse periphery.
    Returns x_0 and two drive-time matrices: 'isolated' (periphery far) and
    'connected' (a route has pulled zone 3 in next to the core)."""
    x0 = torch.tensor([100.0, 90.0, 80.0, 8.0, 8.0])
    far, near, core = 30.0, 6.0, 5.0
    # base: core mutually near, periphery far from everything
    d = torch.full((5, 5), far)
    for i in (0, 1, 2):
        for j in (0, 1, 2):
            d[i, j] = core
    d.fill_diagonal_(1.0)
    isolated = d.clone()
    connected = d.clone()
    # "build a route": zone 3 now near the core (and zone 4 a bit closer too)
    for j in (0, 1, 2):
        connected[3, j] = connected[j, 3] = near
    return x0, isolated, connected


def _run(x0, drive_times, cfg, years=30):
    dyn = LandUseDynamics(initial_activity=x0, config=cfg, seed=0)
    x = x0.clone()
    traj = [x.clone()]
    for _ in range(years):
        x, _ = dyn.step(x, drive_times)
        traj.append(x.clone())
    return torch.stack(traj), dyn.cap


def _report(name, cfg, x0, isolated, connected):
    traj_iso, cap = _run(x0, isolated, cfg)
    traj_con, _ = _run(x0, connected, cfg)
    z = 3  # the peripheral zone we "connected"
    print(f"\n=== {name} ===")
    print(f"  cap[zone3] = {cap[z]:.1f}   (x_0[zone3] = {x0[z]:.1f})")
    print(f"  zone3 activity, isolated  : start {traj_iso[0, z]:7.1f} "
          f"-> year30 {traj_iso[-1, z]:7.1f}")
    print(f"  zone3 activity, CONNECTED : start {traj_con[0, z]:7.1f} "
          f"-> year30 {traj_con[-1, z]:7.1f}")
    gain = traj_con[-1, z] - traj_iso[-1, z]
    print(f"  >> demand UNLOCKED by connecting zone3 (year30): {gain:+7.1f}")
    print(f"  core zone0, CONNECTED     : start {traj_con[0, 0]:7.1f} "
          f"-> year30 {traj_con[-1, 0]:7.1f}")


def _blend_sweep(x0, isolated, connected, add_rate=0.5,
                 lambdas=(1.0, 0.75, 0.5, 0.25, 0.0)):
    """Sweep the cap-blend lambda at fixed add_rate. Shows the trade-off:
    low lambda gives the periphery more room (anticipation gap) but flattens the
    core toward the same cap (lost heterogeneity)."""
    z, core = 3, 0
    print(f"\n=== cap_blend sweep  (add_rate={add_rate}, alpha=0.5) ===")
    print("  lambda | cap[z3] | z3 connected y30 | z3 UNLOCKED | core y30 | "
          "core/z3 ratio")
    print("  -------+---------+------------------+-------------+----------+-----------")
    for lam in lambdas:
        cfg = LandUseConfig(alpha=0.5, base_rate=1.0, cap_multiplier=3.0,
                            cap_blend=lam, add_rate=add_rate)
        traj_iso, cap = _run(x0, isolated, cfg)
        traj_con, _ = _run(x0, connected, cfg)
        z3 = traj_con[-1, z].item()
        unlocked = z3 - traj_iso[-1, z].item()
        core_y30 = traj_con[-1, core].item()
        ratio = core_y30 / max(z3, 1e-9)
        tag = ("  (proportional)" if lam == 1.0 else
               "  (uniform)" if lam == 0.0 else "")
        print(f"   {lam:4.2f}  | {cap[z]:7.1f} | {z3:16.1f} | {unlocked:+11.1f} "
              f"| {core_y30:8.1f} | {ratio:6.2f}x{tag}")
    print("  Pick a lambda where z3 UNLOCKED is large (anticipation has a prize) "
          "AND core/z3 ratio stays > 1 (downtown still dominates -> realistic).")


def main():
    x0, isolated, connected = _city()
    print("Toy city: x_0 =", [round(v, 1) for v in x0.tolist()],
          "(zones 0-2 busy core, 3-4 sparse periphery)")

    original = LandUseConfig(alpha=0.5, base_rate=1.0,
                             cap_multiplier=3.0, cap_mode="proportional",
                             add_rate=0.0)
    shaped = LandUseConfig(alpha=0.5, base_rate=1.0,
                           cap_multiplier=3.0, cap_mode="uniform",
                           add_rate=1.0)
    _report("ORIGINAL dynamics (multiplicative, cap=3*x_0)",
            original, x0, isolated, connected)
    _report("SHAPED dynamics (uniform cap + additive growth)",
            shaped, x0, isolated, connected)

    # The axis we're sweeping first: how much to decouple the cap (lambda),
    # holding the additive injection fixed.
    _blend_sweep(x0, isolated, connected, add_rate=0.5)

    print("\nIf 'demand UNLOCKED by connecting zone3' is ~0 under ORIGINAL but "
          "large under SHAPED, the shaped dynamics give anticipation something "
          "to win: a route into the sparse periphery now grows a demand center "
          "a myopic (today's-demand) baseline would never have prioritized.")


if __name__ == "__main__":
    main()
