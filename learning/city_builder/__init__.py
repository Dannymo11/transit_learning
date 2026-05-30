# CS 224R City Builder package.
#
# Contribution layer on top of Holliday's transit-RL repo (the "library" half
# of the library-not-framework boundary documented in docs/M1_architecture_mapping.md
# and docs/M2_mdp_formalization.md).
#
# v1 surface (M2 / TOP-9):
#   - LandUseDynamics: per-zone activity update under induced demand
#   - hansen_accessibility / normalize_accessibility: A_i = sum_j x_j d_ij^{-beta}
#   - gravity_demand: D_ij = x_i * x_j * d_ij^{-beta}
#   - recompute_demand_in_place: updates CityGraphData.demand + DEMAND_KEY edge_attr
#
# alpha (induced-demand strength) is reserved for this module ONLY. Holliday's
# cost-weight tradeoff is named w_p / demand_time_weight in our code and writeup.
# See docs/M2_mdp_formalization.md notation note.

from .accessibility import hansen_accessibility, normalize_accessibility
from .gravity import gravity_demand
from .land_use_dynamics import AccessibilityFn, LandUseDynamics, LandUseConfig
from .demand_hook import GravityFn, recompute_demand_in_place, step_world

__all__ = [
    "LandUseDynamics",
    "LandUseConfig",
    "AccessibilityFn",
    "hansen_accessibility",
    "normalize_accessibility",
    "gravity_demand",
    "GravityFn",
    "recompute_demand_in_place",
    "step_world",
]
