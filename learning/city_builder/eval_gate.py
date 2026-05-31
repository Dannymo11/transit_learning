"""TOP-11 gate evaluation --- evaluate ONE trained checkpoint at the three
Holliday cost-weight alphas and record C(alpha, G, R).

The TOP-11 decision gate compares our LC-100-style policy against Holliday &
Dudek 2024 Table 2, per cost-weight alpha in {0.0, 0.5, 1.0}. The training run
(modal_runs/train_static_holliday.py) produces a checkpoint but NOT these
per-alpha numbers --- this script produces them.

Method (mirrors LC-100 "best of 100 samples"):
  * load the trained policy,
  * for each cost-weight alpha, force the cost module to fixed weights
    (variable_weights=False; demand_time_weight=alpha, route_time_weight=1-alpha
    --- Holliday's convention: alpha=1 passenger-only, alpha=0 operator-only),
  * sample 100 route sets, keep the best, and record its cost C(alpha) and the
    average passenger trip time (ATT) as a sanity metric.

Writes <gate_out>/<run_name>_gate.json with the per-alpha results. Aggregation
across seeds and the pass/fail comparison against the paper live in
learning/city_builder/gate_report.py (numpy-only, runs without torch).

NOTE on naming: `alpha` here is Holliday's cost-weight (w_p), NOT our
induced-demand alpha, which is fixed at 0 for the whole TOP-11 milestone.

Run from the repo root::

    python -m learning.city_builder.eval_gate \
        --config-name=eval_gate_mandl \
        model.weights=inductive_gae_seed0.pt +run_name=mandl_seed0
"""
from __future__ import annotations

import json
import logging as log
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig
from torch_geometric.loader import DataLoader

from simulation.citygraph_dataset import get_dataset_from_config
import learning.utils as lrnu
from learning.eval_route_generator import eval_model

# Holliday cost-weight alphas to evaluate at (paper Table 2 columns).
COST_WEIGHT_ALPHAS = [0.0, 0.5, 1.0]
# op.yaml / pp.yaml use 1e-10 rather than exact 0 to avoid degenerate weights;
# mirror that here.
WEIGHT_FLOOR = 1e-10


@hydra.main(version_base=None, config_path="../../cfg",
            config_name="eval_gate_mandl")
def main(cfg: DictConfig) -> None:
    assert "model" in cfg and cfg.model.get("weights", None), \
        "Must provide model.weights=<checkpoint.pt>"

    device, run_name, _, cost_obj, model = \
        lrnu.process_standard_experiment_cfg(
            cfg, "gate_eval_", weights_required=True)

    test_ds = get_dataset_from_config(cfg.eval.dataset)
    test_dl = DataLoader(test_ds, batch_size=cfg.batch_size)

    n_samples = int(cfg.get("n_samples", 100))
    sbs = int(cfg.get("sample_batch_size", n_samples))

    # Force fixed cost weights for evaluation (training used variable_weights).
    cost_obj.variable_weights = False

    results: dict[str, dict] = {}
    city = cfg.eval.dataset.city
    label = cfg.get("run_name", Path(str(cfg.model.weights)).stem)

    for a in COST_WEIGHT_ALPHAS:
        cost_obj.set_weights(
            demand_time_weight=max(a, WEIGHT_FLOOR),
            route_time_weight=max(1.0 - a, WEIGHT_FLOOR),
        )
        cost, metrics = eval_model(
            model, test_dl, cfg.eval, cost_obj,
            n_samples=n_samples, sample_batch_size=sbs,
            return_routes=False, device=device,
        )
        c = float(torch.as_tensor(cost).float().mean().item())
        att = None
        if metrics is not None and "ATT" in metrics:
            att = float(torch.as_tensor(metrics["ATT"]).float().mean().item())
        results[f"{a}"] = {
            "cost": c,
            "att_s": att,
            "att_min": (att / 60.0) if att is not None else None,
        }
        log.info("alpha=%.1f  C=%.4f  ATT=%s",
                 a, c, f"{att:.1f}s" if att is not None else "n/a")

    out_dir = Path(str(cfg.get("gate_out", "results/gate")))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{label}_gate.json"
    payload = {
        "label": label,
        "city": city,
        "weights": str(cfg.model.weights),
        "n_samples": n_samples,
        "cost_weight_alphas": COST_WEIGHT_ALPHAS,
        "results": results,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"[eval_gate] {label}: " +
          "  ".join(f"C(a={a})={results[str(a)]['cost']:.4f}"
                    for a in COST_WEIGHT_ALPHAS))
    print(f"[eval_gate] wrote {out_path}")


if __name__ == "__main__":
    main()
