#!/usr/bin/env python3
"""Fresh development-only selection of architecture, C, and operating policy."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

TARGETS = [
    "temperature",
    "ph",
    "electric conductivity",
    "dissolved oxygen",
]
C_GRID = [0.0001, 0.0005, 0.001, 0.005, 0.02]
K_GRID = [0.5, 1.0, 1.5, 2.0]
Q_GRID = [4.0, 8.0, 16.0, 32.0]
DEVELOPMENT_SEEDS = [42, 43, 44]
CONFIRMATION_SEEDS = [88191, 26722, 35230, 45158, 73564]
R0_FLOOR = -0.02
MACRO_RESCUE_FLOOR = 0.20
SAFETY_COVERAGE_FLOOR = 0.80


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("m2p_selection", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def dump(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def boolish(value) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def normalize_architecture(best):
    aliases = {
        "BILSTM_UNITS": ("BILSTM_UNITS", "bilstm_units"),
        "CONV_FILTERS": ("CONV_FILTERS", "conv_filters"),
        "ATTENTION_HEADS": ("ATTENTION_HEADS", "attention_heads"),
        "HEAD_DENSE_UNITS": ("HEAD_DENSE_UNITS", "head_dense_units"),
        "DROPOUT_RATE": ("DROPOUT_RATE", "dropout_rate"),
    }
    architecture = {}
    for canonical, names in aliases.items():
        value = next((best[name] for name in names if name in best), None)
        if value is None:
            raise KeyError(f"Tuner output missing {canonical}: {best}")
        architecture[canonical] = value
    return architecture


def apply_architecture(cfg, architecture):
    for name, value in architecture.items():
        setattr(cfg, name, value)
    cfg.FINAL_USE_ATTENTION = True
    cfg.FINAL_USE_CNN = True
    cfg.FINAL_UNIFIED_ENCODER = False
    return cfg


def select_architecture(module, data, output):
    cfg = module.Config()
    cfg.OUTPUT_DIR = str(output / "architecture")
    cfg.N_ITER_SEARCH = 25
    cfg.FINAL_USE_ATTENTION = True
    cfg.FINAL_USE_CNN = True
    cfg.FINAL_UNIFIED_ENCODER = False
    Path(cfg.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    best = module.tune_m2p_hyperparameters(data, TARGETS, cfg)
    if not isinstance(best, dict):
        raise RuntimeError("Hyperparameter tuner did not return a dictionary.")
    architecture = normalize_architecture(best)
    dump(output / "best_hyperparameters.json", architecture)
    return architecture


def select_c(module, data, observed, fault, architecture, output):
    rows = []
    for c in C_GRID:
        arm_dir = output / "C_selection" / f"C_{c:g}"
        arm_dir.mkdir(parents=True, exist_ok=True)

        cfg = apply_architecture(module.Config(), architecture)
        cfg.OUTPUT_DIR = str(arm_dir)
        cfg.OPTIMAL_LAMBDA = float(c)
        cfg.LAMBDA_ANALYSIS_VALUES = [float(c)]

        experiment = module.PTACExperiment(
            f"M2P-PTAC (C={c:g})",
            cfg,
            base_lambda=float(c),
            conformal_alpha=cfg.DEFAULT_CONFORMAL_ALPHA,
            obs_mask=observed.reindex(index=data.index, columns=TARGETS),
            fault_mask=fault.reindex(index=data.index, columns=TARGETS),
        )
        result, _, _ = experiment.run(data.copy(), TARGETS)
        score = result.get("_val_MAE_for_selection")
        if score is None or not np.isfinite(float(score)):
            raise RuntimeError(f"C={c:g} did not produce a finite validation score.")
        rows.append({"C": float(c), "validation_MAE": float(score)})

    table = pd.DataFrame(rows).sort_values(["validation_MAE", "C"])
    table.to_csv(output / "C_selection.csv", index=False)
    return float(table.iloc[0]["C"])


def configure_policy(module, architecture, c, k, q, output_dir):
    cfg = apply_architecture(module.Config(), architecture)
    cfg.OUTPUT_DIR = str(output_dir)
    cfg.OPTIMAL_LAMBDA = float(c)

    cfg.R17_ENABLE = True
    cfg.USE_FAULT_GATING = True
    cfg.R18_CALIBRATE_DEADBAND = True
    cfg.R20_UTILITY_GATE = True
    cfg.R21_DYNAMIC_SELECTOR = True
    cfg.BENEFIT_AWARE_SELECTOR = False
    cfg.R19_SAFETY_VETO = True
    cfg.R19_T2_DRIFT_RESCUE = True
    cfg.R19_FULL_EVIDENCE_MODE = False
    cfg.NATIVE_SYNTHETIC_LABELS_ONLY = True
    cfg.GATE_THRESHOLD = None
    cfg.R17_GATE_DEADBAND = 0.0
    cfg.FAULT_PROB_DEADBAND = 0.0

    cfg.R17_TAU_KAPPA = float(q)
    cfg.R17_TAU_KAPPA_LEVEL = 0.0
    cfg.R28_HARD_INNOVATION_SWITCH = True
    cfg.R28_HARD_SWITCH_ALPHA = float(k)
    cfg.R28_HARD_SWITCH_FLOOR = 1e-12

    for name in (
        "R29_NOMINAL_CORRECTION_ENVELOPE",
        "R30_HEALTH_STATE_TOURNAMENT",
        "R31_HYBRID_TARGET_GATE",
        "R34_GLOBAL_QUALIFICATION",
        "R34_CHALLENGED_ONLY",
    ):
        if hasattr(cfg, name):
            setattr(cfg, name, False)

    cfg.R22_STRESS_PROTOCOL = True
    cfg.R22_STRESS_SELECT_C = False
    cfg.R17_RESCUE_SEEDS = list(DEVELOPMENT_SEEDS)
    cfg.R27_FIXED_MODEL_REPLAY = True
    cfg.R27_COMPONENT_AUDIT = False
    cfg.R25_MECHANISM_DIAGNOSTIC = False
    cfg.TOURNAMENT_ACTIVE = False
    return cfg


def summarize_policy(curve_path: Path, replay_path: Path):
    curve = pd.read_csv(curve_path)
    curve = curve[curve["drift_type"].astype(str).eq("sudden_shift")].copy()
    if curve.empty:
        raise RuntimeError("No sudden_shift rows in policy selection output.")

    curve["intensity"] = pd.to_numeric(curve["intensity"], errors="coerce")
    curve["rescue"] = pd.to_numeric(curve["rescue"], errors="coerce")
    curve["safety_bool"] = curve["safety_pass"].map(boolish)

    rows = []
    for target in TARGETS:
        data = curve[curve["target"].astype(str).eq(target)].copy()
        nominal = float(data.loc[data["intensity"].eq(0), "rescue"].mean())
        positive = float(data.loc[data["intensity"].gt(0), "rescue"].mean())
        safety = float(
            data.sort_values("intensity")
            .groupby("seed")["safety_bool"]
            .first()
            .mean()
        )
        rows.append(
            {
                "target": target,
                "nominal_rescue": nominal,
                "positive_rescue": positive,
                "safety_coverage": safety,
            }
        )

    targets = pd.DataFrame(rows)

    replay = pd.read_csv(replay_path)
    replay = replay[replay["drift_type"].astype(str).eq("sudden_shift")].copy()
    positive = replay[pd.to_numeric(replay["intensity"], errors="coerce").gt(0)]
    replay_ok = bool(len(positive) and positive["weights_reused"].map(boolish).all())
    if "weight_sha256" in replay.columns:
        replay_ok = replay_ok and bool(
            (
                replay.groupby(["drift_type", "seed"])["weight_sha256"].nunique()
                == 1
            ).all()
        )

    summary = {
        "replay_ok": replay_ok,
        "macro_positive_rescue": float(targets["positive_rescue"].mean()),
        "minimum_positive_rescue": float(targets["positive_rescue"].min()),
        "minimum_nominal_rescue": float(targets["nominal_rescue"].min()),
        "minimum_safety_coverage": float(targets["safety_coverage"].min()),
        "target_summary": targets.to_dict("records"),
    }
    summary["ready"] = bool(
        summary["replay_ok"]
        and summary["minimum_nominal_rescue"] >= R0_FLOOR
        and summary["minimum_positive_rescue"] > 0.0
        and summary["minimum_safety_coverage"] >= SAFETY_COVERAGE_FLOOR
        and summary["macro_positive_rescue"] > MACRO_RESCUE_FLOOR
    )
    return summary


def candidate_rank(item):
    return (
        item["macro_positive_rescue"],
        item["minimum_safety_coverage"],
        item["minimum_positive_rescue"],
        -item["q"],
        -item["k"],
    )


def select_policy(module, data, observed, fault, architecture, c, output):
    records = []
    for q in Q_GRID:
        for k in K_GRID:
            candidate_dir = output / "policy_selection" / f"k_{k:g}_q_{q:g}"
            candidate_dir.mkdir(parents=True, exist_ok=True)
            cfg = configure_policy(module, architecture, c, k, q, candidate_dir)
            module.run_drift_stress_test(
                data,
                TARGETS,
                cfg,
                float(c),
                obs_mask=observed.reindex(index=data.index, columns=TARGETS),
                fault_mask=fault.reindex(index=data.index, columns=TARGETS),
            )
            summary = summarize_policy(
                candidate_dir / "r17_rescue_curve.csv",
                candidate_dir / "r27_fixed_weight_replay_audit.csv",
            )
            summary.update({"k": float(k), "q": float(q)})
            records.append(summary)
            dump(candidate_dir / "candidate_summary.json", summary)

    score = pd.DataFrame(
        [
            {
                "k": r["k"],
                "q": r["q"],
                "ready": r["ready"],
                "macro_positive_rescue": r["macro_positive_rescue"],
                "minimum_positive_rescue": r["minimum_positive_rescue"],
                "minimum_nominal_rescue": r["minimum_nominal_rescue"],
                "minimum_safety_coverage": r["minimum_safety_coverage"],
                "replay_ok": r["replay_ok"],
            }
            for r in records
        ]
    )
    score.to_csv(output / "policy_selection.csv", index=False)

    ready = [r for r in records if r["ready"]]
    if not ready:
        raise RuntimeError(
            "No policy candidate met the predeclared development criteria. "
            "Fresh confirmation must not run."
        )
    return max(ready, key=candidate_rank)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--canonical-manifest", required=True)
    args = p.parse_args()

    if set(DEVELOPMENT_SEEDS) & set(CONFIRMATION_SEEDS):
        raise RuntimeError("Development and confirmation seeds overlap.")

    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)

    canonical = json.loads(Path(args.canonical_manifest).read_text(encoding="utf-8"))
    if canonical.get("status") != "PASS" or canonical.get("dqa_regenerated") is not False:
        raise RuntimeError("Canonical DQA manifest is not valid.")

    module = load_module(Path(args.source).resolve())
    cfg = module.Config()
    cfg.FINAL_USE_ATTENTION = True
    cfg.FINAL_USE_CNN = True
    cfg.FINAL_UNIFIED_ENCODER = False

    data = module.load_water_quality_data(cfg.FILE_PATH_STABLE, cfg)
    observed = module.load_observation_mask(cfg.FILE_PATH_STABLE, cfg)
    fault = module.load_fault_mask(cfg.FILE_PATH_STABLE, cfg)
    if data is None or observed is None or fault is None:
        raise RuntimeError("Stable data or canonical masks are unavailable.")

    architecture = select_architecture(module, data, output)
    selected_c = select_c(module, data, observed, fault, architecture, output)
    policy = select_policy(
        module, data, observed, fault, architecture, selected_c, output
    )

    selected = {
        "protocol": "M2P_PTAC_FRESH_SELECTION_V1",
        "architecture": architecture,
        "C": selected_c,
        "k": policy["k"],
        "q": policy["q"],
        "fixed_method_design": {
            "attention": True,
            "cnn": True,
            "unified_encoder": False,
            "R17_TAU_KAPPA_LEVEL": 0.0,
            "R28_hard_switch": True,
            "R21_dynamic_selector": True,
            "Point_R20": True,
            "R19_safety_veto": True,
        },
        "development_seeds": DEVELOPMENT_SEEDS,
        "confirmation_seed_firewall": CONFIRMATION_SEEDS,
        "architecture_search_space": {
            "BILSTM_UNITS": [64, 80, 96, 112, 128],
            "CONV_FILTERS": [64, 80, 96, 112, 128],
            "ATTENTION_HEADS": [2, 4, 8],
            "HEAD_DENSE_UNITS": [64, 128, 256],
            "DROPOUT_RATE": [0.2, 0.3, 0.4, 0.5],
            "N_ITER_SEARCH": 25,
        },
        "C_search_space": C_GRID,
        "policy_search_space": {"k": K_GRID, "q": Q_GRID},
        "policy_development_summary": policy,
        "canonical_data_manifest_sha256": canonical["canonical_data_manifest_sha256"],
    }
    dump(output / "selected_configuration.json", selected)
    dump(output / "best_hyperparameters.json", architecture)
    print(json.dumps(selected, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
