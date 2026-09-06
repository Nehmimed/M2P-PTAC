#!/usr/bin/env python3
"""Single-use fresh-seed confirmation of the selected four-target model."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

TARGETS = [
    "temperature",
    "ph",
    "electric conductivity",
    "dissolved oxygen",
]
SEEDS = [88191, 26722, 35230, 45158, 73564]
NOMINAL_FLOOR = -0.02
SAFETY_COVERAGE_FLOOR = 0.80
MACRO_RESCUE_FLOOR = 0.20


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("m2p_confirmation", str(path))
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


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def acquire_lock(lock_root: Path, selected_sha: str):
    lock_root.mkdir(parents=True, exist_ok=True)
    lock = lock_root / "confirmation_seeds_consumed.lock.json"
    payload = {
        "seeds": SEEDS,
        "selected_configuration_sha256": selected_sha,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    try:
        fd = os.open(str(lock), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    except FileExistsError as exc:
        raise RuntimeError(f"Confirmation seeds have already been consumed: {lock}") from exc
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return lock


def summarize(curve_path: Path, replay_path: Path):
    curve = pd.read_csv(curve_path)
    curve = curve[curve["drift_type"].astype(str).eq("sudden_shift")].copy()
    curve["intensity"] = pd.to_numeric(curve["intensity"], errors="coerce")
    curve["rescue"] = pd.to_numeric(curve["rescue"], errors="coerce")
    curve["safety_bool"] = curve["safety_pass"].map(boolish)

    rows = []
    for target in TARGETS:
        target_data = curve[curve["target"].astype(str).eq(target)]
        for seed in SEEDS:
            seed_data = target_data[target_data["seed"].eq(seed)]
            nominal = seed_data.loc[seed_data["intensity"].eq(0), "rescue"].dropna()
            positive = seed_data.loc[seed_data["intensity"].gt(0), "rescue"].dropna()
            if nominal.empty or positive.empty:
                raise RuntimeError(f"Incomplete confirmation cell: {target}, seed={seed}")
            rows.append(
                {
                    "target": target,
                    "seed": seed,
                    "nominal_rescue": float(nominal.mean()),
                    "positive_rescue": float(positive.mean()),
                    "safety_pass": bool(
                        seed_data.sort_values("intensity")["safety_bool"].iloc[0]
                    ),
                }
            )

    seed_table = pd.DataFrame(rows)
    target_table = (
        seed_table.groupby("target", as_index=False)
        .agg(
            nominal_rescue_mean=("nominal_rescue", "mean"),
            positive_rescue_mean=("positive_rescue", "mean"),
            positive_rescue_sd=("positive_rescue", "std"),
            safety_coverage=("safety_pass", "mean"),
        )
    )
    macro_seed = (
        seed_table.groupby("seed", as_index=False)["positive_rescue"]
        .mean()
        .rename(columns={"positive_rescue": "macro_positive_rescue"})
    )
    values = macro_seed["macro_positive_rescue"].to_numpy(float)
    mean = float(values.mean())
    sem = float(stats.sem(values))
    tcrit = float(stats.t.ppf(0.975, len(values) - 1))

    replay = pd.read_csv(replay_path)
    replay = replay[replay["drift_type"].astype(str).eq("sudden_shift")].copy()
    positive_replay = replay[pd.to_numeric(replay["intensity"], errors="coerce").gt(0)]
    replay_ok = bool(
        len(positive_replay)
        and positive_replay["weights_reused"].map(boolish).all()
    )
    if "weight_sha256" in replay.columns:
        replay_ok = replay_ok and bool(
            (
                replay.groupby(["drift_type", "seed"])["weight_sha256"].nunique()
                == 1
            ).all()
        )

    aggregate = {
        "macro_positive_rescue": float(target_table["positive_rescue_mean"].mean()),
        "macro_seed_mean": mean,
        "macro_seed_ci95": [mean - tcrit * sem, mean + tcrit * sem],
    }
    return target_table, seed_table, macro_seed, aggregate, replay_ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["preflight", "run"], required=True)
    p.add_argument("--source", required=True)
    p.add_argument("--selected-config", required=True)
    p.add_argument("--canonical-manifest", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--lock-root", required=True)
    args = p.parse_args()

    source = Path(args.source).resolve()
    selected_path = Path(args.selected_config).resolve()
    canonical_path = Path(args.canonical_manifest).resolve()
    output = Path(args.output).resolve()
    lock_root = Path(args.lock_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "model").mkdir(parents=True, exist_ok=True)

    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    canonical = json.loads(canonical_path.read_text(encoding="utf-8"))
    if set(selected["development_seeds"]) & set(SEEDS):
        raise RuntimeError("Development/confirmation seed collision.")
    if canonical.get("dqa_regenerated") is not False:
        raise RuntimeError("Confirmation requires the frozen canonical DQA handoff.")

    manifest = {
        "protocol": "M2P_PTAC_FOUR_TARGET_CONFIRMATION_V1",
        "selected_configuration_sha256": sha256(selected_path),
        "canonical_data_manifest_sha256": canonical["canonical_data_manifest_sha256"],
        "targets": TARGETS,
        "seeds": SEEDS,
        "C": selected["C"],
        "k": selected["k"],
        "q": selected["q"],
        "pass_criteria": {
            "fixed_weight_replay": True,
            "each_target_nominal_rescue_ge": NOMINAL_FLOOR,
            "each_target_positive_rescue_gt": 0.0,
            "each_target_safety_coverage_ge": SAFETY_COVERAGE_FLOOR,
            "macro_positive_rescue_gt": MACRO_RESCUE_FLOOR,
        },
    }
    manifest_path = output / "confirmation_manifest.json"

    if args.mode == "preflight":
        if (lock_root / "confirmation_seeds_consumed.lock.json").exists():
            raise RuntimeError("Confirmation seeds have already been consumed.")
        dump(manifest_path, manifest)
        dump(output / "preflight.json", {"status": "PASS", "seeds_used": False})
        print(json.dumps({"status": "PREFLIGHT_PASS"}, indent=2))
        return

    frozen = json.loads(manifest_path.read_text(encoding="utf-8"))
    if frozen != manifest:
        raise RuntimeError("Confirmation manifest changed after preflight.")

    lock = acquire_lock(lock_root, sha256(selected_path))
    module = load_module(source)
    cfg = module.apply_selected_configuration(module.Config(), str(selected_path))
    cfg.OUTPUT_DIR = str(output / "model")
    cfg.R22_STRESS_PROTOCOL = True
    cfg.R22_STRESS_SELECT_C = False
    cfg.R17_RESCUE_SEEDS = list(SEEDS)
    cfg.R27_FIXED_MODEL_REPLAY = True
    cfg.R27_COMPONENT_AUDIT = False
    cfg.R25_MECHANISM_DIAGNOSTIC = False
    cfg.TOURNAMENT_ACTIVE = False

    try:
        data = module.load_water_quality_data(cfg.FILE_PATH_STABLE, cfg)
        observed = module.load_observation_mask(cfg.FILE_PATH_STABLE, cfg)
        fault = module.load_fault_mask(cfg.FILE_PATH_STABLE, cfg)
        module.run_drift_stress_test(
            data,
            TARGETS,
            cfg,
            float(selected["C"]),
            obs_mask=observed.reindex(index=data.index, columns=TARGETS),
            fault_mask=fault.reindex(index=data.index, columns=TARGETS),
        )

        model_dir = output / "model"
        target, seed, macro, aggregate, replay_ok = summarize(
            model_dir / "r17_rescue_curve.csv",
            model_dir / "r27_fixed_weight_replay_audit.csv",
        )
        target["nominal_ok"] = target["nominal_rescue_mean"] >= NOMINAL_FLOOR
        target["rescue_ok"] = target["positive_rescue_mean"] > 0.0
        target["safety_ok"] = target["safety_coverage"] >= SAFETY_COVERAGE_FLOOR

        criteria = {
            "fixed_weight_replay": replay_ok,
            "all_targets_nominal": bool(target["nominal_ok"].all()),
            "all_targets_positive_rescue": bool(target["rescue_ok"].all()),
            "all_targets_safety_coverage": bool(target["safety_ok"].all()),
            "macro_rescue": bool(
                aggregate["macro_positive_rescue"] > MACRO_RESCUE_FLOOR
            ),
        }
        confirmed = bool(all(criteria.values()))

        target.to_csv(output / "target_summary.csv", index=False)
        seed.to_csv(output / "seed_summary.csv", index=False)
        macro.to_csv(output / "macro_seed_summary.csv", index=False)
        verdict = {
            "status": "CONFIRMED" if confirmed else "NOT_CONFIRMED",
            "confirmed": confirmed,
            "criteria": criteria,
            "aggregate": aggregate,
            "selected_configuration": selected,
            "seed_lock": str(lock),
        }
        dump(output / "confirmation_verdict.json", verdict)
        print(json.dumps(verdict, indent=2, sort_keys=True, default=str))
    except Exception as exc:
        dump(
            output / "technical_error.json",
            {
                "error": repr(exc),
                "confirmation_seeds_consumed": True,
                "seed_lock": str(lock),
            },
        )
        raise


if __name__ == "__main__":
    main()
