#!/usr/bin/env python3
"""Safety-stress evaluation."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd


RUNNER_VERSION = "study_b_v2_paired_risk_reporting_provenance"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                   encoding="utf-8")
    os.replace(tmp, path)


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("m2p_full_for_study_b", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def resolve(args):
    project = Path(args.project_dir).resolve()
    main_script = Path(args.main_script)
    if not main_script.is_absolute():
        main_script = (project / main_script).resolve()
    output = Path(
        args.output_root
        or os.environ.get("M2P_OUTPUT_DIR")
        or (project / "model_results_OJIES_RR_FINAL_STRONG")
    ).resolve()
    return project, main_script, output


def child_root(output: Path, seed: int) -> Path:
    return output / "additional_studies" / "_study_B_seed_runs" / f"seed{seed}"


def child_csv(output: Path, seed: int) -> Path:
    return (child_root(output, seed) / "additional_studies" /
            "study_B_safety_stress" / "study_b_safety_stress_per_seed.csv")


def expected_keys(cfg, seed: int):
    scenarios = list(getattr(cfg, "STUDY_B_SCENARIOS", ()))
    targets = ["temperature", "ph", "electric conductivity", "dissolved oxygen"]
    arms = ["B_anchor", "C_unprotected", "C_bounded_only",
            "C_gated_no_veto", "C_final"]
    return {(int(seed), scenario, arm, target)
            for scenario in scenarios for arm in arms for target in targets}


def seed_file_complete(path: Path, cfg, seed: int, main_sha: str) -> bool:
    if not path.exists():
        return False
    try:
        d = pd.read_csv(path)
    except Exception:
        return False
    required = {"seed", "scenario", "arm", "target", "main_script_sha256"}
    if d.empty or not required.issubset(d.columns):
        return False
    got = {(int(r.seed), str(r.scenario), str(r.arm), str(r.target))
           for r in d.itertuples(index=False)}
    hashes = set(d["main_script_sha256"].astype(str))
    return got == expected_keys(cfg, seed) and hashes == {main_sha}


def write_manifest(output: Path, main_script: Path, cfg, selected_c, selected_arm,
                   provenance_status: str) -> Path:
    final_dir = output / "additional_studies" / "study_B_safety_stress"
    payload = {
        "runner_version": RUNNER_VERSION,
        "runner_script_sha256": sha256_file(Path(__file__).resolve()),
        "main_script": str(main_script.resolve()),
        "main_script_sha256": sha256_file(main_script),
        "selected_c": float(selected_c),
        "selected_arm": str(selected_arm),
        "seeds": list(getattr(cfg, "STUDY_B_SEEDS", [42, 43, 44, 45, 46])),
        "scenarios": list(getattr(cfg, "STUDY_B_SCENARIOS", ())),
        "catastrophic_threshold": float(getattr(
            cfg, "STUDY_B_CATASTROPHIC_MAE_RATIO", 2.0)),
        "attribution_anchor": "XGBoost (differenced target + drift bank)",
        "paired_uncertainty": (
            "paired row bootstrap 95% CI and exact McNemar test"),
        "provenance_status": provenance_status,
    }
    path = final_dir / "study_b_manifest.json"
    atomic_json(payload, path)
    return path


def run_child(args):
    project, main_script, output = resolve(args)
    os.chdir(project)
    m = load_module(main_script)
    cfg, selected_c, selected_arm = m.load_completed_main_configuration(output)

    seed = int(args.seed)
    expected = list(getattr(cfg, "STUDY_B_SEEDS", [42, 43, 44, 45, 46]))
    if seed not in expected:
        raise ValueError(f"Seed {seed} is not in Study-B seeds {expected}")

    root = child_root(output, seed)
    root.mkdir(parents=True, exist_ok=True)
    cfg.OUTPUT_DIR = str(root)
    cfg.STUDY_B_SEEDS = [seed]

    m.install_reviewer_logging(root, f"study_b_seed{seed}")
    print(f"[RUN] Study B seed={seed}; selected arm={selected_arm}; C={selected_c:g}")

    df = m.load_water_quality_data(cfg.FILE_PATH_STABLE)
    if df is None:
        raise FileNotFoundError(cfg.FILE_PATH_STABLE)
    obs = m.load_observation_mask(cfg.FILE_PATH_STABLE, cfg)
    targets = [c for c in ["temperature", "ph", "electric conductivity",
                           "dissolved oxygen"] if c in df.columns]

    m.run_study_b_safety_stress(
        df, targets, cfg, selected_c, obs_mask=obs
    )
    out = child_csv(output, seed)
    if not out.exists():
        raise RuntimeError(f"Study-B seed output missing: {out}")
    d = pd.read_csv(out)
    d["main_script_sha256"] = sha256_file(main_script)
    d["runner_version"] = RUNNER_VERSION
    d.to_csv(out, index=False)
    if not seed_file_complete(out, cfg, seed, sha256_file(main_script)):
        raise RuntimeError(f"Study-B seed output is incomplete: {out}")
    print(f"[DONE] Study B seed={seed}: {out}")


def merge(output: Path, main_script: Path):
    m = load_module(main_script)
    cfg, selected_c, selected_arm = m.load_completed_main_configuration(output)
    seeds = list(getattr(cfg, "STUDY_B_SEEDS", [42, 43, 44, 45, 46]))
    parts = []

    for seed in seeds:
        p = child_csv(output, seed)
        if not p.exists():
            raise FileNotFoundError(
                f"Missing Study-B seed output {p}. Rerun this script to resume.")
        parts.append(pd.read_csv(p))

    raw = pd.concat(parts, ignore_index=True)
    if "main_script_sha256" not in raw.columns:
        raw["main_script_sha256"] = "legacy_unverified"
    if "runner_version" not in raw.columns:
        raw["runner_version"] = "legacy_unverified"
    final_dir = output / "additional_studies" / "study_B_safety_stress"
    final_dir.mkdir(parents=True, exist_ok=True)
    raw.to_csv(final_dir / "study_b_safety_stress_per_seed.csv", index=False)

    summary, verdict = m._summarise_study_b_rows(raw, cfg, str(final_dir))
    current_hash = sha256_file(main_script)
    hashes = set(raw["main_script_sha256"].astype(str))
    provenance = ("verified" if hashes == {current_hash}
                  else "legacy_or_mixed_unverified")
    manifest = write_manifest(
        output, main_script, cfg, selected_c, selected_arm, provenance)

    print("\n### STUDY B COMPLETE ###")
    print(f"[STUDY-B] selected arm={selected_arm}; C={selected_c:g}")
    print(f"[STUDY-B] rows={len(raw)}; seeds={sorted(raw.seed.unique().tolist())}")
    print(f"[STUDY-B] VERDICT: {'GREEN' if verdict.get('green') else 'RED/UNPROVEN'} -- {verdict}")
    print("[STUDY-B] event rates: "
          f"{100 * verdict.get('unprotected_event_rate', float('nan')):.2f}% -> "
          f"{100 * verdict.get('final_event_rate', float('nan')):.2f}%; "
          f"relative reduction={100 * verdict.get('relative_rate_reduction', float('nan')):.2f}%")
    print(f"[STUDY-B] outputs: {final_dir}")
    print(f"[STUDY-B] provenance={provenance}; manifest={manifest}")
    return summary, verdict


def run_master(args):
    project, main_script, output = resolve(args)
    if not main_script.exists():
        raise FileNotFoundError(main_script)
    m = load_module(main_script)
    cfg, selected_c, selected_arm = m.load_completed_main_configuration(output)
    seeds = list(getattr(cfg, "STUDY_B_SEEDS", [42, 43, 44, 45, 46]))

    print("\n### STUDY B | SEPARATE JOB ###")
    print(f"[RUN] main output={output}")
    print(f"[RUN] selected arm={selected_arm}; C={selected_c:g}")
    print(f"[RUN] main SHA256={sha256_file(main_script)}")
    print(f"[RUN] seeds={seeds}; one fresh subprocess per seed")

    for k, seed in enumerate(seeds, 1):
        p = child_csv(output, seed)
        if seed_file_complete(p, cfg, seed, sha256_file(main_script)):
            print(f"[{k}/{len(seeds)}] SKIP seed={seed}: complete and hash-matched")
            continue
        print(f"[{k}/{len(seeds)}] RUN seed={seed}")
        cmd = [
            sys.executable, str(Path(__file__).resolve()),
            "--child", "--seed", str(seed),
            "--project-dir", str(project),
            "--main-script", str(main_script),
            "--output-root", str(output),
        ]
        rc = subprocess.run(cmd).returncode
        if rc != 0:
            raise SystemExit(rc)

    merge(output, main_script)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-dir",
                    default=os.environ.get("M2P_PROJECT_DIR", os.getcwd()))
    ap.add_argument("--main-script",
                    default=os.environ.get("M2P_MAIN_SCRIPT", "full_exp.py"))
    ap.add_argument("--output-root", default=os.environ.get("M2P_OUTPUT_DIR"))
    ap.add_argument("--child", action="store_true")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--merge-only", action="store_true")
    args = ap.parse_args()

    project, main_script, output = resolve(args)
    if args.merge_only:
        merge(output, main_script)
    elif args.child:
        if args.seed is None:
            ap.error("--child requires --seed")
        run_child(args)
    else:
        run_master(args)


if __name__ == "__main__":
    main()
