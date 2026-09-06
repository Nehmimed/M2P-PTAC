#!/usr/bin/env python3
"""Architecture and supervision ablation study."""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


# Frozen strong-MHA Study-C contract.
DEFAULT_SEEDS = [42, 43, 44, 45, 46]
DEFAULT_INTENSITIES = [0.10, 0.15, 0.20, 0.25]
DEFAULT_VARIANTS = [
    "Full",
    "No Attention",
    "No CNN",
    "Unified Encoder",
    "Ordinary Residual Target",
]
DEFAULT_TARGETS = [
    "temperature",
    "ph",
    "electric conductivity",
    "dissolved oxygen",
]

STUDY_C_VERSION = "strong_mha_v2_four_target_post_veto_delta_reporting"


def safe_name(s: str) -> str:
    return "".join(
        ch if ch.isalnum() or ch in ("-", "_") else "_"
        for ch in str(s)
    )


def intensity_tag(x: float) -> str:
    return (
        f"{float(x):.3f}"
        .rstrip("0")
        .rstrip(".")
        .replace(".", "p")
    )


def atomic_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def atomic_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(obj, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def student_t_mean_ci(values, confidence: float = 0.95):
    """Return mean and two-sided Student-t CI (appropriate for five seeds)."""
    x = np.asarray(values, dtype=float).ravel()
    x = x[np.isfinite(x)]
    n = int(x.size)
    if n == 0:
        return np.nan, np.nan, np.nan
    mean = float(np.mean(x))
    if n < 2:
        return mean, np.nan, np.nan
    sem = float(scipy_stats.sem(x, nan_policy="omit"))
    critical = float(scipy_stats.t.ppf((1.0 + confidence) / 2.0, n - 1))
    half_width = critical * sem
    return mean, mean - half_width, mean + half_width


def load_main(path: Path):
    """
    Import the completed main experiment.

    This is called only inside a Study-C cell subprocess, never in the master
    process, so the parent job does not keep an extra TensorFlow runtime alive.
    """
    spec = importlib.util.spec_from_file_location(
        "m2p_rr_studyc",
        str(path),
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import main script: {path}")

    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def resolve_paths(args):
    project = Path(
        args.project_dir
        or os.environ.get("M2P_PROJECT_DIR")
        or os.getcwd()
    ).resolve()

    output = Path(
        args.output_root
        or os.environ.get("M2P_OUTPUT_DIR")
        or (project / "model_results_THESIS_REVISION_FINAL")
    ).resolve()

    main_script = Path(
        args.main_script
        or os.environ.get("M2P_MAIN_SCRIPT")
        or "full_exp.py"
    )
    if not main_script.is_absolute():
        main_script = (project / main_script).resolve()

    return project, output, main_script


def validate_completed_output(output: Path) -> None:
    if not output.exists():
        raise FileNotFoundError(
            f"Output directory does not exist: {output}"
        )

    required = [
        output / "best_hyperparameters.json",
        output / "final_results_metrics_stable.csv",
        output / "drift_stress_results_per_seed.csv",
    ]
    missing = [p for p in required if not p.exists()]

    if missing:
        raise FileNotFoundError(
            "This does not look like the completed main output directory. "
            "Missing:\n  " + "\n  ".join(str(p) for p in missing)
        )

    print("[CHECK] Completed main benchmark outputs found.", flush=True)
    for p in required:
        print(f"[CHECK] Reusing: {p}", flush=True)


def load_hp_json(output: Path) -> dict:
    hp_path = output / "best_hyperparameters.json"
    with hp_path.open("r", encoding="utf-8") as f:
        hp = json.load(f)

    if not isinstance(hp, dict):
        raise TypeError(
            f"Expected a JSON object in {hp_path}, got {type(hp).__name__}"
        )
    return hp


def print_tuned_hyperparameters(output: Path) -> None:
    hp = load_hp_json(output)
    print("[CHECK] Reused tuned hyperparameters:", flush=True)
    for key in [
        "BILSTM_UNITS",
        "CONV_FILTERS",
        "ATTENTION_HEADS",
        "HEAD_DENSE_UNITS",
        "DROPOUT_RATE",
    ]:
        value = hp.get(key, hp.get(key.lower()))
        print(f"        {key} = {value}", flush=True)


def validate_study_c_contract(cfg) -> None:
    """
    Refuse to run if full_exp.py exposes a different Study-C design.

    This protects the strong-MHA ablation from accidentally inheriting the
    earlier compact/no-attention variant list.
    """
    cfg_seeds = list(
        getattr(cfg, "STUDY_C_SEEDS", DEFAULT_SEEDS)
    )
    cfg_intensities = [
        float(x)
        for x in getattr(
            cfg,
            "STUDY_C_INTENSITIES",
            tuple(DEFAULT_INTENSITIES),
        )
    ]
    cfg_targets = list(
        getattr(cfg, "STUDY_C_TARGETS", tuple(DEFAULT_TARGETS))
    )
    cfg_variants = list(
        getattr(cfg, "STUDY_C_VARIANTS", tuple(DEFAULT_VARIANTS))
    )

    problems = []

    if cfg_seeds != DEFAULT_SEEDS:
        problems.append(
            f"STUDY_C_SEEDS={cfg_seeds}, expected {DEFAULT_SEEDS}"
        )

    if len(cfg_intensities) != len(DEFAULT_INTENSITIES) or any(
        abs(a - b) > 1e-12
        for a, b in zip(cfg_intensities, DEFAULT_INTENSITIES)
    ):
        problems.append(
            "STUDY_C_INTENSITIES="
            f"{cfg_intensities}, expected {DEFAULT_INTENSITIES}"
        )

    if cfg_targets != DEFAULT_TARGETS:
        problems.append(
            f"STUDY_C_TARGETS={cfg_targets}, expected {DEFAULT_TARGETS}"
        )

    if cfg_variants != DEFAULT_VARIANTS:
        problems.append(
            f"STUDY_C_VARIANTS={cfg_variants}, expected {DEFAULT_VARIANTS}"
        )

    if problems:
        raise RuntimeError(
            "Study-C contract mismatch. The standalone runner is configured "
            "for the frozen strong-MHA experiment, but full_exp.py exposes a "
            "different Study-C definition:\n  - "
            + "\n  - ".join(problems)
        )


def study_dir_for(output: Path) -> Path:
    return (
        output
        / "additional_studies"
        / "study_C_active_ablations_v2"
    )


def manifest_payload(
    output: Path,
    main_script: Path,
) -> dict:
    return {
        "study_c_version": STUDY_C_VERSION,
        "runner_script_sha256": sha256_file(Path(__file__).resolve()),
        "main_script": str(main_script.resolve()),
        "main_script_sha256": sha256_file(main_script),
        "best_hyperparameters": load_hp_json(output),
        "seeds": DEFAULT_SEEDS,
        "intensities": DEFAULT_INTENSITIES,
        "targets": DEFAULT_TARGETS,
        "variants": DEFAULT_VARIANTS,
        "row_contract": 400,
        "metric_output": "post_veto",
        "confidence_interval": "paired/two-sided Student-t with df=n-1",
        "attribution_anchor": "XGBoost (differenced target + drift bank)",
        "full_arm": {
            "use_attention": True,
            "use_cnn": True,
            "unified_encoder": False,
            "t2_target": True,
        },
    }


def ensure_manifest(
    output: Path,
    main_script: Path,
) -> Path:
    """
    Create or validate Study-C provenance.

    If chunks already exist without a manifest, fail rather than assume they
    belong to this architecture.
    """
    study = study_dir_for(output)
    study.mkdir(parents=True, exist_ok=True)

    manifest_path = study / "study_c_manifest.json"
    expected = manifest_payload(output, main_script)

    if manifest_path.exists():
        current = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )
        if current != expected:
            raise RuntimeError(
                "Existing Study-C manifest does not match the current strong "
                "MHA experiment. Archive or remove the old "
                f"{study} directory before running this configuration."
            )
        print(
            f"[CHECK] Study-C manifest matches: {manifest_path}",
            flush=True,
        )
        return manifest_path

    chunk_dir = study / "chunks"
    existing_chunks = (
        list(chunk_dir.glob("study_c_seed*_I*.csv"))
        if chunk_dir.exists()
        else []
    )
    if existing_chunks:
        raise RuntimeError(
            "Study-C chunks already exist but no provenance manifest is "
            "present. They may belong to another architecture. Archive or "
            f"remove {study} before rerunning."
        )

    atomic_json(expected, manifest_path)
    print(
        f"[CHECK] Created Study-C manifest: {manifest_path}",
        flush=True,
    )
    return manifest_path


def cell_paths(
    output: Path,
    seed: int,
    intensity: float,
):
    study_dir = study_dir_for(output)
    chunk_dir = study_dir / "chunks"
    cell_dir = (
        study_dir
        / "work"
        / f"seed{seed}_I{intensity_tag(intensity)}"
    )
    chunk = (
        chunk_dir
        / f"study_c_seed{seed}_I{intensity_tag(intensity)}.csv"
    )
    done = cell_dir / "DONE"
    return study_dir, chunk_dir, cell_dir, chunk, done


def is_cell_complete(
    output: Path,
    seed: int,
    intensity: float,
    variants=DEFAULT_VARIANTS,
    targets=DEFAULT_TARGETS,
) -> bool:
    _, _, _, chunk, done = cell_paths(
        output,
        seed,
        intensity,
    )

    if not chunk.exists():
        return False

    try:
        d = pd.read_csv(chunk)
    except Exception:
        return False

    required_cols = {
        "variant", "target", "xgb_diff_anchor_mae", "nn_delta_mae",
        "m2p_final_mae", "nominal_penalty_rel", "overall_final_mae",
        "gate_applied", "metric_output", "main_script_sha256",
    }
    if d.empty or not required_cols.issubset(d.columns):
        return False

    got = set(
        zip(
            d["variant"].astype(str),
            d["target"].astype(str),
        )
    )
    expected = {
        (variant, target)
        for variant in variants
        for target in targets
    }

    ok = (
        got == expected
        and d["metric_output"].astype(str).eq("post_veto").all()
        and not d.duplicated(["variant", "target"]).any()
    )
    if ok and not done.exists():
        done.parent.mkdir(parents=True, exist_ok=True)
        done.write_text(
            f"seed={seed}\n"
            f"intensity={intensity}\n"
            f"rows={len(d)}\n",
            encoding="utf-8",
        )
    return ok


def variant_settings(variant: str):
    """
    Return the frozen strong-MHA architecture for one ablation.

    Full is always the strong MHA model. Each architectural ablation changes
    only the named component.
    """
    use_attention = True
    use_cnn = True
    unified_encoder = False
    t2_target = True

    if variant == "Full":
        pass
    elif variant == "No Attention":
        use_attention = False
    elif variant == "No CNN":
        use_cnn = False
    elif variant == "Unified Encoder":
        unified_encoder = True
    elif variant == "Ordinary Residual Target":
        t2_target = False
    else:
        raise ValueError(f"Unknown Study-C variant: {variant}")

    return (
        use_attention,
        use_cnn,
        unified_encoder,
        t2_target,
    )


def run_cell(args) -> None:
    project, output, main_script = resolve_paths(args)
    os.chdir(project)

    validate_completed_output(output)

    if not main_script.exists():
        raise FileNotFoundError(
            f"Main experiment script not found: {main_script}"
        )

    ensure_manifest(output, main_script)

    seed = int(args.seed)
    intensity = float(args.intensity)

    if seed not in DEFAULT_SEEDS:
        raise ValueError(
            f"Seed {seed} not in predeclared Study-C seeds: "
            f"{DEFAULT_SEEDS}"
        )

    if not any(
        abs(intensity - x) < 1e-12
        for x in DEFAULT_INTENSITIES
    ):
        raise ValueError(
            f"Intensity {intensity} not in predeclared "
            f"Study-C intensities: {DEFAULT_INTENSITIES}"
        )

    print(
        f"[CELL] importing main experiment for "
        f"seed={seed}, I={intensity:g} ...",
        flush=True,
    )
    m = load_main(main_script)
    print("[CELL] main experiment import complete.", flush=True)

    cfg, optimal_c, selected_arm = (
        m.load_completed_main_configuration(output)
    )
    validate_study_c_contract(cfg)

    # Load the same stable reference testbed used by the original Study C.
    df = m.load_water_quality_data(cfg.FILE_PATH_STABLE)
    if df is None:
        raise FileNotFoundError(
            "Stable dataset could not be loaded from "
            f"{cfg.FILE_PATH_STABLE}"
        )

    obs_mask = m.load_observation_mask(
        cfg.FILE_PATH_STABLE,
        cfg,
    )

    # The model remains multivariate; Study C reports all four prespecified
    # water-quality targets used by the main benchmark.
    target_vars = [
        c
        for c in [
            "temperature",
            "ph",
            "electric conductivity",
            "dissolved oxygen",
        ]
        if c in df.columns
    ]

    missing_targets = [
        t for t in DEFAULT_TARGETS
        if t not in target_vars
    ]
    if missing_targets:
        raise RuntimeError(
            "Stable dataset is missing Study-C claim targets: "
            f"{missing_targets}"
        )

    claim_targets = list(DEFAULT_TARGETS)
    variants = list(DEFAULT_VARIANTS)

    (
        study_dir,
        chunk_dir,
        cell_dir,
        chunk_path,
        done_marker,
    ) = cell_paths(
        output,
        seed,
        intensity,
    )

    chunk_dir.mkdir(parents=True, exist_ok=True)
    cell_dir.mkdir(parents=True, exist_ok=True)

    m.install_reviewer_logging(
        cell_dir,
        f"study_c_seed{seed}_I{intensity_tag(intensity)}",
    )

    print(
        f"[RUN] Study C seed={seed}, I={intensity:g}; "
        f"selected arm={selected_arm}; C={optimal_c:g}",
        flush=True,
    )

    if is_cell_complete(
        output,
        seed,
        intensity,
        variants,
        claim_targets,
    ):
        print(
            f"[DONE/SKIP] seed={seed}, I={intensity:g} "
            "already complete.",
            flush=True,
        )
        return

    rows_df = (
        pd.read_csv(chunk_path)
        if chunk_path.exists()
        else pd.DataFrame()
    )

    completed_variants = set()
    _resume_required = {
        "variant", "target", "xgb_diff_anchor_mae", "nn_delta_mae",
        "nominal_penalty_rel", "overall_final_mae", "gate_applied",
        "metric_output", "main_script_sha256",
    }
    if not rows_df.empty and _resume_required.issubset(rows_df.columns):
        rows_df = rows_df[
            rows_df["main_script_sha256"].astype(str).eq(
                sha256_file(main_script)
            )
            & rows_df["metric_output"].astype(str).eq("post_veto")
        ].copy()
        counts = (
            rows_df.groupby("variant")["target"]
            .nunique()
            .to_dict()
        )
        completed_variants = {
            v
            for v, n in counts.items()
            if n >= len(claim_targets)
        }

    print(
        f"[RESUME] seed={seed}, I={intensity:g}; "
        f"completed={sorted(completed_variants)}",
        flush=True,
    )

    # Reproduce the original Study-C active regime exactly: one held-out
    # gradual degradation realization is generated per seed/intensity cell and
    # reused across all variants.
    base_cfg = copy.deepcopy(cfg)
    base_cfg.OUTPUT_DIR = str(cell_dir)
    base_cfg._dataset_name = "study_C_active_ablations"
    base_cfg.GATE_THRESHOLD = None
    base_cfg.STUDY_A_ENABLE_JUMP_CONTEXT = False
    base_cfg.R20_UTILITY_GATE = True

    m.ensure_cadence(
        df,
        base_cfg,
        "study_C_active_ablations",
    )
    m.set_all_seeds(seed)

    injector = m.SyntheticDriftInjector(base_cfg)
    degraded, clean = injector.apply_drift(
        df,
        "gradual_bias",
        intensity,
        target_vars,
        history_episodes=int(
            getattr(
                base_cfg,
                "R18_HISTORY_EPISODES",
                8,
            )
        ),
        history_seed=seed + 9001,
    )
    drift_mask = injector.last_drift_mask.copy()

    for variant in variants:
        if variant in completed_variants:
            print(
                f"[SKIP] {variant}: already checkpointed",
                flush=True,
            )
            continue

        (
            use_attention,
            use_cnn,
            unified_encoder,
            t2_target,
        ) = variant_settings(variant)

        variant_dir = cell_dir / safe_name(variant)
        variant_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        c = copy.deepcopy(base_cfg)
        c.OUTPUT_DIR = str(variant_dir)
        c._dataset_name = "study_C_active_ablations"

        m.set_all_seeds(seed)

        exp = None
        try:
            print(
                "\n" + "=" * 80,
                flush=True,
            )
            print(
                f"STUDY C ONLY | seed={seed} | "
                f"I={intensity:g} | {variant}",
                flush=True,
            )
            print(
                "architecture: "
                f"attention={use_attention}, "
                f"cnn={use_cnn}, "
                f"unified_encoder={unified_encoder}, "
                f"t2_target={t2_target}",
                flush=True,
            )
            print(
                "=" * 80,
                flush=True,
            )

            exp = m.PTACExperiment(
                (
                    f"StudyC_{safe_name(variant)}_"
                    f"I{intensity:g}_seed{seed}"
                ),
                c,
                base_lambda=optimal_c,
                unified_encoder=unified_encoder,
                use_attention=use_attention,
                use_cnn=use_cnn,
                conformal_alpha=c.DEFAULT_CONFORMAL_ALPHA,
                eval_truth_df=clean,
                obs_mask=obs_mask,
                fault_mask=drift_mask,
                train_truth_df=None,
                corrector_target_df=(
                    clean if t2_target else None
                ),
                seed=seed,
                gate_threshold=None,
            )

            exp.run(
                degraded,
                target_vars,
            )

            arr = getattr(
                exp,
                "_r19_arrays",
                {},
            )
            if not arr or "test_index" not in arr:
                raise RuntimeError(
                    "Study-C experiment did not expose "
                    "_r19_arrays."
                )

            idx = pd.DatetimeIndex(
                arr["test_index"]
            )
            base = np.asarray(
                arr["test_base"],
                dtype=float,
            )
            corr = np.asarray(
                arr["test_corr"],
                dtype=float,
            )
            true = np.asarray(
                arr["test_true"],
                dtype=float,
            )
            tg = list(arr["targets"])

            mm = (
                drift_mask
                .reindex(
                    index=idx,
                    columns=tg,
                )
                .fillna(False)
                .values
                .astype(bool)
            )

            safety_mask = np.asarray(
                getattr(
                    exp,
                    "_r19_safety_mask",
                    np.zeros(len(tg), dtype=bool),
                ),
                dtype=bool,
            )

            gate_mask = np.asarray(
                arr.get("gate_mask", np.ones(len(tg), dtype=bool)),
                dtype=bool,
            )
            if gate_mask.size != len(tg):
                raise RuntimeError(
                    "Study-C gate mask width does not match target width."
                )
            if (~gate_mask).any() and not np.allclose(
                corr[:, ~gate_mask], 0.0, rtol=0.0, atol=1e-12
            ):
                raise RuntimeError(
                    "Post-veto assertion failed: a gated-off target retained "
                    "a nonzero correction."
                )

            observed_mask = (
                obs_mask.reindex(index=idx, columns=tg)
                .fillna(False).values.astype(bool)
                if obs_mask is not None
                else np.ones_like(mm, dtype=bool)
            )
            pred = base + corr
            delta_true = true - base

            new_rows = []

            for j, target in enumerate(tg):
                if target not in claim_targets:
                    continue

                degraded_eval = (
                    mm[:, j]
                    & observed_mask[:, j]
                    & np.isfinite(true[:, j])
                    & np.isfinite(base[:, j])
                    & np.isfinite(corr[:, j])
                )
                valid_eval = (
                    observed_mask[:, j]
                    & np.isfinite(true[:, j])
                    & np.isfinite(base[:, j])
                    & np.isfinite(corr[:, j])
                )
                nominal_eval = valid_eval & (~mm[:, j])

                if not degraded_eval.any():
                    raise RuntimeError(
                        "No degraded test rows for Study-C "
                        f"target {target}."
                    )

                anchor_mae = float(
                    np.mean(
                        np.abs(
                            true[degraded_eval, j]
                            - base[degraded_eval, j]
                        )
                    )
                )

                corrected_mae = float(
                    np.mean(
                        np.abs(
                            true[degraded_eval, j]
                            - pred[degraded_eval, j]
                        )
                    )
                )

                nn_delta_mae = float(np.mean(np.abs(
                    delta_true[degraded_eval, j]
                    - corr[degraded_eval, j]
                )))
                if not np.isclose(
                    nn_delta_mae, corrected_mae,
                    rtol=0.0, atol=1e-12,
                ):
                    raise RuntimeError(
                        "Neural-delta MAE does not equal final forecast MAE; "
                        "anchor/correction alignment failed."
                    )

                nominal_anchor_mae = (
                    float(np.mean(np.abs(
                        true[nominal_eval, j] - base[nominal_eval, j]
                    ))) if nominal_eval.any() else np.nan
                )
                nominal_final_mae = (
                    float(np.mean(np.abs(
                        true[nominal_eval, j] - pred[nominal_eval, j]
                    ))) if nominal_eval.any() else np.nan
                )
                overall_anchor_mae = float(np.mean(np.abs(
                    true[valid_eval, j] - base[valid_eval, j]
                )))
                overall_final_mae = float(np.mean(np.abs(
                    true[valid_eval, j] - pred[valid_eval, j]
                )))

                rescue = (
                    1.0
                    - corrected_mae
                    / max(anchor_mae, 1e-12)
                )

                new_rows.append(
                    dict(
                        seed=seed,
                        intensity=intensity,
                        target=target,
                        variant=variant,
                        comparison_anchor=(
                            "XGBoost (differenced target + drift bank)"
                        ),
                        anchor_mae=anchor_mae,
                        corrected_mae=corrected_mae,
                        xgb_diff_anchor_mae=anchor_mae,
                        nn_delta_mae=nn_delta_mae,
                        m2p_final_mae=corrected_mae,
                        abs_gain_vs_xgb_diff=anchor_mae - corrected_mae,
                        pct_gain_vs_xgb_diff=(
                            100.0 * (anchor_mae - corrected_mae) / anchor_mae
                            if anchor_mae > 1e-12 else np.nan
                        ),
                        rescue=rescue,
                        nominal_anchor_mae=nominal_anchor_mae,
                        nominal_final_mae=nominal_final_mae,
                        nominal_penalty_rel=(
                            (nominal_final_mae - nominal_anchor_mae)
                            / nominal_anchor_mae
                            if np.isfinite(nominal_anchor_mae)
                            and nominal_anchor_mae > 1e-12 else np.nan
                        ),
                        overall_anchor_mae=overall_anchor_mae,
                        overall_final_mae=overall_final_mae,
                        overall_abs_gain=(
                            overall_anchor_mae - overall_final_mae
                        ),
                        overall_pct_gain=(
                            100.0 * (overall_anchor_mae - overall_final_mae)
                            / overall_anchor_mae
                            if overall_anchor_mae > 1e-12 else np.nan
                        ),
                        n_drift_observed=int(degraded_eval.sum()),
                        n_nominal_observed=int(nominal_eval.sum()),
                        n_overall_observed=int(valid_eval.sum()),
                        max_abs_correction=float(
                            np.nanmax(
                                np.abs(corr[:, j])
                            )
                        ),
                        safety_pass=(
                            bool(safety_mask[j])
                            if j < len(safety_mask)
                            else False
                        ),
                        gate_applied=bool(gate_mask[j]),
                        metric_output="post_veto",
                        use_attention=use_attention,
                        use_cnn=use_cnn,
                        unified_encoder=unified_encoder,
                        t2_target=t2_target,
                        selected_c=float(optimal_c),
                        main_script_sha256=sha256_file(
                            main_script
                        ),
                    )
                )

            if len(new_rows) != len(claim_targets):
                got_targets = {
                    r["target"]
                    for r in new_rows
                }
                raise RuntimeError(
                    f"Variant {variant} produced "
                    f"{len(new_rows)}/{len(claim_targets)} "
                    "claim-target rows. Got "
                    f"{sorted(got_targets)}"
                )

            if rows_df.empty:
                rows_df = pd.DataFrame(
                    new_rows
                )
            else:
                rows_df = (
                    rows_df[
                        rows_df["variant"] != variant
                    ]
                    .copy()
                )
                rows_df = pd.concat(
                    [
                        rows_df,
                        pd.DataFrame(new_rows),
                    ],
                    ignore_index=True,
                )

            atomic_csv(
                rows_df,
                chunk_path,
            )
            print(
                f"[CHECKPOINT] {variant}: "
                f"saved -> {chunk_path}",
                flush=True,
            )

        except Exception:
            traceback.print_exc()
            raise

        finally:
            try:
                del exp
            except Exception:
                pass

            try:
                m.tf.keras.backend.clear_session()
            except Exception:
                pass

            gc.collect()

    final = pd.read_csv(
        chunk_path
    )

    expected = (
        len(variants)
        * len(claim_targets)
    )
    got = len(
        final.drop_duplicates(
            ["variant", "target"]
        )
    )

    if got != expected:
        raise RuntimeError(
            f"Cell incomplete: expected {expected} "
            f"variant-target rows, got {got}"
        )

    done_marker.write_text(
        f"seed={seed}\n"
        f"intensity={intensity}\n"
        f"rows={got}\n",
        encoding="utf-8",
    )

    print(
        f"[DONE] Study-C cell seed={seed}, "
        f"I={intensity:g}: {got}/{expected} rows.",
        flush=True,
    )


def merge_results(output: Path) -> None:
    study = study_dir_for(output)
    chunk_dir = study / "chunks"
    chunks = sorted(
        chunk_dir.glob(
            "study_c_seed*_I*.csv"
        )
    )

    if not chunks:
        raise FileNotFoundError(
            f"No Study-C chunks found in {chunk_dir}"
        )

    raw = pd.concat(
        [
            pd.read_csv(p)
            for p in chunks
        ],
        ignore_index=True,
    )

    raw = (
        raw.drop_duplicates(
            [
                "seed",
                "intensity",
                "target",
                "variant",
            ],
            keep="last",
        )
        .sort_values(
            [
                "seed",
                "intensity",
                "variant",
                "target",
            ]
        )
        .reset_index(drop=True)
    )

    expected_keys = {
        (
            seed,
            float(intensity),
            variant,
            target,
        )
        for seed in DEFAULT_SEEDS
        for intensity in DEFAULT_INTENSITIES
        for variant in DEFAULT_VARIANTS
        for target in DEFAULT_TARGETS
    }

    actual_keys = {
        (
            int(r.seed),
            float(r.intensity),
            str(r.variant),
            str(r.target),
        )
        for r in raw.itertuples(
            index=False
        )
    }

    missing = (
        expected_keys
        - actual_keys
    )

    if missing:
        print(
            f"[INCOMPLETE] rows={len(raw)}/400; "
            f"missing={len(missing)}",
            flush=True,
        )
        for item in sorted(missing)[:40]:
            print(
                "  missing:",
                item,
                flush=True,
            )
        raise SystemExit(2)

    # Keep only the frozen 400-row contract even if unrelated stale rows exist.
    raw = raw[
        raw.apply(
            lambda r: (
                int(r["seed"]),
                float(r["intensity"]),
                str(r["variant"]),
                str(r["target"]),
            )
            in expected_keys,
            axis=1,
        )
    ].copy()

    if len(raw) != 400:
        raise RuntimeError(
            f"Expected exactly 400 Study-C rows after "
            f"contract filtering, got {len(raw)}"
        )

    atomic_csv(
        raw,
        study
        / "study_c_active_ablations_per_seed.csv",
    )

    summary = (
        raw.groupby(
            [
                "variant",
                "intensity",
                "target",
            ]
        )
        .agg(
            n_seeds=("seed", "nunique"),
            rescue_mean=("rescue", "mean"),
            rescue_sd=("rescue", "std"),
            positive_seeds=(
                "rescue",
                lambda x: int(
                    np.sum(
                        np.asarray(x) > 0
                    )
                ),
            ),
            anchor_mae=("anchor_mae", "mean"),
            corrected_mae=(
                "corrected_mae",
                "mean",
            ),
            nn_delta_mae=("nn_delta_mae", "mean"),
            abs_gain_vs_xgb_diff=("abs_gain_vs_xgb_diff", "mean"),
            pct_gain_vs_xgb_diff=("pct_gain_vs_xgb_diff", "mean"),
            nominal_anchor_mae=("nominal_anchor_mae", "mean"),
            nominal_final_mae=("nominal_final_mae", "mean"),
            nominal_penalty_rel=("nominal_penalty_rel", "mean"),
            overall_anchor_mae=("overall_anchor_mae", "mean"),
            overall_final_mae=("overall_final_mae", "mean"),
            overall_abs_gain=("overall_abs_gain", "mean"),
            overall_pct_gain=("overall_pct_gain", "mean"),
            gate_applied_rate=("gate_applied", "mean"),
            safety_pass_rate=(
                "safety_pass",
                "mean",
            ),
        )
        .reset_index()
    )

    group_keys = ["variant", "intensity", "target"]
    ci = (
        raw.groupby(group_keys)["rescue"]
        .apply(
            lambda x: pd.Series(
                student_t_mean_ci(x),
                index=["rescue_t_mean", "rescue_ci_lo", "rescue_ci_hi"],
            )
        )
        .unstack()
        .reset_index()
    )
    summary = summary.merge(
        ci, on=group_keys, how="left", validate="one_to_one"
    )
    if not np.allclose(
        summary["rescue_mean"], summary["rescue_t_mean"],
        equal_nan=True, rtol=0.0, atol=1e-12,
    ):
        raise RuntimeError("Study-C mean/CI sample mismatch.")
    summary.drop(columns=["rescue_t_mean"], inplace=True)
    summary["ci_method"] = "two-sided Student-t, df=n_seeds-1"

    atomic_csv(
        summary,
        study
        / "study_c_active_ablations_summary.csv",
    )

    piv = raw.pivot_table(
        index=[
            "seed",
            "intensity",
            "target",
        ],
        columns="variant",
        values="rescue",
        aggfunc="first",
    )

    comp_rows = []

    if "Full" not in piv.columns:
        raise RuntimeError(
            "Full arm missing from Study-C merged results."
        )

    for variant in [
        v
        for v in DEFAULT_VARIANTS
        if v != "Full"
    ]:
        if variant not in piv.columns:
            raise RuntimeError(
                f"Ablation arm missing from merged results: "
                f"{variant}"
            )

        z = (
            (
                piv["Full"]
                - piv[variant]
            )
            .dropna()
            .rename("delta")
            .reset_index()
        )

        for (
            intensity,
            target,
        ), group in z.groupby(
            [
                "intensity",
                "target",
            ]
        ):
            values = (
                group["delta"]
                .values
                .astype(float)
            )
            n = len(values)
            mu, ci_lo, ci_hi = student_t_mean_ci(values)

            comp_rows.append(
                dict(
                    variant=variant,
                    intensity=float(
                        intensity
                    ),
                    target=target,
                    n_seeds=n,
                    full_minus_variant_rescue=mu,
                    ci_lo=ci_lo,
                    ci_hi=ci_hi,
                    ci_method=(
                        "paired two-sided Student-t, df=n_seeds-1"
                    ),
                    full_better_seeds=int(
                        np.sum(
                            values > 0
                        )
                    ),
                )
            )

    atomic_csv(
        pd.DataFrame(
            comp_rows
        ),
        study
        / "study_c_full_vs_ablation.csv",
    )

    print(
        "\n### STUDY C COMPLETE ###",
        flush=True,
    )
    print(
        "400 / 400 expected rows present.",
        flush=True,
    )
    print(
        f"Saved under: {study}",
        flush=True,
    )


def run_master(args) -> None:
    """
    Launch the 20 Study-C cells.

    Deliberately does NOT import full_exp.py. Importing the main experiment in
    both the parent and every child can leave two TensorFlow runtimes resident
    in the same allocation. Only cell subprocesses import the model.
    """
    (
        project,
        output,
        main_script,
    ) = resolve_paths(args)

    validate_completed_output(output)

    if not main_script.exists():
        raise FileNotFoundError(
            f"Main experiment script not found: {main_script}"
        )

    ensure_manifest(
        output,
        main_script,
    )
    print_tuned_hyperparameters(
        output
    )

    self_script = Path(
        __file__
    ).resolve()

    cells = [
        (seed, intensity)
        for seed in DEFAULT_SEEDS
        for intensity in DEFAULT_INTENSITIES
    ]

    print(
        "\n" + "#" * 80,
        flush=True,
    )
    print(
        "### STUDY-C-ONLY MASTER | STRONG MHA ###",
        flush=True,
    )
    print(
        "#" * 80,
        flush=True,
    )
    print(
        f"Project: {project}",
        flush=True,
    )
    print(
        f"Output : {output}",
        flush=True,
    )
    print(
        f"Main   : {main_script}",
        flush=True,
    )
    print(
        f"SHA256 : {sha256_file(main_script)[:16]}...",
        flush=True,
    )
    print(
        "Full arm: attention=True, cnn=True, "
        "unified_encoder=False, t2_target=True",
        flush=True,
    )
    print(
        "20 fresh subprocesses; "
        "one TensorFlow process per cell.",
        flush=True,
    )

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault(
        "TF_FORCE_GPU_ALLOW_GROWTH",
        "true",
    )

    for k, (
        seed,
        intensity,
    ) in enumerate(
        cells,
        1,
    ):
        if is_cell_complete(
            output,
            seed,
            intensity,
        ):
            print(
                f"[{k:02d}/20] SKIP "
                f"seed={seed}, I={intensity:g}: "
                "complete",
                flush=True,
            )
            continue

        print(
            f"\n[{k:02d}/20] RUN "
            f"seed={seed}, I={intensity:g}",
            flush=True,
        )

        cmd = [
            sys.executable,
            "-u",
            str(self_script),
            "--cell",
            "--seed",
            str(seed),
            "--intensity",
            str(intensity),
            "--project-dir",
            str(project),
            "--output-root",
            str(output),
            "--main-script",
            str(main_script),
        ]

        result = subprocess.run(
            cmd,
            env=env,
        )

        if result.returncode != 0:
            print(
                f"[FAIL] seed={seed}, "
                f"I={intensity:g}, "
                f"exit={result.returncode}",
                flush=True,
            )
            print(
                "Rerun the exact same master command "
                "to resume from saved variants.",
                flush=True,
            )
            raise SystemExit(
                result.returncode
            )

    merge_results(
        output
    )


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--project-dir",
        default=os.environ.get(
            "M2P_PROJECT_DIR",
            os.getcwd(),
        ),
    )
    ap.add_argument(
        "--output-root",
        default=os.environ.get(
            "M2P_OUTPUT_DIR"
        ),
    )
    ap.add_argument(
        "--main-script",
        default=os.environ.get(
            "M2P_MAIN_SCRIPT",
            "full_exp.py",
        ),
    )

    ap.add_argument(
        "--cell",
        action="store_true",
    )
    ap.add_argument(
        "--seed",
        type=int,
    )
    ap.add_argument(
        "--intensity",
        type=float,
    )
    ap.add_argument(
        "--merge-only",
        action="store_true",
    )

    args = ap.parse_args()

    (
        _project,
        output,
        main_script,
    ) = resolve_paths(args)

    if args.merge_only:
        validate_completed_output(
            output
        )
        if not main_script.exists():
            raise FileNotFoundError(
                f"Main experiment script not found: "
                f"{main_script}"
            )
        ensure_manifest(
            output,
            main_script,
        )
        merge_results(
            output
        )
        return

    if args.cell:
        if (
            args.seed is None
            or args.intensity is None
        ):
            ap.error(
                "--cell requires "
                "--seed and --intensity"
            )
        run_cell(
            args
        )
        return

    run_master(
        args
    )


if __name__ == "__main__":
    main()
