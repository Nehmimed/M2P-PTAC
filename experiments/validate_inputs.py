#!/usr/bin/env python3
"""Validate and fingerprint the frozen canonical data-quality handoff."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("m2p_release", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    p.add_argument("--project-dir", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    project = Path(args.project_dir).resolve()
    cleaned = project / "cleaned_data"
    source = Path(args.source).resolve()
    output = Path(args.output).resolve()

    module = load_module(source)
    cfg = module.Config()

    cleaned_inputs = [
        Path(cfg.FILE_PATH_STABLE),
        Path(cfg.FILE_PATH_CHALLENGED),
        Path(cfg.FILE_PATH_CHALLENGED_2),
        Path(cfg.FILE_PATH_BELGIAN),
    ]

    required = []
    for relative in cleaned_inputs:
        data_path = relative if relative.is_absolute() else project / relative
        required.append(data_path)
        base = data_path.name
        if base.startswith("cleaned_"):
            base = base[len("cleaned_"):]
        required.append(cleaned / f"interp_mask_{base}")
        required.append(cleaned / f"fault_mask_{base}")

    required.append(cleaned / "target_exclusions.csv")

    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit("Missing canonical DQA artifacts:\n  " + "\n  ".join(missing))

    artifacts = {}
    for path in sorted(set(required)):
        artifacts[str(path.relative_to(project))] = {
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
        }

    for pattern in (
        "fault_mask_drift_*.csv",
        "fault_mask_stuck_*.csv",
        "fault_mask_excursion_*.csv",
        "fault_mask_zero_*.csv",
        "fault_mask_sentinel_*.csv",
    ):
        for path in sorted(cleaned.glob(pattern)):
            artifacts[str(path.relative_to(project))] = {
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }

    payload = {
        "status": "PASS",
        "canonical_only": True,
        "dqa_regenerated": False,
        "cleaned_data_directory": str(cleaned),
        "artifacts": artifacts,
    }
    canonical = json.dumps(artifacts, sort_keys=True, separators=(",", ":")).encode()
    payload["canonical_data_manifest_sha256"] = hashlib.sha256(canonical).hexdigest()

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
