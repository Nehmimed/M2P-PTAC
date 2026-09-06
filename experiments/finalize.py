#!/usr/bin/env python3
"""Create a hash manifest for a completed paper experiment run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root")
    args = p.parse_args()
    root = Path(args.root).resolve()

    required = [
        root / "canonical_data_manifest.json",
        root / "selection" / "selected_configuration.json",
        root / "benchmark",
        root / "model_sensitivity" / "sensitivity_summary.json",
        root / "confirmation" / "confirmation_verdict.json",
    ]
    missing = [str(path) for path in required if not path.exists()]

    files = {}
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".json", ".csv"}:
            files[str(path.relative_to(root))] = {
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }

    payload = {
        "status": "COMPLETE" if not missing else "INCOMPLETE",
        "missing": missing,
        "artifacts": files,
    }
    (root / "paper_run_manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    if missing:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
