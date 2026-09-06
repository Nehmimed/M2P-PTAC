#!/usr/bin/env python3
"""Inventory model-side sensitivity outputs without selecting a preferred arm."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ARMS = [
    "canonical",
    "residual_robust_nocenter",
    "residual_none",
    "variance_inverse_sqrt",
    "variance_none",
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root")
    args = p.parse_args()
    root = Path(args.root).resolve()

    rows = []
    for arm in ARMS:
        result_dir = root / arm / "results"
        csvs = sorted(result_dir.glob("*.csv")) if result_dir.exists() else []
        rows.append(
            {
                "arm": arm,
                "completed": bool(csvs),
                "csv_count": len(csvs),
                "csv_files": ";".join(path.name for path in csvs),
            }
        )

    table = pd.DataFrame(rows)
    table.to_csv(root / "sensitivity_inventory.csv", index=False)
    payload = {
        "selection_performed": False,
        "canonical_DQA_only": True,
        "DQA_regenerated": False,
        "arms": table.to_dict("records"),
    }
    (root / "sensitivity_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
