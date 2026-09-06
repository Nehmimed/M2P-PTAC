#!/usr/bin/env python3
"""Run the full paper benchmark with the development-selected configuration."""

from __future__ import annotations

import argparse
import json
import os
import runpy
import shutil
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    p.add_argument("--selected-config", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    source = Path(args.source).resolve()
    selected = Path(args.selected_config).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)

    payload = json.loads(selected.read_text(encoding="utf-8"))
    for key in ("architecture", "C", "k", "q"):
        if key not in payload:
            raise RuntimeError(f"Selected configuration is missing {key}")

    shutil.copy2(selected, output / "selected_configuration.json")
    (output / "best_hyperparameters.json").write_text(
        json.dumps(payload["architecture"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    os.environ["M2P_OUTPUT_DIR"] = str(output)
    os.environ["M2P_SELECTED_CONFIG"] = str(output / "selected_configuration.json")
    os.environ["MASK_SUFFIX"] = ""
    os.environ["REQUIRE_DQA_HANDOFF"] = "1"
    os.environ["M2P_RETUNE_COMPACT"] = "0"
    os.environ["PTAC_STRESS_FIXED_C"] = str(payload["C"])

    for name in (
        "PTAC_STRESS_ONLY",
        "PTAC_ARCH_TOURNAMENT",
        "PTAC_MECHANISM_DIAGNOSTIC",
        "PTAC_POLICY_LOCALIZATION",
        "PTAC_R27_COMPONENT_AUDIT",
    ):
        os.environ.pop(name, None)

    runpy.run_path(str(source), run_name="__main__")


if __name__ == "__main__":
    main()
