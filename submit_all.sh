#!/bin/bash
set -euo pipefail

sbatch() { command sbatch "$@" | cut -d";" -f1; }

: "${VSC_DATA:?VSC_DATA is not set}"

WORKDIR="${M2P_WORKDIR:-$VSC_DATA/m2p_RR_Backup_delta}"
RELEASE="${PAPER_RELEASE:-$WORKDIR/paper_release}"
RUN_TAG="${1:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="$WORKDIR/paper_results/$RUN_TAG"
LOG_DIR="$RUN_ROOT/logs"

mkdir -p "$LOG_DIR"
cd "$WORKDIR"

export M2P_WORKDIR="$WORKDIR"
export PAPER_RELEASE="$RELEASE"
export PAPER_RUN_ROOT="$RUN_ROOT"

python "$RELEASE/experiments/validate_inputs.py" \
  --source "$RELEASE/src/m2p_ptac.py" \
  --project-dir "$WORKDIR" \
  --output "$RUN_ROOT/canonical_data_manifest.json"

cd "$LOG_DIR"

selection=$(sbatch --parsable \
  --export=ALL,M2P_WORKDIR="$WORKDIR",PAPER_RELEASE="$RELEASE",PAPER_RUN_ROOT="$RUN_ROOT" \
  "$RELEASE/slurm/model_selection.slurm")

benchmark=$(sbatch --parsable --dependency=afterok:$selection \
  --export=ALL,M2P_WORKDIR="$WORKDIR",PAPER_RELEASE="$RELEASE",PAPER_RUN_ROOT="$RUN_ROOT" \
  "$RELEASE/slurm/benchmark.slurm")

study_a=$(sbatch --parsable --dependency=afterok:$benchmark \
  --export=ALL,M2P_WORKDIR="$WORKDIR",PAPER_RELEASE="$RELEASE",PAPER_RUN_ROOT="$RUN_ROOT" \
  "$RELEASE/slurm/study_a.slurm")

study_b=$(sbatch --parsable --dependency=afterok:$benchmark \
  --export=ALL,M2P_WORKDIR="$WORKDIR",PAPER_RELEASE="$RELEASE",PAPER_RUN_ROOT="$RUN_ROOT" \
  "$RELEASE/slurm/study_b.slurm")

study_c=$(sbatch --parsable --dependency=afterok:$benchmark \
  --export=ALL,M2P_WORKDIR="$WORKDIR",PAPER_RELEASE="$RELEASE",PAPER_RUN_ROOT="$RUN_ROOT" \
  "$RELEASE/slurm/study_c.slurm")

sensitivity=$(sbatch --parsable --dependency=afterok:$benchmark \
  --export=ALL,M2P_WORKDIR="$WORKDIR",PAPER_RELEASE="$RELEASE",PAPER_RUN_ROOT="$RUN_ROOT",M2P_SELECTED_CONFIG="$RUN_ROOT/selection/selected_configuration.json" \
  "$RELEASE/slurm/model_sensitivity.slurm")

sensitivity_summary=$(sbatch --parsable --dependency=afterok:$sensitivity \
  --export=ALL,PAPER_RELEASE="$RELEASE",PAPER_RUN_ROOT="$RUN_ROOT" \
  "$RELEASE/slurm/model_sensitivity_summary.slurm")

confirmation=$(sbatch --parsable \
  --dependency=afterok:$study_a:$study_b:$study_c:$sensitivity_summary \
  --export=ALL,M2P_WORKDIR="$WORKDIR",PAPER_RELEASE="$RELEASE",PAPER_RUN_ROOT="$RUN_ROOT" \
  "$RELEASE/slurm/confirmation.slurm")

finalize=$(sbatch --parsable --dependency=afterok:$confirmation \
  --export=ALL,PAPER_RELEASE="$RELEASE",PAPER_RUN_ROOT="$RUN_ROOT" \
  "$RELEASE/slurm/finalize.slurm")

cat > "$RUN_ROOT/jobs.json" <<EOF
{
  "run_tag": "$RUN_TAG",
  "selection": "$selection",
  "benchmark": "$benchmark",
  "study_a": "$study_a",
  "study_b": "$study_b",
  "study_c": "$study_c",
  "model_sensitivity": "$sensitivity",
  "model_sensitivity_summary": "$sensitivity_summary",
  "confirmation": "$confirmation",
  "finalize": "$finalize"
}
EOF

echo "Run submitted: $RUN_TAG"
echo "Results: $RUN_ROOT"
cat "$RUN_ROOT/jobs.json"
