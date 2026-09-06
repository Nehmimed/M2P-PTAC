#!/bin/bash
#SBATCH -M anansi
#SBATCH --job-name=m2p_sens_sum
#SBATCH --account=bvo00001
#SBATCH --partition=ada_gpu
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --output=model_sensitivity_summary_%j.log

set -euo pipefail
module load Python/3.12.3-GCCcore-13.3.0
source "${M2P_VENV:-$VSC_DATA/m2p/m2p_after_IQR/m2p_venv/bin/activate}"
: "${PAPER_RELEASE:?}"
: "${PAPER_RUN_ROOT:?}"
python -u "$PAPER_RELEASE/experiments/summarize_model_sensitivity.py" \
  "$PAPER_RUN_ROOT/model_sensitivity"
