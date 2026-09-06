#!/bin/bash
#SBATCH -M anansi
#SBATCH --job-name=m2p_finalize
#SBATCH --account=bvo00001
#SBATCH --partition=ada_gpu
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --output=finalize_%j.log

set -euo pipefail
module load Python/3.12.3-GCCcore-13.3.0
: "${PAPER_RELEASE:?}"
: "${PAPER_RUN_ROOT:?}"
python -u "$PAPER_RELEASE/experiments/finalize.py" "$PAPER_RUN_ROOT"
