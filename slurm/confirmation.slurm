#!/bin/bash
#SBATCH -M anansi
#SBATCH --job-name=m2p_confirm
#SBATCH --account=bvo00001
#SBATCH --partition=ada_gpu
#SBATCH --gres=shard:2
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --output=confirmation_%j.log

set -euo pipefail
module load Python/3.12.3-GCCcore-13.3.0
module load PyTorch-Geometric/2.6.1-foss-2024a-CUDA-12.6.0
source "${M2P_VENV:-$VSC_DATA/m2p/m2p_after_IQR/m2p_venv/bin/activate}"
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_CPP_MIN_LOG_LEVEL=2
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1

: "${M2P_WORKDIR:?}"
: "${PAPER_RELEASE:?}"
: "${PAPER_RUN_ROOT:?}"
cd "$M2P_WORKDIR"

ARGS=(
  --source "$PAPER_RELEASE/src/m2p_ptac.py"
  --selected-config "$PAPER_RUN_ROOT/selection/selected_configuration.json"
  --canonical-manifest "$PAPER_RUN_ROOT/canonical_data_manifest.json"
  --output "$PAPER_RUN_ROOT/confirmation"
  --lock-root "$M2P_WORKDIR/paper_results"
)

python -u "$PAPER_RELEASE/experiments/confirmation.py" --mode preflight "${ARGS[@]}"
python -u "$PAPER_RELEASE/experiments/confirmation.py" --mode run "${ARGS[@]}"
