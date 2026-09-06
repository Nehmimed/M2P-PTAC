#!/bin/bash
#SBATCH -M anansi
#SBATCH --job-name=m2p_model_sens
#SBATCH --account=bvo00001
#SBATCH --partition=ada_gpu
#SBATCH --gres=shard:2
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --array=0-4
#SBATCH --output=model_sensitivity_%A_%a.log

set -euo pipefail

: "${M2P_WORKDIR:?}"
: "${PAPER_RELEASE:?}"
: "${PAPER_RUN_ROOT:?}"
M2P_SELECTED_CONFIG="${M2P_SELECTED_CONFIG:-$PAPER_RUN_ROOT/selection/selected_configuration.json}"

[[ -f "$M2P_SELECTED_CONFIG" ]] || { echo "[FAIL] Missing selected config: $M2P_SELECTED_CONFIG" >&2; exit 2; }

export M2P_SELECTED_CONFIG

export PTAC_PAPER_FROZEN_SELECTION="$M2P_SELECTED_CONFIG"

SOURCE="$PAPER_RELEASE/src/m2p_ptac.py"
TASK="${SLURM_ARRAY_TASK_ID:-0}"

case "$TASK" in
  0) ARM="canonical"; RESIDUAL="canonical"; VARIANCE="canonical" ;;
  1) ARM="residual_robust_nocenter"; RESIDUAL="robust_nocenter"; VARIANCE="canonical" ;;
  2) ARM="residual_none"; RESIDUAL="none"; VARIANCE="canonical" ;;
  3) ARM="variance_inverse_sqrt"; RESIDUAL="canonical"; VARIANCE="inverse_sqrt" ;;
  4) ARM="variance_none"; RESIDUAL="canonical"; VARIANCE="none" ;;
  *) exit 2 ;;
esac

OUT="$PAPER_RUN_ROOT/model_sensitivity/$ARM"
mkdir -p "$OUT"
PATCHED="$OUT/m2p_ptac_${ARM}.py"
cp "$SOURCE" "$PATCHED"

python - "$PATCHED" "$RESIDUAL" "$VARIANCE" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
residual_mode = sys.argv[2]
variance_mode = sys.argv[3]
text = path.read_text(encoding="utf-8")

residual_old = (
    "        _r17_on = bool(getattr(self.config, 'R17_ENABLE', False))\n"
    "        scaler_residuals = StandardScaler(with_mean=not _r17_on)"
)
variance_old = (
    "        stds = y_train_tab.std()\n"
    "        normalized_stds = stds / stds.mean() + 1e-6\n"
    "        lambdas = self.base_lambda / normalized_stds"
)

if residual_old not in text:
    raise SystemExit("Residual-scaling block not recognized.")
if variance_old not in text:
    raise SystemExit("Variance-normalization block not recognized.")

if "from sklearn.preprocessing import StandardScaler, RobustScaler" not in text:
    text = text.replace(
        "from sklearn.preprocessing import StandardScaler",
        "from sklearn.preprocessing import StandardScaler, RobustScaler",
        1,
    )

if residual_mode == "canonical":
    residual_new = residual_old
elif residual_mode == "robust_nocenter":
    residual_new = (
        "        _r17_on = bool(getattr(self.config, 'R17_ENABLE', False))\n"
        "        scaler_residuals = RobustScaler("
        "with_centering=False, with_scaling=True, unit_variance=True)"
    )
elif residual_mode == "none":
    residual_new = (
        "        _r17_on = bool(getattr(self.config, 'R17_ENABLE', False))\n"
        "        scaler_residuals = StandardScaler(with_mean=False, with_std=False)"
    )
else:
    raise SystemExit("Unknown residual mode.")

if variance_mode == "canonical":
    variance_new = variance_old
elif variance_mode == "inverse_sqrt":
    variance_new = (
        "        stds = y_train_tab.std()\n"
        "        normalized_stds = np.sqrt(stds / stds.mean() + 1e-6)\n"
        "        lambdas = self.base_lambda / normalized_stds"
    )
elif variance_mode == "none":
    variance_new = (
        "        stds = y_train_tab.std()\n"
        "        normalized_stds = pd.Series(1.0, index=stds.index)\n"
        "        lambdas = pd.Series(float(self.base_lambda), index=stds.index)"
    )
else:
    raise SystemExit("Unknown variance mode.")

text = text.replace(residual_old, residual_new, 1)
text = text.replace(variance_old, variance_new, 1)
compile(text, str(path), "exec")
path.write_text(text, encoding="utf-8")
PY

export M2P_OUTPUT_DIR="$OUT/results"
export M2P_SELECTED_CONFIG
export MASK_SUFFIX=""
export REQUIRE_DQA_HANDOFF=1
export M2P_RETUNE_COMPACT=0
unset REVIEWER_IQR_MULTIPLIER || true
unset REVIEWER_STUCK_WINDOW || true
unset REVIEWER_DQA_SUFFIX || true

module load Python/3.12.3-GCCcore-13.3.0
module load PyTorch-Geometric/2.6.1-foss-2024a-CUDA-12.6.0
source "${M2P_VENV:-$VSC_DATA/m2p/m2p_after_IQR/m2p_venv/bin/activate}"

cd "$M2P_WORKDIR"
python -u "$PATCHED"
