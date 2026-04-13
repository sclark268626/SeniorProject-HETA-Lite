#!/bin/bash
#SBATCH --partition=hpg-turin
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

INPUT_JSONL="${INPUT_JSONL:-}"
OUTPUT_DIR="${OUTPUT_DIR:-}"

echo "INPUT_JSONL=${INPUT_JSONL}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
if [[ -z "${INPUT_JSONL}" || -z "${OUTPUT_DIR}" ]]; then
  echo "Missing INPUT_JSONL or OUTPUT_DIR"
  exit 2
fi

MODEL_NAME="${MODEL_NAME:-}"
MODE="${MODE:-fidelity}"
BETA="${BETA:-0.5}"
GAMMA="${GAMMA:-0.5}"
TARGET_K="${TARGET_K:-1}"
MAX_EXAMPLES="${MAX_EXAMPLES:-500}"
SEED="${SEED:-0}"
SAVE_EVERY="${SAVE_EVERY:-50}"

export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONUNBUFFERED=1

mkdir -p "${OUTPUT_DIR}/logs"
JOBID="${SLURM_JOB_ID:-manual}"
OUT="${OUTPUT_DIR}/logs/run_${JOBID}.out"
ERR="${OUTPUT_DIR}/logs/run_${JOBID}.err"
echo "[BOOT] host=$(hostname) jobid=${JOBID} time=$(date)"
echo "[BOOT] input=${INPUT_JSONL}"
echo "[BOOT] outdir=${OUTPUT_DIR}"
exec 1>>"${OUT}"
exec 2>>"${ERR}"

source "${REPO_ROOT}/.venv/bin/activate"
cd "${REPO_ROOT}"

CMD=(
  python -u scripts/run_faithfulness_hotpot.py
  --mode "${MODE}"
  --input_jsonl "${INPUT_JSONL}"
  --output_dir "${OUTPUT_DIR}"
  --beta "${BETA}"
  --gamma "${GAMMA}"
  --target_k "${TARGET_K}"
  --max_examples "${MAX_EXAMPLES}"
  --seed "${SEED}"
  --num_workers 1
  --save_every "${SAVE_EVERY}"
)

if [[ -n "${MODEL_NAME}" ]]; then
  CMD+=(--model "${MODEL_NAME}")
fi

"${CMD[@]}"