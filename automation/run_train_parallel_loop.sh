#!/bin/bash
# Usage: ./run_train_parallel_loop.sh <SRC_LOOP> <DST_LOOP> <BUCKET_NAME> [BATCH_SIZE] [--batch-size <N>] [--entropy-reg-weight <V>] \
#        [--kl-penalty-weight <V>] [--arg1-loss-weight <V>] [--arg2-loss-weight <V>] [--extra-arg <ARG>]...
# Example: ./run_train_parallel_loop.sh RL1 RL2 fof-data-20251010-milano 32 --entropy-reg-weight 0.05 --kl-penalty-weight 0.1

set -e  # 途中でエラーが出たら終了

# プロジェクトルートに移動
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

SRC_LOOP=$1
DST_LOOP=$2
BUCKET=$3
shift 3

BATCH_SIZE=32  # デフォルトバッチサイズ
ENTROPY_REG_WEIGHT=0.0
KL_PENALTY_WEIGHT=0.0
ARG1_LOSS_WEIGHT=0.8
ARG2_LOSS_WEIGHT=0.8
EXTRA_PYTHON_ARGS=()

# オプション解析
if [[ $# -gt 0 && "$1" != --* ]]; then
  BATCH_SIZE=$1
  shift
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --batch-size)
      BATCH_SIZE=$2
      shift 2
      ;;
    --batch-size=*)
      BATCH_SIZE="${1#*=}"
      shift
      ;;
    --entropy-reg-weight)
      ENTROPY_REG_WEIGHT=$2
      shift 2
      ;;
    --entropy-reg-weight=*)
      ENTROPY_REG_WEIGHT="${1#*=}"
      shift
      ;;
    --kl-penalty-weight)
      KL_PENALTY_WEIGHT=$2
      shift 2
      ;;
    --kl-penalty-weight=*)
      KL_PENALTY_WEIGHT="${1#*=}"
      shift
      ;;
    --arg1-loss-weight)
      ARG1_LOSS_WEIGHT=$2
      shift 2
      ;;
    --arg1-loss-weight=*)
      ARG1_LOSS_WEIGHT="${1#*=}"
      shift
      ;;
    --arg2-loss-weight)
      ARG2_LOSS_WEIGHT=$2
      shift 2
      ;;
    --arg2-loss-weight=*)
      ARG2_LOSS_WEIGHT="${1#*=}"
      shift
      ;;
    --extra-arg)
      EXTRA_PYTHON_ARGS+=("$2")
      shift 2
      ;;
    --extra-arg=*)
      EXTRA_PYTHON_ARGS+=("${1#*=}")
      shift
      ;;
    --)
      shift
      EXTRA_PYTHON_ARGS+=("$@")
      break
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
  esac
done

# === 設定 ===
TEMPS=("1" "1.25" "1.5" "2")
LOG_DIR="logs_${DST_LOOP}"
mkdir -p "${LOG_DIR}"

# === トレーニング実行（順次実行） ===
echo "=== Starting parallel training for ${DST_LOOP} ==="
echo "Using batch size: ${BATCH_SIZE}"
echo "Entropy regularization weight: ${ENTROPY_REG_WEIGHT}"
echo "KL penalty weight: ${KL_PENALTY_WEIGHT}"
echo "Arg1 loss weight: ${ARG1_LOSS_WEIGHT}"
echo "Arg2 loss weight: ${ARG2_LOSS_WEIGHT}"

for TEMP in "${TEMPS[@]}"; do
  DATA_DIR="generated_data_${DST_LOOP}/temperature_${TEMP}_mixture"
  LOAD_MODEL="models/${SRC_LOOP}_temperature_${TEMP}_low_lr.pth"
  SAVE_MODEL="models/${DST_LOOP}_temperature_${TEMP}_low_lr.pth"
  LOG_FILE="${LOG_DIR}/train_${TEMP}.log"

  echo "→ Starting training for temperature=${TEMP}"
  python src/training/train_with_generated_data.py \
    --data_dir "${DATA_DIR}" \
    --batch_size "${BATCH_SIZE}" \
    --num_epochs 1 \
    --learning_rate 1e-7 \
    --log_frequency 10000 \
    --use_wandb \
    --load_model_path "${LOAD_MODEL}" \
    --save_path "${SAVE_MODEL}" \
    --arg1_loss_weight "${ARG1_LOSS_WEIGHT}" \
    --arg2_loss_weight "${ARG2_LOSS_WEIGHT}" \
    --entropy_reg_weight "${ENTROPY_REG_WEIGHT}" \
    --kl_penalty_weight "${KL_PENALTY_WEIGHT}" \
    "${EXTRA_PYTHON_ARGS[@]}" \
    > "${LOG_FILE}" 2>&1
  
  echo "✓ Completed training for temperature=${TEMP}"
done

echo "All training jobs completed."

# === GCSへアップロード ===
echo "Uploading models to gs://${BUCKET}/models/ ..."

for TEMP in "${TEMPS[@]}"; do
  SAVE_MODEL="models/${DST_LOOP}_temperature_${TEMP}_low_lr.pth"
  if [ -f "${SAVE_MODEL}" ]; then
    echo "→ Uploading ${SAVE_MODEL}"
    gsutil cp "${SAVE_MODEL}" "gs://${BUCKET}/models/"
  else
    echo "⚠️ Warning: ${SAVE_MODEL} not found, skipping upload."
  fi
done

echo "✅ All uploads completed successfully!"
