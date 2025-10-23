#!/bin/bash
# Usage: ./run_train_parallel.sh <SRC_LOOP> <DST_LOOP> <BUCKET_NAME> [BATCH_SIZE]
#        [--temperature <V>] [--entropy-reg-weight <V>] [--kl-penalty-weight <V>]
#        [--kl-reference-model-template <PATH>] [--arg1-loss-weight <V>] [--arg2-loss-weight <V>]
#        [-- <additional python args...>]
# Example: ./run_train_parallel.sh RL1 RL2 fof-data-20251010-milano 8 --temperature 1.5 --entropy-reg-weight 0.05 --kl-penalty-weight 0.05 --kl-reference-model-template "models/pretrained_model.pth"

set -e  # 途中でエラーが出たら終了

# プロジェクトルートに移動
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

SRC_LOOP=$1
DST_LOOP=$2
BUCKET=$3
shift 3

BATCH_SIZE=8  # デフォルトバッチサイズ
TEMPERATURE=1  # デフォルト温度
ENTROPY_REG_WEIGHT=0.0
KL_PENALTY_WEIGHT=0.0
KL_REFERENCE_MODEL_TEMPLATE=""
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
    --temperature)
      TEMPERATURE=$2
      shift 2
      ;;
    --entropy-reg-weight)
      ENTROPY_REG_WEIGHT=$2
      shift 2
      ;;
    --kl-penalty-weight)
      KL_PENALTY_WEIGHT=$2
      shift 2
      ;;
    --kl-reference-model-template)
      KL_REFERENCE_MODEL_TEMPLATE=$2
      shift 2
      ;;
    --arg1-loss-weight)
      ARG1_LOSS_WEIGHT=$2
      shift 2
      ;;
    --arg2-loss-weight)
      ARG2_LOSS_WEIGHT=$2
      shift 2
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
TEMPS=("${TEMPERATURE}")
LOG_DIR="logs_${DST_LOOP}"
mkdir -p "${LOG_DIR}"

# === トレーニング実行（順次実行） ===
echo "=== Starting parallel training for ${DST_LOOP} ==="
echo "Using batch size: ${BATCH_SIZE}"
echo "Temperature: ${TEMPERATURE}"
echo "Entropy regularization weight: ${ENTROPY_REG_WEIGHT}"
echo "KL penalty weight: ${KL_PENALTY_WEIGHT}"
if [[ -n "${KL_REFERENCE_MODEL_TEMPLATE}" ]]; then
  echo "KL reference model template: ${KL_REFERENCE_MODEL_TEMPLATE}"
fi
echo "Arg1 loss weight: ${ARG1_LOSS_WEIGHT}"
echo "Arg2 loss weight: ${ARG2_LOSS_WEIGHT}"

for TEMP in "${TEMPS[@]}"; do
  DATA_DIR="generated_data_${DST_LOOP}/temperature_${TEMP}_mixture"
  
  # RL0の場合はpretrained_model.pthを使用
  if [[ "${SRC_LOOP}" == "RL0" ]]; then
    LOAD_MODEL="models/pretrained_model.pth"
  else
    LOAD_MODEL="models/${SRC_LOOP}_temperature_${TEMP}_low_lr.pth"
  fi
  
  SAVE_MODEL="models/${DST_LOOP}_temperature_${TEMP}_KL_${KL_PENALTY_WEIGHT}_entropy_${ENTROPY_REG_WEIGHT}.pth"
  LOG_FILE="${LOG_DIR}/train_${TEMP}.log"
  REFERENCE_MODEL_PATH="${KL_REFERENCE_MODEL_TEMPLATE}"
  if [[ -n "${REFERENCE_MODEL_PATH}" ]]; then
    REFERENCE_MODEL_PATH="${REFERENCE_MODEL_PATH//\{TEMP\}/$TEMP}"
    REFERENCE_MODEL_PATH="${REFERENCE_MODEL_PATH//\{SRC_LOOP\}/$SRC_LOOP}"
    REFERENCE_MODEL_PATH="${REFERENCE_MODEL_PATH//\{DST_LOOP\}/$DST_LOOP}"
  else
    REFERENCE_MODEL_PATH="${LOAD_MODEL}"
  fi

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
    --kl_reference_model_path "${REFERENCE_MODEL_PATH}" \
    "${EXTRA_PYTHON_ARGS[@]}" \
    > "${LOG_FILE}" 2>&1
  
  echo "✓ Completed training for temperature=${TEMP}"
done

echo "All training jobs completed."

# === GCSへアップロード ===
echo "Uploading models to gs://${BUCKET}/models/ ..."

for TEMP in "${TEMPS[@]}"; do
  SAVE_MODEL="models/${DST_LOOP}_temperature_${TEMP}_KL_${KL_PENALTY_WEIGHT}_entropy_${ENTROPY_REG_WEIGHT}.pth"
  if [ -f "${SAVE_MODEL}" ]; then
    echo "→ Uploading ${SAVE_MODEL}"
    gsutil cp "${SAVE_MODEL}" "gs://${BUCKET}/models/"
  else
    echo "⚠️ Warning: ${SAVE_MODEL} not found, skipping upload."
  fi
done

echo "✅ All uploads completed successfully!"
