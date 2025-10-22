#!/bin/bash
# Usage: ./run_train_simple_loop.sh <SRC_LOOP> <DST_LOOP> <BUCKET_NAME>
# Example: ./run_train_simple_loop.sh RL1 RL2 fof-data-20251010-milano

set -e  # 途中でエラーが出たら終了

# プロジェクトルートに移動
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

SRC_LOOP=$1
DST_LOOP=$2
BUCKET=$3

# === 設定 ===
TEMPS=("1" "1.25" "1.5" "2")
LOG_DIR="logs_${DST_LOOP}"
mkdir -p "${LOG_DIR}"

# === トレーニング実行 ===
echo "=== Launching training jobs for ${DST_LOOP} ==="

for TEMP in "${TEMPS[@]}"; do
  DATA_DIR="generated_data_${DST_LOOP}/temperature_${TEMP}_mixture"
  LOAD_MODEL="models/${SRC_LOOP}_temperature_${TEMP}_low_lr.pth"
  SAVE_MODEL="models/${DST_LOOP}_temperature_${TEMP}_low_lr.pth"
  LOG_FILE="${LOG_DIR}/train_${TEMP}.log"

  echo "→ Starting training for temperature=${TEMP}"
  nohup python src/training/train_simple.py \
    --data_dir "${DATA_DIR}" \
    --num_epochs 1 \
    --learning_rate 1e-7 \
    --log_frequency 10000 \
    --use_wandb \
    --load_model_path "${LOAD_MODEL}" \
    --save_path "${SAVE_MODEL}" \
    > "${LOG_FILE}" 2>&1 &
done

echo "All training jobs launched. Waiting for them to finish..."

# === 全ジョブの終了を待機 ===
wait
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
