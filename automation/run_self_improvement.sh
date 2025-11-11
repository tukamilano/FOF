#!/bin/bash
# Usage: ./run_self_improvement.sh RL3(3 round)
# Specify cycle name like "RL1" "RL2" as argument to use that cycle name

# プロジェクトルート 移動
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

CYCLE=${1:-RL1}  # Default is RL1
PREV_CYCLE="RL$(( ${CYCLE:2} - 1 ))"  # Convert "RL2" → "RL1"
BUCKET="fof-data-20251010-milano"
NUM_WORKERS=12

# List of temperature and model name mappings
declare -a TEMPS=("1" "1.25" "1.5" "2")

for T in "${TEMPS[@]}"; do
    MODEL_PATH="models/${PREV_CYCLE}_temperature_${T}_mixture_low_lr.pth"
    PREFIX="generated_data_${CYCLE}/temperature_${T}"
    echo "Starting temperature ${T} for ${CYCLE}..."
    
    nohup python src/interaction/self_improvement_data_parallel_collector.py \
        --model_path "${MODEL_PATH}" \
        --num_workers ${NUM_WORKERS} \
        --gcs_bucket ${BUCKET} \
        --gcs_prefix "${PREFIX}" \
        --generated_data_dir "generated_data_${CYCLE}/temperature_${T}_mixture" \
        --temperature ${T} \
        > /dev/null 2>&1 &
done

echo "All temperature jobs for ${CYCLE} started."