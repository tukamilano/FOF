#!/bin/bash
# 使い方: ./run_self_improvement.sh RL2 models/RL1_temperature_1_KL_0.05_entropy_0.05.pth
# 第一引数: サイクル名 (RL2, RL3, etc.) - 自動的にtrial番号が決定される
# 第二引数: 使用するモデルパス

# プロジェクトルートに移動
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

CYCLE=${1:-RL2}  # デフォルトはRL2
MODEL_PATH=${2:-"models/RL1_temperature_1_KL_0.05_entropy_0.05.pth"}  # デフォルトモデル
BUCKET="fof-data-20251010-milano"
NUM_WORKERS=12

# サイクル名からtrial番号を自動決定（現在は使用しない）
# RL2 -> trial2, RL3 -> trial3, etc.
TRIAL_NUMBER=${CYCLE:2}  # "RL2" -> "2", "RL3" -> "3"

echo "Processing cycle ${CYCLE} (trial ${TRIAL_NUMBER} - not used in current implementation)"

# 温度リスト（単一温度で実行）
declare -a TEMPS=("1")

for T in "${TEMPS[@]}"; do
    PREFIX="generated_data_${CYCLE}/temperature_${T}_self_improvement"
    echo "Starting temperature ${T} for ${CYCLE} with model ${MODEL_PATH} using trial${TRIAL_NUMBER} tautologies..."
    
    nohup python src/interaction/self_improvement_data_parallel_collector.py \
        --model_path "${MODEL_PATH}" \
        --num_workers ${NUM_WORKERS} \
        --gcs_bucket ${BUCKET} \
        --gcs_prefix "${PREFIX}" \
        --generated_data_dir "tautology/trial${TRIAL_NUMBER}" \
        --temperature ${T}
done

echo "All temperature jobs for ${CYCLE} started."