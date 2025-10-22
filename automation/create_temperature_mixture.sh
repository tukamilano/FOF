#!/bin/bash
# 使い方: ./create_temperature_mixture.sh RL3
# RL3などのサイクル名を引数に指定すると、自動的にtemperature_1〜2まで実行されます

# プロジェクトルートに移動
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

CYCLE=${1:-RL1}
BASE_DIR="$PROJECT_ROOT"
OUTPUT_DIR="generated_data_${CYCLE}"
INPUT_DIR="generated_data_${CYCLE}"

# 作成する温度リスト
declare -a TEMPS=("temperature_1" "temperature_1.25" "temperature_1.5" "temperature_2")

echo "Using input directory: ${INPUT_DIR}"
echo "Output directory: ${OUTPUT_DIR}"

for T in "${TEMPS[@]}"; do
    echo "Creating mixture for ${T} into ${OUTPUT_DIR}..."
    python3 create_temperature_mixtures.py \
        --temperatures ${T} \
        --output-dir ${OUTPUT_DIR} \
        --input-dir ${INPUT_DIR} \
        --base-dir ${BASE_DIR}
done

echo "✅ All mixtures created in ${OUTPUT_DIR}"
