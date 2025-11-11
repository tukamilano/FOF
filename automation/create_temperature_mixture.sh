#!/bin/bash
# Usage: ./create_temperature_mixture.sh RL3
# Specify cycle name like RL3 as argument to automatically execute temperature_1~2

# Move to project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

CYCLE=${1:-RL1}
BASE_DIR="$PROJECT_ROOT"
OUTPUT_DIR="generated_data_${CYCLE}"
INPUT_DIR="generated_data_${CYCLE}"

# List of temperatures to create
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
