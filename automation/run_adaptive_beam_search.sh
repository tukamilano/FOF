#!/bin/bash

# 適応的ビームサーチによる強化学習データ収集
# 成功率に基づいて動的にビーム幅を調整

set -e

# 設定
MODEL_PATH="models/pretrained_model.pth"
GENERATED_DATA_DIR="generated_data"
OUTPUT_BASE_DIR="self_improvement_data_adaptive"
GCS_BUCKET="fof-data-20251010-milano"
GCS_PREFIX="generated_data_RL1_adaptive/"
NUM_WORKERS=12
BATCH_SIZE=10000

# ビーム設定の配列 (beam_width, top_k, target_success_rate)
BEAM_CONFIGS=(
    "3 5 0.65"      # 高速設定
    "5 8 0.75"      # 中速設定  
    "10 15 0.80"    # 高品質設定
    "15 20 0.85"    # 最高品質設定
)

# 各設定での処理例数
EXAMPLES_PER_CONFIG=250000  # 100万例を4つの設定で分割

echo "=== 適応的ビームサーチ強化学習データ収集 ==="
echo "総例数: 1,000,000"
echo "並列ワーカー数: $NUM_WORKERS"
echo "各設定の例数: $EXAMPLES_PER_CONFIG"
echo ""

# 出力ディレクトリを作成
mkdir -p $OUTPUT_BASE_DIR

# 各ビーム設定で並列実行
for i in "${!BEAM_CONFIGS[@]}"; do
    IFS=' ' read -r beam_width top_k target_rate <<< "${BEAM_CONFIGS[$i]}"
    
    echo "=== 設定 $((i+1)): beam_width=$beam_width, top_k=$top_k (目標成功率: ${target_rate}) ==="
    
    OUTPUT_DIR="${OUTPUT_BASE_DIR}/beam_${beam_width}_${top_k}"
    GCS_PREFIX_CONFIG="${GCS_PREFIX}beam_${beam_width}_${top_k}/"
    
    # バックグラウンドで実行
    python3 src/interaction/self_improvement_data_beam_search_collector.py \
        --model_path $MODEL_PATH \
        --count $EXAMPLES_PER_CONFIG \
        --max_steps 30 \
        --beam_width $beam_width \
        --top_k $top_k \
        --device auto \
        --max_seq_len 256 \
        --output_dir $OUTPUT_DIR \
        --batch_size $BATCH_SIZE \
        --generated_data_dir $GENERATED_DATA_DIR \
        --num_workers $NUM_WORKERS \
        --gcs_bucket $GCS_BUCKET \
        --gcs_prefix $GCS_PREFIX_CONFIG \
        > "logs/adaptive_beam_${beam_width}_${top_k}.log" 2>&1 &
    
    echo "設定 $((i+1)) をバックグラウンドで開始 (PID: $!)"
    echo "ログファイル: logs/adaptive_beam_${beam_width}_${top_k}.log"
    echo ""
done

echo "=== 全設定を並列実行中 ==="
echo "進行状況を確認するには:"
echo "  tail -f logs/adaptive_beam_*.log"
echo ""
echo "プロセス確認:"
echo "  ps aux | grep beam_search_collector"
echo ""
echo "完了まで待機中..."

# 全プロセスの完了を待機
wait

echo "=== 全設定の実行完了 ==="
echo "結果の確認:"
echo "  ls -la $OUTPUT_BASE_DIR/"
echo ""
echo "GCSアップロード状況:"
echo "  gsutil ls gs://$GCS_BUCKET/$GCS_PREFIX"

