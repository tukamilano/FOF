# 並列化訓練ガイド

このドキュメントでは、FOFプロジェクトでの並列化訓練の使用方法について説明します。

## 利用可能な並列化手法

### 1. DataLoader並列化
データローディングの並列化により、データの前処理を高速化できます。

```bash
python src/training/train_with_generated_data.py \
    --data_dir generated_data \
    --batch_size 32 \
    --num_workers 8 \
    --use_wandb
```

### 2. GPU並列化（DataParallel）
単一マシンで複数のGPUを使用して訓練を並列化します。

```bash
# すべてのGPUを使用
python src/training/train_with_generated_data.py \
    --data_dir generated_data \
    --batch_size 32 \
    --use_data_parallel \
    --use_wandb

# 特定のGPUのみを使用
python src/training/train_with_generated_data.py \
    --data_dir generated_data \
    --batch_size 32 \
    --use_data_parallel \
    --gpu_ids "0,1,2" \
    --use_wandb

# 特定のGPUを1つだけ使用
python src/training/train_with_generated_data.py \
    --data_dir generated_data \
    --batch_size 32 \
    --use_data_parallel \
    --gpu_ids "0" \
    --use_wandb
```

### 3. 混合精度訓練（AMP）
メモリ使用量を削減し、訓練速度を向上させます。

```bash
python src/training/train_with_generated_data.py \
    --data_dir generated_data \
    --batch_size 32 \
    --use_amp \
    --use_wandb
```

### 4. 勾配累積
大きなバッチサイズをシミュレートして、メモリ効率を向上させます。

```bash
python src/training/train_with_generated_data.py \
    --data_dir generated_data \
    --batch_size 16 \
    --gradient_accumulation_steps 4 \
    --use_wandb
```

## 組み合わせ例

### 高パフォーマンス設定
```bash
python src/training/train_with_generated_data.py \
    --data_dir generated_data \
    --batch_size 32 \
    --num_workers 8 \
    --use_data_parallel \
    --use_amp \
    --gradient_accumulation_steps 2 \
    --use_wandb \
    --wandb_project fof-high-performance
```

### メモリ効率設定
```bash
python src/training/train_with_generated_data.py \
    --data_dir generated_data \
    --batch_size 8 \
    --gradient_accumulation_steps 8 \
    --use_amp \
    --use_wandb
```

### 最大パフォーマンス設定
```bash
python src/training/train_with_generated_data.py \
    --data_dir generated_data \
    --batch_size 32 \
    --num_workers 8 \
    --use_data_parallel \
    --use_amp \
    --gradient_accumulation_steps 2 \
    --use_wandb \
    --wandb_project fof-max-performance
```

## パラメータ説明

### 並列化関連パラメータ

- `--num_workers`: データローディングのワーカー数（デフォルト: 4）
- `--use_data_parallel`: DataParallelを使用（複数GPU）
- `--gpu_ids`: 使用するGPU ID（例: "0,1,2" または "all"）
- `--use_amp`: 混合精度訓練を使用
- `--gradient_accumulation_steps`: 勾配累積のステップ数（デフォルト: 1）
- `--validation_frequency`: バリデーション実行頻度（デフォルト: 10000データポイントごと）

## 注意事項

1. **メモリ使用量**: 並列化によりメモリ使用量が増加する可能性があります
2. **バッチサイズ調整**: 並列化時はバッチサイズを調整することを推奨します
3. **勾配累積**: 大きなバッチサイズが必要な場合は勾配累積を使用してください
4. **GPU競合**: `--use_data_parallel`は自動的にすべてのGPUを使用するため、他のプロセスと競合する可能性があります
5. **明示的なGPU指定**: 本番環境では`--gpu_ids`で明示的にGPUを指定することを推奨します

## トラブルシューティング

### メモリ不足エラー
- バッチサイズを小さくする
- 勾配累積を使用する
- 混合精度訓練を使用する

### データローディングが遅い
- `num_workers`を増やす
- `pin_memory=True`が有効になっているか確認

### GPU競合エラー
- `--gpu_ids`で明示的にGPUを指定する
- `nvidia-smi`でGPU使用状況を確認する
- 他のプロセスがGPUを使用していないか確認する
