# FOF (First-Order Formula) - Transformer-based Theorem Prover

Transformerモデルで命題論理の定理証明を自動化するシステムです。[pyprover](https://github.com/kaicho8636/pyprover) と組み合わせ、数式生成→学習→推論→自己改善まで一貫したワークフローを提供します。

## 🚀 主な特徴

- **階層分類アーキテクチャ**: タクティクの種類と引数を独立に予測
- **推論評価スイート**: さまざまな推論手法を比較・検証
- **大規模データ運用**: GCS統合と重複排除で効率化
- **並列データ収集/学習**: マルチプロセス・マルチGPU・AMP対応
- **実験追跡**: wandb で詳細なログ・可視化

## 🔰 クイックスタート（推論のみ）

学習済みモデルで推論を素早く試す：

```bash
python validation/inference_hierarchical.py \
  --model_path models/pretrained_model.pth \
  --count 100 \
  --max_steps 30 \
  --verbose
```

- 追加のベンチマークは `validation/pretrained_model_validation.txt` を参照

## 環境設定

```bash
# 仮想環境の作成と有効化
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate  # Windows

# 依存関係をインストール
pip install -r requirements.txt
```

### 依存関係

- Python 3.8+（推奨: 3.9〜3.11）
- PyTorch
- [pyprover](https://github.com/kaicho8636/pyprover)
- wandb（任意）
- GCS を使う場合は `google-cloud-storage`

## プロジェクト構造（抜粋）

```
FOF/
├── automation/                   # 自動化スクリプト
│   ├── create_temperature_mixture.sh
│   ├── run_self_improvement.sh
│   ├── run_train_simple_loop.sh
│   └── README.md
├── src/
│   ├── core/                     # Transformer/エンコーダ/パラメータ
│   ├── data_generation/          # 生成・収集（並列あり）
│   ├── interaction/              # 自己改善データ収集
│   ├── training/                 # 学習・分析・重複排除
│   └── compression/              # 圧縮ユーティリティ
├── validation/                   # 推論・比較
├── tests/                        # テスト一式
├── models/                       # 学習済み/チェックポイント
└── pyprover/                     # 証明器
```

## 使用方法

### 1) データ生成

```bash
# 並列データ収集（ローカル保存）
python src/data_generation/auto_data_parallel_collector.py \
  --count 1000 \
  --workers 4 \
  --examples_per_file 100

# 直接 GCS に保存
python src/data_generation/auto_data_parallel_collector.py \
  --count 10000 \
  --workers 8 \
  --gcs_bucket your-bucket \
  --gcs_prefix generated_data/
```

### 2) 重複排除と分析

```bash
python src/training/deduplicate_generated_data.py \
  --input_dir generated_data \
  --output_dir deduplicated_data

python src/training/analyze_generated_data.py
```

### 3) 学習（シンプル）

```bash
python src/training/train_simple.py \
  --data_dir deduplicated_data \
  --batch_size 32 \
  --learning_rate 3e-4 \
  --num_epochs 10

# wandb で追跡
python src/training/train_simple.py --use_wandb --wandb_project fof-training
```

より詳細なワークフローや二段階重複排除は `src/training/README.md` を参照。

### 4) 推論と比較

```bash
# 階層分類推論
python validation/inference_hierarchical.py \
  --model_path models/pretrained_model.pth \
  --count 100 \
  --max_steps 30

# ビームサーチなどの比較
python validation/inference_beam_search.py --help
python validation/compare_inference_methods.py --help
```

## 並列学習・高速化オプション

- DataLoader 並列化、複数 GPU（DataParallel）、AMP、勾配累積に対応
- 具体例・推奨設定は `src/training/PARALLEL_TRAINING.md` を参照

## 自動化スクリプト（automation/）

`automation/README.md` に簡易ガイドあり。実行前に実行権限を付与：

```bash
chmod +x automation/*.sh
```

例：

```bash
# 温度ミクスチャ生成
./automation/create_temperature_mixture.sh RL3

# 学習ループ（例: RL1→RL2）
./automation/run_train_simple_loop.sh RL1 RL2 your-gcs-bucket-prefix

# 自己改善データ収集
./automation/run_self_improvement.sh RL3
```

## モデル/チェックポイント

- `models/pretrained_model.pth`: 事前学習済みモデル
- `models/RL*_*.pth`: SFTサイクル（温度・ビームサーチ・top_k 等の条件）で得たモデル

## 推奨ワークフロー（要約）

```bash
# 1. 生成
python src/data_generation/auto_data_parallel_collector.py --count 1000 --workers 4

# 2. 重複排除
python src/training/deduplicate_generated_data.py --input_dir generated_data --output_dir deduplicated_data

# 3. 学習
python src/training/train_simple.py --data_dir deduplicated_data --use_wandb

# 4. 推論
python validation/inference_hierarchical.py --verbose
```

## テスト

```bash
python tests/test_integration.py
python tests/test_parameter_sync.py
python tests/test_duplicate_check.py
python tests/test_deduplicated_data_hashes.py
python tests/test_tactic_tokens.py
python tests/test_inference_evaluation.py
```

注意: `tests/test_wandb_connection.py` は wandb ログインが必要です（`wandb login` または `WANDB_API_KEY` 環境変数）。

## トラブルシューティング

- **wandb にログインできない**: `pip install wandb && wandb login`
- **GCS に書き込めない**: `GOOGLE_APPLICATION_CREDENTIALS` を設定し、`google-cloud-storage` をインストール
- **CUDA メモリ不足**: バッチサイズを減らす / `--use_amp` / 勾配累積を利用
- **データローディングが遅い**: `--num_workers` を増やす

## 謝辞

このプロジェクトは以下の OSS を利用しています：

- [pyprover](https://github.com/kaicho8636/pyprover)
- PyTorch
- wandb
