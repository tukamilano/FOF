# FOF (First-Order Formula) - Transformer-based Theorem Prover

Transformerモデルを使用して命題論理の定理証明を自動化するシステムです。[pyprover](https://github.com/kaicho8636/pyprover)ライブラリと組み合わせて、数式生成から証明戦略の予測まで一貫したワークフローを提供します。

## 🚀 主な特徴

- **階層分類アーキテクチャ**: タクティクの種類と引数を独立して予測
- **推論性能評価システム**: 実際の問題解決能力を測定
- **大規模データ処理**: GCS統合による効率的なデータ管理
- **並列データ収集**: マルチプロセス対応の高速データ生成
- **実験追跡**: wandbによる詳細な学習・推論ログ

## 環境設定

```bash
# 仮想環境の作成と有効化
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows

# 依存関係をインストール
pip install -r requirements.txt
```

### 依存関係

- Python 3.8+
- PyTorch
- [pyprover](https://github.com/kaicho8636/pyprover) - 命題論理証明器ライブラリ
- wandb (オプション) - 実験追跡

## プロジェクト構造

```
FOF/
├── src/                          # メインソースコード
│   ├── core/                     # コア機能
│   │   ├── transformer_classifier.py  # Transformerモデル
│   │   ├── state_encoder.py           # 証明状態のエンコーディング
│   │   ├── parameter.py               # ハイパーパラメータ管理
│   │   └── fof_tokens.py              # トークン定義
│   ├── data_generation/          # データ生成
│   │   ├── auto_data_parallel_collector.py  # 並列データ収集
│   │   └── tautology_generator.py          # トートロジー生成
│   ├── training/                 # 学習関連
│   │   ├── train_simple.py              # シンプル学習スクリプト
│   │   ├── train_with_generated_data.py # 生成データ学習
│   │   └── deduplicate_generated_data.py # 重複排除
│   ├── interaction/              # インタラクション
│   │   └── self_improvement_data_parallel_collector.py
│   └── compression/              # データ圧縮
├── validation/                   # 推論・評価
│   └── inference_hierarchical.py # 階層分類推論
├── tests/                        # テストファイル
├── generated_data/               # 生成された学習データ
├── deduplicated_data/            # 重複排除済みデータ
├── models/                       # 学習済みモデル
└── pyprover/                     # pyproverライブラリ
```

## 使用方法

### 1. データ生成

```bash
# 並列データ収集（推奨）
python src/data_generation/auto_data_parallel_collector.py \
  --count 1000 \
  --workers 4 \
  --examples_per_file 100

# GCSに直接アップロード
python src/data_generation/auto_data_parallel_collector.py \
  --count 10000 \
  --workers 8 \
  --gcs_bucket your-bucket \
  --gcs_prefix generated_data/
```

### 2. モデル学習

```bash
# シンプル学習（重複排除済みデータ）
python src/training/train_simple.py \
  --data_dir deduplicated_data \
  --batch_size 32 \
  --learning_rate 3e-4 \
  --num_epochs 10

# wandbを使用した学習追跡
python src/training/train_simple.py \
  --use_wandb \
  --wandb_project fof-training
```

### 3. 推論実行

```bash
# 基本的な推論
python validation/inference_hierarchical.py \
  --model_path models/pretrained_model.pth \
  --count 100 \
  --max_steps 30

# wandbを使用した推論追跡
python validation/inference_hierarchical.py \
  --use_wandb \
  --wandb_project fof-inference
```

### 4. データ管理

```bash
# 重複排除
python src/training/deduplicate_generated_data.py \
  --input_dir generated_data \
  --output_dir deduplicated_data

# データ分析
python src/training/analyze_generated_data.py
```

## 学習システムの特徴

### 全データ学習 + 推論性能評価

- **全データ学習**: 利用可能なすべてのデータを学習に使用
- **推論性能評価**: 実際の問題解決能力を測定
- **ランダム問題選択**: 毎回異なる問題で評価
- **実用的メトリクス**: 推論成功率と平均ステップ数

### 階層分類アーキテクチャ

```python
# 3つの独立した分類ヘッド
main_logits, arg1_logits, arg2_logits = model(input_ids, attention_mask)

# タクティクの種類に応じた引数要件
TACTIC_ARG_MASK = {
    "intro": (False, False),      # 引数不要
    "apply": (True, False),       # arg1のみ必要
    "specialize": (True, True),   # arg1, arg2両方必要
}
```

## 証明戦略

| 戦略 | main | arg1 | arg2 | 説明 |
|------|------|------|------|------|
| `assumption` | "assumption" | null | null | 前提の直接適用 |
| `intro` | "intro" | null | null | 含意導入 |
| `split` | "split" | null | null | 連言の分解 |
| `left` | "left" | null | null | 選言の左側選択 |
| `right` | "right" | null | null | 選言の右側選択 |
| `apply N` | "apply" | "N" | null | 前提Nの適用 |
| `destruct N` | "destruct" | "N" | null | 前提Nの分解 |
| `specialize N M` | "specialize" | "N" | "M" | 前提NをMで特殊化 |

## 推奨ワークフロー

```bash
# 1. データ生成
python src/data_generation/auto_data_parallel_collector.py --count 1000 --workers 4

# 2. 重複排除
python src/training/deduplicate_generated_data.py \
  --input_dir generated_data \
  --output_dir deduplicated_data

# 3. 学習
python src/training/train_simple.py \
  --data_dir deduplicated_data \
  --use_wandb

# 4. 推論
python validation/inference_hierarchical.py --verbose
```

## テスト

```bash
# 基本機能テスト
python tests/test_integration.py
python tests/test_parameter_sync.py

# 重複排除テスト
python tests/test_duplicate_check.py
python tests/test_deduplicated_data_hashes.py
```

## 謝辞

このプロジェクトは以下のオープンソースライブラリを使用しています：

- **[pyprover](https://github.com/kaicho8636/pyprover)** - 命題論理証明器ライブラリ
- **PyTorch** - 深層学習フレームワーク
- **wandb** - 実験追跡プラットフォーム