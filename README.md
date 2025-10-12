# FOF (First-Order Formula) - Transformer-based Theorem Prover

このプロジェクトは、Transformerモデルを使用して命題論理の定理証明を自動化するシステムです。[pyprover](https://github.com/kaicho8636/pyprover)ライブラリと組み合わせて、数式生成から証明戦略の予測まで一貫したワークフローを提供します。

## 🚀 主な特徴

- **階層分類アーキテクチャ**: タクティクの種類と引数を独立して予測
- **推論性能評価システム**: 実際の問題解決能力を測定する評価システム
- **大規模データ処理**: GCS統合による効率的なデータ管理
- **高度な重複排除**: 効率的な重複排除システム
- **並列データ収集**: マルチプロセス対応の高速データ生成
- **実験追跡**: wandbによる詳細な学習・推論ログ

## 環境設定

### 仮想環境の作成と有効化

```bash
# 仮想環境を作成
python -m venv .venv

# 仮想環境を有効化
# macOS/Linux:
source .venv/bin/activate
# Windows:
# .venv\Scripts\activate

# 依存関係をインストール
pip install -r requirements.txt
```

### 依存関係

- Python 3.8+
- PyTorch
- [pyprover](https://github.com/kaicho8636/pyprover) - 命題論理証明器ライブラリ
- wandb (オプション) - 実験追跡
- その他の依存関係は `requirements.txt` を参照

### pyproverについて

このプロジェクトは[pyprover](https://github.com/kaicho8636/pyprover)ライブラリを使用して命題論理の定理証明を行います。pyproverは以下の特徴を持ちます：

- **Coqライクなインタラクティブインターフェース**
- **古典論理のサポート**
- **直感的な戦略システム**（assumption, intro, apply, split, left, right, destruct, specialize, add_dn, auto）
- **命題記号のサポート**（→, ∧, ∨, ¬）

pyproverの詳細な使用方法については、[公式リポジトリ](https://github.com/kaicho8636/pyprover)を参照してください。

## プロジェクト構造

```
FOF/
├── src/                          # メインソースコード
│   ├── interaction/              # インタラクション（オンライン学習）
│   │   └── run_interaction.py    # メインの実行ファイル。数式生成、Transformer予測、証明実行の統合ワークフロー
│   ├── data_generation/          # 事前学習データ生成
│   │   ├── auto_data_collector.py        # auto_classical()を使用したデータ収集システム
│   │   └── auto_data_parallel_collector.py # 並列処理対応の高速データ収集システム（GCS統合）
│   ├── training/                 # 学習関連
│   │   ├── train_with_generated_data.py  # 生成データを使用した学習スクリプト（推奨）
│   │   ├── inference_hierarchical.py     # 階層分類対応の推論スクリプト
│   │   ├── analyze_generated_data.py     # 生成データの分析
│   │   ├── check_duplicates.py           # 重複チェック
│   │   ├── deduplicate_generated_data.py # 生成データの重複排除
│   │   ├── deduplicate_gcs_data.py       # GCSデータの重複排除
│   │   ├── run_training.py               # 学習実行スクリプト
│   │   └── README.md                     # 学習システムの詳細ドキュメント
│   ├── compression/              # 圧縮関連
│   │   ├── create_compressed_training_data.py # 圧縮されたタクティクで新しいtraining_data.jsonを作成
│   │   └── extract_tactics.py        # BPEアルゴリズムでタクティクシーケンスを圧縮
│   └── core/                     # コア機能（共通）
│       ├── transformer_classifier.py # Transformerモデルとトークナイザーの実装
│       ├── state_encoder.py         # 証明状態のエンコーディング
│       ├── generate_prop.py         # 命題論理式の生成器
│       ├── parameter.py             # ハイパーパラメータ管理
│       ├── utils.py                 # 共通ユーティリティ関数
│       └── fof_tokens.py            # 入力トークンと出力ラベルの定義
├── tests/                        # テストファイル
│   ├── test_parameter_sync.py    # パラメータ同期テスト
│   ├── test_tactic_tokens.py     # タクティクトークンテスト
│   ├── test_integration.py       # 統合テスト
│   ├── test_wandb_connection.py  # wandb接続テスト
│   ├── test_single_file_training.py # 単一ファイル学習テスト
│   ├── test_duplicate_check.py   # 重複チェックテスト
│   ├── test_deduplicated_data_hashes.py # 重複排除データハッシュテスト
│   ├── test_gcs_cross_file_duplicates.py # GCSクロスファイル重複テスト
│   └── debug_duplicate_counting.py # 重複カウントデバッグ
├── examples/                     # 使用例
│   └── example_parameter_usage.py # parameter.pyの使用例
├── data/                         # データファイル
│   ├── training_data.json
│   ├── training_data_compressed.json
│   └── tactic_compression_*.json
├── generated_data/               # 生成された学習データ
│   ├── test_output_00001.json
│   ├── test_output_00002.json
│   └── ...
├── deduplicated_data/            # 重複排除済みデータ
│   ├── deduplicated_batch_00001.json
│   ├── deduplicated_batch_00002.json
│   └── ...
├── models/                       # 学習済みモデル
│   ├── hierarchical_model.pth
│   ├── hierarchical_model_generated.pth
│   └── test_*.pth                # テスト用モデル
├── pyprover/                     # pyprover（既存のまま）
├── test_inference_randomness.py  # 推論ランダム性テスト
├── test_problem_selection.py     # 問題選択テスト
├── deduplication_report.json     # 重複排除レポート
├── gcs_deduplication_report.json # GCS重複排除レポート
└── README.md
```

## 使用方法

### 1. データ生成（推奨ワークフロー）

#### 並列データ収集（本番環境推奨）

```bash
# 大規模データ収集（GCS直接アップロード）
python src/data_generation/auto_data_parallel_collector.py \
  --count 1000 \
  --examples_per_file 100 \
  --workers 2 \
  --gcs_bucket fof-data-20251009-milano \
  --gcs_prefix generated_data/ \
  --dataset_file training_data

# 中規模データ収集（ローカル保存）
python src/data_generation/auto_data_parallel_collector.py --count 100 --workers 4

# 高難易度・深い探索
python src/data_generation/auto_data_parallel_collector.py \
  --count 200 \
  --difficulty 0.9 \
  --max_depth 12 \
  --workers 8 \
  --examples_per_file 50 \
  --dataset_file high_difficulty_data

# 超大量データ収集（GCS + 高並列度）
python src/data_generation/auto_data_parallel_collector.py \
  --count 10000 \
  --examples_per_file 1000 \
  --workers 16 \
  --gcs_bucket fof-data-20251009-milano \
  --gcs_prefix generated_data/ \
  --dataset_file large_scale_data \
  --buffer_size 5000
```

#### 並列データ収集の特徴

- **マルチプロセス処理**: CPUコア数を活用した高速データ生成
- **GCS統合**: 大規模データセットの効率的な管理
- **重複排除**: Example重複の自動検出とスキップ
- **ストリーミング処理**: メモリ効率的な大規模データ処理
- **バッチ分割**: 指定されたサイズでファイルを自動分割

#### 基本的なデータ収集

```bash
# 基本的なデータ収集（Transformer不要）
python src/data_generation/auto_data_collector.py --count 10

# 探索の深さを調整
python src/data_generation/auto_data_collector.py --count 10 --max_depth 10
```

### 2. モデル学習

#### 生成データを使用した学習（推奨）

このシステムは**全データを学習に使用**し、**推論性能評価**でモデルの性能を測定します。従来のvalidation分割は行わず、実際の問題解決能力を評価します。

```bash
# 基本的な学習（全データ使用 + 推論性能評価）
python src/training/train_with_generated_data.py

# 重複排除済みデータを使用（推奨）
python src/training/train_with_generated_data.py \
  --data_dir deduplicated_data \
  --batch_size 32 \
  --learning_rate 3e-4

# カスタム設定での学習
python src/training/train_with_generated_data.py \
  --batch_size 64 \
  --learning_rate 1e-4 \
  --num_epochs 20 \
  --max_seq_len 512

# wandbを使用した学習追跡
python src/training/train_with_generated_data.py \
  --use_wandb \
  --wandb_project fof-training \
  --wandb_run_name experiment_001

# 重複を保持した学習
python src/training/train_with_generated_data.py --keep_duplicates

# カスタムデータディレクトリ
python src/training/train_with_generated_data.py --data_dir my_generated_data
```

#### 学習オプション

```bash
python src/training/train_with_generated_data.py [オプション]

オプション:
  --data_dir DIR                   生成データディレクトリ (デフォルト: generated_data)
  --batch_size SIZE                バッチサイズ (デフォルト: 32)
  --learning_rate RATE             学習率 (デフォルト: 3e-4)
  --num_epochs EPOCHS              エポック数 (デフォルト: 10)
  --device DEVICE                  デバイス選択 auto/cpu/cuda (デフォルト: auto)
  --save_path PATH                 モデル保存パス (デフォルト: models/hierarchical_model_generated.pth)
  --max_seq_len LEN                最大シーケンス長 (デフォルト: 512)
  --remove_duplicates              重複削除を有効化 (デフォルト: True)
  --keep_duplicates                重複を保持 (--remove_duplicatesを無効化)
  --use_wandb                      wandbを使用した学習追跡
  --wandb_project PROJECT          wandbプロジェクト名 (デフォルト: fof-training)
  --wandb_run_name NAME            wandb実行名
  --arg1_loss_weight WEIGHT        arg1損失重み (デフォルト: 0.8)
  --arg2_loss_weight WEIGHT        arg2損失重み (デフォルト: 0.8)
  --inference_eval_examples COUNT  推論評価例数 (デフォルト: 50)
  --inference_max_steps STEPS      推論最大ステップ数 (デフォルト: 30)
  --inference_temperature TEMP     推論温度 (デフォルト: 1.0)
  --validation_frequency FREQ      推論評価の実行頻度 (デフォルト: 1000データポイント)
  --skip_inference_eval            推論性能評価をスキップ（高速学習）
```

### 3. 推論実行

#### 階層分類推論

```bash
# 基本的な推論
python src/training/inference_hierarchical.py

# カスタム設定での推論
python src/training/inference_hierarchical.py \
  --model_path models/hierarchical_model_generated.pth \
  --num_examples 100 \
  --max_steps 20 \
  --temperature 0.8 \
  --verbose

# wandbを使用した推論追跡
python src/training/inference_hierarchical.py \
  --use_wandb \
  --wandb_project fof-inference \
  --wandb_run_name inference_test
```

#### インタラクティブ実行

```bash
# 基本的な実行（selftest）
python src/interaction/run_interaction.py --selftest

# より多くの例を生成
python src/interaction/run_interaction.py --count 10

# 難易度を調整
python src/interaction/run_interaction.py --difficulty 0.7

# 最大ステップ数を設定
python src/interaction/run_interaction.py --max_steps 10
```

### 4. データ分析

#### 生成データの分析

```bash
# 生成データの統計分析
python src/training/analyze_generated_data.py

# 重複チェック
python src/training/check_duplicates.py

# 特定のファイルを分析
python src/training/analyze_generated_data.py --data_dir generated_data
```

## 学習システムの特徴

### 全データ学習 + 推論性能評価

このシステムは従来のvalidation分割を廃止し、より効率的で実用的な学習アプローチを採用しています：

#### 新しい学習アプローチ
- **全データ学習**: 利用可能なすべてのデータを学習に使用（データの無駄を排除）
- **推論性能評価**: 実際の問題解決能力を測定する評価システム
- **ランダム問題選択**: 毎回異なる問題で評価（公平性の確保）
- **実用的メトリクス**: 推論成功率と平均ステップ数で性能を測定

#### 推論性能評価の特徴
- **毎回異なる問題**: 現在時刻をシードとして使用し、毎回ランダムに問題を選択
- **実際の証明実行**: pyproverを使用した実際の定理証明で性能を測定
- **純粋な言語モデル性能**: 人工的な精度向上要因を排除した真の性能評価

### 重複排除済みデータ対応

#### 新しいデータ形式サポート
- **DeduplicatedDataDataset**: 重複排除済みデータ専用のデータセットクラス
- **効率的なメモリ使用**: 単純なstepの集合形式でメモリ効率を向上
- **バッチ処理**: 大規模データセットの効率的な処理

#### 学習オプション
- **重複排除済みデータ使用**: `--data_dir deduplicated_data`で重複排除済みデータを指定
- **従来データ使用**: `--data_dir generated_data`で従来のデータ形式を使用
- **重複保持**: `--keep_duplicates`で重複を保持した学習

### 階層分類アーキテクチャ

```python
# 3つの独立した分類ヘッド
main_logits, arg1_logits, arg2_logits = model(input_ids, attention_mask)

# タクティクの種類に応じた引数要件
TACTIC_ARG_MASK = {
    "intro": (False, False),      # 引数不要
    "apply": (True, False),       # arg1のみ必要
    "specialize": (True, True),   # arg1, arg2両方必要
    # ...
}
```

### データ形式

#### 学習データ形式
```json
[
  {
    "example_hash": "1743dfaf9bb101e51276719e50ba05ce",
    "meta": {
      "goal_original": "((((b ∧ a) ∧ ((b ∨ c) → False)) ∧ (c → (b → c))) → b)",
      "is_proved": true
    },
    "steps": [
      {
        "step_index": 0,
        "premises": [],
        "goal": "((((b ∧ a) ∧ ((b ∨ c) → False)) ∧ (c → (b → c))) → b)",
        "tactic": {
          "main": "intro",
          "arg1": null,
          "arg2": null
        },
        "tactic_apply": true,
        "state_hash": "29326faf43695967bc47255fc73a580c"
      }
    ]
  }
]
```

#### 入力エンコーディング
```
[CLS] Goal [SEP] Premise₁ [SEP] Premise₂ [SEP] Premise₃ [SEP] ... [EOS]
```

## 実験追跡（wandb）

### 学習メトリクス

学習中に以下のメトリクスが記録されます：

- `train_loss`: 訓練損失
- `inference/success_rate`: 推論成功率（実際の問題解決能力）
- `inference/avg_steps`: 推論平均ステップ数（解決時の効率性）
- `learning_rate`: 学習率
- `best_inference_success_rate`: 最高推論成功率（モデル保存の基準）

#### 推論性能評価の詳細
- **問題選択**: 毎回ランダムに異なる問題を選択（公平性確保）
- **評価頻度**: 指定されたデータポイント数ごとに実行（デフォルト: 1000）
- **評価例数**: デフォルト50例（カスタマイズ可能）
- **最大ステップ数**: 各問題の最大証明ステップ数（デフォルト: 30）

### 使用方法

```bash
# wandbを使用した学習
python src/training/train_with_generated_data.py --use_wandb

# カスタムプロジェクト名
python src/training/train_with_generated_data.py \
  --use_wandb \
  --wandb_project my-fof-experiment \
  --wandb_run_name run_001
```

## 重複排除システム

### 概要

大規模データセットの効率的な処理のため、重複排除機能を提供しています。これにより、学習データの品質向上とメモリ使用量の最適化を実現します。

### 重複排除の実行

#### 生成データの重複排除

```bash
# 生成データの重複排除
python src/training/deduplicate_generated_data.py \
    --input_dir generated_data \
    --output_dir deduplicated_data \
    --report_file deduplication_report.json \
    --verbose
```

#### GCSデータの重複排除（大規模データセット用）

```bash
# GCSデータの重複排除
python src/training/deduplicate_gcs_data.py \
    --gcs_bucket fof-data-20251009-milano \
    --gcs_prefix generated_data/ \
    --output_dir deduplicated_data \
    --report_file gcs_deduplication_report.json \
    --batch_size 20000 \
    --max_workers 8 \
    --verbose
```

### 重複排除済みデータの使用

重複排除済みデータを使用して学習：

```bash
# 重複排除済みデータを使用
python src/training/train_with_generated_data.py \
    --data_dir deduplicated_data \
    --batch_size 32 \
    --learning_rate 3e-4

# または従来通り（重複排除を学習時に実行）
python src/training/train_with_generated_data.py \
    --data_dir generated_data \
    --batch_size 32 \
    --learning_rate 3e-4
```

### 重複排除の種類

1. **Example重複**: 同じ`example_hash`の例（完全に同一の問題）
2. **State重複**: 同じ`state_hash`の例（同じ証明状態）
3. **State-Tactic重複**: 同じ`state_tactic_hash`の例（同じ状態+戦略の組み合わせ）

### 出力形式

重複排除後は証明の連続性が失われるため、単純なstepの集合として保存されます：

```json
[
  {
    "step_index": 0,
    "premises": [],
    "goal": "((((b ∧ a) ∧ ((b ∨ c) → False)) ∧ (c → (b → c))) → b)",
    "tactic": {
      "main": "intro",
      "arg1": null,
      "arg2": null
    },
    "tactic_apply": true,
    "state_hash": "29326faf43695967bc47255fc73a580c"
  }
]
```

### GCS統合

大規模データセットの効率的な処理のため、Google Cloud Storageとの統合を提供：

- **直接アップロード**: データ生成時にGCSに直接アップロード
- **並列ダウンロード**: 複数ファイルの並列ダウンロード
- **メモリ効率**: 大規模データセットでもメモリ効率的に処理
- **バッチ処理**: 指定されたサイズでバッチ分割

## データ圧縮システム

### 概要

BPE（Byte Pair Encoding）アルゴリズムを使用してタクティクシーケンスを圧縮し、より効率的な学習データを作成します。

### 圧縮プロセス

```bash
# タクティクシーケンスの圧縮
python src/compression/extract_tactics.py

# 圧縮された学習データの作成
python src/compression/create_compressed_training_data.py
```

## 推論性能評価システム

### 概要

従来のvalidation分割に代わり、実際の問題解決能力を測定する推論性能評価システムを採用しています。

### 特徴

- **全データ学習**: 利用可能なすべてのデータを学習に使用
- **ランダム問題選択**: 毎回異なる問題で評価（公平性確保）
- **実際の証明実行**: pyproverを使用した実際の定理証明
- **実用的メトリクス**: 推論成功率と平均ステップ数

### 評価プロセス

1. **問題選択**: 現在時刻をシードとしてランダムに問題を選択
2. **証明実行**: 選択された問題に対して実際に証明を実行
3. **性能測定**: 成功率と平均ステップ数を計算
4. **モデル保存**: 最高推論成功率でモデルを保存

## テスト

### テスト実行

#### 基本機能テスト

```bash
# 統合テスト
python tests/test_integration.py

# パラメータテスト
python tests/test_parameter_sync.py

# タクティクトークンテスト
python tests/test_tactic_tokens.py

# wandb接続テスト
python tests/test_wandb_connection.py

# 単一ファイル学習テスト
python tests/test_single_file_training.py
```

#### 重複排除テスト

```bash
# 重複チェックテスト
python tests/test_duplicate_check.py

# 重複排除データハッシュテスト
python tests/test_deduplicated_data_hashes.py

# GCSクロスファイル重複テスト
python tests/test_gcs_cross_file_duplicates.py

# 重複カウントデバッグ
python tests/debug_duplicate_counting.py
```

#### 推論・問題選択テスト

```bash
# 推論ランダム性テスト
python test_inference_randomness.py

# 問題選択テスト
python test_problem_selection.py
```

### テストカバレッジ

- **基本機能**: インポート、パラメータ同期、トークナイザー
- **学習システム**: 単一ファイル学習、wandb統合
- **重複排除**: 各種重複チェック、ハッシュ検証
- **GCS統合**: クロスファイル重複検出
- **推論システム**: ランダム性、問題選択の公平性

## 証明戦略

システムは以下の戦略をサポートします：

| 戦略 | main | arg1 | arg2 | 説明 |
|------|------|------|------|------|
| `assumption` | "assumption" | null | null | 前提の直接適用 |
| `intro` | "intro" | null | null | 含意導入 |
| `split` | "split" | null | null | 連言の分解 |
| `left` | "left" | null | null | 選言の左側選択 |
| `right` | "right" | null | null | 選言の右側選択 |
| `add_dn` | "add_dn" | null | null | 二重否定の追加 |
| `apply N` | "apply" | "N" | null | 前提Nの適用 |
| `destruct N` | "destruct" | "N" | null | 前提Nの分解 |
| `specialize N M` | "specialize" | "N" | "M" | 前提NをMで特殊化 |

## 開発

### モデルの改善

- `src/core/transformer_classifier.py`でモデルアーキテクチャを調整
- `src/core/state_encoder.py`でエンコーディング方法を調整
- `src/core/parameter.py`でハイパーパラメータを管理

### 新しい入力形式のカスタマイズ

- `src/core/transformer_classifier.py`の`encode()`メソッドで入力形式を調整
- `src/core/state_encoder.py`の`encode_prover_state()`で状態エンコーディングを調整
- セグメントIDの割り当て（0=special, 1=goal, 2+=premises）を変更可能

## 推奨ワークフロー

### 完全なワークフロー（大規模データセット）

```bash
# 1. 大規模データ生成（GCS直接アップロード）
python src/data_generation/auto_data_parallel_collector.py \
  --count 10000 \
  --examples_per_file 1000 \
  --workers 8 \
  --gcs_bucket fof-data-20251009-milano \
  --gcs_prefix generated_data/ \
  --dataset_file large_scale_data

# 2. GCSデータの重複排除
python src/training/deduplicate_gcs_data.py \
  --gcs_bucket fof-data-20251009-milano \
  --gcs_prefix generated_data/ \
  --output_dir deduplicated_data \
  --report_file gcs_deduplication_report.json \
  --batch_size 20000 \
  --max_workers 8 \
  --verbose

# 3. 重複排除済みデータで学習
python src/training/train_with_generated_data.py \
  --data_dir deduplicated_data \
  --batch_size 32 \
  --learning_rate 3e-4 \
  --num_epochs 10 \
  --use_wandb \
  --wandb_project fof-training \
  --wandb_run_name large_scale_experiment

# 4. 推論性能評価
python src/training/inference_hierarchical.py \
  --model_path models/hierarchical_model_generated.pth \
  --num_examples 100 \
  --max_steps 30 \
  --use_wandb \
  --wandb_project fof-inference
```

### 簡易ワークフロー（小規模データセット）

```bash
# 1. データ生成
python src/data_generation/auto_data_parallel_collector.py --count 100 --workers 4

# 2. 重複排除
python src/training/deduplicate_generated_data.py \
  --input_dir generated_data \
  --output_dir deduplicated_data \
  --verbose

# 3. 学習
python src/training/train_with_generated_data.py \
  --data_dir deduplicated_data \
  --use_wandb

# 4. 推論
python src/training/inference_hierarchical.py --verbose
```

### 開発・テストワークフロー

```bash
# 1. テスト実行
python tests/test_integration.py
python tests/test_duplicate_check.py
python tests/test_wandb_connection.py

# 2. 小規模データで動作確認
python src/data_generation/auto_data_collector.py --count 10
python src/training/train_with_generated_data.py --data_dir generated_data --num_epochs 1

# 3. 推論テスト
python test_inference_randomness.py
python test_problem_selection.py
```

## 謝辞

このプロジェクトは以下のオープンソースライブラリを使用しています：

- **[pyprover](https://github.com/kaicho8636/pyprover)** - 命題論理証明器ライブラリ
- **PyTorch** - 深層学習フレームワーク
- **wandb** - 実験追跡プラットフォーム