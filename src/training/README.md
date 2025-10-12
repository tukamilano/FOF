# Generated Data Training

このディレクトリには、`generated_data`ディレクトリのデータを使用してモデルを学習するためのスクリプトが含まれています。

## ファイル構成

### コア機能
- `train_with_generated_data.py`: メインの学習スクリプト（`generated_data`用）
- `run_training.py`: 学習を簡単に実行するためのラッパースクリプト
- `inference_hierarchical.py`: 階層分類モデルの推論スクリプト

### 分析・ユーティリティ
- `analyze_generated_data.py`: 生成されたデータの内容を分析するスクリプト
- `check_duplicates.py`: データの重複をチェックするスクリプト

### ドキュメント
- `README.md`: このファイル

## 使用方法

### 1. データ分析

まず、生成されたデータの内容を確認します：

```bash
python src/training/analyze_generated_data.py
```

重複の詳細を確認したい場合：

```bash
python src/training/check_duplicates.py
```

### 2. 学習実行

#### 簡単な実行（推奨）
```bash
python src/training/run_training.py
```

#### 詳細な設定で実行
```bash
python src/training/train_with_generated_data.py \
    --data_dir generated_data \
    --batch_size 16 \
    --learning_rate 3e-4 \
    --num_epochs 5 \
    --save_path models/hierarchical_model_generated.pth \
    --eval_split 0.2 \
    --max_seq_len 256 \
    --use_wandb \
    --wandb_project fof-training
```

### 3. 推論実行

学習したモデルを使用して推論を実行：

```bash
python src/training/inference_hierarchical.py \
    --model_path models/hierarchical_model_generated.pth \
    --count 10 \
    --max_steps 20 \
    --temperature 1.0 \
    --verbose \
    --use_wandb \
    --wandb_project fof-inference
```

## パラメータ説明

- `--data_dir`: 生成されたデータが格納されているディレクトリ（デフォルト: `generated_data`）
- `--batch_size`: バッチサイズ（デフォルト: 16）
- `--learning_rate`: 学習率（デフォルト: 3e-4）
- `--num_epochs`: エポック数（デフォルト: 5）
- `--save_path`: モデルの保存パス（デフォルト: `models/hierarchical_model_generated.pth`）
- `--eval_split`: 評価用データの割合（デフォルト: 0.2）
- `--max_seq_len`: 最大シーケンス長（デフォルト: 256）
- `--remove_duplicates`: 同じstate_hashの重複例を削除（デフォルト: 有効）
- `--keep_duplicates`: 重複例を保持（`--remove_duplicates`を無効化）
- `--use_wandb`: wandbでログを記録
- `--wandb_project`: wandbプロジェクト名（デフォルト: `fof-training`/`fof-inference`）
- `--wandb_run_name`: wandbラン名（デフォルト: 自動生成）

## wandbの使用方法

### 1. wandbのインストール
```bash
pip install wandb
```

### 2. wandbにログイン
```bash
wandb login
```

### 3. 学習時のログ記録
```bash
python src/training/train_with_generated_data.py --use_wandb --wandb_project fof-training
```

### 4. 推論時のログ記録
```bash
python src/training/inference_hierarchical.py --use_wandb --wandb_project fof-inference
```

### 記録される情報
- **学習時**: エポックごとの損失、精度、学習率
- **推論時**: 各例の成功/失敗、ステップ数、信頼度、タクティク使用頻度

## データ形式

`generated_data`ディレクトリには、以下の形式のJSONファイルが含まれている必要があります：

```json
[
  {
    "example_hash": "unique_hash",
    "meta": {
      "goal_original": "original_goal",
      "is_proved": true
    },
    "steps": [
      {
        "step_index": 0,
        "premises": ["premise1", "premise2"],
        "goal": "current_goal",
        "tactic": {
          "main": "tactic_name",
          "arg1": "argument1",
          "arg2": "argument2"
        },
        "tactic_apply": true,
        "state_hash": "state_hash"
      }
    ]
  }
]
```

## 二段階重複排除ワークフロー

学習時の重複排除を効率化するため、二段階のワークフローを提供しています：

### 段階1: 事前重複排除（推奨）

学習前に重複を事前に除去し、重複排除済みデータを生成：

```bash
# 重複排除を実行
python src/training/deduplicate_generated_data.py \
    --input_dir generated_data \
    --output_dir deduplicated_data \
    --report_file deduplication_report.json \
    --verbose
```

**出力形式:**
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

**メリット:**
- 重複排除と学習が分離され、効率的
- 重複排除済みデータを複数回再利用可能
- 詳細な重複統計レポートを生成
- 学習時のメモリ使用量を削減
- 単純なstepの集合形式でメモリ効率が向上

### 段階2: 学習実行

重複排除済みデータを使用して学習：

```bash
# 重複排除済みデータを使用（推奨）
python src/training/train_with_generated_data.py \
    --use_deduplicated_data \
    --data_dir deduplicated_data \
    --batch_size 32 \
    --learning_rate 3e-4

# または従来通り（重複排除を学習時に実行）
python src/training/train_with_generated_data.py \
    --data_dir generated_data \
    --batch_size 32 \
    --learning_rate 3e-4
```

### 従来の重複削除について

`generated_data`には同じ状態（`state_hash`が同じ）の重複例が含まれている場合があります。従来の方法では学習時に重複を自動的に削除します：

- **重複削除あり**（デフォルト）: 同じ`state_hash`の例は最初の1つだけを保持
- **重複保持**: `--keep_duplicates`オプションで重複を保持

重複削除により学習データの品質が向上し、より効率的な学習が可能になります。

## 注意事項

1. **データディレクトリ**: `generated_data`ディレクトリが存在し、JSONファイルが含まれていることを確認してください
2. **重複排除済みデータ**: `--use_deduplicated_data`を使用する場合は、事前に`deduplicate_generated_data.py`を実行してください
3. **メモリ使用量**: `batch_size`と`max_seq_len`を適切に設定してください
4. **学習時間**: 学習には時間がかかる場合があります。GPUが利用可能な場合は自動的に使用されます
5. **モデル保存**: モデルは`models`ディレクトリに保存されます（ディレクトリが存在しない場合は自動作成されます）
6. **重複削除効果**: 重複削除により学習データ数が減少する場合があります（通常7-10%程度）
7. **ワークフロー推奨**: 効率的な学習のため、二段階ワークフロー（事前重複排除→学習）の使用を推奨します
