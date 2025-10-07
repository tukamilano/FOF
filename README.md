# FOF (First-Order Formula) - Transformer-based Theorem Prover

このプロジェクトは、Transformerモデルを使用して命題論理の定理証明を自動化するシステムです。pyproverライブラリと組み合わせて、数式生成から証明戦略の予測まで一貫したワークフローを提供します。

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
- その他の依存関係は `requirements.txt` を参照

## ファイル構成

### メインファイル

- **`run_interaction.py`** - メインの実行ファイル。数式生成、Transformer予測、証明実行の統合ワークフロー
- **`transformer_classifier.py`** - Transformerモデルとトークナイザーの実装
- **`generate_prop.py`** - 命題論理式の生成器
- **`fof_tokens.py`** - 入力トークンと出力ラベルの定義
- **`pyprover/`** - 命題論理証明器ライブラリ

### テストファイル

- **`test_example_train.py`** - Transformerモデルのトレーニングテスト用
- **`test_evaluate_time.py`** - パフォーマンス評価とタイミング分析用

## 使用方法

### 1. 基本的な実行

```bash
# 仮想環境を有効化
source .venv/bin/activate

# 基本的な実行（3つの例を生成）
python run_interaction.py

# より多くの例を生成
python run_interaction.py --count 10

# 難易度を調整
python run_interaction.py --difficulty 0.7

# 最大ステップ数を設定
python run_interaction.py --max_steps 10
```

### 2. コマンドラインオプション

```bash
python run_interaction.py [オプション]

オプション:
  --count COUNT           実行するインタラクション数 (デフォルト: 3)
  --difficulty DIFFICULTY 数式の難易度 0.0-1.0 (デフォルト: 0.5)
  --seed SEED             乱数シード (デフォルト: 7)
  --device DEVICE         デバイス選択 auto/cpu/cuda (デフォルト: auto)
  --max_steps MAX_STEPS   1つの例あたりの最大戦略適用回数（成功・失敗含む） (デフォルト: 5)
  --selftest              自己テストを実行して終了
  --collect_data          学習データをJSON形式で収集
  --work_file FILE        作業用一時ファイル (デフォルト: temp_work.json)
  --dataset_file FILE     データセットファイル (デフォルト: training_data.json)
```

### 3. 自己テスト

```bash
# 基本的な戦略（intro, apply）の動作確認
python run_interaction.py --selftest
```

### 4. 学習データ収集

```bash
# 学習データを収集（10例、最大5ステップ）
python run_interaction.py --count 10 --collect_data --max_steps 5

# カスタムファイル名でデータ収集
python run_interaction.py --count 5 --collect_data --work_file my_work.json --dataset_file my_dataset.json
```

### 5. テストファイルの実行

```bash
# Transformerモデルのトレーニングテスト
python test_example_train.py

# 数式生成のテスト
python generate_prop.py --count 5 --difficulty 0.3

# パフォーマンス評価とタイミング分析
python test_evaluate_time.py --count 10 --max_interactions 5

# 学習データ収集機能のテスト
python test_data_collection.py
```

## システムの動作

1. **数式生成**: `FormulaGenerator`が命題論理式を生成
2. **トークン化**: `CharTokenizer`が文字レベルでトークン化
3. **予測**: `TransformerClassifier`が次の証明戦略を予測
4. **実行**: pyproverが実際の証明戦略を実行
5. **データ収集**: 学習データをJSON形式で蓄積（`--collect_data`オプション時）

## 学習データ収集

### データ形式

学習データは以下のJSON形式で保存されます：

```json
[
  {
    "premises": ["(a → b)", "(b → c)", "a"],
    "goal": "c",
    "tactic": "apply 0",
    "tactic_apply": true,
    "is_proved": true
  }
]
```

### フィールド説明

- `premises`: 前提の配列（最大3つ、不足分は空文字列）
- `goal`: 現在のゴール
- `tactic`: 適用された戦略
- `tactic_apply`: 戦略の適用が成功したかどうか
- `is_proved`: その例全体で証明が成功したかどうか

### ファイル管理

- **作業ファイル**: 各例の進行状況を一時保存（`temp_work.json`）
- **データセットファイル**: 完了した例の学習データを蓄積（`training_data.json`）

## パフォーマンス評価

`test_evaluate_time.py`を使用して、システムのパフォーマンスを詳細に分析できます：

### タイミング分析

```bash
# 基本的なパフォーマンス評価
python test_evaluate_time.py

# より詳細な分析
python test_evaluate_time.py --count 20 --max_interactions 10 --warmup 3

# GPU使用時のパフォーマンス比較
python test_evaluate_time.py --device cuda --count 15
```

### 測定されるコンポーネント

- **transformer_inference**: Transformerモデルの推論時間
- **tactic_application**: pyproverの戦略実行時間
- **state_extraction**: 証明状態からの入力抽出時間
- **tokenization**: 入力のトークン化時間

## 証明戦略

システムは以下の戦略をサポートします：

- `assumption` - 前提の直接適用
- `intro` - 含意導入
- `split` - 連言の分解
- `left`/`right` - 選言の左/右側選択
- `add_dn` - 二重否定の追加
- `apply N` - 前提Nの適用
- `destruct N` - 前提Nの分解
- `specialize N M` - 前提NをMで特殊化

## 開発

### 新しい戦略の追加

1. `fof_tokens.py`の`output`リストに戦略名を追加
2. `run_interaction.py`の`apply_tactic_from_label`関数に実装を追加

### モデルの改善

- `transformer_classifier.py`でモデルアーキテクチャを調整
- `test_example_train.py`でトレーニングデータを拡張

## トラブルシューティング

### 仮想環境の問題

```bash
# 仮想環境を再作成
rm -rf .venv
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 依存関係の問題

```bash
# 依存関係を再インストール
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```

### GPU使用時の問題

```bash
# CPUのみで実行
python run_interaction.py --device cpu
```

## ライセンス

このプロジェクトのライセンスについては、各ファイルのヘッダーを確認してください。