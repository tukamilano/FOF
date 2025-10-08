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
- **`auto_data_collector.py`** - auto_classical()を使用したデータ収集システム（Transformer不要）
- **`transformer_classifier.py`** - Transformerモデルとトークナイザーの実装（制限なしの新しい形式）
- **`state_encoder.py`** - 証明状態のエンコーディング（制限なし）
- **`generate_prop.py`** - 命題論理式の生成器
- **`fof_tokens.py`** - 入力トークンと出力ラベルの定義（[EOS]トークン追加）
- **`pyprover/`** - 命題論理証明器ライブラリ

### テストファイル

- **`test_example_train.py`** - Transformerモデルのトレーニングテスト用
- **`test_evaluate_time.py`** - パフォーマンス評価とタイミング分析用

## 使用方法

### 1. Transformerベースの実行（従来の方法）

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

### 2. auto_classical()ベースのデータ収集（推奨）

```bash
# 基本的なデータ収集（Transformer不要）
python auto_data_collector.py --count 10

# 成功したテクティクのみを保存
python auto_data_collector.py --count 10 --filter_tactic_success_only

# より多くのデータを収集
python auto_data_collector.py --count 50 --filter_tactic_success_only

# 探索の深さを調整
python auto_data_collector.py --count 10 --max_depth 10
```

### 3. コマンドラインオプション

#### run_interaction.py のオプション

```bash
python run_interaction.py [オプション]

オプション:
  --count COUNT           実行するインタラクション数 (デフォルト: 3)
  --difficulty DIFFICULTY 数式の難易度 0.0-1.0 (デフォルト: 0.3)
  --seed SEED             乱数シード (デフォルト: 7)
  --device DEVICE         デバイス選択 auto/cpu/cuda (デフォルト: auto)
  --max_steps MAX_STEPS   1つの例あたりの最大戦略適用回数（成功・失敗含む） (デフォルト: 5)
  --selftest              自己テストを実行して終了
  --collect_data          学習データをJSON形式で収集
  --work_file FILE        作業用一時ファイル (デフォルト: temp_work.json)
  --dataset_file FILE     データセットファイル (デフォルト: training_data.json)
  --filter_tactic_success_only  成功した戦略のみを保存 (tactic_apply=true)
  --filter_successful_only      成功した戦略かつ証明完了のみを保存 (tactic_apply=true かつ is_proved=true)
```

#### auto_data_collector.py のオプション

```bash
python auto_data_collector.py [オプション]

オプション:
  --count COUNT                 処理する式の数 (デフォルト: 10)
  --difficulty DIFFICULTY       式生成の難易度 0.0-1.0 (デフォルト: 0.7)
  --seed SEED                   乱数シード (デフォルト: 7)
  --max_depth MAX_DEPTH         auto_classical()の最大探索深さ (デフォルト: 8)
  --dataset_file FILE           出力データセットファイル (デフォルト: training_data.json)
  --filter_tactic_success_only  成功したテクティクのみを保存 (tactic_apply=true)
```

### 4. 自己テスト

```bash
# 基本的な戦略（intro, apply）の動作確認
python run_interaction.py --selftest
```

### 5. 新しい形式のテスト

```bash
# 新しい制限なし形式のテスト
python -c "
from transformer_classifier import CharTokenizer
from fof_tokens import input_token

# 制限なしの前提でテスト
tokenizer = CharTokenizer(base_tokens=input_token)
goal = 'a'
premises = ['(a→b)', '(b→c)', '(c→d)', '(d→e)', '(e→f)']  # 5つの前提

# 新しい形式でエンコード
ids, mask, seg = tokenizer.encode(goal, premises, max_seq_len=512)
print(f'Goal: {goal}')
print(f'Premises: {premises}')
print(f'Number of premises: {len(premises)}')
print(f'Input format: [CLS] Goal [SEP] Premise₁ [SEP] Premise₂ [SEP] ... [EOS]')
print('Success: Unlimited premises format works!')
"
```

### 6. 学習データ収集

#### Transformerベースのデータ収集（従来の方法）

```bash
# 基本的なデータ収集（すべてのレコードを保存）
python run_interaction.py --count 10 --collect_data --max_steps 5

# 成功した戦略のみを保存
python run_interaction.py --count 10 --collect_data --max_steps 5 --filter_tactic_success_only

# 成功した戦略かつ証明完了のみを保存
python run_interaction.py --count 10 --collect_data --max_steps 5 --filter_successful_only
```

#### auto_classical()ベースのデータ収集（推奨）

```bash
# 基本的なデータ収集（すべてのレコードを保存）
python auto_data_collector.py --count 10

# 成功したテクティクのみを保存（推奨）
python auto_data_collector.py --count 10 --filter_tactic_success_only

# より多くのデータを収集
python auto_data_collector.py --count 50 --filter_tactic_success_only

# カスタムファイル名でデータ収集
python auto_data_collector.py --count 5 --dataset_file my_dataset.json
```

### 7. テストファイルの実行

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

### 基本的なワークフロー（run_interaction.py）

1. **数式生成**: `FormulaGenerator`が命題論理式を生成
2. **状態エンコーディング**: `encode_prover_state()`が制限なしで前提とゴールをエンコード
3. **トークン化**: `CharTokenizer.encode()`が新しい形式`[CLS] Goal [SEP] Premise₁ [SEP] Premise₂ [SEP] ... [EOS]`でトークン化
4. **予測**: `TransformerClassifier`が次の証明戦略を予測
5. **実行**: pyproverが実際の証明戦略を実行
6. **データ収集**: 学習データをJSON形式で蓄積（`--collect_data`オプション時）

### 新しい入力形式

Transformerへの入力は以下の形式でエンコードされます：

```
[CLS] Goal [SEP] Premise₁ [SEP] Premise₂ [SEP] Premise₃ [SEP] ... [EOS]
```


## 学習データ収集

### データ形式

学習データは以下のJSON形式で保存されます：

```json
[
  {
    "premises": ["(a → b)", "(b → c)", "(c → d)", "a"],
    "goal": "d",
    "tactic": {
      "main": "apply",
      "arg1": "0",
      "arg2": null
    },
    "tactic_apply": true,
    "is_proved": true
  }
]
```

### フィールド説明

- `premises`: 前提の配列（制限なし、任意の数）
- `goal`: 現在のゴール
- `tactic`: 構造化された戦略オブジェクト
  - `main`: 戦略の種類（"apply", "intro", "split"など）
  - `arg1`: 第1引数（インデックスやパラメータ）
  - `arg2`: 第2引数（specialize戦略などで使用）
- `tactic_apply`: 戦略の適用が成功したかどうか
- `is_proved`: その例全体で証明が成功したかどうか

### 戦略の種類

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

### ファイル管理

- **作業ファイル**: 各例の進行状況を一時保存（`temp_work.json`）
- **データセットファイル**: 完了した例の学習データを蓄積（`training_data.json`）

### データフィルタリングオプション

学習データ収集時に、以下のフィルタリングオプションを使用できます：

#### 1. フィルタリングなし（デフォルト）
```bash
python run_interaction.py --collect_data --count 10
```
- すべてのレコード（成功・失敗問わず）を保存
- 統計: `Examples: 10 (proved: 0, failed: 10), Records: 50`

#### 2. 成功した戦略のみ（`--filter_tactic_success_only`）
```bash
python run_interaction.py --collect_data --count 10 --filter_tactic_success_only
```
- `tactic_apply: true`のレコードのみを保存
- 証明の完了に関係なく、成功した戦略のデータを収集
- 統計: `Examples: 10 (proved: 0, failed: 10), Records: 15, Successful tactics: 15`

#### 3. 成功した戦略かつ証明完了のみ（`--filter_successful_only`）
```bash
python run_interaction.py --collect_data --count 10 --filter_successful_only
```
- `tactic_apply: true` かつ `is_proved: true`のレコードのみを保存
- 最も厳しい条件で、完全に成功した例のみを収集
- 統計: `Examples: 10 (proved: 0, failed: 10), Records: 0, Successful tactics: 0`

**注意**: 現在の設定では証明が完了する例が少ないため、`--filter_successful_only`オプションでは0件になる可能性があります。`--filter_tactic_success_only`オプションの使用を推奨します。


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

## データ収集の比較

### Transformerベース vs auto_classical()ベース

| 特徴 | Transformerベース | auto_classical()ベース |
|------|------------------|----------------------|
| **依存関係** | PyTorch、Transformerモデル | pyproverのみ |
| **速度** | モデル推論が必要 | 高速（最適化済み） |
| **データ品質** | 予測に依存 | 確実な証明パス |
| **設定** | 複雑（モデル、デバイス等） | シンプル |
| **推奨用途** | 研究・実験 | 実用的なデータ収集 |

### 推奨される使用方法

- **研究・実験**: `run_interaction.py`（Transformerベース）
- **実用的なデータ収集**: `auto_data_collector.py`（auto_classical()ベース）

## 開発

### 新しい戦略の追加

1. `fof_tokens.py`の`output`リストに戦略名を追加
2. `run_interaction.py`の`apply_tactic_from_label`関数に実装を追加
3. `auto_data_collector.py`の`apply_tactic_from_label`関数にも同様の実装を追加

### モデルの改善

- `transformer_classifier.py`でモデルアーキテクチャを調整（制限なしの新しい形式）
- `state_encoder.py`でエンコーディング方法を調整
- `test_example_train.py`でトレーニングデータを拡張

### 新しい入力形式のカスタマイズ

- `transformer_classifier.py`の`encode()`メソッドで入力形式を調整
- `state_encoder.py`の`encode_prover_state()`で状態エンコーディングを調整
- セグメントIDの割り当て（0=special, 1=goal, 2+=premises）を変更可能

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