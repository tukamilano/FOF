# FOF (First-Order Formula) - Transformer-based Theorem Prover

このプロジェクトは、Transformerモデルを使用して命題論理の定理証明を自動化するシステムです。[pyprover](https://github.com/kaicho8636/pyprover)ライブラリと組み合わせて、数式生成から証明戦略の予測まで一貫したワークフローを提供します。

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
│   │   └── auto_data_parallel_collector.py # 並列処理対応の高速データ収集システム
│   ├── training/                 # 学習関連
│   │   ├── train_hierarchical.py     # 階層分類対応の学習スクリプト
│   │   └── inference_hierarchical.py # 階層分類対応の推論スクリプト
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
│   └── test_integration.py       # 統合テスト
├── examples/                     # 使用例
│   └── example_parameter_usage.py # parameter.pyの使用例
├── data/                         # データファイル
│   ├── training_data.json
│   ├── training_data_compressed.json
│   └── tactic_compression_*.json
├── models/                       # 学習済みモデル
│   └── hierarchical_model.pth
├── pyprover/                     # pyprover（既存のまま）
└── README.md
```

## 使用方法

### 1. Transformerベースの実行（従来の方法）

```bash
# 仮想環境を有効化
source .venv/bin/activate

# 基本的な実行（selftest）
python src/interaction/run_interaction.py --selftest

# より多くの例を生成
python src/interaction/run_interaction.py --count 10

# 難易度を調整
python src/interaction/run_interaction.py --difficulty 0.7

# 最大ステップ数を設定
python src/interaction/run_interaction.py --max_steps 10
```

### 2. auto_classical()ベースのデータ収集（推奨）

```bash
# 基本的なデータ収集（Transformer不要）
python src/data_generation/auto_data_collector.py --count 10

# 探索の深さを調整
python src/data_generation/auto_data_collector.py --count 10 --max_depth 10
```

### 3. コマンドラインオプション

#### run_interaction.py のオプション

```bash
python src/interaction/run_interaction.py [オプション]

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
  --filter_successful_only      成功した戦略かつ証明完了のみを保存 (tactic_apply=true かつ is_proved=true)
```

#### auto_data_collector.py のオプション

```bash
python src/data_generation/auto_data_collector.py [オプション]

オプション:
  --count COUNT                 処理する式の数 (デフォルト: 10)
  --difficulty DIFFICULTY       式生成の難易度 0.0-1.0 (デフォルト: 0.7)
  --seed SEED                   乱数シード (デフォルト: 7)
  --max_depth MAX_DEPTH         auto_classical()の最大探索深さ (デフォルト: 8)
  --dataset_file FILE           出力データセットファイル (デフォルト: training_data.json)
```

#### auto_data_parallel_collector.py のオプション

```bash
python src/data_generation/auto_data_parallel_collector.py [オプション]

オプション:
  --count COUNT                 処理する式の数 (デフォルト: 10)
  --difficulty DIFFICULTY       式生成の難易度 0.0-1.0 (デフォルト: 0.7)
  --seed SEED                   乱数シード (デフォルト: 7)
  --max_depth MAX_DEPTH         auto_classical()の最大探索深さ (デフォルト: 8)
  --dataset_file FILE           出力データセットファイル (デフォルト: training_data.json)
  --workers WORKERS             並列ワーカー数 (デフォルト: min(cpu_count, 8))
```

### 4. 自己テスト

```bash
# 基本的な戦略（intro, apply）の動作確認
python src/interaction/run_interaction.py --selftest
```

### 5. 新しい形式のテスト

```bash
# 新しい制限なし形式のテスト
python -c "
import sys
import os
sys.path.insert(0, '.')
from src.core.transformer_classifier import CharTokenizer
from src.core.fof_tokens import input_token

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

#### Transformerベースのデータ収集

```bash
# 基本的なデータ収集（すべてのレコードを保存）
python src/interaction/run_interaction.py --count 10 --collect_data --max_steps 5

# 成功した戦略かつ証明完了のみを保存
python src/interaction/run_interaction.py --count 10 --collect_data --max_steps 5 --filter_successful_only
```

#### auto_classical()ベースのデータ収集

```bash
# 基本的なデータ収集（すべてのレコードを保存）
python src/data_generation/auto_data_collector.py --count 10

# カスタムファイル名でデータ収集
python src/data_generation/auto_data_collector.py --count 5 --dataset_file data/my_dataset.json
```

#### 並列データ収集（高速処理）

```bash
# 高速並列モード（証明発見のみ）
python src/data_generation/auto_data_parallel_collector.py --count 100 --workers 4

# カスタムワーカー数での並列処理
python src/data_generation/auto_data_parallel_collector.py --count 50 --workers 2

# カスタム設定での並列処理
python src/data_generation/auto_data_parallel_collector.py --count 200 --max_depth 10 --workers 8 --dataset_file data/parallel_dataset.json
```

### 7. データ圧縮の実行

#### タクティクシーケンスの圧縮

```bash
# 基本的なBPE圧縮（デフォルト200回のマージ）
python src/compression/extract_tactics.py

# カスタムマージ回数で圧縮
python src/compression/extract_tactics.py 100

# 圧縮結果の確認
ls -la data/tactic_compression_bpe_analysis.json
```

#### 圧縮された学習データの作成

```bash
# 圧縮されたタクティクで新しいtraining_data.jsonを作成
python src/compression/create_compressed_training_data.py

# 圧縮結果の確認
ls -la data/training_data_compressed_bpe.json
```

### 8. テストファイルの実行

```bash

# 数式生成のテスト
python src/core/generate_prop.py --count 5 --difficulty 0.3


```

## システムの動作

### 基本的なワークフロー（run_interaction.py）

1. **数式生成**: `FormulaGenerator`が命題論理式を生成
2. **状態エンコーディング**: `encode_prover_state()`が制限なしで前提とゴールをエンコード
3. **トークン化**: `CharTokenizer.encode()`が新しい形式`[CLS] Goal [SEP] Premise₁ [SEP] Premise₂ [SEP] ... [EOS]`でトークン化
4. **予測**: `TransformerClassifier`が次の証明戦略を予測
5. **実行**: [pyprover](https://github.com/kaicho8636/pyprover)が実際の証明戦略を実行
6. **データ収集**: 学習データをJSON形式で蓄積（`--collect_data`オプション時）

### 新しい入力形式

Transformerへの入力は以下の形式でエンコードされます：

```
[CLS] Goal [SEP] Premise₁ [SEP] Premise₂ [SEP] Premise₃ [SEP] ... [EOS]
```


## データ圧縮システム

### 概要

このシステムは、BPE（Byte Pair Encoding）アルゴリズムを使用してタクティクシーケンスを圧縮し、より効率的な学習データを作成します。圧縮により、冗長なタクティクの組み合わせを単一のトークンに統合し、モデルの学習効率を向上させます。

### 圧縮プロセス

1. **タクティク抽出**: `training_data.json`からタクティクシーケンスを抽出
2. **BPE適用**: 最も頻繁に出現するタクティクペアを繰り返しマージ
3. **圧縮データ生成**: 圧縮されたタクティクで新しい学習データを作成

### 圧縮ファイル

- **`tactic_compression_bpe_analysis.json`** - BPE圧縮の分析結果とマッピング情報
- **`training_data_compressed_bpe.json`** - 圧縮されたタクティクで作成された学習データ

### 圧縮効果

- タクティクシーケンスの長さを大幅に短縮
- 頻出するタクティクの組み合わせを単一トークンに統合
- モデルの学習効率と推論速度の向上

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
python src/interaction/run_interaction.py --collect_data --count 10
```
- すべてのレコード（成功・失敗問わず）を保存
- 統計: `Examples: 10 (proved: 0, failed: 10), Records: 50`

#### 2. 成功した戦略かつ証明完了のみ（`--filter_successful_only`）
```bash
python src/interaction/run_interaction.py --collect_data --count 10 --filter_successful_only
```
- `tactic_apply: true` かつ `is_proved: true`のレコードのみを保存
- 最も厳しい条件で、完全に成功した例のみを収集
- 統計: `Examples: 10 (proved: 0, failed: 10), Records: 0, Successful tactics: 0`

**注意**: 現在の設定では証明が完了する例が少ないため、`--filter_successful_only`オプションでは0件になる可能性があります。基本的なデータ収集（フィルタなし）の使用を推奨します。

## 並列データ収集システム

### 概要

`auto_data_parallel_collector.py`は、マルチプロセシングを使用して大規模なデータ収集を高速化するシステムです。CPUコアを活用して複数の数式を並列処理し、従来のシーケンシャル処理と比較して大幅な処理時間短縮を実現します。

### 主要機能

#### 1. 並列処理
- **高速処理**: 証明発見のみに焦点を当てた高速並列処理
- **メモリ効率**: ワーカー数制限によるメモリ使用量の最適化

#### 2. ワーカー管理
- **自動ワーカー数決定**: `min(cpu_count, 8)`でメモリ使用量を制限
- **カスタムワーカー数**: `--workers`オプションで手動設定
- **プロセスプール**: `ProcessPoolExecutor`による効率的な並列実行

#### 3. 進捗表示
- **リアルタイム進捗**: `tqdm`によるプログレスバー
- **成功統計**: リアルタイムでの証明成功数表示
- **エラーハンドリング**: 個別プロセスのエラーを適切に処理

### 処理フロー

1. **数式生成**: 指定された数の数式を事前生成
2. **並列処理**: 各ワーカープロセスで数式を並列処理
3. **証明発見**: `auto_classical()`を使用して証明パスを発見
4. **結果集約**: 全プロセスの結果を統合して保存

### 出力形式

```json
[
  {
    "example_id": 0,
    "formula": "(a → b) → (b → c) → (a → c)",
    "proof_found": true,
    "proof_path": ["intro", "intro", "apply 0", "apply 1"],
    "total_steps": 4,
    "time_taken": 0.123,
    "worker_id": 12345
  }
]
```

### パフォーマンス特性

#### メモリ使用量
- **ワーカー制限**: デフォルトで最大8ワーカー（メモリ保護）
- **プロセス分離**: 各ワーカーが独立したメモリ空間で動作
- **ガベージコレクション**: プロセス終了時の自動メモリ解放

#### 処理速度
- **並列効率**: CPUコア数に比例した処理速度向上
- **I/O最適化**: バッチ処理による効率的なファイル操作
- **エラー耐性**: 個別プロセスの失敗が全体に影響しない

### 使用例

#### 大規模データ収集
```bash
# 1000個の数式を8ワーカーで並列処理
python src/data_generation/auto_data_parallel_collector.py --count 1000 --workers 8
```

#### 中規模データ収集
```bash
# 50個の数式を4ワーカーで並列処理
python src/data_generation/auto_data_parallel_collector.py --count 50 --workers 4
```

#### カスタム設定
```bash
# 高難易度、深い探索、カスタム出力ファイル
python src/data_generation/auto_data_parallel_collector.py \
  --count 500 \
  --difficulty 0.9 \
  --max_depth 12 \
  --workers 6 \
  --dataset_file data/high_difficulty_dataset.json
```

### トラブルシューティング

#### メモリ不足
- ワーカー数を減らす: `--workers 2`
- 処理する数式数を減らす: `--count 100`

#### プロセスエラー
- ログでエラー詳細を確認
- 個別の数式で問題を特定
- ワーカー数を調整して再実行

## パフォーマンス評価

テストファイルを使用して、システムのパフォーマンスを分析できます：

### テスト実行

```bash
# 統合テストの実行
python tests/test_integration.py

# パラメータテストの実行
python tests/test_parameter_sync.py

# タクティクトークンテストの実行
python tests/test_tactic_tokens.py
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

### モデルの改善

- `transformer_classifier.py`でモデルアーキテクチャを調整（制限なしの新しい形式）
- `state_encoder.py`でエンコーディング方法を調整

### 新しい入力形式のカスタマイズ

- `transformer_classifier.py`の`encode()`メソッドで入力形式を調整
- `state_encoder.py`の`encode_prover_state()`で状態エンコーディングを調整
- セグメントIDの割り当て（0=special, 1=goal, 2+=premises）を変更可能

## 謝辞

このプロジェクトは以下のオープンソースライブラリを使用しています：

- **[pyprover](https://github.com/kaicho8636/pyprover)** - 命題論理証明器ライブラリ
