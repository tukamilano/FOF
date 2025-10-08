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
- stable-baselines3 (PPO強化学習)
- gym (強化学習環境)
- その他の依存関係は `requirements.txt` を参照

## ファイル構成

### メインファイル

- **`run_interaction.py`** - メインの実行ファイル。数式生成、Transformer予測、証明実行の統合ワークフロー
- **`auto_data_collector.py`** - auto_classical()を使用したデータ収集システム（Transformer不要）
- **`ppo_trainer.py`** - PPO強化学習によるTransformerモデルのバッチ学習システム
- **`transformer_classifier.py`** - Transformerモデルとトークナイザーの実装
- **`generate_prop.py`** - 命題論理式の生成器
- **`fof_tokens.py`** - 入力トークンと出力ラベルの定義
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

### 5. 学習データ収集

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

### 5. PPO強化学習によるバッチ学習

```bash
# 基本的なPPO学習（10000ステップ、バッチサイズ32）
python ppo_trainer.py --total_timesteps 10000 --batch_size 32

# 短時間テスト（100ステップ、バッチサイズ8）
python ppo_trainer.py --total_timesteps 100 --batch_size 8 --max_steps 3

# 学習の再開
python ppo_trainer.py --resume_from ppo_models/ppo_model_5000

# 学習済みモデルの評価
python ppo_trainer.py --evaluate ppo_models/ppo_model_final --eval_episodes 100

# カスタム設定での学習
python ppo_trainer.py --total_timesteps 50000 --batch_size 64 --learning_rate 1e-4 --difficulty 0.7
```

### 6. テストファイルの実行

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
2. **トークン化**: `CharTokenizer`が文字レベルでトークン化
3. **予測**: `TransformerClassifier`が次の証明戦略を予測
4. **実行**: pyproverが実際の証明戦略を実行
5. **データ収集**: 学習データをJSON形式で蓄積（`--collect_data`オプション時）

### PPO強化学習ワークフロー（ppo_trainer.py）

1. **環境設定**: Gym環境として証明問題を定義
2. **バッチ生成**: 各バッチで新しい問題を生成（メモリ効率的）
3. **PPO学習**: stable-baselines3を使用した強化学習
4. **報酬計算**: 戦略成功かつ証明完了時のみ報酬1.0
5. **モデル保存**: 定期的なチェックポイント保存と学習再開機能

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

## PPO強化学習の詳細

### 報酬設計

PPO学習では以下の報酬設計を使用：

```python
def calculate_reward(tactic_apply: bool, is_proved: bool) -> float:
    if tactic_apply and is_proved:
        return 1.0  # 完全成功のみ報酬
    else:
        return 0.0  # その他は無報酬
```

### バッチ学習の特徴

- **メモリ効率**: 各バッチで新しい問題を生成し、使用後は破棄
- **バッチサイズ**: デフォルト32（小型Transformerに最適化）
- **学習再開**: 任意のチェックポイントから学習を再開可能
- **進捗ログ**: リアルタイムの学習進捗と統計情報

### モデル保存

- **チェックポイント**: `ppo_models/ppo_model_{timesteps}`で定期保存
- **最終モデル**: `ppo_models/ppo_model_final`で最終保存
- **学習再開**: `--resume_from`オプションで任意のチェックポイントから再開

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
4. `ppo_trainer.py`の`apply_tactic_from_label`関数にも同様の実装を追加

### モデルの改善

- `transformer_classifier.py`でモデルアーキテクチャを調整
- `test_example_train.py`でトレーニングデータを拡張
- `ppo_trainer.py`でPPOハイパーパラメータを調整

### PPO学習の調整

- **バッチサイズ**: `--batch_size`で調整（デフォルト32）
- **学習率**: `--learning_rate`で調整（デフォルト3e-4）
- **難易度**: `--difficulty`で問題の難易度を調整（デフォルト0.3）
- **最大ステップ**: `--max_steps`でエピソードの最大ステップ数を調整（デフォルト5）

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
python ppo_trainer.py --device cpu
```

### PPO学習時の問題

```bash
# 短時間テストで動作確認
python ppo_trainer.py --total_timesteps 100 --batch_size 4

# 学習の再開
python ppo_trainer.py --resume_from ppo_models/ppo_model_1000

# モデルの評価
python ppo_trainer.py --evaluate ppo_models/ppo_model_final --eval_episodes 10
```

## ライセンス

このプロジェクトのライセンスについては、各ファイルのヘッダーを確認してください。