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
  --max_steps MAX_STEPS   1つの例あたりの最大戦略適用回数 (デフォルト: 5)
  --selftest              自己テストを実行して終了
```

### 3. 自己テスト

```bash
# 基本的な戦略（intro, apply）の動作確認
python run_interaction.py --selftest
```

### 4. テストファイルの実行

```bash
# Transformerモデルのトレーニングテスト
python test_example_train.py

# 数式生成のテスト
python generate_prop.py --count 5 --difficulty 0.3
```

## システムの動作

1. **数式生成**: `FormulaGenerator`が命題論理式を生成
2. **トークン化**: `CharTokenizer`が文字レベルでトークン化
3. **予測**: `TransformerClassifier`が次の証明戦略を予測
4. **実行**: pyproverが実際の証明戦略を実行

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
