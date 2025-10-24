# Actor-Critic シャッフル学習機能

actor_critic_dataのfailed_tacticsとsuccessful_tacticsを使ってシャッフルした学習を実装しました。

## 機能概要

- **全データシャッフル**: 成功データと失敗データの全てを使用してシャッフル
- **ランダムシード**: 再現可能なシャッフル結果
- **データ保持**: 全てのデータが保持され、ラベル情報も維持

## 使用方法

### 基本的な使用方法

```bash
# デフォルト設定で学習（シャッフル比率50%）
python train_actor_critic.py --data_dir actor_critic_data --output_dir actor_critic_models

# シャッフル比率を指定
python train_actor_critic.py --shuffle_ratio 0.3 --random_seed 42

# シャッフルなしで学習
python train_actor_critic.py --shuffle_ratio 0.0
```

### パラメータ説明

- `--shuffle_ratio`: シャッフルする割合 (0.0-1.0, デフォルト: 0.5)
- `--random_seed`: ランダムシード (デフォルト: 42)
- `--data_dir`: データディレクトリ (デフォルト: actor_critic_data)
- `--output_dir`: 出力ディレクトリ (デフォルト: actor_critic_models)

### シャッフル機能の詳細

1. **全データ使用**: 成功データと失敗データの全てを結合してシャッフル
2. **ラベル保持**: 元のデータの成功/失敗ラベル情報を維持
3. **再現性**: 同じ`random_seed`で同じ結果を再現

## テスト

シャッフル機能のテストを実行:

```bash
python test_shuffle_training.py
```

テスト内容:
- シャッフル機能の基本動作
- 異なるシャッフル比率での動作
- ランダムシードの一貫性
- エッジケース（空データ、片方のみのデータなど）

## 実装詳細

### シャッフルアルゴリズム

1. 成功データと失敗データを全て結合
2. 結合されたデータをランダムにシャッフル
3. シャッフルされたデータを元の成功/失敗ラベルで分類
4. ラベル情報を保持しながらシャッフルされたデータを返す

### コード例

```python
from train_actor_critic import shuffle_training_data

# データを読み込み
successful_tactics, failed_tactics = load_actor_critic_data("actor_critic_data")

# シャッフル実行（30%のデータを交換）
shuffled_successful, shuffled_failed = shuffle_training_data(
    successful_tactics, failed_tactics, 
    shuffle_ratio=0.3, random_seed=42
)
```

## 注意事項

- 全てのデータが使用されます（シャッフル比率は無視されます）
- ランダムシードを変更すると異なる結果になります
- データの総数はシャッフル後も変わりません
- 元のデータのラベル情報（成功/失敗）が保持されます

## 学習への影響

シャッフル機能により:
- モデルが成功・失敗の境界をより柔軟に学習
- 過学習の防止
- より汎用的な戦略の獲得

シャッフル比率は実験的に調整することを推奨します。
