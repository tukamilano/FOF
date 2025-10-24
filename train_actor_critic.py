#!/usr/bin/env python3
"""
Actor-Critic学習スクリプト
生成されたデータを使ってActor-Criticモデルを学習
"""
import os
import sys
import json
import torch
import argparse
import numpy as np
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple

# プロジェクトルートをパスに追加
project_root = os.path.dirname(__file__)
sys.path.insert(0, project_root)

from src.core.transformer_classifier import (
    load_tokens_and_labels_from_token_py,
    CharTokenizer,
    TransformerClassifier,
    build_hierarchical_label_mappings,
)
from src.core.parameter import get_model_params, get_hierarchical_labels
from src.core.actor_critic_model import ActorCriticModel
from src.training.actor_critic_trainer import (
    train_actor_critic_epoch,
)


def load_actor_critic_data(data_dir: str = "actor_critic_data") -> Tuple[List[Dict], List[Dict]]:
    """
    Actor-Critic学習用データを読み込み
    複数のファイル（successful_tactics_*.json, failed_tactics_*.json）を読み込む
    
    Args:
        data_dir: データディレクトリ
    
    Returns:
        (successful_tactics, failed_tactics)
    """
    print(f"📁 Loading data from {data_dir}...")
    
    successful_tactics = []
    failed_tactics = []
    
    # 成功データファイルを検索
    success_files = []
    failed_files = []
    
    if os.path.exists(data_dir):
        for filename in os.listdir(data_dir):
            if filename.startswith("successful_tactics_") and filename.endswith(".json"):
                success_files.append(os.path.join(data_dir, filename))
            elif filename.startswith("failed_tactics_") and filename.endswith(".json"):
                failed_files.append(os.path.join(data_dir, filename))
    
    # 成功データを読み込み
    for success_file in sorted(success_files):
        try:
            with open(success_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    successful_tactics.extend(data)
                else:
                    print(f"⚠️  Unexpected data format in {success_file}")
        except Exception as e:
            print(f"⚠️  Error loading {success_file}: {e}")
    
    # 失敗データを読み込み
    for failed_file in sorted(failed_files):
        try:
            with open(failed_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    failed_tactics.extend(data)
                else:
                    print(f"⚠️  Unexpected data format in {failed_file}")
        except Exception as e:
            print(f"⚠️  Error loading {failed_file}: {e}")
    
    print(f"✅ Loaded {len(successful_tactics)} successful tactics from {len(success_files)} files")
    print(f"✅ Loaded {len(failed_tactics)} failed tactics from {len(failed_files)} files")
    
    if not successful_tactics and not failed_tactics:
        print(f"❌ No training data found in {data_dir}")
        print(f"   Expected files: successful_tactics_*.json, failed_tactics_*.json")
    
    return successful_tactics, failed_tactics


def shuffle_training_data(
    successful_tactics: List[Dict],
    failed_tactics: List[Dict],
    shuffle_ratio: float = 0.5,
    random_seed: int = 42
) -> Tuple[List[Dict], List[Dict]]:
    """
    成功データと失敗データをシャッフルして混合する
    全てのデータを使用してシャッフルを実行
    
    Args:
        successful_tactics: 成功タクティクデータ
        failed_tactics: 失敗タクティクデータ
        shuffle_ratio: シャッフルする割合 (0.0-1.0) - 全てのデータを使用する場合は無視
        random_seed: ランダムシード
    
    Returns:
        (shuffled_successful_tactics, shuffled_failed_tactics)
    """
    import random
    
    # ランダムシードを設定
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    print(f"🔀 Shuffling ALL training data")
    print(f"  Original successful tactics: {len(successful_tactics)}")
    print(f"  Original failed tactics: {len(failed_tactics)}")
    
    # 元のデータをコピーしてシャッフル
    shuffled_successful = successful_tactics.copy()
    shuffled_failed = failed_tactics.copy()
    
    # 各リストを個別にシャッフル
    random.shuffle(shuffled_successful)
    random.shuffle(shuffled_failed)
    
    print(f"  Shuffled successful tactics: {len(shuffled_successful)}")
    print(f"  Shuffled failed tactics: {len(shuffled_failed)}")
    print(f"  Used ALL {len(successful_tactics) + len(failed_tactics)} data points for shuffling")
    
    return shuffled_successful, shuffled_failed


def prepare_training_data(
    successful_tactics: List[Dict],
    failed_tactics: List[Dict],
    tokenizer: CharTokenizer,
    label_mappings: Dict[str, Any],
    max_seq_len: int = 256
) -> Tuple[List[Dict], List[Dict]]:
    """
    学習用データを準備
    
    Args:
        successful_tactics: 成功タクティクデータ
        failed_tactics: 失敗タクティクデータ
        tokenizer: トークナイザー
        label_mappings: ラベルマッピング
        max_seq_len: 最大シーケンス長
    
    Returns:
        (prepared_success_data, prepared_failed_data)
    """
    print("🔄 Preparing training data...")
    
    def prepare_single_data(data_list: List[Dict], is_success: bool) -> List[Dict]:
        prepared_data = []
        
        for item in data_list:
            try:
                # データの妥当性チェック
                premises = item.get('premises', [])
                goal = item.get('goal', '')
                tactic = item.get('tactic', {})
                
                # 必須フィールドのチェック（premisesは空でもOK）
                if premises is None or not goal or not tactic:
                    print(f"⚠️  Skipping item with missing data: premises={premises is not None}, goal={bool(goal)}, tactic={bool(tactic)}")
                    continue
                
                # premisesがNoneでないことをチェック（空のリストはOK）
                if premises is not None and not isinstance(premises, list):
                    print(f"⚠️  Skipping item with invalid premises type: {type(premises)}")
                    continue
                
                # goalが空でないことをチェック
                if not isinstance(goal, str) or len(goal.strip()) == 0:
                    print(f"⚠️  Skipping item with empty goal")
                    continue
                
                prepared_item = {
                    'premises': premises,
                    'goal': goal,
                    'tactic': tactic,  # RLDatasetが期待するtacticキー
                    'reward': item.get('reward', 0.0),
                    'log_prob': item.get('log_prob', 0.0),
                    'is_success': is_success,
                    'step_index': item.get('step_index', 0),
                    'state_tactic_hash': item.get('state_tactic_hash', '')
                }
                
                prepared_data.append(prepared_item)
                
            except Exception as e:
                print(f"⚠️  Error preparing data item: {e}")
                continue
        
        return prepared_data
    
    # 成功データを準備
    prepared_success = prepare_single_data(successful_tactics, True)
    print(f"✅ Prepared {len(prepared_success)} successful examples")
    
    # 失敗データを準備
    prepared_failed = prepare_single_data(failed_tactics, False)
    print(f"✅ Prepared {len(prepared_failed)} failed examples")
    
    return prepared_success, prepared_failed


def create_actor_critic_optimizer(model: ActorCriticModel, learning_rate: float):
    """Actor-Critic用のオプティマイザーを作成"""
    # ActorとCriticで別々のオプティマイザーを作成
    actor_optimizer = torch.optim.AdamW(model.shared_encoder.parameters(), lr=learning_rate)
    critic_optimizer = torch.optim.AdamW(model.critic.parameters(), lr=learning_rate)
    
    return actor_optimizer, critic_optimizer


def train_actor_critic(
    pretrained_model_path: str = "models/pretrained_model.pth",
    data_dir: str = "actor_critic_data",
    output_dir: str = "actor_critic_models",
    num_epochs: int = 1,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    kl_weight: float = 0.1,
    entropy_weight: float = 0.01,
    value_weight: float = 0.5,
    gamma: float = 0.99,
    lambda_gae: float = 0.95,
    device: str = "auto",
    use_wandb: bool = False,
    wandb_project: str = "fof-actor-critic",
    wandb_run_name: str = None,
    shuffle_ratio: float = 0.5,
    random_seed: int = 42
):
    """
    Actor-Criticモデルを学習
    
    Args:
        pretrained_model_path: 事前学習済みモデルのパス
        data_dir: データディレクトリ
        output_dir: 出力ディレクトリ
        num_epochs: エポック数
        batch_size: バッチサイズ
        learning_rate: 学習率
        kl_weight: KL制約の重み
        entropy_weight: エントロピー正則化の重み
        value_weight: 価値関数の重み
        gamma: 割引率
        lambda_gae: GAEのλパラメータ
        device: デバイス
        shuffle_ratio: シャッフルする割合
        random_seed: ランダムシード
    """
    print("🚀 Starting Actor-Critic training...")
    
    # デバイス設定
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    print(f"Using device: {device}")
    
    # 出力ディレクトリを作成
    os.makedirs(output_dir, exist_ok=True)
    
    # wandb初期化
    if use_wandb:
        try:
            import wandb
            run_name = wandb_run_name or f"actor_critic_{int(time.time())}"
            wandb.init(
                project=wandb_project,
                name=run_name,
                config={
                    "pretrained_model_path": pretrained_model_path,
                    "data_dir": data_dir,
                    "output_dir": output_dir,
                    "num_epochs": num_epochs,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "kl_weight": kl_weight,
                    "entropy_weight": entropy_weight,
                    "value_weight": value_weight,
                    "gamma": gamma,
                    "lambda_gae": lambda_gae,
                    "device": str(device),
                    "shuffle_ratio": shuffle_ratio,
                    "random_seed": random_seed
                }
            )
            print(f"✅ Wandb initialized: {wandb_project}/{run_name}")
        except ImportError:
            print("⚠️  Wandb not available. Continuing without logging.")
            use_wandb = False
    
    # パラメータを初期化
    model_params = get_model_params()
    hierarchical_labels = get_hierarchical_labels()
    
    # トークンとラベルを読み込み
    token_py_path = os.path.join(project_root, "src", "core", "fof_tokens.py")
    base_tokens, _ = load_tokens_and_labels_from_token_py(token_py_path)
    
    # 階層分類用のラベルマッピングを構築
    main_to_id, arg1_to_id, arg2_to_id, id_to_main, id_to_arg1, id_to_arg2 = build_hierarchical_label_mappings(
        hierarchical_labels.main_tactics,
        hierarchical_labels.arg1_values,
        hierarchical_labels.arg2_values
    )
    
    label_mappings = {
        'main_to_id': main_to_id,
        'arg1_to_id': arg1_to_id,
        'arg2_to_id': arg2_to_id,
        'id_to_main': id_to_main,
        'id_to_arg1': id_to_arg1,
        'id_to_arg2': id_to_arg2
    }
    
    # トークナイザーを作成
    tokenizer = CharTokenizer(
        base_tokens=base_tokens,
        add_tactic_tokens=model_params.add_tactic_tokens,
        num_tactic_tokens=model_params.num_tactic_tokens
    )
    
    # 事前学習済みモデルを読み込み
    print("🔄 Loading pretrained model...")
    pretrained_model = TransformerClassifier(
        vocab_size=tokenizer.vocab_size,
        pad_id=tokenizer.pad_id,
        max_seq_len=256,
        d_model=model_params.d_model,
        nhead=model_params.nhead,
        num_layers=model_params.num_layers,
        dim_feedforward=model_params.dim_feedforward,
        dropout=model_params.dropout,
        num_main_classes=len(id_to_main),
        num_arg1_classes=len(id_to_arg1),
        num_arg2_classes=len(id_to_arg2),
    )
    
    if os.path.exists(pretrained_model_path):
        state_dict = torch.load(pretrained_model_path, map_location=device)
        pretrained_model.load_state_dict(state_dict)
        pretrained_model = pretrained_model.to(device)
        print("✅ Pretrained model loaded successfully!")
    else:
        print(f"❌ Pretrained model not found: {pretrained_model_path}")
        return
    
    # Actor-Criticモデルを作成
    print("🔄 Creating Actor-Critic model...")
    actor_critic_model = ActorCriticModel(
        base_transformer=pretrained_model,
        pretrained_model=pretrained_model,
        critic_hidden_dim=512
    ).to(device)
    
    # データを読み込み
    successful_tactics, failed_tactics = load_actor_critic_data(data_dir)
    
    if not successful_tactics and not failed_tactics:
        print("❌ No training data available. Exiting.")
        return
    
    # データをシャッフル
    if shuffle_ratio > 0.0:
        successful_tactics, failed_tactics = shuffle_training_data(
            successful_tactics, failed_tactics, shuffle_ratio, random_seed
        )
    
    # データを準備
    prepared_success, prepared_failed = prepare_training_data(
        successful_tactics, failed_tactics, tokenizer, label_mappings
    )
    
    # 全データを結合
    all_data = prepared_success + prepared_failed
    print(f"📊 Total training examples: {len(all_data)}")
    
    # オプティマイザーを作成
    actor_optimizer, critic_optimizer = create_actor_critic_optimizer(actor_critic_model, learning_rate)
    
    # 学習履歴を記録
    training_history = {
        'epoch': [],
        'actor_loss': [],
        'critic_loss': [],
        'total_loss': [],
        'kl_loss': [],
        'entropy_loss': [],
        'value_loss': [],
        'success_rate': []
    }
    
    print(f"\n🎯 Starting training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        print(f"\n📚 Epoch {epoch + 1}/{num_epochs}")
        
        # データセットとデータローダーを作成
        from src.training.actor_critic_trainer import RLDataset
        from torch.utils.data import DataLoader
        
        dataset = RLDataset(prepared_success, prepared_failed, tokenizer, 256)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,  # マルチプロセシングを無効化してデバッグ
            pin_memory=True if device.type == 'cuda' else False
        )
        
        # 1エポックの学習を実行
        epoch_losses = train_actor_critic_epoch(
            model=actor_critic_model,
            dataloader=dataloader,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
            device=device,
            kl_penalty_weight=kl_weight,
            entropy_weight=entropy_weight,
            ppo_epochs=2,
            clip_ratio=0.2,
            value_coef=value_weight,
            gamma=gamma,
            lam=lambda_gae,
            use_amp=False,
            scaler=None,
            use_wandb=use_wandb,
            epoch=epoch,
            log_frequency=100
        )
        
        # 統計を記録
        training_history['epoch'].append(epoch + 1)
        training_history['actor_loss'].append(epoch_losses['actor_loss'])
        training_history['critic_loss'].append(epoch_losses['critic_loss'])
        training_history['total_loss'].append(epoch_losses['total_loss'])
        training_history['kl_loss'].append(epoch_losses['kl_loss'])
        training_history['entropy_loss'].append(epoch_losses['entropy_loss'])
        training_history['value_loss'].append(epoch_losses.get('value_loss', 0.0))
        training_history['success_rate'].append(epoch_losses.get('success_rate', 0.0))
        
        print(f"📈 Epoch {epoch + 1} Results:")
        print(f"  Actor Loss: {epoch_losses['actor_loss']:.4f}")
        print(f"  Critic Loss: {epoch_losses['critic_loss']:.4f}")
        print(f"  Total Loss: {epoch_losses['total_loss']:.4f}")
        print(f"  KL Loss: {epoch_losses['kl_loss']:.4f}")
        print(f"  Entropy Loss: {epoch_losses['entropy_loss']:.4f}")
        print(f"  Value Loss: {epoch_losses.get('value_loss', 0.0):.4f}")
        print(f"  Success Rate: {epoch_losses.get('success_rate', 0.0):.4f}")
        
        # wandbにログを送信
        if use_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "actor_loss": epoch_losses['actor_loss'],
                "critic_loss": epoch_losses['critic_loss'],
                "total_loss": epoch_losses['total_loss'],
                "kl_loss": epoch_losses['kl_loss'],
                "entropy_loss": epoch_losses['entropy_loss'],
                "value_loss": epoch_losses.get('value_loss', 0.0),
                "success_rate": epoch_losses.get('success_rate', 0.0)
            })
        
        # エポックごとのモデル保存は無効化（最終モデルのみ保存）
    
    # 最終モデルを保存
    final_model_path = os.path.join(output_dir, "actor_critic_final.pth")
    torch.save(actor_critic_model.state_dict(), final_model_path)
    print(f"💾 Final model saved to: {final_model_path}")
    
    # 学習履歴を保存
    history_path = os.path.join(output_dir, "training_history.json")
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump(training_history, f, ensure_ascii=False, indent=2)
    print(f"📊 Training history saved to: {history_path}")
    
    # 学習曲線はwandbで記録されるため、ローカルプロットは不要
    
    # wandbを終了
    if use_wandb:
        wandb.finish()
    
    print(f"\n🎉 Actor-Critic training completed!")
    print(f"📁 Output directory: {output_dir}")




def main():
    parser = argparse.ArgumentParser(description="Train Actor-Critic model")
    parser.add_argument("--pretrained_model", type=str, default="models/pretrained_model.pth", help="pretrained model path")
    parser.add_argument("--data_dir", type=str, default="actor_critic_data", help="data directory")
    parser.add_argument("--output_dir", type=str, default="actor_critic_models", help="output directory")
    parser.add_argument("--num_epochs", type=int, default=1, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--kl_weight", type=float, default=0.1, help="KL constraint weight")
    parser.add_argument("--entropy_weight", type=float, default=0.01, help="entropy regularization weight")
    parser.add_argument("--value_weight", type=float, default=0.5, help="value function weight")
    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
    parser.add_argument("--lambda_gae", type=float, default=0.95, help="GAE lambda parameter")
    parser.add_argument("--device", type=str, default="auto", help="device (auto/cuda/cpu)")
    parser.add_argument("--use_wandb", action="store_true", help="use wandb for logging")
    parser.add_argument("--wandb_project", type=str, default="fof-actor-critic", help="wandb project name")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="wandb run name")
    parser.add_argument("--shuffle_ratio", type=float, default=0.5, help="shuffle ratio for training data (0.0-1.0)")
    parser.add_argument("--random_seed", type=int, default=42, help="random seed for shuffling")
    
    args = parser.parse_args()
    
    train_actor_critic(
        pretrained_model_path=args.pretrained_model,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        kl_weight=args.kl_weight,
        entropy_weight=args.entropy_weight,
        value_weight=args.value_weight,
        gamma=args.gamma,
        lambda_gae=args.lambda_gae,
        device=args.device,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        shuffle_ratio=args.shuffle_ratio,
        random_seed=args.random_seed
    )


if __name__ == "__main__":
    main()