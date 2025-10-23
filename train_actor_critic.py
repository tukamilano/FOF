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
from pathlib import Path
from typing import Dict, List, Any, Tuple
import matplotlib.pyplot as plt

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
    compute_gae_advantages,
    compute_actor_critic_loss,
    create_actor_critic_optimizer,
)


def load_actor_critic_data(data_dir: str = "actor_critic_data") -> Tuple[List[Dict], List[Dict]]:
    """
    Actor-Critic学習用データを読み込み
    
    Args:
        data_dir: データディレクトリ
    
    Returns:
        (successful_tactics, failed_tactics)
    """
    success_file = os.path.join(data_dir, "successful_tactics.json")
    failed_file = os.path.join(data_dir, "failed_tactics.json")
    
    print(f"📁 Loading data from {data_dir}...")
    
    # 成功データを読み込み
    if os.path.exists(success_file):
        with open(success_file, 'r', encoding='utf-8') as f:
            successful_tactics = json.load(f)
        print(f"✅ Loaded {len(successful_tactics)} successful tactics")
    else:
        print(f"❌ Success file not found: {success_file}")
        successful_tactics = []
    
    # 失敗データを読み込み
    if os.path.exists(failed_file):
        with open(failed_file, 'r', encoding='utf-8') as f:
            failed_tactics = json.load(f)
        print(f"✅ Loaded {len(failed_tactics)} failed tactics")
    else:
        print(f"❌ Failed file not found: {failed_file}")
        failed_tactics = []
    
    return successful_tactics, failed_tactics


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
                # 状態をエンコード
                state_text = f"Premises: {item['premises']}\nGoal: {item['goal']}"
                state_tokens = tokenizer.encode(state_text, max_length=max_seq_len)
                
                # タクティクをエンコード
                tactic = item['tactic']
                main_id = label_mappings['main_to_id'].get(tactic['main'], 0)
                arg1_id = label_mappings['arg1_to_id'].get(tactic['arg1'], 0) if tactic['arg1'] else 0
                arg2_id = label_mappings['arg2_to_id'].get(tactic['arg2'], 0) if tactic['arg2'] else 0
                
                prepared_item = {
                    'state_tokens': state_tokens,
                    'main_tactic_id': main_id,
                    'arg1_id': arg1_id,
                    'arg2_id': arg2_id,
                    'reward': item['reward'],
                    'log_prob': item['log_prob'],
                    'is_success': is_success,
                    'step_index': item['step_index'],
                    'state_tactic_hash': item['state_tactic_hash']
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


def train_actor_critic(
    pretrained_model_path: str = "models/pretrained_model.pth",
    data_dir: str = "actor_critic_data",
    output_dir: str = "actor_critic_models",
    num_epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    kl_weight: float = 0.1,
    entropy_weight: float = 0.01,
    value_weight: float = 0.5,
    gamma: float = 0.99,
    lambda_gae: float = 0.95,
    device: str = "auto",
    save_every: int = 2
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
        save_every: モデル保存間隔
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
        shared_encoder=pretrained_model,
        num_main_classes=len(id_to_main),
        num_arg1_classes=len(id_to_arg1),
        num_arg2_classes=len(id_to_arg2),
        critic_hidden_dim=512
    ).to(device)
    
    # データを読み込み
    successful_tactics, failed_tactics = load_actor_critic_data(data_dir)
    
    if not successful_tactics and not failed_tactics:
        print("❌ No training data available. Exiting.")
        return
    
    # データを準備
    prepared_success, prepared_failed = prepare_training_data(
        successful_tactics, failed_tactics, tokenizer, label_mappings
    )
    
    # 全データを結合
    all_data = prepared_success + prepared_failed
    print(f"📊 Total training examples: {len(all_data)}")
    
    # オプティマイザーを作成
    optimizer = create_actor_critic_optimizer(actor_critic_model, learning_rate)
    
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
        
        # エポック学習を実行
        epoch_stats = train_actor_critic_epoch(
            model=actor_critic_model,
            pretrained_model=pretrained_model,
            data=all_data,
            optimizer=optimizer,
            batch_size=batch_size,
            kl_weight=kl_weight,
            entropy_weight=entropy_weight,
            value_weight=value_weight,
            gamma=gamma,
            lambda_gae=lambda_gae,
            device=device,
            verbose=True
        )
        
        # 統計を記録
        training_history['epoch'].append(epoch + 1)
        training_history['actor_loss'].append(epoch_stats['actor_loss'])
        training_history['critic_loss'].append(epoch_stats['critic_loss'])
        training_history['total_loss'].append(epoch_stats['total_loss'])
        training_history['kl_loss'].append(epoch_stats['kl_loss'])
        training_history['entropy_loss'].append(epoch_stats['entropy_loss'])
        training_history['value_loss'].append(epoch_stats['value_loss'])
        training_history['success_rate'].append(epoch_stats['success_rate'])
        
        print(f"📈 Epoch {epoch + 1} Results:")
        print(f"  Actor Loss: {epoch_stats['actor_loss']:.4f}")
        print(f"  Critic Loss: {epoch_stats['critic_loss']:.4f}")
        print(f"  Total Loss: {epoch_stats['total_loss']:.4f}")
        print(f"  KL Loss: {epoch_stats['kl_loss']:.4f}")
        print(f"  Entropy Loss: {epoch_stats['entropy_loss']:.4f}")
        print(f"  Value Loss: {epoch_stats['value_loss']:.4f}")
        print(f"  Success Rate: {epoch_stats['success_rate']:.4f}")
        
        # モデルを保存
        if (epoch + 1) % save_every == 0 or epoch == num_epochs - 1:
            model_path = os.path.join(output_dir, f"actor_critic_epoch_{epoch + 1}.pth")
            torch.save(actor_critic_model.state_dict(), model_path)
            print(f"💾 Model saved to: {model_path}")
    
    # 最終モデルを保存
    final_model_path = os.path.join(output_dir, "actor_critic_final.pth")
    torch.save(actor_critic_model.state_dict(), final_model_path)
    print(f"💾 Final model saved to: {final_model_path}")
    
    # 学習履歴を保存
    history_path = os.path.join(output_dir, "training_history.json")
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump(training_history, f, ensure_ascii=False, indent=2)
    print(f"📊 Training history saved to: {history_path}")
    
    # 学習曲線をプロット
    plot_training_curves(training_history, output_dir)
    
    print(f"\n🎉 Actor-Critic training completed!")
    print(f"📁 Output directory: {output_dir}")


def plot_training_curves(history: Dict, output_dir: str):
    """
    学習曲線をプロットして保存
    
    Args:
        history: 学習履歴
        output_dir: 出力ディレクトリ
    """
    try:
        plt.figure(figsize=(15, 10))
        
        # 損失曲線
        plt.subplot(2, 3, 1)
        plt.plot(history['epoch'], history['actor_loss'], label='Actor Loss', marker='o')
        plt.plot(history['epoch'], history['critic_loss'], label='Critic Loss', marker='s')
        plt.plot(history['epoch'], history['total_loss'], label='Total Loss', marker='^')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Losses')
        plt.legend()
        plt.grid(True)
        
        # KL損失
        plt.subplot(2, 3, 2)
        plt.plot(history['epoch'], history['kl_loss'], label='KL Loss', marker='o', color='red')
        plt.xlabel('Epoch')
        plt.ylabel('KL Loss')
        plt.title('KL Constraint Loss')
        plt.legend()
        plt.grid(True)
        
        # エントロピー損失
        plt.subplot(2, 3, 3)
        plt.plot(history['epoch'], history['entropy_loss'], label='Entropy Loss', marker='o', color='green')
        plt.xlabel('Epoch')
        plt.ylabel('Entropy Loss')
        plt.title('Entropy Regularization Loss')
        plt.legend()
        plt.grid(True)
        
        # 価値関数損失
        plt.subplot(2, 3, 4)
        plt.plot(history['epoch'], history['value_loss'], label='Value Loss', marker='o', color='purple')
        plt.xlabel('Epoch')
        plt.ylabel('Value Loss')
        plt.title('Value Function Loss')
        plt.legend()
        plt.grid(True)
        
        # 成功率
        plt.subplot(2, 3, 5)
        plt.plot(history['epoch'], history['success_rate'], label='Success Rate', marker='o', color='orange')
        plt.xlabel('Epoch')
        plt.ylabel('Success Rate')
        plt.title('Training Success Rate')
        plt.legend()
        plt.grid(True)
        
        # 総合損失
        plt.subplot(2, 3, 6)
        plt.plot(history['epoch'], history['total_loss'], label='Total Loss', marker='o', color='blue')
        plt.xlabel('Epoch')
        plt.ylabel('Total Loss')
        plt.title('Overall Training Progress')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        
        # プロットを保存
        plot_path = os.path.join(output_dir, "training_curves.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"📊 Training curves saved to: {plot_path}")
        
        plt.close()
        
    except Exception as e:
        print(f"⚠️  Error plotting training curves: {e}")


def main():
    parser = argparse.ArgumentParser(description="Train Actor-Critic model")
    parser.add_argument("--pretrained_model", type=str, default="models/pretrained_model.pth", help="pretrained model path")
    parser.add_argument("--data_dir", type=str, default="actor_critic_data", help="data directory")
    parser.add_argument("--output_dir", type=str, default="actor_critic_models", help="output directory")
    parser.add_argument("--num_epochs", type=int, default=10, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--kl_weight", type=float, default=0.1, help="KL constraint weight")
    parser.add_argument("--entropy_weight", type=float, default=0.01, help="entropy regularization weight")
    parser.add_argument("--value_weight", type=float, default=0.5, help="value function weight")
    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
    parser.add_argument("--lambda_gae", type=float, default=0.95, help="GAE lambda parameter")
    parser.add_argument("--device", type=str, default="auto", help="device (auto/cuda/cpu)")
    parser.add_argument("--save_every", type=int, default=2, help="save model every N epochs")
    
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
        save_every=args.save_every
    )


if __name__ == "__main__":
    main()

