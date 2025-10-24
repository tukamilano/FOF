#!/usr/bin/env python3
"""
Actor-Criticモデルの初期化された仮のモデルファイルを作成
"""
import os
import sys
import torch
import argparse

# プロジェクトルートをパスに追加
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_root)

from src.core.transformer_classifier import TransformerClassifier
from src.core.actor_critic_model import ActorCriticModel

def create_dummy_actor_critic_model(output_path: str, vocab_size: int = 65, d_model: int = 128):
    """初期化されたActor-Criticモデルを作成"""
    print(f"🏗️  Creating dummy Actor-Critic model...")
    print(f"  Output path: {output_path}")
    print(f"  Vocab size: {vocab_size}")
    print(f"  D model: {d_model}")
    
    # デフォルトのクラス数
    num_main_classes = 59
    num_arg1_classes = 10
    num_arg2_classes = 10
    
    # ベースTransformerを作成
    base_transformer = TransformerClassifier(
        vocab_size=vocab_size,
        pad_id=0,
        max_seq_len=256,
        d_model=d_model,
        nhead=8,
        num_layers=2,
        dim_feedforward=256,
        dropout=0.1,
        num_main_classes=num_main_classes,
        num_arg1_classes=num_arg1_classes,
        num_arg2_classes=num_arg2_classes,
    )
    
    # Actor-Criticモデルを作成
    actor_critic_model = ActorCriticModel(
        base_transformer=base_transformer,
        pretrained_model=base_transformer,  # 同じモデルをpretrained_modelとして使用
        critic_hidden_dim=512
    )
    
    # チェックポイントを作成
    checkpoint = {
        'model_state_dict': actor_critic_model.state_dict(),
        'vocab_size': vocab_size,
        'pad_id': 0,
        'max_seq_len': 256,
        'model_params': {
            'd_model': d_model,
            'nhead': 8,
            'num_layers': 2,
            'dim_feedforward': 256,
            'dropout': 0.1,
        }
    }
    
    # 保存
    torch.save(checkpoint, output_path)
    print(f"✅ Dummy Actor-Critic model saved to: {output_path}")
    
    # モデル情報を表示
    total_params = sum(p.numel() for p in actor_critic_model.parameters())
    trainable_params = sum(p.numel() for p in actor_critic_model.parameters() if p.requires_grad)
    
    print(f"\n📊 Model Information:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Vocab size: {vocab_size}")
    print(f"  D model: {d_model}")
    print(f"  Main classes: {num_main_classes}")
    print(f"  Arg1 classes: {num_arg1_classes}")
    print(f"  Arg2 classes: {num_arg2_classes}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create dummy Actor-Critic model")
    parser.add_argument("--output_path", type=str, default="../models/actor_critic_model.pth", help="Output path for the model")
    parser.add_argument("--vocab_size", type=int, default=65, help="Vocabulary size")
    parser.add_argument("--d_model", type=int, default=128, help="Model dimension")
    
    args = parser.parse_args()
    
    # 出力ディレクトリを作成
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    create_dummy_actor_critic_model(args.output_path, args.vocab_size, args.d_model)
