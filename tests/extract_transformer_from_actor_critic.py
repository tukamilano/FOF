#!/usr/bin/env python3
"""
Actor-CriticモデルからTransformerClassifierを抽出して評価用に保存
"""
import os
import sys
import torch
import argparse
from pathlib import Path

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


def extract_transformer_from_actor_critic(
    actor_critic_path: str,
    output_path: str,
    pretrained_model_path: str = "models/pretrained_model.pth"
):
    """
    Actor-CriticモデルからTransformerClassifierを抽出
    
    Args:
        actor_critic_path: Actor-Criticモデルのパス
        output_path: 出力パス
        pretrained_model_path: 事前学習済みモデルのパス（ラベルマッピング用）
    """
    print(f"🔄 Extracting TransformerClassifier from Actor-Critic model...")
    print(f"  Input: {actor_critic_path}")
    print(f"  Output: {output_path}")
    
    # デバイス設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
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
    
    # 事前学習済みモデルを読み込み（ラベルマッピング用）
    pretrained_model = TransformerClassifier(
        vocab_size=len(base_tokens),
        pad_id=0,
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
        print("✅ Pretrained model loaded for reference")
    
    # Actor-Criticモデルを作成
    actor_critic_model = ActorCriticModel(
        base_transformer=pretrained_model,
        pretrained_model=pretrained_model,
        critic_hidden_dim=512
    ).to(device)
    
    # Actor-Criticモデルの重みを読み込み
    if os.path.exists(actor_critic_path):
        actor_critic_state_dict = torch.load(actor_critic_path, map_location=device)
        actor_critic_model.load_state_dict(actor_critic_state_dict)
        print("✅ Actor-Critic model loaded")
    else:
        print(f"❌ Actor-Critic model not found: {actor_critic_path}")
        return
    
    # TransformerClassifierを抽出（shared_encoderの重みを使用）
    transformer_model = TransformerClassifier(
        vocab_size=len(base_tokens),
        pad_id=0,
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
    
    # Actor-Criticのshared_encoderの重みをTransformerClassifierにコピー
    transformer_model.load_state_dict(actor_critic_model.shared_encoder.state_dict())
    transformer_model = transformer_model.to(device)
    
    # 評価用のチェックポイントを作成
    checkpoint = {
        'model_params': {
            'vocab_size': len(base_tokens),
            'pad_id': 0,
            'max_seq_len': 256,
            'd_model': model_params.d_model,
            'nhead': model_params.nhead,
            'num_layers': model_params.num_layers,
            'dim_feedforward': model_params.dim_feedforward,
            'dropout': model_params.dropout,
        },
        'vocab_size': len(base_tokens),
        'pad_id': 0,
        'max_seq_len': 256,
        'main_to_id': main_to_id,
        'arg1_to_id': arg1_to_id,
        'arg2_to_id': arg2_to_id,
        'id_to_main': id_to_main,
        'id_to_arg1': id_to_arg1,
        'id_to_arg2': id_to_arg2,
    }
    
    # モデルの重みを追加
    checkpoint.update(transformer_model.state_dict())
    
    # チェックポイントを保存
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(checkpoint, output_path)
    
    print(f"✅ TransformerClassifier extracted and saved to: {output_path}")
    print(f"📊 Model info:")
    print(f"  Vocab size: {len(base_tokens)}")
    print(f"  Main tactics: {len(id_to_main)}")
    print(f"  Arg1 values: {len(id_to_arg1)}")
    print(f"  Arg2 values: {len(id_to_arg2)}")


def main():
    parser = argparse.ArgumentParser(description="Extract TransformerClassifier from Actor-Critic model")
    parser.add_argument("--actor_critic_path", type=str, required=True, help="Actor-Critic model path")
    parser.add_argument("--output_path", type=str, required=True, help="Output path for TransformerClassifier")
    parser.add_argument("--pretrained_model", type=str, default="models/pretrained_model.pth", help="Pretrained model path")
    
    args = parser.parse_args()
    
    extract_transformer_from_actor_critic(
        actor_critic_path=args.actor_critic_path,
        output_path=args.output_path,
        pretrained_model_path=args.pretrained_model
    )


if __name__ == "__main__":
    main()
"""
Actor-CriticモデルからTransformerClassifierを抽出して評価用に保存
"""
import os
import sys
import torch
import argparse
from pathlib import Path

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


def extract_transformer_from_actor_critic(
    actor_critic_path: str,
    output_path: str,
    pretrained_model_path: str = "models/pretrained_model.pth"
):
    """
    Actor-CriticモデルからTransformerClassifierを抽出
    
    Args:
        actor_critic_path: Actor-Criticモデルのパス
        output_path: 出力パス
        pretrained_model_path: 事前学習済みモデルのパス（ラベルマッピング用）
    """
    print(f"🔄 Extracting TransformerClassifier from Actor-Critic model...")
    print(f"  Input: {actor_critic_path}")
    print(f"  Output: {output_path}")
    
    # デバイス設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
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
    
    # 事前学習済みモデルを読み込み（ラベルマッピング用）
    pretrained_model = TransformerClassifier(
        vocab_size=len(base_tokens),
        pad_id=0,
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
        print("✅ Pretrained model loaded for reference")
    
    # Actor-Criticモデルを作成
    actor_critic_model = ActorCriticModel(
        base_transformer=pretrained_model,
        pretrained_model=pretrained_model,
        critic_hidden_dim=512
    ).to(device)
    
    # Actor-Criticモデルの重みを読み込み
    if os.path.exists(actor_critic_path):
        actor_critic_state_dict = torch.load(actor_critic_path, map_location=device)
        actor_critic_model.load_state_dict(actor_critic_state_dict)
        print("✅ Actor-Critic model loaded")
    else:
        print(f"❌ Actor-Critic model not found: {actor_critic_path}")
        return
    
    # TransformerClassifierを抽出（shared_encoderの重みを使用）
    transformer_model = TransformerClassifier(
        vocab_size=len(base_tokens),
        pad_id=0,
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
    
    # Actor-Criticのshared_encoderの重みをTransformerClassifierにコピー
    transformer_model.load_state_dict(actor_critic_model.shared_encoder.state_dict())
    transformer_model = transformer_model.to(device)
    
    # 評価用のチェックポイントを作成
    checkpoint = {
        'model_params': {
            'vocab_size': len(base_tokens),
            'pad_id': 0,
            'max_seq_len': 256,
            'd_model': model_params.d_model,
            'nhead': model_params.nhead,
            'num_layers': model_params.num_layers,
            'dim_feedforward': model_params.dim_feedforward,
            'dropout': model_params.dropout,
        },
        'vocab_size': len(base_tokens),
        'pad_id': 0,
        'max_seq_len': 256,
        'main_to_id': main_to_id,
        'arg1_to_id': arg1_to_id,
        'arg2_to_id': arg2_to_id,
        'id_to_main': id_to_main,
        'id_to_arg1': id_to_arg1,
        'id_to_arg2': id_to_arg2,
    }
    
    # モデルの重みを追加
    checkpoint.update(transformer_model.state_dict())
    
    # チェックポイントを保存
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(checkpoint, output_path)
    
    print(f"✅ TransformerClassifier extracted and saved to: {output_path}")
    print(f"📊 Model info:")
    print(f"  Vocab size: {len(base_tokens)}")
    print(f"  Main tactics: {len(id_to_main)}")
    print(f"  Arg1 values: {len(id_to_arg1)}")
    print(f"  Arg2 values: {len(id_to_arg2)}")


def main():
    parser = argparse.ArgumentParser(description="Extract TransformerClassifier from Actor-Critic model")
    parser.add_argument("--actor_critic_path", type=str, required=True, help="Actor-Critic model path")
    parser.add_argument("--output_path", type=str, required=True, help="Output path for TransformerClassifier")
    parser.add_argument("--pretrained_model", type=str, default="models/pretrained_model.pth", help="Pretrained model path")
    
    args = parser.parse_args()
    
    extract_transformer_from_actor_critic(
        actor_critic_path=args.actor_critic_path,
        output_path=args.output_path,
        pretrained_model_path=args.pretrained_model
    )


if __name__ == "__main__":
    main()
