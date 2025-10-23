#!/usr/bin/env python3
"""
Actor-Critic使用例
"""
import os
import sys
import torch

# プロジェクトルートをパスに追加
project_root = os.path.dirname(__file__)
sys.path.insert(0, project_root)

from src.core.actor_critic_model import create_actor_critic_model
from src.core.transformer_classifier import (
    load_tokens_and_labels_from_token_py,
    CharTokenizer,
    build_hierarchical_label_mappings,
)
from src.core.parameter import get_model_params, get_hierarchical_labels


def example_actor_critic_usage():
    """Actor-Criticモデルの使用例"""
    print("📚 Actor-Critic Model Usage Example")
    
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
    
    # トークナイザーを作成
    tokenizer = CharTokenizer(
        base_tokens=base_tokens,
        add_tactic_tokens=model_params.add_tactic_tokens,
        num_tactic_tokens=model_params.num_tactic_tokens
    )
    
    # Actor-Criticモデルを作成
    print("🔄 Creating Actor-Critic model...")
    model = create_actor_critic_model(
        pretrained_model_path="models/pretrained_model.pth",
        vocab_size=tokenizer.vocab_size,
        pad_id=tokenizer.pad_id,
        max_seq_len=256,
        num_main_classes=len(id_to_main),
        num_arg1_classes=len(id_to_arg1),
        num_arg2_classes=len(id_to_arg2),
        d_model=model_params.d_model,
        nhead=model_params.nhead,
        num_layers=model_params.num_layers,
        dim_feedforward=model_params.dim_feedforward,
        dropout=model_params.dropout,
        critic_hidden_dim=128,
        device=device
    )
    
    print("✅ Actor-Critic model created successfully!")
    
    # モデル情報を表示
    print(f"\n📊 Model Information:")
    print(f"  Vocab size: {model.vocab_size}")
    print(f"  Max sequence length: {model.max_seq_len}")
    print(f"  Main tactic classes: {model.num_main_classes}")
    print(f"  Arg1 classes: {model.num_arg1_classes}")
    print(f"  Arg2 classes: {model.num_arg2_classes}")
    
    # テスト用の入力を作成
    test_cases = [
        ("p -> p", []),  # トートロジー
        ("p & q -> p", []),  # 論理積
        ("p -> p | q", []),  # 論理和
    ]
    
    print(f"\n🔍 Testing inference on {len(test_cases)} test cases:")
    
    for i, (goal, premises) in enumerate(test_cases):
        print(f"\nTest case {i+1}: {goal}")
        print(f"  Premises: {premises}")
        
        # 入力をエンコード
        input_ids, attention_mask, segment_ids = tokenizer.encode(
            goal, premises, 256
        )
        input_ids = input_ids.unsqueeze(0).to(device)
        attention_mask = attention_mask.unsqueeze(0).to(device)
        segment_ids = segment_ids.unsqueeze(0).to(device)
        
        # 推論モードに設定
        model.eval()
        
        with torch.no_grad():
            # アクション選択
            action_dict, log_prob, value = model.select_action(
                input_ids, attention_mask, segment_ids, temperature=1.0
            )
            
            # 価値関数の出力も取得
            main_logits, arg1_logits, arg2_logits, critic_value, _ = model(
                input_ids, attention_mask, segment_ids, return_pretrained_logits=False
            )
            
            print(f"  Selected action: {action_dict}")
            print(f"  Log probability: {log_prob:.4f}")
            print(f"  Value estimate: {value:.4f}")
            print(f"  Critic value: {critic_value[0].item():.4f}")
            
            # アクションを人間が読める形式に変換
            main_tactic = id_to_main[action_dict['main']]
            arg1_value = id_to_arg1[action_dict['arg1']] if action_dict['arg1'] != 0 else None
            arg2_value = id_to_arg2[action_dict['arg2']] if action_dict['arg2'] != 0 else None
            
            tactic_string = main_tactic
            if arg1_value:
                tactic_string += f" {arg1_value}"
            if arg2_value:
                tactic_string += f" {arg2_value}"
            
            print(f"  Tactic string: {tactic_string}")
    
    # KL制約のテスト
    print(f"\n🔒 Testing KL constraint...")
    model.eval()
    
    with torch.no_grad():
        # テスト用の入力でKL divergenceを計算
        input_ids, attention_mask, segment_ids = tokenizer.encode(
            "p -> p", [], 256
        )
        input_ids = input_ids.unsqueeze(0).to(device)
        attention_mask = attention_mask.unsqueeze(0).to(device)
        segment_ids = segment_ids.unsqueeze(0).to(device)
        
        # Actorとpretrained modelの両方のlogitsを取得
        main_logits, arg1_logits, arg2_logits, critic_value, pretrained_logits = model(
            input_ids, attention_mask, segment_ids, return_pretrained_logits=True
        )
        
        if pretrained_logits is not None:
            kl_loss = model.compute_kl_divergence(
                (main_logits, arg1_logits, arg2_logits),
                pretrained_logits
            )
            print(f"  KL divergence: {kl_loss.item():.4f}")
        else:
            print("  Pretrained logits not available")
    
    print(f"\n🎉 Actor-Critic usage example completed!")


if __name__ == "__main__":
    example_actor_critic_usage()
