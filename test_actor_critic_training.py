#!/usr/bin/env python3
"""
Actor-Critic学習のテストスクリプト
"""
import os
import sys
import torch
import json

# プロジェクトルートをパスに追加
project_root = os.path.dirname(__file__)
sys.path.insert(0, project_root)

from src.core.transformer_classifier import (
    load_tokens_and_labels_from_token_py,
    CharTokenizer,
    build_hierarchical_label_mappings,
)
from src.core.parameter import get_model_params, get_hierarchical_labels
from src.core.actor_critic_model import create_actor_critic_model
from src.training.actor_critic_trainer import train_actor_critic
from src.interaction.self_improvement_data_collector import collect_comprehensive_rl_data


def test_actor_critic_training():
    """Actor-Critic学習をテスト"""
    print("🧪 Testing Actor-Critic training...")
    
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
    
    # データ収集（テスト用に少なめ）
    print("\n📊 Collecting training data...")
    successful_tactics, failed_tactics = collect_comprehensive_rl_data(
        model=model.shared_encoder,  # 共有エンコーダーを使用
        tokenizer=tokenizer,
        label_mappings=label_mappings,
        device=device,
        max_seq_len=256,
        num_examples=10,  # テスト用に少なめ
        max_steps=5,      # テスト用に少なめ
        verbose=True,
        generated_data_dir="generated_data",
        temperature=1.0,
        include_failures=True,
        success_reward=1.0,
        step_penalty=0.01,
        failure_penalty=-0.1
    )
    
    print(f"📈 Data collection completed:")
    print(f"  Successful tactics: {len(successful_tactics)}")
    print(f"  Failed tactics: {len(failed_tactics)}")
    
    if len(successful_tactics) == 0 and len(failed_tactics) == 0:
        print("❌ No training data collected. Please check your data directory.")
        return
    
    # Actor-Critic学習を実行
    print("\n🚀 Starting Actor-Critic training...")
    history = train_actor_critic(
        model=model,
        successful_tactics=successful_tactics,
        failed_tactics=failed_tactics,
        tokenizer=tokenizer,
        device=device,
        num_epochs=3,  # テスト用に少なめ
        batch_size=8,  # テスト用に少なめ
        learning_rate=1e-4,
        kl_penalty_weight=0.1,
        entropy_weight=0.01,
        ppo_epochs=2,
        clip_ratio=0.2,
        value_coef=0.5,
        gamma=0.99,
        lam=0.95,
        use_amp=False,  # テスト用に無効
        use_wandb=False,
        max_seq_len=256
    )
    
    print("\n📊 Training completed!")
    print("Final losses:")
    for key, values in history.items():
        if values:
            print(f"  {key}: {values[-1]:.4f}")
    
    # モデルを保存
    model_path = "models/actor_critic_test.pth"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"💾 Model saved to: {model_path}")
    
    # 学習履歴を保存
    history_path = "logs/actor_critic_training_history.json"
    os.makedirs(os.path.dirname(history_path), exist_ok=True)
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"📈 Training history saved to: {history_path}")
    
    # 簡単な推論テスト
    print("\n🔍 Testing inference...")
    model.eval()
    
    # テスト用の入力を作成
    test_goal = "p -> p"
    test_premises = []
    input_ids, attention_mask, segment_ids = tokenizer.encode(
        test_goal, test_premises, 256
    )
    input_ids = input_ids.unsqueeze(0).to(device)
    attention_mask = attention_mask.unsqueeze(0).to(device)
    segment_ids = segment_ids.unsqueeze(0).to(device)
    
    # アクション選択
    action_dict, log_prob, value = model.select_action(
        input_ids, attention_mask, segment_ids, temperature=1.0
    )
    
    print(f"Test inference result:")
    print(f"  Goal: {test_goal}")
    print(f"  Selected action: {action_dict}")
    print(f"  Log probability: {log_prob:.4f}")
    print(f"  Value: {value:.4f}")
    
    print("\n🎉 Actor-Critic training test completed successfully!")


if __name__ == "__main__":
    test_actor_critic_training()
