#!/usr/bin/env python3
"""
包括的データ収集のテストスクリプト
"""
import os
import sys
import torch

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
from src.interaction.self_improvement_data_collector import collect_comprehensive_rl_data


def test_comprehensive_data_collection():
    """包括的データ収集をテスト"""
    print("🧪 Testing comprehensive data collection...")
    
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
    
    # モデルを作成
    model = TransformerClassifier(
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
    
    # 事前学習済みモデルを読み込み
    model_path = "models/pretrained_model.pth"
    if os.path.exists(model_path):
        print(f"🔄 Loading pretrained model from: {model_path}")
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model = model.to(device)
        print("✅ Pretrained model loaded successfully!")
    else:
        print(f"❌ Pretrained model not found: {model_path}")
        return
    
    # 包括的データ収集をテスト
    print("\n📊 Starting comprehensive data collection...")
    successful_tactics, failed_tactics = collect_comprehensive_rl_data(
        model=model,
        tokenizer=tokenizer,
        label_mappings=label_mappings,
        device=device,
        max_seq_len=256,
        num_examples=5,  # テスト用に少なめ
        max_steps=10,    # テスト用に少なめ
        verbose=True,
        generated_data_dir="generated_data",
        temperature=1.0,
        include_failures=True,
        success_reward=1.0,
        step_penalty=0.01,
        failure_penalty=-0.1
    )
    
    print(f"\n📈 Results:")
    print(f"  Successful tactics: {len(successful_tactics)}")
    print(f"  Failed tactics: {len(failed_tactics)}")
    
    # 成功データの例を表示
    if successful_tactics:
        print(f"\n✅ Sample successful tactic:")
        sample = successful_tactics[0]
        print(f"  Step: {sample['step_index']}")
        print(f"  Premises: {sample['premises']}")
        print(f"  Goal: {sample['goal']}")
        print(f"  Tactic: {sample['tactic']}")
        print(f"  Reward: {sample['reward']}")
        print(f"  Log prob: {sample['log_prob']:.4f}")
    
    # 失敗データの例を表示
    if failed_tactics:
        print(f"\n❌ Sample failed tactic:")
        sample = failed_tactics[0]
        print(f"  Step: {sample['step_index']}")
        print(f"  Premises: {sample['premises']}")
        print(f"  Goal: {sample['goal']}")
        print(f"  Tactic: {sample['tactic']}")
        print(f"  Reward: {sample['reward']}")
        print(f"  Log prob: {sample['log_prob']:.4f}")
    
    print("\n🎉 Test completed successfully!")


if __name__ == "__main__":
    test_comprehensive_data_collection()
