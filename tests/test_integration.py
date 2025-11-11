#!/usr/bin/env python3
"""
統合テスト
"""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_root)

from src.core.parameter import get_model_params, get_training_params
from src.core.transformer_classifier import CharTokenizer, load_tokens_and_labels_from_token_py
from src.core.state_encoder import encode_prover_state


def test_imports():
    """インポートのテスト"""
    print("Testing imports...")
    
    try:
        from src.interaction.run_interaction import apply_tactic_from_label
        from src.data_generation.auto_data_collector import AutoDataCollector
        from src.training.train_hierarchical import HierarchicalTacticDataset
        from src.compression.extract_tactics import extract_tactic_sequences
        print("✓ All imports successful")
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False
    
    return True


def test_basic_functionality():
    """基本的な機能のテスト"""
    print("Testing basic functionality...")
    
    # パラメータのget
    model_params = get_model_params()
    training_params = get_training_params()
    
    assert model_params.d_model > 0, "Model parameters not loaded"
    assert training_params.batch_size > 0, "Training parameters not loaded"
    
    # トークナイザーの作成
    token_py_path = os.path.join(project_root, "src", "core", "fof_tokens.py")
    base_tokens, _ = load_tokens_and_labels_from_token_py(token_py_path)
    tokenizer = CharTokenizer(base_tokens=base_tokens)
    
    # エンコードテスト
    goal = "a"
    premises = ["(a → b)"]
    input_ids, attention_mask, segment_ids = tokenizer.encode(goal, premises)
    
    assert len(input_ids) > 0, "Encoding failed"
    
    print("✓ Basic functionality test passed")
    return True


if __name__ == "__main__":
    print("Running integration tests...")
    print("=" * 50)
    
    success = True
    success &= test_imports()
    success &= test_basic_functionality()
    
    print("=" * 50)
    if success:
        print("✓ All integration tests passed!")
    else:
        print("✗ Some integration tests failed!")
        sys.exit(1)
