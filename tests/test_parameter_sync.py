#!/usr/bin/env python3
"""
parameter.pyの同期テスト
"""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_root)

from src.core.parameter import (
    default_params, get_model_params, get_training_params, 
    get_generation_params, get_system_params
)


def test_parameter_consistency():
    """パラメータの一貫性 テスト"""
    print("Testing parameter consistency...")
    
    # Default parameters get
    model_params = get_model_params()
    training_params = get_training_params()
    generation_params = get_generation_params()
    system_params = get_system_params()
    
    # Check basic values
    assert model_params.d_model > 0, "d_model should be positive"
    assert training_params.batch_size > 0, "batch_size should be positive"
    assert generation_params.count > 0, "count should be positive"
    
    print("✓ Parameter consistency test passed")


def test_parameter_updates():
    """Test parameter updates"""
    print("Testing parameter updates...")
    
    # 元の値 保存
    original_d_model = get_model_params().d_model
    
    # パラメータ 更新
    default_params.update_model_params(d_model=256)
    assert get_model_params().d_model == 256, "Model parameter update failed"
    
    # 元 戻す
    default_params.update_model_params(d_model=original_d_model)
    assert get_model_params().d_model == original_d_model, "Parameter rollback failed"
    
    print("✓ Parameter update test passed")


if __name__ == "__main__":
    test_parameter_consistency()
    test_parameter_updates()
    print("\nAll parameter tests passed!")
