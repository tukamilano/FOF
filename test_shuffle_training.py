#!/usr/bin/env python3
"""
シャッフル機能のテストスクリプト
actor_critic_dataのシャッフル機能をテストする
"""
import os
import sys
import json
import random
import numpy as np
from typing import List, Dict, Any, Tuple

# プロジェクトルートをパスに追加
project_root = os.path.dirname(__file__)
sys.path.insert(0, project_root)

from train_actor_critic import shuffle_training_data, load_actor_critic_data


def create_test_data(num_success: int = 100, num_failed: int = 100) -> Tuple[List[Dict], List[Dict]]:
    """テスト用のデータを作成"""
    successful_tactics = []
    failed_tactics = []
    
    for i in range(num_success):
        successful_tactics.append({
            'premises': f'premise_{i}',
            'goal': f'goal_{i}',
            'tactic': {'main': 'assumption', 'arg1': None, 'arg2': None},
            'reward': 1.0,
            'log_prob': -0.1,
            'step_index': i,
            'state_tactic_hash': f'success_hash_{i}'
        })
    
    for i in range(num_failed):
        failed_tactics.append({
            'premises': f'premise_{i}',
            'goal': f'goal_{i}',
            'tactic': {'main': 'intro', 'arg1': None, 'arg2': None},
            'reward': -0.1,
            'log_prob': -0.5,
            'step_index': i,
            'state_tactic_hash': f'failed_hash_{i}'
        })
    
    return successful_tactics, failed_tactics


def test_shuffle_functionality():
    """シャッフル機能をテスト（全てのデータを使用）"""
    print("🧪 Testing shuffle functionality with ALL data...")
    
    # テストデータを作成
    successful_tactics, failed_tactics = create_test_data(50, 50)
    
    print(f"Original data:")
    print(f"  Successful tactics: {len(successful_tactics)}")
    print(f"  Failed tactics: {len(failed_tactics)}")
    
    # シャッフル前のハッシュを記録
    original_success_hashes = [item['state_tactic_hash'] for item in successful_tactics]
    original_failed_hashes = [item['state_tactic_hash'] for item in failed_tactics]
    original_all_hashes = set(original_success_hashes + original_failed_hashes)
    
    # シャッフルを実行
    shuffled_successful, shuffled_failed = shuffle_training_data(
        successful_tactics, failed_tactics, shuffle_ratio=0.3, random_seed=42
    )
    
    print(f"\nAfter shuffling:")
    print(f"  Shuffled successful tactics: {len(shuffled_successful)}")
    print(f"  Shuffled failed tactics: {len(shuffled_failed)}")
    
    # シャッフル後のハッシュを記録
    shuffled_success_hashes = [item['state_tactic_hash'] for item in shuffled_successful]
    shuffled_failed_hashes = [item['state_tactic_hash'] for item in shuffled_failed]
    shuffled_all_hashes = set(shuffled_success_hashes + shuffled_failed_hashes)
    
    # 検証
    print(f"\n🔍 Verification:")
    
    # 1. データ数が変わらないことを確認
    assert len(shuffled_successful) + len(shuffled_failed) == len(successful_tactics) + len(failed_tactics), \
        "Total data count should remain the same"
    
    # 2. 全てのデータが使用されていることを確認
    assert original_all_hashes == shuffled_all_hashes, "All data should be preserved"
    
    # 3. シャッフルが実際に行われたことを確認
    success_changed = len(set(original_success_hashes) & set(shuffled_success_hashes)) < len(original_success_hashes)
    failed_changed = len(set(original_failed_hashes) & set(shuffled_failed_hashes)) < len(original_failed_hashes)
    
    print(f"  Success data changed: {success_changed}")
    print(f"  Failed data changed: {failed_changed}")
    
    # 4. データが交換されたことを確認
    success_in_failed = len(set(shuffled_success_hashes) & set(original_failed_hashes))
    failed_in_success = len(set(shuffled_failed_hashes) & set(original_success_hashes))
    
    print(f"  Success items moved to failed: {success_in_failed}")
    print(f"  Failed items moved to success: {failed_in_success}")
    
    # 5. 全てのデータが使用されていることを確認
    print(f"  All data preserved: {len(original_all_hashes)} == {len(shuffled_all_hashes)}")
    
    print(f"\n✅ Shuffle functionality test passed!")
    return True


def test_different_shuffle_ratios():
    """異なるシャッフル比率をテスト（全てのデータを使用）"""
    print("\n🧪 Testing different shuffle ratios with ALL data...")
    
    successful_tactics, failed_tactics = create_test_data(100, 100)
    
    ratios = [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]
    
    for ratio in ratios:
        print(f"\nTesting shuffle ratio: {ratio} (ALL data will be used)")
        
        shuffled_successful, shuffled_failed = shuffle_training_data(
            successful_tactics.copy(), failed_tactics.copy(), 
            shuffle_ratio=ratio, random_seed=42
        )
        
        # データ数が変わらないことを確認
        total_original = len(successful_tactics) + len(failed_tactics)
        total_shuffled = len(shuffled_successful) + len(shuffled_failed)
        
        assert total_original == total_shuffled, f"Data count mismatch for ratio {ratio}"
        
        # 全てのデータが使用されていることを確認
        original_all_hashes = set(item['state_tactic_hash'] for item in successful_tactics + failed_tactics)
        shuffled_all_hashes = set(item['state_tactic_hash'] for item in shuffled_successful + shuffled_failed)
        
        assert original_all_hashes == shuffled_all_hashes, f"All data should be preserved for ratio {ratio}"
        
        # 交換されたデータ数を計算
        original_success_hashes = set(item['state_tactic_hash'] for item in successful_tactics)
        original_failed_hashes = set(item['state_tactic_hash'] for item in failed_tactics)
        
        shuffled_success_hashes = set(item['state_tactic_hash'] for item in shuffled_successful)
        shuffled_failed_hashes = set(item['state_tactic_hash'] for item in shuffled_failed)
        
        # 元々成功だったものが失敗に移動した数
        success_to_failed = len(original_success_hashes & shuffled_failed_hashes)
        # 元々失敗だったものが成功に移動した数
        failed_to_success = len(original_failed_hashes & shuffled_success_hashes)
        
        print(f"  All data preserved: {len(original_all_hashes)} == {len(shuffled_all_hashes)}")
        print(f"  Success -> Failed: {success_to_failed}")
        print(f"  Failed -> Success: {failed_to_success}")
        print(f"  Total exchange: {success_to_failed + failed_to_success}")
    
    print(f"\n✅ Different shuffle ratios test passed!")
    return True


def test_random_seed_consistency():
    """ランダムシードの一貫性をテスト"""
    print("\n🧪 Testing random seed consistency...")
    
    successful_tactics, failed_tactics = create_test_data(50, 50)
    
    # 同じシードで2回シャッフル
    shuffled1_success, shuffled1_failed = shuffle_training_data(
        successful_tactics.copy(), failed_tactics.copy(), 
        shuffle_ratio=0.5, random_seed=123
    )
    
    shuffled2_success, shuffled2_failed = shuffle_training_data(
        successful_tactics.copy(), failed_tactics.copy(), 
        shuffle_ratio=0.5, random_seed=123
    )
    
    # 結果が同じであることを確認
    success1_hashes = set(item['state_tactic_hash'] for item in shuffled1_success)
    success2_hashes = set(item['state_tactic_hash'] for item in shuffled2_success)
    
    failed1_hashes = set(item['state_tactic_hash'] for item in shuffled1_failed)
    failed2_hashes = set(item['state_tactic_hash'] for item in shuffled2_failed)
    
    assert success1_hashes == success2_hashes, "Success data should be identical with same seed"
    assert failed1_hashes == failed2_hashes, "Failed data should be identical with same seed"
    
    print(f"  Same seed produces identical results: ✅")
    
    # 異なるシードで異なる結果になることを確認
    shuffled3_success, shuffled3_failed = shuffle_training_data(
        successful_tactics.copy(), failed_tactics.copy(), 
        shuffle_ratio=0.5, random_seed=456
    )
    
    success3_hashes = set(item['state_tactic_hash'] for item in shuffled3_success)
    failed3_hashes = set(item['state_tactic_hash'] for item in shuffled3_failed)
    
    assert success1_hashes != success3_hashes, "Different seeds should produce different results"
    assert failed1_hashes != failed3_hashes, "Different seeds should produce different results"
    
    print(f"  Different seeds produce different results: ✅")
    
    print(f"\n✅ Random seed consistency test passed!")
    return True


def test_edge_cases():
    """エッジケースをテスト"""
    print("\n🧪 Testing edge cases...")
    
    # 1. 空のデータ
    print("  Testing empty data...")
    empty_success, empty_failed = shuffle_training_data([], [], shuffle_ratio=0.5, random_seed=42)
    assert len(empty_success) == 0 and len(empty_failed) == 0, "Empty data should remain empty"
    
    # 2. 片方だけのデータ
    print("  Testing single-sided data...")
    success_only, failed_only = create_test_data(50, 0)
    shuffled_success, shuffled_failed = shuffle_training_data(
        success_only, failed_only, shuffle_ratio=0.5, random_seed=42
    )
    assert len(shuffled_success) == 50 and len(shuffled_failed) == 0, "Single-sided data should remain unchanged"
    
    # 3. シャッフル比率0
    print("  Testing shuffle ratio 0...")
    successful_tactics, failed_tactics = create_test_data(50, 50)
    shuffled_success, shuffled_failed = shuffle_training_data(
        successful_tactics, failed_tactics, shuffle_ratio=0.0, random_seed=42
    )
    
    original_success_hashes = set(item['state_tactic_hash'] for item in successful_tactics)
    original_failed_hashes = set(item['state_tactic_hash'] for item in failed_tactics)
    shuffled_success_hashes = set(item['state_tactic_hash'] for item in shuffled_success)
    shuffled_failed_hashes = set(item['state_tactic_hash'] for item in shuffled_failed)
    
    assert original_success_hashes == shuffled_success_hashes, "No shuffle should preserve original data"
    assert original_failed_hashes == shuffled_failed_hashes, "No shuffle should preserve original data"
    
    print(f"\n✅ Edge cases test passed!")
    return True


def main():
    """メイン関数"""
    print("🚀 Starting shuffle functionality tests...")
    
    try:
        # 各テストを実行
        test_shuffle_functionality()
        test_different_shuffle_ratios()
        test_random_seed_consistency()
        test_edge_cases()
        
        print(f"\n🎉 All tests passed successfully!")
        print(f"✅ Shuffle functionality is working correctly!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
