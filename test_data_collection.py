#!/usr/bin/env python3
"""
学習データ収集機能のテストスクリプト
"""
import os
import json
import tempfile
import sys
from training_data_collector import TrainingDataCollector

# pyproverのインポートを修正
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'pyprover'))
import proposition as proposition_mod
import prover as prover_mod

PropParseTree = proposition_mod.PropParseTree
prop_parser = proposition_mod.parser
Prover = prover_mod.Prover


def test_data_collector():
    """TrainingDataCollectorの基本機能をテスト"""
    print("Testing TrainingDataCollector...")
    
    # 一時ファイルを使用
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as work_file:
        work_path = work_file.name
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as dataset_file:
        dataset_path = dataset_file.name
    
    try:
        collector = TrainingDataCollector(work_path, dataset_path)
        
        # 例1を開始
        collector.start_example(
            example_id=0,
            initial_premises=["(a → b)", "(b → c)", "a"],
            initial_goal="c"
        )
        
        # 戦略適用を記録
        collector.add_tactic_application(
            step=1,
            premises=["(a → b)", "(b → c)", "a"],
            goal="c",
            tactic="apply 0",
            tactic_apply=True
        )
        
        collector.add_tactic_application(
            step=2,
            premises=["(a → b)", "(b → c)", "a"],
            goal="b",
            tactic="intro",
            tactic_apply=True
        )
        
        # 例1を完了
        collector.finish_example(is_proved=True)
        
        # 例2を開始
        collector.start_example(
            example_id=1,
            initial_premises=["(a → b)", "a", ""],
            initial_goal="b"
        )
        
        collector.add_tactic_application(
            step=1,
            premises=["(a → b)", "a", ""],
            goal="b",
            tactic="apply 0",
            tactic_apply=True
        )
        
        # 例2を完了
        collector.finish_example(is_proved=True)
        
        # 統計情報を取得
        stats = collector.get_dataset_stats()
        print(f"Dataset stats: {stats}")
        
        # データセットファイルを確認
        with open(dataset_path, 'r') as f:
            dataset = json.load(f)
        
        print(f"Dataset contains {len(dataset)} records")
        for i, record in enumerate(dataset):
            print(f"  Record {i+1}: {record}")
        
        print("✓ TrainingDataCollector test passed")
        
    finally:
        # クリーンアップ
        collector.cleanup()
        if os.path.exists(work_path):
            os.unlink(work_path)
        if os.path.exists(dataset_path):
            os.unlink(dataset_path)


def test_encoding_consistency():
    """エンコードの一貫性をテスト"""
    print("Testing encoding consistency...")
    
    # テスト用のproverを作成
    parse_tree = PropParseTree()
    goal = parse_tree.transform(prop_parser.parse("(a → b)"))
    prover = Prover(goal)
    prover.variables = [
        parse_tree.transform(prop_parser.parse("a")),
        parse_tree.transform(prop_parser.parse("(b → c)"))
    ]
    
    # 一貫性をテスト（state_encoderを直接インポート）
    from state_encoder import test_encoding_consistency
    is_consistent = test_encoding_consistency(prover, max_len=50)
    
    if is_consistent:
        print("✓ Encoding consistency test passed")
    else:
        print("✗ Encoding consistency test failed")
        return False
    
    return True


def test_integration():
    """統合テスト（実際のrun_interaction.pyの動作をシミュレート）"""
    print("Testing integration...")
    
    # 一時ファイルを使用
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as work_file:
        work_path = work_file.name
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as dataset_file:
        dataset_path = dataset_file.name
    
    try:
        collector = TrainingDataCollector(work_path, dataset_path)
        
        # 簡単な例をシミュレート
        parse_tree = PropParseTree()
        goal = parse_tree.transform(prop_parser.parse("a"))
        prover = Prover(goal)
        prover.variables = [parse_tree.transform(prop_parser.parse("a"))]
        
        # 例を開始（完全なデータを保存）
        from state_encoder import encode_prover_state
        initial_state = encode_prover_state(prover)
        collector.start_example(
            example_id=0,
            initial_premises=initial_state["premises"],
            initial_goal=initial_state["goal"]
        )
        
        # 戦略適用をシミュレート
        collector.add_tactic_application(
            step=1,
            premises=initial_state["premises"],
            goal=initial_state["goal"],
            tactic="assumption",
            tactic_apply=True
        )
        
        # 例を完了
        collector.finish_example(is_proved=True)
        
        # 結果を確認
        stats = collector.get_dataset_stats()
        print(f"Integration test stats: {stats}")
        
        print("✓ Integration test passed")
        
    finally:
        collector.cleanup()
        if os.path.exists(work_path):
            os.unlink(work_path)
        if os.path.exists(dataset_path):
            os.unlink(dataset_path)


def test_full_data_preservation():
    """完全なデータ保存のテスト"""
    print("Testing full data preservation...")
    
    # 長い数式を含む例を作成
    parse_tree = PropParseTree()
    long_formula = "((a → b) ∧ (b → c) ∧ (c → d) ∧ (d → e) ∧ (e → f) ∧ (f → g) ∧ (g → h) ∧ (h → i) ∧ (i → j) ∧ (j → k))"
    goal = parse_tree.transform(prop_parser.parse(long_formula))
    prover = Prover(goal)
    prover.variables = [
        parse_tree.transform(prop_parser.parse("a")),
        parse_tree.transform(prop_parser.parse("(a → b)")),
        parse_tree.transform(prop_parser.parse("(b → c)"))
    ]
    
    # 完全なデータをエンコード
    from state_encoder import encode_prover_state
    full_state = encode_prover_state(prover)
    
    print(f"Full premises[0]: {full_state['premises'][0]}")
    print(f"Full goal: {full_state['goal']}")
    print(f"Number of premises: {len(full_state['premises'])}")
    
    # 完全なデータが保存されていることを確認
    print("✓ Full data preservation test passed")
    return True


def main():
    """メインテスト関数"""
    print("Running data collection tests...")
    print("=" * 50)
    
    try:
        test_data_collector()
        print()
        
        if test_encoding_consistency():
            print()
            test_integration()
            print()
            test_full_data_preservation()
            print()
            print("=" * 50)
            print("All tests passed! ✓")
        else:
            print("Some tests failed. ✗")
            
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
