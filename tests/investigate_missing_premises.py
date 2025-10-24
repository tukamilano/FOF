#!/usr/bin/env python3
"""
premises=Falseになるデータを調査するスクリプト
actor_critic_dataのデータ構造を分析して問題のあるデータを特定
"""
import os
import sys
import json
from typing import List, Dict, Any

# プロジェクトルートをパスに追加
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_root)


def load_actor_critic_data(data_dir: str = "actor_critic_data") -> tuple[List[Dict], List[Dict]]:
    """Actor-Critic学習用データを読み込み"""
    print(f"📁 Loading data from {data_dir}...")
    
    successful_tactics = []
    failed_tactics = []
    
    # 成功データファイルを検索
    success_files = []
    failed_files = []
    
    if os.path.exists(data_dir):
        for filename in os.listdir(data_dir):
            if filename.startswith("successful_tactics_") and filename.endswith(".json"):
                success_files.append(os.path.join(data_dir, filename))
            elif filename.startswith("failed_tactics_") and filename.endswith(".json"):
                failed_files.append(os.path.join(data_dir, filename))
    
    # 成功データを読み込み
    for success_file in sorted(success_files):
        try:
            with open(success_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    successful_tactics.extend(data)
                else:
                    print(f"⚠️  Unexpected data format in {success_file}")
        except Exception as e:
            print(f"⚠️  Error loading {success_file}: {e}")
    
    # 失敗データを読み込み
    for failed_file in sorted(failed_files):
        try:
            with open(failed_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    failed_tactics.extend(data)
                else:
                    print(f"⚠️  Unexpected data format in {failed_file}")
        except Exception as e:
            print(f"⚠️  Error loading {failed_file}: {e}")
    
    print(f"✅ Loaded {len(successful_tactics)} successful tactics from {len(success_files)} files")
    print(f"✅ Loaded {len(failed_tactics)} failed tactics from {len(failed_files)} files")
    
    return successful_tactics, failed_tactics


def analyze_data_structure(data_list: List[Dict], data_type: str) -> Dict[str, Any]:
    """データ構造を分析"""
    print(f"\n🔍 Analyzing {data_type} data structure...")
    
    total_items = len(data_list)
    missing_premises = []
    missing_goals = []
    missing_tactics = []
    empty_premises = []
    empty_goals = []
    invalid_premises = []
    invalid_goals = []
    
    for i, item in enumerate(data_list):
        # premisesの分析
        premises = item.get('premises', None)
        if premises is None:
            missing_premises.append((i, item))
        elif not premises:
            empty_premises.append((i, item))
        elif not isinstance(premises, list):
            invalid_premises.append((i, item))
        elif len(premises) == 0:
            empty_premises.append((i, item))
        
        # goalの分析
        goal = item.get('goal', None)
        if goal is None:
            missing_goals.append((i, item))
        elif not goal:
            empty_goals.append((i, item))
        elif not isinstance(goal, str):
            invalid_goals.append((i, item))
        elif len(goal.strip()) == 0:
            empty_goals.append((i, item))
        
        # tacticの分析
        tactic = item.get('tactic', None)
        if tactic is None:
            missing_tactics.append((i, item))
    
    return {
        'total_items': total_items,
        'missing_premises': missing_premises,
        'missing_goals': missing_goals,
        'missing_tactics': missing_tactics,
        'empty_premises': empty_premises,
        'empty_goals': empty_goals,
        'invalid_premises': invalid_premises,
        'invalid_goals': invalid_goals
    }


def print_problematic_data(problematic_items: List[tuple], data_type: str, problem_type: str, max_examples: int = 5):
    """問題のあるデータの例を表示"""
    if not problematic_items:
        print(f"✅ No {problem_type} found in {data_type}")
        return
    
    print(f"\n❌ Found {len(problematic_items)} items with {problem_type} in {data_type}:")
    
    for i, (idx, item) in enumerate(problematic_items[:max_examples]):
        print(f"\n  Example {i+1} (Index {idx}):")
        print(f"    Keys: {list(item.keys())}")
        
        # 各フィールドの詳細を表示
        for key, value in item.items():
            if key in ['premises', 'goal', 'tactic']:
                print(f"    {key}: {type(value).__name__} = {repr(value)}")
            else:
                print(f"    {key}: {type(value).__name__} = {value}")


def investigate_specific_cases(data_list: List[Dict], data_type: str):
    """特定のケースを詳細調査"""
    print(f"\n🔬 Detailed investigation of {data_type} data...")
    
    # premises=Falseになるケースを特定
    premises_false_cases = []
    
    for i, item in enumerate(data_list):
        premises = item.get('premises', None)
        goal = item.get('goal', None)
        tactic = item.get('tactic', None)
        
        # premises=Falseの条件をチェック
        premises_ok = premises and isinstance(premises, list) and len(premises) > 0
        goal_ok = goal and isinstance(goal, str) and len(goal.strip()) > 0
        tactic_ok = tactic is not None
        
        if not premises_ok and goal_ok and tactic_ok:
            premises_false_cases.append((i, item))
    
    print(f"\n📊 Found {len(premises_false_cases)} cases where premises=False, goal=True, tactic=True")
    
    if premises_false_cases:
        print("\n🔍 Examples of problematic data:")
        for i, (idx, item) in enumerate(premises_false_cases[:10]):  # 最初の10例を表示
            print(f"\n  Case {i+1} (Index {idx}):")
            print(f"    premises: {repr(item.get('premises'))}")
            print(f"    goal: {repr(item.get('goal'))}")
            print(f"    tactic: {repr(item.get('tactic'))}")
            print(f"    All keys: {list(item.keys())}")
            
            # 他のフィールドも表示
            for key, value in item.items():
                if key not in ['premises', 'goal', 'tactic']:
                    print(f"    {key}: {repr(value)}")


def main():
    """メイン関数"""
    print("🔍 Investigating missing premises data...")
    
    # データを読み込み
    successful_tactics, failed_tactics = load_actor_critic_data("actor_critic_data")
    
    if not successful_tactics and not failed_tactics:
        print("❌ No data found!")
        return
    
    # 成功データの分析
    success_analysis = analyze_data_structure(successful_tactics, "successful")
    
    # 失敗データの分析
    failed_analysis = analyze_data_structure(failed_tactics, "failed")
    
    # 結果を表示
    print(f"\n📊 Analysis Results:")
    print(f"  Total successful tactics: {success_analysis['total_items']}")
    print(f"  Total failed tactics: {failed_analysis['total_items']}")
    
    # 問題のあるデータの詳細を表示
    print_problematic_data(success_analysis['missing_premises'], "successful", "missing premises")
    print_problematic_data(success_analysis['empty_premises'], "successful", "empty premises")
    print_problematic_data(success_analysis['invalid_premises'], "successful", "invalid premises")
    
    print_problematic_data(failed_analysis['missing_premises'], "failed", "missing premises")
    print_problematic_data(failed_analysis['empty_premises'], "failed", "empty premises")
    print_problematic_data(failed_analysis['invalid_premises'], "failed", "invalid premises")
    
    # 特定のケースを調査
    investigate_specific_cases(successful_tactics, "successful")
    investigate_specific_cases(failed_tactics, "failed")
    
    print(f"\n✅ Investigation completed!")


if __name__ == "__main__":
    main()
