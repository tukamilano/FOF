#!/usr/bin/env python3
"""
generated_dataの内容を分析するスクリプト
"""
import os
import sys
import json
import glob
from collections import Counter, defaultdict

def analyze_generated_data(data_dir="generated_data"):
    """generated_dataディレクトリの内容を分析"""
    
    if not os.path.exists(data_dir):
        print(f"Directory {data_dir} not found")
        return
    
    # 全JSONファイルを読み込み
    json_files = glob.glob(os.path.join(data_dir, "*.json"))
    print(f"Found {len(json_files)} JSON files")
    
    all_examples = []
    all_steps = []
    tactic_counter = Counter()
    step_lengths = []
    
    for json_file in json_files:
        print(f"Analyzing {json_file}...")
        with open(json_file, 'r') as f:
            file_data = json.load(f)
        
        for example in file_data:
            all_examples.append(example)
            steps = example.get('steps', [])
            step_lengths.append(len(steps))
            
            for step in steps:
                all_steps.append(step)
                tactic = step.get('tactic', {})
                if isinstance(tactic, dict):
                    main_tactic = tactic.get('main', 'unknown')
                    tactic_counter[main_tactic] += 1
    
    print(f"\n=== Generated Data Analysis ===")
    print(f"Total examples: {len(all_examples)}")
    print(f"Total steps: {len(all_steps)}")
    print(f"Average steps per example: {sum(step_lengths) / len(step_lengths):.2f}")
    print(f"Max steps in an example: {max(step_lengths)}")
    print(f"Min steps in an example: {min(step_lengths)}")
    
    # 成功したステップの統計
    successful_steps = [step for step in all_steps if step.get('tactic_apply', False)]
    print(f"Successful steps: {len(successful_steps)}")
    print(f"Success rate: {len(successful_steps) / len(all_steps) * 100:.2f}%")
    
    # タクティクの分布
    print(f"\n=== Tactic Distribution ===")
    for tactic, count in tactic_counter.most_common():
        print(f"{tactic}: {count} ({count/len(all_steps)*100:.2f}%)")
    
    # 引数の分析
    arg1_values = Counter()
    arg2_values = Counter()
    
    for step in all_steps:
        tactic = step.get('tactic', {})
        if isinstance(tactic, dict):
            arg1 = tactic.get('arg1')
            arg2 = tactic.get('arg2')
            if arg1 is not None:
                arg1_values[arg1] += 1
            if arg2 is not None:
                arg2_values[arg2] += 1
    
    print(f"\n=== Arg1 Distribution ===")
    for arg, count in arg1_values.most_common(10):
        print(f"{arg}: {count}")
    
    print(f"\n=== Arg2 Distribution ===")
    for arg, count in arg2_values.most_common(10):
        print(f"{arg}: {count}")
    
    # ゴールの複雑さ分析
    goal_lengths = [len(step.get('goal', '')) for step in all_steps]
    print(f"\n=== Goal Complexity ===")
    print(f"Average goal length: {sum(goal_lengths) / len(goal_lengths):.2f}")
    print(f"Max goal length: {max(goal_lengths)}")
    print(f"Min goal length: {min(goal_lengths)}")
    
    # 前提の数分析
    premise_counts = [len(step.get('premises', [])) for step in all_steps]
    print(f"\n=== Premise Count ===")
    print(f"Average premises per step: {sum(premise_counts) / len(premise_counts):.2f}")
    print(f"Max premises: {max(premise_counts)}")
    print(f"Min premises: {min(premise_counts)}")
    
    return {
        'total_examples': len(all_examples),
        'total_steps': len(all_steps),
        'successful_steps': len(successful_steps),
        'tactic_distribution': dict(tactic_counter),
        'avg_steps_per_example': sum(step_lengths) / len(step_lengths),
        'avg_goal_length': sum(goal_lengths) / len(goal_lengths),
        'avg_premise_count': sum(premise_counts) / len(premise_counts)
    }

def main():
    # プロジェクトルートに移動
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    os.chdir(project_root)
    
    data_dir = "generated_data"
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    
    analyze_generated_data(data_dir)

if __name__ == "__main__":
    main()
