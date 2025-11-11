#!/usr/bin/env python3
"""
Script to check duplicates in generated_data
"""
import os
import sys
import json
import glob
from collections import Counter

def check_duplicates(data_dir="generated_data"):
    """Check duplicates in generated_data"""
    
    if not os.path.exists(data_dir):
        print(f"Directory {data_dir} not found")
        return
    
    # Load all JSON files
    json_files = glob.glob(os.path.join(data_dir, "*.json"))
    print(f"Found {len(json_files)} JSON files")
    
    all_steps = []
    state_hashes = []
    
    for json_file in json_files:
        print(f"Loading {json_file}...")
        with open(json_file, 'r') as f:
            file_data = json.load(f)
        
        for example in file_data:
            for step in example.get('steps', []):
                if step.get('tactic_apply', False):
                    all_steps.append(step)
                    state_hashes.append(step.get('state_hash', ''))
    
    print(f"\n=== Duplicate Analysis ===")
    print(f"Total steps: {len(all_steps)}")
    print(f"Unique state hashes: {len(set(state_hashes))}")
    print(f"Duplicates: {len(all_steps) - len(set(state_hashes))}")
    print(f"Duplicate rate: {(len(all_steps) - len(set(state_hashes))) / len(all_steps) * 100:.2f}%")
    
    # 重複の詳細
    hash_counts = Counter(state_hashes)
    duplicates = {h: c for h, c in hash_counts.items() if c > 1}
    
    if duplicates:
        print(f"\n=== Top 10 Most Duplicated States ===")
        for hash_val, count in sorted(duplicates.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"Hash: {hash_val[:16]}... Count: {count}")
            
            # thisハッシュの例 表示
            examples = [step for step in all_steps if step.get('state_hash') == hash_val]
            if examples:
                example = examples[0]
                print(f"  Example: premises={example.get('premises', [])}, goal={example.get('goal', '')}")
                print(f"  Tactic: {example.get('tactic', {})}")
                print()
    
    return {
        'total_steps': len(all_steps),
        'unique_hashes': len(set(state_hashes)),
        'duplicates': len(all_steps) - len(set(state_hashes)),
        'duplicate_rate': (len(all_steps) - len(set(state_hashes))) / len(all_steps) * 100
    }

def main():
    # Move to project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    os.chdir(project_root)
    
    data_dir = "generated_data"
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    
    check_duplicates(data_dir)

if __name__ == "__main__":
    main()
