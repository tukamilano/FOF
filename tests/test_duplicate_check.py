#!/usr/bin/env python3
"""
重複チェックのテストスクリプト
"""

import json
import hashlib
import os
import glob
from collections import defaultdict, Counter
from typing import Dict, List, Set, Tuple

def example_hash(original_goal: str) -> str:
    """Example全体の重複チェック用ハッシュを生成（元の目標式のみ）"""
    return hashlib.md5(original_goal.encode()).hexdigest()

def extract_hashes_from_examples(examples: List[Dict]) -> List[Tuple[str, str, str]]:
    """例からhashを抽出（example_id, hash_type, hash_value）"""
    hashes = []
    
    for example in examples:
        example_id = example.get('example_id', 'unknown')
        
        # Example全体の重複チェック（example_hashフィールドから直接取得、なければ計算）
        example_hash_val = example.get('example_hash', '')
        if not example_hash_val:
            # 既存のexample_hashがない場合は計算
            original_goal = example.get('meta', {}).get('goal_original', '')
            if original_goal:
                example_hash_val = example_hash(original_goal)
        
        if example_hash_val:
            hashes.append((example_id, 'example_hash', example_hash_val))
        
        # stepsからstate_hashを抽出
        for step in example.get('steps', []):
            state_hash = step.get('state_hash', '')
            if state_hash:
                hashes.append((example_id, 'state_hash', state_hash))
    
    return hashes

def check_duplicates_in_file(file_path: str) -> Dict:
    """単一ファイルの重複hashをチェック"""
    print(f"Reading {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        examples = json.load(f)
    
    print(f"Processing {len(examples)} examples...")
    
    # example_hashのみをチェック（state_hashの重複は自然な現象）
    example_hash_counter = Counter()
    
    total_examples = len(examples)
    total_steps = 0
    
    # hashを抽出
    file_hashes = extract_hashes_from_examples(examples)
    total_steps = len(file_hashes)
    
    # example_hashのみをカウント
    for example_id, hash_type, hash_value in file_hashes:
        if hash_type == 'example_hash':
            example_hash_counter[hash_value] += 1
    
    # 重複を検出（example_hashのみ）
    duplicate_example_hashes = {h: count for h, count in example_hash_counter.items() if count > 1}
    
    return {
        'file_path': file_path,
        'total_examples': total_examples,
        'total_steps': total_steps,
        'unique_example_hashes': len(example_hash_counter),
        'duplicate_example_hashes': len(duplicate_example_hashes),
        'duplicate_example_hash_details': duplicate_example_hashes
    }

def check_generated_data_duplicates():
    """generated_data内のすべてのファイルで重複チェックを実行（ファイル間重複も検出）"""
    # generated_dataディレクトリ内のJSONファイルを取得
    pattern = "generated_data/*.json"
    files = glob.glob(pattern)
    
    if not files:
        print("No JSON files found in generated_data/")
        return
    
    print(f"Found {len(files)} files in generated_data/")
    
    all_results = []
    total_examples = 0
    total_steps = 0
    
    # 全ファイルのハッシュを集める（ファイル間重複検出用）
    global_example_hash_counter = Counter()
    global_example_hash_files = defaultdict(set)  # ハッシュがどのファイルに含まれているかを追跡
    
    for file_path in sorted(files):
        try:
            result = check_duplicates_in_file(file_path)
            all_results.append(result)
            
            total_examples += result['total_examples']
            total_steps += result['total_steps']
            
            # ファイル内のすべてのハッシュをグローバルカウンターに追加
            with open(file_path, 'r', encoding='utf-8') as f:
                examples = json.load(f)
            
            for example in examples:
                example_hash_val = example.get('example_hash', '')
                if not example_hash_val:
                    original_goal = example.get('meta', {}).get('goal_original', '')
                    if original_goal:
                        example_hash_val = example_hash(original_goal)
                
                if example_hash_val:
                    global_example_hash_counter[example_hash_val] += 1
                    global_example_hash_files[example_hash_val].add(os.path.basename(file_path))
            
            print(f"  {os.path.basename(file_path)}: {result['total_examples']} examples, {result['duplicate_example_hashes']} duplicate examples")
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue
    
    # ファイル間重複を検出
    cross_file_duplicates = {h: count for h, count in global_example_hash_counter.items() if count > 1}
    cross_file_duplicate_count = len(cross_file_duplicates)
    
    # 全体の結果を表示
    print(f"\n" + "="*60)
    print("全体の重複チェック結果")
    print("="*60)
    
    print(f"処理ファイル数: {len(all_results)}")
    print(f"総例数: {total_examples:,}")
    print(f"総ステップ数: {total_steps:,}")
    
    print(f"\nハッシュ統計:")
    print(f"  ユニーク example_hash: {len(global_example_hash_counter):,}")
    
    print(f"\n重複検出:")
    print(f"  重複 example_hash: {cross_file_duplicate_count:,}")
    
    # 重複の詳細表示
    if cross_file_duplicate_count > 0:
        print(f"\n重複Exampleの詳細:")
        for hash_value, count in list(cross_file_duplicates.items())[:10]:
            files_list = sorted(global_example_hash_files[hash_value])
            print(f"  {hash_value} (出現回数: {count}, ファイル: {', '.join(files_list)})")
        if cross_file_duplicate_count > 10:
            print(f"  ... 他 {cross_file_duplicate_count - 10} 個")
    else:
        print(f"\n✓ 重複は見つかりませんでした！")
    

def check_single_file(file_path: str):
    """単一ファイルの重複チェックを実行"""
    try:
        result = check_duplicates_in_file(file_path)
        
        print(f"\n" + "="*60)
        print(f"重複チェック結果: {os.path.basename(file_path)}")
        print("="*60)
        
        print(f"総例数: {result['total_examples']:,}")
        print(f"総ステップ数: {result['total_steps']:,}")
        
        print(f"\nハッシュ統計:")
        print(f"  ユニーク example_hash: {result['unique_example_hashes']:,}")
        
        print(f"\n重複検出:")
        print(f"  重複 example_hash: {result['duplicate_example_hashes']:,}")
        
        if result['duplicate_example_hashes'] > 0:
            print(f"\n重複Exampleの詳細:")
            for hash_value, count in list(result['duplicate_example_hash_details'].items())[:10]:
                print(f"  {hash_value} (出現回数: {count})")
            if result['duplicate_example_hashes'] > 10:
                print(f"  ... 他 {result['duplicate_example_hashes'] - 10} 個")
        else:
            print(f"\n✓ 重複は見つかりませんでした！")
            
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # 単一ファイルのチェック
        file_path = sys.argv[1]
        check_single_file(file_path)
    else:
        # generated_data全体のチェック
        check_generated_data_duplicates()
