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
    
    # hashの種類別にカウンターを作成
    example_hash_counter = Counter()
    state_hash_counter = Counter()
    all_hashes = defaultdict(list)  # hash -> [(example_id, file_name), ...]
    
    total_examples = len(examples)
    total_steps = 0
    
    # hashを抽出
    file_hashes = extract_hashes_from_examples(examples)
    total_steps = len(file_hashes)
    
    # hashをカウント
    for example_id, hash_type, hash_value in file_hashes:
        if hash_type == 'example_hash':
            example_hash_counter[hash_value] += 1
        elif hash_type == 'state_hash':
            state_hash_counter[hash_value] += 1
        
        all_hashes[hash_value].append((example_id, file_path))
    
    # 重複を検出
    duplicate_example_hashes = {h: count for h, count in example_hash_counter.items() if count > 1}
    duplicate_state_hashes = {h: count for h, count in state_hash_counter.items() if count > 1}
    
    # 全hashの重複チェック
    duplicate_all_hashes = {h: locations for h, locations in all_hashes.items() if len(locations) > 1}
    
    return {
        'file_path': file_path,
        'total_examples': total_examples,
        'total_steps': total_steps,
        'unique_example_hashes': len(example_hash_counter),
        'unique_state_hashes': len(state_hash_counter),
        'unique_all_hashes': len(all_hashes),
        'duplicate_example_hashes': len(duplicate_example_hashes),
        'duplicate_state_hashes': len(duplicate_state_hashes),
        'duplicate_all_hashes': len(duplicate_all_hashes),
        'duplicate_example_hash_details': duplicate_example_hashes,
        'duplicate_state_hash_details': duplicate_state_hashes,
        'duplicate_all_hash_details': duplicate_all_hashes
    }

def check_generated_data_duplicates():
    """generated_data内のすべてのファイルで重複チェックを実行"""
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
    total_unique_examples = 0
    total_unique_states = 0
    total_duplicate_examples = 0
    total_duplicate_states = 0
    
    for file_path in sorted(files):
        try:
            result = check_duplicates_in_file(file_path)
            all_results.append(result)
            
            total_examples += result['total_examples']
            total_steps += result['total_steps']
            total_unique_examples += result['unique_example_hashes']
            total_unique_states += result['unique_state_hashes']
            total_duplicate_examples += result['duplicate_example_hashes']
            total_duplicate_states += result['duplicate_state_hashes']
            
            print(f"  {os.path.basename(file_path)}: {result['total_examples']} examples, {result['duplicate_example_hashes']} duplicate examples, {result['duplicate_state_hashes']} duplicate states")
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue
    
    # 全体の結果を表示
    print(f"\n" + "="*60)
    print("全体の重複チェック結果")
    print("="*60)
    
    print(f"処理ファイル数: {len(all_results)}")
    print(f"総例数: {total_examples:,}")
    print(f"総ステップ数: {total_steps:,}")
    
    print(f"\nハッシュ統計:")
    print(f"  ユニーク example_hash: {total_unique_examples:,}")
    print(f"  ユニーク state_hash: {total_unique_states:,}")
    
    print(f"\n重複検出:")
    print(f"  重複 example_hash: {total_duplicate_examples:,}")
    print(f"  重複 state_hash: {total_duplicate_states:,}")
    
    # 重複の詳細表示
    if total_duplicate_examples > 0:
        print(f"\n重複Exampleの詳細:")
        for result in all_results:
            if result['duplicate_example_hashes'] > 0:
                print(f"  {os.path.basename(result['file_path'])}:")
                for hash_value, count in list(result['duplicate_example_hash_details'].items())[:5]:
                    print(f"    {hash_value} (出現回数: {count})")
                if result['duplicate_example_hashes'] > 5:
                    print(f"    ... 他 {result['duplicate_example_hashes'] - 5} 個")
    
    if total_duplicate_states > 0:
        print(f"\n重複Stateの詳細:")
        for result in all_results:
            if result['duplicate_state_hashes'] > 0:
                print(f"  {os.path.basename(result['file_path'])}:")
                for hash_value, count in list(result['duplicate_state_hash_details'].items())[:5]:
                    print(f"    {hash_value} (出現回数: {count})")
                if result['duplicate_state_hashes'] > 5:
                    print(f"    ... 他 {result['duplicate_state_hashes'] - 5} 個")
    
    if total_duplicate_examples == 0 and total_duplicate_states == 0:
        print(f"\n✓ 重複は見つかりませんでした！")
    
    # 結果をJSONファイルに保存
    output_file = f"duplicate_check_results_{int(__import__('time').time())}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'summary': {
                'total_files': len(all_results),
                'total_examples': total_examples,
                'total_steps': total_steps,
                'unique_example_hashes': total_unique_examples,
                'unique_state_hashes': total_unique_states,
                'duplicate_example_hashes': total_duplicate_examples,
                'duplicate_state_hashes': total_duplicate_states
            },
            'file_results': all_results
        }, f, ensure_ascii=False, indent=2)
    print(f"\n詳細結果を {output_file} に保存しました。")

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
        print(f"  ユニーク state_hash: {result['unique_state_hashes']:,}")
        
        print(f"\n重複検出:")
        print(f"  重複 example_hash: {result['duplicate_example_hashes']:,}")
        print(f"  重複 state_hash: {result['duplicate_state_hashes']:,}")
        
        if result['duplicate_example_hashes'] > 0:
            print(f"\n重複Exampleの詳細:")
            for hash_value, count in list(result['duplicate_example_hash_details'].items())[:10]:
                print(f"  {hash_value} (出現回数: {count})")
            if result['duplicate_example_hashes'] > 10:
                print(f"  ... 他 {result['duplicate_example_hashes'] - 10} 個")
        
        if result['duplicate_state_hashes'] > 0:
            print(f"\n重複Stateの詳細:")
            for hash_value, count in list(result['duplicate_state_hash_details'].items())[:10]:
                print(f"  {hash_value} (出現回数: {count})")
            if result['duplicate_state_hashes'] > 10:
                print(f"  ... 他 {result['duplicate_state_hashes'] - 10} 個")
        
        if result['duplicate_example_hashes'] == 0 and result['duplicate_state_hashes'] == 0:
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
