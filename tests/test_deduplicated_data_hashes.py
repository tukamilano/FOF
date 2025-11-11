#!/usr/bin/env python3
"""
Test to check duplicates of state_hash and state_tactic_hash in deduplicated_data
"""

import json
import os
import glob
from collections import Counter, defaultdict
from typing import Dict, List, Set, Tuple

def check_deduplicated_data_hashes():
    """Check state_hash and state_tactic_hash duplicates in all files in deduplicated_data"""
    # Get JSON files in deduplicated_data directory
    pattern = "deduplicated_data/*.json"
    files = glob.glob(pattern)
    
    if not files:
        print("No JSON files found in deduplicated_data/")
        return
    
    print(f"Found {len(files)} files in deduplicated_data/")
    
    # 全ファイルのハッシュ 集める
    global_state_hash_counter = Counter()
    global_state_tactic_hash_counter = Counter()
    global_state_hash_files = defaultdict(set)  # ハッシュ どのファイル 含まれてexistsか 追跡
    global_state_tactic_hash_files = defaultdict(set)
    
    total_steps = 0
    total_files = 0
    
    for file_path in sorted(files):
        try:
            print(f"Reading {file_path}...")
            with open(file_path, 'r', encoding='utf-8') as f:
                steps = json.load(f)
            
            print(f"Processing {len(steps)} steps...")
            total_steps += len(steps)
            total_files += 1
            
            # 各ステップのハッシュ チェック
            for step in steps:
                state_hash = step.get('state_hash', '')
                state_tactic_hash = step.get('state_tactic_hash', '')
                
                if state_hash:
                    global_state_hash_counter[state_hash] += 1
                    global_state_hash_files[state_hash].add(os.path.basename(file_path))
                
                if state_tactic_hash:
                    global_state_tactic_hash_counter[state_tactic_hash] += 1
                    global_state_tactic_hash_files[state_tactic_hash].add(os.path.basename(file_path))
            
            print(f"  {os.path.basename(file_path)}: {len(steps)} steps processed")
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue
    
    # 重複 検出
    duplicate_state_hashes = {h: count for h, count in global_state_hash_counter.items() if count > 1}
    duplicate_state_tactic_hashes = {h: count for h, count in global_state_tactic_hash_counter.items() if count > 1}
    
    # Detect cross-file duplicates（複数ファイル また る重複）
    cross_file_state_hash_duplicates = {h: files for h, files in global_state_hash_files.items() if len(files) > 1}
    cross_file_state_tactic_hash_duplicates = {h: files for h, files in global_state_tactic_hash_files.items() if len(files) > 1}
    
    # 結果 表示
    print(f"\n" + "="*80)
    print("deduplicated_data ハッシュ重複チェック結果")
    print("="*80)
    
    print(f"処理ファイル数: {total_files}")
    print(f"総ステップ数: {total_steps:,}")
    
    print(f"\nstate_hash統計:")
    print(f"  ユニーク state_hash: {len(global_state_hash_counter):,}")
    print(f"  重複 state_hash: {len(duplicate_state_hashes):,}")
    print(f"  ファイル間重複: {len(cross_file_state_hash_duplicates):,}")
    
    print(f"\nstate_tactic_hash統計:")
    print(f"  ユニーク state_tactic_hash: {len(global_state_tactic_hash_counter):,}")
    print(f"  重複 state_tactic_hash: {len(duplicate_state_tactic_hashes):,}")
    print(f"  ファイル間重複: {len(cross_file_state_tactic_hash_duplicates):,}")
    
    # state_hash重複の詳細表示
    if duplicate_state_hashes:
        print(f"\n重複state_hashの詳細 (上位10件):")
        for i, (hash_value, count) in enumerate(list(duplicate_state_hashes.items())[:10]):
            files_list = list(global_state_hash_files[hash_value])
            cross_file_indicator = " (ファイル間)" if len(files_list) > 1 else " (ファイル内)"
            print(f"  {i+1}. Hash: {hash_value[:16]}... Count: {count}{cross_file_indicator}, Files: {files_list}")
    else:
        print(f"\n✓ state_hash重複は見かりません with/at did！")
    
    # state_tactic_hash重複の詳細表示
    if duplicate_state_tactic_hashes:
        print(f"\n重複state_tactic_hashの詳細 (上位10件):")
        for i, (hash_value, count) in enumerate(list(duplicate_state_tactic_hashes.items())[:10]):
            files_list = list(global_state_tactic_hash_files[hash_value])
            cross_file_indicator = " (ファイル間)" if len(files_list) > 1 else " (ファイル内)"
            print(f"  {i+1}. Hash: {hash_value[:16]}... Count: {count}{cross_file_indicator}, Files: {files_list}")
    else:
        print(f"\n✓ state_tactic_hash重複は見かりません with/at did！")
    
    # 期待is done結果の確認
    print(f"\n" + "="*80)
    print("期待is done結果の確認")
    print("="*80)
    
    # state_hashの重複は正常（同じ状態 with/at 異becometactic 試すため）
    if len(duplicate_state_hashes) > 0:
        print(f"ℹ️  state_hash: {len(duplicate_state_hashes)}の重複（正常 - 同じ状態 with/at 異becometactic 試すため）")
        print(f"   - ファイル間重複: {len(cross_file_state_hash_duplicates)}")
    else:
        print("✅ state_hash: 重複なし")
    
    # state_tactic_hashの重複は除去is doneべき
    if len(duplicate_state_tactic_hashes) == 0:
        print("✅ state_tactic_hash: 重複なし（期待通り - 重複除去 正しく動作）")
    else:
        print(f"❌ state_tactic_hash: {len(duplicate_state_tactic_hashes)}の重複 検出されまdid（重複除去 問題あり）")
        print(f"   - ファイル間重複: {len(cross_file_state_tactic_hash_duplicates)}")
    
    return {
        'total_files': total_files,
        'total_steps': total_steps,
        'unique_state_hashes': len(global_state_hash_counter),
        'unique_state_tactic_hashes': len(global_state_tactic_hash_counter),
        'duplicate_state_hashes': len(duplicate_state_hashes),
        'duplicate_state_tactic_hashes': len(duplicate_state_tactic_hashes),
        'cross_file_state_hash_duplicates': len(cross_file_state_hash_duplicates),
        'cross_file_state_tactic_hash_duplicates': len(cross_file_state_tactic_hash_duplicates)
    }

def main():
    """メイン関数"""
    print("🔍 deduplicated_data ハッシュ重複チェック 開始...")
    result = check_deduplicated_data_hashes()
    
    if result:
        print(f"\n📊 チェック完了:")
        print(f"  処理ファイル数: {result['total_files']}")
        print(f"  総ステップ数: {result['total_steps']:,}")
        print(f"  ユニーク state_hash: {result['unique_state_hashes']:,}")
        print(f"  ユニーク state_tactic_hash: {result['unique_state_tactic_hashes']:,}")
        print(f"  重複 state_hash: {result['duplicate_state_hashes']}")
        print(f"  重複 state_tactic_hash: {result['duplicate_state_tactic_hashes']}")

if __name__ == "__main__":
    main()
