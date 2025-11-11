#!/usr/bin/env python3
"""
GCSの全ファイル 横断and重複チェック 実行do/performスクリプト
"""

import json
import hashlib
import os
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Set, Tuple
from google.cloud import storage
import argparse

def example_hash(original_goal: str) -> str:
    """Example全体の重複チェック用ハッシュ Generation（元の目標式only）"""
    return hashlib.md5(original_goal.encode()).hexdigest()

def list_gcs_files(bucket_name: str, prefix: str) -> List[str]:
    """GCSバケット from 指定プレフィックスのファイル一覧 get"""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    
    blobs = bucket.list_blobs(prefix=prefix)
    files = []
    
    for blob in blobs:
        if blob.name.endswith('.json'):
            files.append(blob.name)
    
    return sorted(files)

def download_gcs_file(bucket_name: str, file_path: str) -> List[Dict]:
    """GCS from ファイル ダウンロードandJSON as 読み込み"""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(file_path)
    
    content = blob.download_as_text()
    return json.loads(content)

def check_gcs_cross_file_duplicates(bucket_name: str, prefix: str, limit: int = None):
    """GCSの全ファイル 横断and重複チェック 実行"""
    print(f"GCS Bucket: {bucket_name}")
    print(f"Prefix: {prefix}")
    if limit:
        print(f"Limit: 最初の {limit} ファイルonly処理")
    print("="*60)
    
    # GCSファイル一覧 get
    print("GCSファイル一覧 get中...")
    files = list_gcs_files(bucket_name, prefix)
    
    # ファイル数制限 適用
    if limit and limit > 0:
        files = files[:limit]
        print(f"制限適用: {len(files)} ファイル 処理do/does")
    
    if not files:
        print("No JSON files found in GCS bucket")
        return
    
    print(f"Found {len(files)} files in GCS bucket")
    
    # 全ファイルのハッシュ 集める
    global_example_hash_counter = Counter()
    global_example_hash_files = defaultdict(set)  # ハッシュ どのファイル 含まれてexistsか 追跡
    file_stats = []  # ファイルごとの統計 保存
    
    total_examples = 0
    total_steps = 0
    processed_files = 0
    
    for file_path in files:
        try:
            print(f"Processing {os.path.basename(file_path)}...")
            examples = download_gcs_file(bucket_name, file_path)
            
            file_examples = len(examples)
            file_steps = sum(len(example.get('steps', [])) for example in examples)
            
            total_examples += file_examples
            total_steps += file_steps
            processed_files += 1
            
            # ファイル内のallのハッシュ グローバルカウンター 追加
            for example in examples:
                example_hash_val = example.get('example_hash', '')
                if not example_hash_val:
                    original_goal = example.get('meta', {}).get('goal_original', '')
                    if original_goal:
                        example_hash_val = example_hash(original_goal)
                
                if example_hash_val:
                    global_example_hash_counter[example_hash_val] += 1
                    global_example_hash_files[example_hash_val].add(os.path.basename(file_path))
            
            print(f"  {os.path.basename(file_path)}: {file_examples} examples, {file_steps} steps")
            
            # ファイル統計 保存
            file_stats.append({
                'file_path': file_path,
                'examples': file_examples,
                'steps': file_steps
            })
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue
    
    # 重複 検出
    duplicates = {h: count for h, count in global_example_hash_counter.items() if count > 1}
    duplicate_count = len(duplicates)
    
    # 全体の結果 表示
    print(f"\n" + "="*60)
    print("GCS全体の重複チェック結果")
    print("="*60)
    
    print(f"処理ファイル数: {processed_files}")
    print(f"総例数: {total_examples:,}")
    print(f"総ステップ数: {total_steps:,}")
    
    print(f"\nハッシュ統計:")
    print(f"  ユニーク example_hash: {len(global_example_hash_counter):,}")
    
    print(f"\n重複検出:")
    print(f"  重複 example_hash: {duplicate_count:,}")
    
    # 重複の詳細表示
    if duplicate_count > 0:
        print(f"\n重複Exampleの詳細:")
        for hash_value, count in list(duplicates.items())[:20]:
            files_list = sorted(global_example_hash_files[hash_value])
            print(f"  {hash_value} (出現 times数: {count}, ファイル: {', '.join(files_list)})")
        if duplicate_count > 20:
            print(f"  ... 他 {duplicate_count - 20} ")
    else:
        print(f"\n✓ 重複は見かりません with/at did！")
    
    # 重複率の計算
    if total_examples > 0:
        duplicate_examples = sum(count - 1 for count in duplicates.values())
        duplicate_rate = (duplicate_examples / total_examples) * 100
        print(f"\n重複率: {duplicate_rate:.2f}% ({duplicate_examples:,}/{total_examples:,})")
        
        # ファイルごとの重複統計
        print(f"\n📊 ファイルごとの重複統計:")
        for i, stat in enumerate(file_stats, 1):
            file_duplicates = sum(1 for hash_val, (count, files_list) in duplicates.items() 
                                if count > 1 and os.path.basename(stat['file_path']) in files_list)
            file_examples = stat['examples']
            if file_examples > 0:
                file_duplicate_rate = (file_duplicates / file_examples) * 100
                print(f"  File {i:02d}: {file_duplicates}/{file_examples} duplicates ({file_duplicate_rate:.1f}%) - {os.path.basename(stat['file_path'])}")
            else:
                print(f"  File {i:02d}: 0/0 duplicates (N/A) - {os.path.basename(stat['file_path'])}")

def main():
    parser = argparse.ArgumentParser(description='GCSの全ファイル 横断and重複チェック 実行')
    parser.add_argument('--bucket', default='fof-data-20251009-milano', help='GCS bucket name')
    parser.add_argument('--prefix', default='generated_data/', help='GCS prefix')
    parser.add_argument('--limit', type=int, help='処理do/performファイル数の上限（最初のNのファイルonly）')
    
    args = parser.parse_args()
    
    check_gcs_cross_file_duplicates(args.bucket, args.prefix, args.limit)

if __name__ == "__main__":
    main()
