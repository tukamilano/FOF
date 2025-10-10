#!/usr/bin/env python3
"""
GCSの全ファイルを横断して重複チェックを実行するスクリプト
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
    """Example全体の重複チェック用ハッシュを生成（元の目標式のみ）"""
    return hashlib.md5(original_goal.encode()).hexdigest()

def list_gcs_files(bucket_name: str, prefix: str) -> List[str]:
    """GCSバケットから指定プレフィックスのファイル一覧を取得"""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    
    blobs = bucket.list_blobs(prefix=prefix)
    files = []
    
    for blob in blobs:
        if blob.name.endswith('.json'):
            files.append(blob.name)
    
    return sorted(files)

def download_gcs_file(bucket_name: str, file_path: str) -> List[Dict]:
    """GCSからファイルをダウンロードしてJSONとして読み込み"""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(file_path)
    
    content = blob.download_as_text()
    return json.loads(content)

def check_gcs_cross_file_duplicates(bucket_name: str, prefix: str):
    """GCSの全ファイルを横断して重複チェックを実行"""
    print(f"GCS Bucket: {bucket_name}")
    print(f"Prefix: {prefix}")
    print("="*60)
    
    # GCSファイル一覧を取得
    print("GCSファイル一覧を取得中...")
    files = list_gcs_files(bucket_name, prefix)
    
    if not files:
        print("No JSON files found in GCS bucket")
        return
    
    print(f"Found {len(files)} files in GCS bucket")
    
    # 全ファイルのハッシュを集める
    global_example_hash_counter = Counter()
    global_example_hash_files = defaultdict(set)  # ハッシュがどのファイルに含まれているかを追跡
    
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
            
            # ファイル内のすべてのハッシュをグローバルカウンターに追加
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
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue
    
    # 重複を検出
    duplicates = {h: count for h, count in global_example_hash_counter.items() if count > 1}
    duplicate_count = len(duplicates)
    
    # 全体の結果を表示
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
            print(f"  {hash_value} (出現回数: {count}, ファイル: {', '.join(files_list)})")
        if duplicate_count > 20:
            print(f"  ... 他 {duplicate_count - 20} 個")
    else:
        print(f"\n✓ 重複は見つかりませんでした！")
    
    # 重複率の計算
    if total_examples > 0:
        duplicate_examples = sum(count - 1 for count in duplicates.values())
        duplicate_rate = (duplicate_examples / total_examples) * 100
        print(f"\n重複率: {duplicate_rate:.2f}% ({duplicate_examples:,}/{total_examples:,})")

def main():
    parser = argparse.ArgumentParser(description='GCSの全ファイルを横断して重複チェックを実行')
    parser.add_argument('--bucket', default='fof-data-20251009-milano', help='GCS bucket name')
    parser.add_argument('--prefix', default='generated_data/', help='GCS prefix')
    
    args = parser.parse_args()
    
    check_gcs_cross_file_duplicates(args.bucket, args.prefix)

if __name__ == "__main__":
    main()
