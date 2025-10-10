#!/usr/bin/env python3
"""
GCSに格納されたexampleのhashに重複がないかチェックするテストスクリプト
"""

import json
import argparse
from collections import defaultdict, Counter
from typing import Dict, List, Set, Tuple
import time
from google.cloud import storage
import os
import hashlib

def list_gcs_files(bucket_name: str, prefix: str) -> List[str]:
    """GCSバケットから指定されたプレフィックスのファイル一覧を取得"""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    
    files = []
    for blob in bucket.list_blobs(prefix=prefix):
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
        
        # 古い形式のrecord_hashもチェック（もしあれば）
        for step in example.get('steps', []):
            record_hash = step.get('record_hash', '')
            if record_hash:
                hashes.append((example_id, 'record_hash', record_hash))
    
    return hashes

def check_duplicate_hashes(bucket_name: str, prefix: str, max_files: int = None) -> Dict:
    """重複hashをチェック"""
    print(f"GCSバケット '{bucket_name}' からファイル一覧を取得中...")
    files = list_gcs_files(bucket_name, prefix)
    
    if max_files:
        files = files[:max_files]
    
    print(f"処理対象ファイル数: {len(files)}")
    
    # hashの種類別にカウンターを作成
    example_hash_counter = Counter()
    state_hash_counter = Counter()
    record_hash_counter = Counter()
    all_hashes = defaultdict(list)  # hash -> [(example_id, file_name), ...]
    
    total_examples = 0
    total_steps = 0
    processed_files = 0
    
    start_time = time.time()
    
    for i, file_path in enumerate(files):
        try:
            print(f"処理中 ({i+1}/{len(files)}): {file_path}")
            
            # ファイルをダウンロード
            examples = download_gcs_file(bucket_name, file_path)
            total_examples += len(examples)
            
            # hashを抽出
            file_hashes = extract_hashes_from_examples(examples)
            total_steps += len(file_hashes)
            
            # hashをカウント
            for example_id, hash_type, hash_value in file_hashes:
                if hash_type == 'example_hash':
                    example_hash_counter[hash_value] += 1
                elif hash_type == 'state_hash':
                    state_hash_counter[hash_value] += 1
                elif hash_type == 'record_hash':
                    record_hash_counter[hash_value] += 1
                
                all_hashes[hash_value].append((example_id, file_path))
            
            processed_files += 1
            
            # 進捗表示
            if (i + 1) % 10 == 0:
                elapsed = time.time() - start_time
                print(f"  進捗: {i+1}/{len(files)} ファイル処理完了 (経過時間: {elapsed:.1f}秒)")
                
        except Exception as e:
            print(f"エラー: {file_path} の処理中にエラーが発生: {e}")
            continue
    
    # 重複を検出
    duplicate_example_hashes = {h: count for h, count in example_hash_counter.items() if count > 1}
    duplicate_state_hashes = {h: count for h, count in state_hash_counter.items() if count > 1}
    duplicate_record_hashes = {h: count for h, count in record_hash_counter.items() if count > 1}
    
    # 全hashの重複チェック
    duplicate_all_hashes = {h: locations for h, locations in all_hashes.items() if len(locations) > 1}
    
    elapsed_time = time.time() - start_time
    
    return {
        'total_files': len(files),
        'processed_files': processed_files,
        'total_examples': total_examples,
        'total_steps': total_steps,
        'unique_example_hashes': len(example_hash_counter),
        'unique_state_hashes': len(state_hash_counter),
        'unique_record_hashes': len(record_hash_counter),
        'unique_all_hashes': len(all_hashes),
        'duplicate_example_hashes': len(duplicate_example_hashes),
        'duplicate_state_hashes': len(duplicate_state_hashes),
        'duplicate_record_hashes': len(duplicate_record_hashes),
        'duplicate_all_hashes': len(duplicate_all_hashes),
        'duplicate_example_hash_details': duplicate_example_hashes,
        'duplicate_state_hash_details': duplicate_state_hashes,
        'duplicate_record_hash_details': duplicate_record_hashes,
        'duplicate_all_hash_details': duplicate_all_hashes,
        'processing_time': elapsed_time
    }

def print_results(results: Dict):
    """結果を表示"""
    print("\n" + "="*60)
    print("重複チェック結果")
    print("="*60)
    
    print(f"処理ファイル数: {results['processed_files']}/{results['total_files']}")
    print(f"総例数: {results['total_examples']:,}")
    print(f"総ステップ数: {results['total_steps']:,}")
    print(f"処理時間: {results['processing_time']:.1f}秒")
    
    print(f"\nハッシュ統計:")
    print(f"  ユニーク example_hash: {results['unique_example_hashes']:,}")
    print(f"  ユニーク state_hash: {results['unique_state_hashes']:,}")
    print(f"  ユニーク record_hash: {results['unique_record_hashes']:,}")
    print(f"  ユニーク全ハッシュ: {results['unique_all_hashes']:,}")
    
    print(f"\n重複検出:")
    print(f"  重複 example_hash: {results['duplicate_example_hashes']:,}")
    print(f"  重複 state_hash: {results['duplicate_state_hashes']:,}")
    print(f"  重複 record_hash: {results['duplicate_record_hashes']:,}")
    print(f"  重複全ハッシュ: {results['duplicate_all_hashes']:,}")
    
    # Example重複の詳細表示
    if results['duplicate_example_hashes'] > 0:
        print(f"\n重複Exampleの詳細 (最初の10個):")
        for i, (hash_value, count) in enumerate(list(results['duplicate_example_hash_details'].items())[:10]):
            print(f"  {i+1}. {hash_value} (出現回数: {count})")
        if results['duplicate_example_hashes'] > 10:
            print(f"  ... 他 {results['duplicate_example_hashes'] - 10} 個の重複Example")
        print()
    
    # 全ハッシュ重複の詳細表示
    if results['duplicate_all_hashes'] > 0:
        print(f"\n重複ハッシュの詳細 (最初の10個):")
        for i, (hash_value, locations) in enumerate(list(results['duplicate_all_hash_details'].items())[:10]):
            print(f"  {i+1}. {hash_value}")
            print(f"     出現回数: {len(locations)}")
            for example_id, file_name in locations[:5]:  # 最初の5個の出現箇所
                print(f"     - {example_id} in {file_name}")
            if len(locations) > 5:
                print(f"     ... 他 {len(locations) - 5} 箇所")
            print()
    else:
        print("\n✓ 重複ハッシュは見つかりませんでした！")

def main():
    parser = argparse.ArgumentParser(description="GCSに格納されたexampleのhash重複をチェック（Example重複チェック対応）")
    parser.add_argument("--bucket", type=str, required=True, help="GCSバケット名")
    parser.add_argument("--prefix", type=str, required=True, help="GCSプレフィックス")
    parser.add_argument("--max_files", type=int, default=None, help="処理する最大ファイル数（テスト用）")
    
    args = parser.parse_args()
    
    print(f"GCSバケット: {args.bucket}")
    print(f"プレフィックス: {args.prefix}")
    if args.max_files:
        print(f"最大ファイル数: {args.max_files}")
    
    # 認証情報の確認
    if not os.path.exists(os.path.expanduser("~/.config/gcloud/application_default_credentials.json")):
        print("警告: GCS認証情報が見つかりません。gcloud auth application-default login を実行してください。")
        return
    
    try:
        results = check_duplicate_hashes(args.bucket, args.prefix, args.max_files)
        print_results(results)
        
        # 結果をJSONファイルに保存
        output_file = f"hash_duplicate_check_{int(time.time())}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n詳細結果を {output_file} に保存しました。")
        
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
