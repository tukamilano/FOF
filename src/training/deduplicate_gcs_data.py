#!/usr/bin/env python3
"""
GCS上のデータの重複排除スクリプト
state_hashからstate_tactic_hashを再計算して重複排除を実行
1Mオーダーのデータに対応する効率的なアルゴリズムを実装
"""
import argparse
import json
import os
import sys
import glob
import time
import hashlib
from typing import List, Dict, Any, Set, Optional, Tuple
from collections import Counter, defaultdict
from tqdm import tqdm
from google.cloud import storage
import tempfile
import shutil
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import threading
from queue import Queue

# プロジェクトルートをパスに追加
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, project_root)

from src.core.state_encoder import format_tactic_string


def state_tactic_hash(premises: List[str], goal: str, tactic: str) -> str:
    """
    状態とtacticの組み合わせのハッシュ
    重複チェックやデータ管理に使用
    
    Args:
        premises: 前提のリスト
        goal: ゴール
        tactic: 戦略文字列
        
    Returns:
        状態とtacticの組み合わせのハッシュ値
    """
    record_str = f"{'|'.join(premises)}|{goal}|{tactic}"
    return hashlib.md5(record_str.encode()).hexdigest()


def download_single_file(args: Tuple) -> Optional[str]:
    """
    単一ファイルのダウンロード（並列処理用）
    
    Args:
        args: (blob, local_dir) のタプル
        
    Returns:
        ダウンロードしたファイルのパス（失敗時はNone）
    """
    blob, local_dir = args
    
    try:
        # ローカルファイルパスを生成
        local_filename = os.path.basename(blob.name)
        local_path = os.path.join(local_dir, local_filename)
        
        # ファイルをダウンロード
        blob.download_to_filename(local_path)
        return local_path
        
    except Exception as e:
        print(f"❌ Error downloading {blob.name}: {e}")
        return None


def download_gcs_data_parallel(gcs_bucket: str, gcs_prefix: str, local_dir: str, 
                              max_workers: int = 4, verbose: bool = False) -> List[str]:
    """
    GCSからデータを並列ダウンロード
    
    Args:
        gcs_bucket: GCSバケット名
        gcs_prefix: GCSプレフィックス
        local_dir: ローカルダウンロード先ディレクトリ
        max_workers: 並列ダウンロードの最大ワーカー数
        verbose: 詳細ログを表示するかどうか
        
    Returns:
        ダウンロードしたファイルのパスリスト
    """
    print(f"📥 Downloading data from GCS (parallel)...")
    print(f"   Bucket: {gcs_bucket}")
    print(f"   Prefix: {gcs_prefix}")
    print(f"   Local directory: {local_dir}")
    print(f"   Max workers: {max_workers}")
    
    # ローカルディレクトリを作成
    os.makedirs(local_dir, exist_ok=True)
    
    # GCSクライアントを初期化
    client = storage.Client()
    bucket = client.bucket(gcs_bucket)
    
    # プレフィックスにマッチするファイルを取得
    blobs = bucket.list_blobs(prefix=gcs_prefix)
    json_files = [blob for blob in blobs if blob.name.endswith('.json')]
    
    if not json_files:
        print(f"❌ No JSON files found in gs://{gcs_bucket}/{gcs_prefix}")
        return []
    
    print(f"📁 Found {len(json_files)} JSON files in GCS")
    
    # 並列ダウンロードを実行
    downloaded_files = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # ダウンロードタスクを準備
        download_tasks = [(blob, local_dir) for blob in json_files]
        
        # 並列ダウンロードを実行
        with tqdm(total=len(download_tasks), desc="Downloading files", unit="file") as pbar:
            future_to_blob = {executor.submit(download_single_file, task): task[0] for task in download_tasks}
            
            for future in as_completed(future_to_blob):
                blob = future_to_blob[future]
                try:
                    result = future.result()
                    if result:
                        downloaded_files.append(result)
                        if verbose:
                            print(f"   ✅ Downloaded: {os.path.basename(blob.name)}")
                except Exception as e:
                    print(f"❌ Error downloading {blob.name}: {e}")
                finally:
                    pbar.update(1)
    
    print(f"✅ Downloaded {len(downloaded_files)} files")
    return downloaded_files


def download_gcs_data(gcs_bucket: str, gcs_prefix: str, local_dir: str, verbose: bool = False) -> List[str]:
    """
    GCSからデータをダウンロード（シーケンシャル版、後方互換性のため）
    
    Args:
        gcs_bucket: GCSバケット名
        gcs_prefix: GCSプレフィックス
        local_dir: ローカルダウンロード先ディレクトリ
        verbose: 詳細ログを表示するかどうか
        
    Returns:
        ダウンロードしたファイルのパスリスト
    """
    return download_gcs_data_parallel(gcs_bucket, gcs_prefix, local_dir, max_workers=1, verbose=verbose)


def process_single_file_worker(args: Tuple) -> Dict[str, Any]:
    """
    単一ファイルの処理（並列処理用、重複排除なし）
    
    Args:
        args: (file_path,) のタプル
        
    Returns:
        処理結果の辞書
    """
    file_path, = args
    
    try:
        # ファイルを読み込み
        with open(file_path, 'r', encoding='utf-8') as f:
            file_data = json.load(f)
        
        processed_steps = []
        file_stats = {
            'file_name': os.path.basename(file_path),
            'total_steps': 0,
            'processing_time': 0.0
        }
        
        start_time = time.time()
        
        # ファイル内の全ステップを処理（重複排除なし）
        for example in file_data:
            for step in example.get('steps', []):
                if step.get('tactic_apply', False):
                    file_stats['total_steps'] += 1
                    
                    # state_tactic_hashを生成
                    premises = step.get('premises', [])
                    goal = step.get('goal', '')
                    tactic_dict = step.get('tactic', {})
                    
                    try:
                        tactic_str = format_tactic_string(tactic_dict)
                        state_tactic_hash_val = state_tactic_hash(premises, goal, tactic_str)
                        
                        # state_tactic_hashをステップに追加
                        step['state_tactic_hash'] = state_tactic_hash_val
                        processed_steps.append(step)
                        
                    except Exception as e:
                        if verbose:
                            print(f"⚠️ Error processing step in {file_path}: {e}")
                        continue
        
        file_stats['processing_time'] = time.time() - start_time
        file_stats['processed_steps'] = processed_steps
        
        return file_stats
        
    except Exception as e:
        return {
            'file_name': os.path.basename(file_path),
            'error': str(e),
            'total_steps': 0,
            'processing_time': 0.0,
            'processed_steps': []
        }


def deduplicate_gcs_data(
    gcs_bucket: str,
    gcs_prefix: str,
    output_dir: str,
    temp_dir: Optional[str] = None,
    report_file: str = None,
    verbose: bool = False,
    batch_size: int = 10000,
    memory_efficient: bool = True,
    max_workers: int = None,
    parallel_download: bool = True
) -> Dict[str, Any]:
    """
    GCS上のデータの重複を除去し、重複排除済みデータを生成
    
    Args:
        gcs_bucket: GCSバケット名
        gcs_prefix: GCSプレフィックス
        output_dir: 出力ディレクトリ
        temp_dir: 一時ダウンロードディレクトリ（Noneの場合は自動生成）
        report_file: 統計レポートファイルのパス
        verbose: 詳細ログを表示するかどうか
        batch_size: バッチサイズ（デフォルト: 10000）
        memory_efficient: メモリ効率モード（デフォルト: True）
        max_workers: 並列処理の最大ワーカー数（Noneの場合は自動設定）
        parallel_download: 並列ダウンロードを使用するかどうか（デフォルト: True）
    
    Returns:
        重複排除統計情報
    """
    print(f"🔍 Starting GCS data deduplication process...")
    print(f"   GCS bucket: {gcs_bucket}")
    print(f"   GCS prefix: {gcs_prefix}")
    print(f"   Output directory: {output_dir}")
    print(f"   Batch size: {batch_size}")
    print(f"   Memory efficient: {memory_efficient}")
    print(f"   Parallel download: {parallel_download}")
    
    # 並列処理のワーカー数を設定
    if max_workers is None:
        max_workers = min(mp.cpu_count(), 8)  # 最大8プロセス
    print(f"   Max workers: {max_workers}")
    
    # 出力ディレクトリを作成
    os.makedirs(output_dir, exist_ok=True)
    
    # 一時ディレクトリを設定
    if temp_dir is None:
        temp_dir = tempfile.mkdtemp(prefix="gcs_dedup_")
        cleanup_temp = True
    else:
        os.makedirs(temp_dir, exist_ok=True)
        cleanup_temp = False
    
    try:
        # GCSからデータをダウンロード
        if parallel_download:
            input_files = download_gcs_data_parallel(gcs_bucket, gcs_prefix, temp_dir, 
                                                   max_workers=max_workers, verbose=verbose)
        else:
            input_files = download_gcs_data(gcs_bucket, gcs_prefix, temp_dir, verbose)
        
        if not input_files:
            print(f"❌ No files downloaded from GCS")
            return {}
        
        print(f"📁 Processing {len(input_files)} downloaded files")
        
        # グローバル重複排除用のセット（シーケンシャル処理用）
        seen_hashes: Set[str] = set()
        
        # バッチ処理用の変数
        current_batch = []
        file_counter = 0
        
        # 統計情報
        stats = {
            'total_files': len(input_files),
            'total_steps_before': 0,
            'total_steps_after': 0,
            'duplicate_steps': 0,
            'duplicate_rate': 0.0,
            'output_files': 0,
            'duplicate_hash_counts': Counter(),
            'processing_time': 0.0,
            'memory_usage_mb': 0.0,
            'parallel_processing': True,
            'max_workers': max_workers
        }
        
        start_time = time.time()
        
        def save_batch(batch_data: List[Dict], batch_num: int) -> None:
            """バッチデータをファイルに保存"""
            output_file = os.path.join(output_dir, f"deduplicated_batch_{batch_num:05d}.json")
            try:
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(batch_data, f, ensure_ascii=False, indent=2)
                if verbose:
                    print(f"   💾 Saved batch {batch_num:05d}: {len(batch_data)} steps")
            except Exception as e:
                print(f"❌ Error saving batch {batch_num:05d}: {e}")
        
        def process_step(step: Dict) -> Optional[str]:
            """ステップからstate_tactic_hashを生成"""
            try:
                premises = step.get('premises', [])
                goal = step.get('goal', '')
                tactic_dict = step.get('tactic', {})
                
                # tactic文字列を生成
                tactic_str = format_tactic_string(tactic_dict)
                
                # state_tactic_hashを計算
                return state_tactic_hash(premises, goal, tactic_str)
                
            except Exception as e:
                if verbose:
                    print(f"⚠️ Error processing step: {e}")
                return None
        
        # 並列処理モードの実装
        if max_workers > 1:
            print(f"🔄 Processing files in parallel mode with {max_workers} workers...")
            
            # 並列処理用のタスクを準備
            file_tasks = [(file_path,) for file_path in input_files]
            
            all_processed_steps = []
            
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # 並列処理を実行
                with tqdm(total=len(file_tasks), desc="Processing files", unit="file") as pbar:
                    future_to_file = {executor.submit(process_single_file_worker, task): task[0] for task in file_tasks}
                    
                    for future in as_completed(future_to_file):
                        file_path = future_to_file[future]
                        try:
                            result = future.result()
                            
                            # 統計情報を更新
                            stats['total_steps_before'] += result['total_steps']
                            
                            # 処理されたステップを収集
                            if 'processed_steps' in result:
                                all_processed_steps.extend(result['processed_steps'])
                            
                            if verbose:
                                print(f"   ✅ Processed {result['file_name']}: "
                                      f"{result['total_steps']} steps "
                                      f"({result['processing_time']:.2f}s)")
                            
                        except Exception as e:
                            print(f"❌ Error processing {file_path}: {e}")
                        finally:
                            pbar.update(1)
            
            # グローバル重複排除を実行（並列処理後）
            print(f"🔄 Performing global deduplication on {len(all_processed_steps)} steps...")
            global_unique_steps = []
            global_duplicate_count = 0
            
            for step in tqdm(all_processed_steps, desc="Global deduplication"):
                state_tactic_hash_val = step.get('state_tactic_hash', '')
                
                if state_tactic_hash_val in seen_hashes:
                    global_duplicate_count += 1
                    stats['duplicate_hash_counts'][state_tactic_hash_val] += 1
                else:
                    seen_hashes.add(state_tactic_hash_val)
                    global_unique_steps.append(step)
            
            # グローバル重複統計を更新
            stats['duplicate_steps'] = global_duplicate_count
            stats['total_steps_after'] = len(global_unique_steps)
            
            print(f"💾 Saving {len(global_unique_steps)} unique steps in batches...")
            
            # バッチに分割して保存（逐次書き込み）
            for i in range(0, len(global_unique_steps), batch_size):
                batch = global_unique_steps[i:i + batch_size]
                save_batch(batch, file_counter)
                file_counter += 1
                
                if verbose and file_counter % 10 == 0:
                    print(f"   💾 Saved batch {file_counter}: {len(batch)} steps")
        
        else:
            # シーケンシャル処理モード（元の実装）
            print(f"🔄 Processing files in sequential mode...")
            
            for file_idx, input_file in enumerate(tqdm(input_files, desc="Processing files")):
                if verbose:
                    print(f"\n📄 Processing {os.path.basename(input_file)}...")
                
                # JSONファイルを読み込み
                try:
                    with open(input_file, 'r', encoding='utf-8') as f:
                        file_data = json.load(f)
                except Exception as e:
                    print(f"❌ Error reading {input_file}: {e}")
                    continue
                
                # ファイル内の全ステップを処理
                for example in file_data:
                    for step in example.get('steps', []):
                        if step.get('tactic_apply', False):
                            stats['total_steps_before'] += 1
                            
                            # state_tactic_hashを生成
                            premises = step.get('premises', [])
                            goal = step.get('goal', '')
                            tactic_dict = step.get('tactic', {})
                            
                            try:
                                tactic_str = format_tactic_string(tactic_dict)
                                state_tactic_hash_val = state_tactic_hash(premises, goal, tactic_str)
                                
                                if state_tactic_hash_val in seen_hashes:
                                    # 重複をスキップ
                                    stats['duplicate_steps'] += 1
                                    stats['duplicate_hash_counts'][state_tactic_hash_val] += 1
                                else:
                                    # 新しいステップを追加
                                    seen_hashes.add(state_tactic_hash_val)
                                    
                                    # state_tactic_hashをステップに追加
                                    step['state_tactic_hash'] = state_tactic_hash_val
                                    current_batch.append(step)
                                    stats['total_steps_after'] += 1
                                    
                                    # バッチサイズに達したらファイルに保存
                                    if len(current_batch) >= batch_size:
                                        save_batch(current_batch, file_counter)
                                        file_counter += 1
                                        current_batch = []
                                        
                                        # メモリ使用量をチェック
                                        if file_idx % 10 == 0:  # 10ファイルごとにチェック
                                            import psutil
                                            process = psutil.Process()
                                            memory_mb = process.memory_info().rss / 1024 / 1024
                                            stats['memory_usage_mb'] = memory_mb
                                            if verbose:
                                                print(f"   📊 Memory usage: {memory_mb:.1f} MB")
                                
                            except Exception as e:
                                if verbose:
                                    print(f"⚠️ Error processing step: {e}")
                                continue
        
        # 残りのバッチを保存（シーケンシャルモードの場合のみ）
        if max_workers == 1 and current_batch:
            save_batch(current_batch, file_counter)
            file_counter += 1
        
        stats['output_files'] = file_counter
        
        # 全体統計を計算
        stats['processing_time'] = time.time() - start_time
        stats['duplicate_rate'] = (
            stats['duplicate_steps'] / stats['total_steps_before'] * 100
            if stats['total_steps_before'] > 0 else 0.0
        )
        
        # 統計レポートを表示
        print(f"\n📊 Deduplication Summary")
        print(f"   Input files processed: {stats['total_files']}")
        print(f"   Output files created: {stats['output_files']}")
        print(f"   Total steps before: {stats['total_steps_before']:,}")
        print(f"   Total steps after: {stats['total_steps_after']:,}")
        print(f"   Duplicates removed: {stats['duplicate_steps']:,}")
        print(f"   Duplicate rate: {stats['duplicate_rate']:.2f}%")
        print(f"   Processing time: {stats['processing_time']:.2f}s")
        print(f"   Average steps per output file: {stats['total_steps_after'] / stats['output_files']:.0f}")
        if stats['memory_usage_mb'] > 0:
            print(f"   Peak memory usage: {stats['memory_usage_mb']:.1f} MB")
        if stats.get('parallel_processing', False):
            print(f"   Parallel processing: {stats['max_workers']} workers")
            print(f"   Processing speed: {stats['total_steps_before'] / stats['processing_time']:.0f} steps/sec")
        
        # 最も重複が多いハッシュを表示
        if stats['duplicate_hash_counts']:
            print(f"\n🔍 Top 10 Most Duplicated States:")
            for hash_val, count in stats['duplicate_hash_counts'].most_common(10):
                print(f"   Hash: {hash_val[:16]}... Count: {count}")
        
        # 統計レポートを保存
        if report_file:
            try:
                with open(report_file, 'w', encoding='utf-8') as f:
                    json.dump(stats, f, ensure_ascii=False, indent=2)
                print(f"\n📄 Statistics saved to: {report_file}")
            except Exception as e:
                print(f"❌ Error saving report: {e}")
        
        return stats
        
    finally:
        # 一時ディレクトリをクリーンアップ
        if cleanup_temp and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
                if verbose:
                    print(f"🧹 Cleaned up temporary directory: {temp_dir}")
            except Exception as e:
                print(f"⚠️ Warning: Could not clean up temporary directory {temp_dir}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Deduplicate GCS data before training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic GCS deduplication
  python src/training/deduplicate_gcs_data.py \\
    --gcs_bucket fof-data-20251009-milano \\
    --gcs_prefix generated_data/ \\
    --output_dir deduplicated_data
  
  # With detailed report and custom settings
  python src/training/deduplicate_gcs_data.py \\
    --gcs_bucket fof-data-20251009-milano \\
    --gcs_prefix generated_data/ \\
    --output_dir deduplicated_data \\
    --report_file gcs_deduplication_report.json \\
    --batch_size 20000 \\
    --max_workers 4 \\
    --verbose

  # High-performance parallel processing
  python src/training/deduplicate_gcs_data.py \\
    --gcs_bucket fof-data-20251009-milano \\
    --gcs_prefix generated_data/ \\
    --output_dir deduplicated_data \\
    --max_workers 8 \\
    --batch_size 50000 \\
    --verbose
        """
    )
    
    parser.add_argument(
        "--gcs_bucket", 
        type=str, 
        required=True,
        help="GCS bucket name"
    )
    parser.add_argument(
        "--gcs_prefix", 
        type=str, 
        required=True,
        help="GCS prefix for data files"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="deduplicated_data",
        help="Output directory for deduplicated files (default: deduplicated_data)"
    )
    parser.add_argument(
        "--temp_dir", 
        type=str, 
        default=None,
        help="Temporary directory for downloading GCS files (default: auto-generated)"
    )
    parser.add_argument(
        "--report_file", 
        type=str, 
        default="gcs_deduplication_report.json",
        help="Path to save deduplication statistics report (default: gcs_deduplication_report.json)"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Show detailed processing information"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=10000,
        help="Number of steps per output file (default: 10000)"
    )
    parser.add_argument(
        "--no_memory_efficient", 
        action="store_true",
        help="Disable memory-efficient mode (load all data into memory)"
    )
    parser.add_argument(
        "--max_workers", 
        type=int, 
        default=None,
        help="Maximum number of parallel workers (default: min(cpu_count, 8))"
    )
    parser.add_argument(
        "--no_parallel_download", 
        action="store_true",
        help="Disable parallel download from GCS"
    )
    
    args = parser.parse_args()
    
    # 重複排除を実行
    try:
        stats = deduplicate_gcs_data(
            gcs_bucket=args.gcs_bucket,
            gcs_prefix=args.gcs_prefix,
            output_dir=args.output_dir,
            temp_dir=args.temp_dir,
            report_file=args.report_file,
            verbose=args.verbose,
            batch_size=args.batch_size,
            memory_efficient=not args.no_memory_efficient,
            max_workers=args.max_workers,
            parallel_download=not args.no_parallel_download
        )
        
        if stats:
            print(f"\n✅ GCS deduplication completed successfully!")
            print(f"   Output directory: {args.output_dir}")
            if args.report_file:
                print(f"   Report file: {args.report_file}")
        else:
            print(f"❌ GCS deduplication failed!")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Error during GCS deduplication: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
