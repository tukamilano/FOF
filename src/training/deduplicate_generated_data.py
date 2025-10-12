#!/usr/bin/env python3
"""
generated_dataディレクトリの重複排除スクリプト
学習前に重複を事前に除去し、重複排除済みデータを生成する
"""
import argparse
import json
import os
import sys
import glob
import time
from typing import List, Dict, Any, Set
from collections import Counter, defaultdict
from tqdm import tqdm

# プロジェクトルートをパスに追加
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, project_root)


def deduplicate_generated_data(
    input_dir: str,
    output_dir: str,
    report_file: str = None,
    verbose: bool = False,
    batch_size: int = 10000
) -> Dict[str, Any]:
    """
    generated_dataディレクトリの重複を除去し、重複排除済みデータを生成
    
    Args:
        input_dir: 入力ディレクトリ（generated_data）
        output_dir: 出力ディレクトリ（deduplicated_data）
        report_file: 統計レポートファイルのパス
        verbose: 詳細ログを表示するかどうか
        batch_size: バッチサイズ（デフォルト: 10000）
    
    Returns:
        重複排除統計情報
    """
    print(f"🔍 Starting deduplication process...")
    print(f"   Input directory: {input_dir}")
    print(f"   Output directory: {output_dir}")
    print(f"   Batch size: {batch_size}")
    
    # 出力ディレクトリを作成
    os.makedirs(output_dir, exist_ok=True)
    
    # 入力ファイルを取得
    input_files = glob.glob(os.path.join(input_dir, "*.json"))
    if not input_files:
        print(f"❌ No JSON files found in {input_dir}")
        return {}
    
    print(f"📁 Found {len(input_files)} JSON files")
    
    # グローバル重複排除用のセット
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
        'processing_time': 0.0
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
    
    # 全ファイルを統合してグローバル重複排除を実行
    print(f"🔄 Processing all files for global deduplication...")
    
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
                    
                    state_tactic_hash = step.get('state_tactic_hash', '')
                    
                    if state_tactic_hash in seen_hashes:
                        # 重複をスキップ
                        stats['duplicate_steps'] += 1
                        stats['duplicate_hash_counts'][state_tactic_hash] += 1
                    else:
                        # 新しいステップを追加
                        seen_hashes.add(state_tactic_hash)
                        current_batch.append(step)
                        stats['total_steps_after'] += 1
                        
                        # バッチサイズに達したらファイルに保存
                        if len(current_batch) >= batch_size:
                            save_batch(current_batch, file_counter)
                            file_counter += 1
                            current_batch = []
    
    # 残りのバッチを保存
    if current_batch:
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
    print(f"   Total steps before: {stats['total_steps_before']}")
    print(f"   Total steps after: {stats['total_steps_after']}")
    print(f"   Duplicates removed: {stats['duplicate_steps']}")
    print(f"   Duplicate rate: {stats['duplicate_rate']:.2f}%")
    print(f"   Processing time: {stats['processing_time']:.2f}s")
    print(f"   Average steps per output file: {stats['total_steps_after'] / stats['output_files']:.0f}")
    
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


def main():
    parser = argparse.ArgumentParser(
        description="Deduplicate generated data before training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic deduplication
  python src/training/deduplicate_generated_data.py --input_dir generated_data --output_dir deduplicated_data
  
  # With detailed report
  python src/training/deduplicate_generated_data.py \\
    --input_dir generated_data \\
    --output_dir deduplicated_data \\
    --report_file deduplication_report.json \\
    --verbose
        """
    )
    
    parser.add_argument(
        "--input_dir", 
        type=str, 
        default="generated_data",
        help="Input directory containing JSON files (default: generated_data)"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="deduplicated_data",
        help="Output directory for deduplicated files (default: deduplicated_data)"
    )
    parser.add_argument(
        "--report_file", 
        type=str, 
        default="deduplication_report.json",
        help="Path to save deduplication statistics report (default: deduplication_report.json)"
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
    
    args = parser.parse_args()
    
    # 入力ディレクトリの存在確認
    if not os.path.exists(args.input_dir):
        print(f"❌ Input directory not found: {args.input_dir}")
        sys.exit(1)
    
    # 重複排除を実行
    try:
        stats = deduplicate_generated_data(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            report_file=args.report_file,
            verbose=args.verbose,
            batch_size=args.batch_size
        )
        
        if stats:
            print(f"\n✅ Deduplication completed successfully!")
            print(f"   Output directory: {args.output_dir}")
            if args.report_file:
                print(f"   Report file: {args.report_file}")
        else:
            print(f"❌ Deduplication failed!")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Error during deduplication: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
