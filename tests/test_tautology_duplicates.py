#!/usr/bin/env python3
"""
GCSからトートロジーファイルをダウンロードして重複判定を行うテスト

使用方法:
python tests/test_tautology_duplicates.py --gcs_bucket fof-data-20251010-milano --gcs_prefix tautology
"""

import argparse
import json
import os
import subprocess
import tempfile
from typing import List, Dict, Set
from collections import Counter
import hashlib


def download_gcs_files(gcs_bucket: str, gcs_prefix: str, local_dir: str) -> List[str]:
    """GCSからファイルをダウンロードしてローカルファイルパスのリストを返す"""
    print(f"Downloading files from gs://{gcs_bucket}/{gcs_prefix}")
    
    # GCSのファイル一覧を取得
    try:
        result = subprocess.run([
            'gcloud', 'storage', 'ls', f'gs://{gcs_bucket}/{gcs_prefix}'
        ], capture_output=True, text=True, check=True)
        
        gcs_files = [line.strip() for line in result.stdout.split('\n') if line.strip().endswith('.json')]
        print(f"Found {len(gcs_files)} JSON files in GCS")
        
    except subprocess.CalledProcessError as e:
        print(f"Error listing GCS files: {e}")
        return []
    
    # ローカルにダウンロード
    local_files = []
    for gcs_file in gcs_files:
        filename = os.path.basename(gcs_file)
        local_path = os.path.join(local_dir, filename)
        
        try:
            subprocess.run([
                'gcloud', 'storage', 'cp', gcs_file, local_path
            ], capture_output=True, text=True, check=True)
            local_files.append(local_path)
            print(f"Downloaded: {filename}")
        except subprocess.CalledProcessError as e:
            print(f"Error downloading {filename}: {e}")
    
    return local_files


def load_formulas_from_files(file_paths: List[str]) -> List[str]:
    """複数のJSONファイルから論理式を読み込む"""
    all_formulas = []
    
    for file_path in file_paths:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    all_formulas.extend(data)
                    print(f"Loaded {len(data)} formulas from {os.path.basename(file_path)}")
                else:
                    print(f"Warning: {file_path} is not a list format")
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading {file_path}: {e}")
    
    return all_formulas


def analyze_duplicates(formulas: List[str]) -> Dict[str, any]:
    """論理式の重複を分析"""
    print(f"\nAnalyzing {len(formulas)} formulas...")
    
    # ハッシュを計算
    formula_hashes = []
    for formula in formulas:
        formula_hash = hashlib.md5(formula.encode()).hexdigest()
        formula_hashes.append(formula_hash)
    
    # 重複分析
    hash_counter = Counter(formula_hashes)
    unique_hashes = len(hash_counter)
    total_formulas = len(formulas)
    duplicates = total_formulas - unique_hashes
    
    # 重複の詳細
    duplicate_hashes = {h: count for h, count in hash_counter.items() if count > 1}
    duplicate_formulas = []
    
    for hash_val, count in duplicate_hashes.items():
        # このハッシュに対応する論理式を1つ取得
        for i, f_hash in enumerate(formula_hashes):
            if f_hash == hash_val:
                duplicate_formulas.append({
                    'formula': formulas[i],
                    'hash': hash_val,
                    'count': count
                })
                break
    
    # 統計
    duplicate_rate = (duplicates / total_formulas) * 100 if total_formulas > 0 else 0
    
    return {
        'total_formulas': total_formulas,
        'unique_formulas': unique_hashes,
        'duplicates': duplicates,
        'duplicate_rate': duplicate_rate,
        'duplicate_formulas': duplicate_formulas,
        'hash_counter': hash_counter
    }


def print_duplicate_analysis(analysis: Dict[str, any], show_examples: bool = True):
    """重複分析結果を表示"""
    print(f"\n{'='*60}")
    print(f"DUPLICATE ANALYSIS RESULTS")
    print(f"{'='*60}")
    print(f"Total formulas: {analysis['total_formulas']}")
    print(f"Unique formulas: {analysis['unique_formulas']}")
    print(f"Duplicates: {analysis['duplicates']}")
    print(f"Duplicate rate: {analysis['duplicate_rate']:.2f}%")
    
    if show_examples and analysis['duplicate_formulas']:
        print(f"\n{'='*60}")
        print(f"DUPLICATE EXAMPLES (showing first 10)")
        print(f"{'='*60}")
        
        for i, dup in enumerate(analysis['duplicate_formulas'][:10]):
            print(f"{i+1:2d}. Count: {dup['count']:2d} | Formula: {dup['formula']}")
            print(f"     Hash: {dup['hash']}")
            print()
    
    # 重複数の分布
    if analysis['duplicate_formulas']:
        print(f"{'='*60}")
        print(f"DUPLICATE COUNT DISTRIBUTION")
        print(f"{'='*60}")
        
        count_distribution = Counter([dup['count'] for dup in analysis['duplicate_formulas']])
        for count in sorted(count_distribution.keys()):
            print(f"  {count}x duplicates: {count_distribution[count]} formulas")


def analyze_file_distribution(file_paths: List[str]):
    """ファイルごとの分布を分析"""
    print(f"\n{'='*60}")
    print(f"FILE DISTRIBUTION ANALYSIS")
    print(f"{'='*60}")
    
    for file_path in file_paths:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    print(f"{os.path.basename(file_path):20s}: {len(data):4d} formulas")
                else:
                    print(f"{os.path.basename(file_path):20s}: Invalid format")
        except Exception as e:
            print(f"{os.path.basename(file_path):20s}: Error - {e}")


def main():
    parser = argparse.ArgumentParser(description="Analyze duplicates in GCS tautology files")
    parser.add_argument("--gcs_bucket", type=str, required=True, help="GCS bucket name")
    parser.add_argument("--gcs_prefix", type=str, required=True, help="GCS prefix")
    parser.add_argument("--show_examples", action="store_true", help="Show duplicate examples")
    parser.add_argument("--keep_files", action="store_true", help="Keep downloaded files after analysis")
    args = parser.parse_args()
    
    # 一時ディレクトリを作成
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Using temporary directory: {temp_dir}")
        
        # GCSからファイルをダウンロード
        local_files = download_gcs_files(args.gcs_bucket, args.gcs_prefix, temp_dir)
        
        if not local_files:
            print("No files downloaded. Exiting.")
            return
        
        # ファイル分布を分析
        analyze_file_distribution(local_files)
        
        # 論理式を読み込み
        formulas = load_formulas_from_files(local_files)
        
        if not formulas:
            print("No formulas loaded. Exiting.")
            return
        
        # 重複を分析
        analysis = analyze_duplicates(formulas)
        
        # 結果を表示
        print_duplicate_analysis(analysis, args.show_examples)
        
        # ファイルを保持する場合
        if args.keep_files:
            keep_dir = "downloaded_tautology_files"
            os.makedirs(keep_dir, exist_ok=True)
            for local_file in local_files:
                filename = os.path.basename(local_file)
                keep_path = os.path.join(keep_dir, filename)
                subprocess.run(['cp', local_file, keep_path])
            print(f"\nFiles kept in: {keep_dir}/")


if __name__ == "__main__":
    main()
