from __future__ import annotations

import argparse
import os
import sys
import json
import hashlib
import time
import multiprocessing as mp
import subprocess
from typing import List, Tuple, Dict, Any, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# プロジェクトルートをパスに追加
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, project_root)

from src.core.generate_prop import FormulaGenerator, filter_formulas
from src.core.parameter import (
    default_params, get_generation_params, get_training_params, 
    get_system_params, DeviceType, DataFilterType
)


def example_hash(formula: str) -> str:
    """論理式の重複チェック用ハッシュを生成"""
    return hashlib.md5(formula.encode()).hexdigest()


def process_single_formula_worker(args: Tuple) -> Dict[str, Any]:
    """単一の論理式を処理するワーカー関数
    
    Args:
        args: Tuple containing (formula_data, worker_id)
    
    Returns:
        Dictionary containing the processing result
    """
    formula_data, worker_id = args
    
    try:
        formula = formula_data['formula']
        
        # 論理式が空の場合はスキップ
        if not formula:
            return {
                'formula': '',
                'is_tautology': False,
                'worker_id': worker_id,
                'error': 'Empty formula'
            }
        
        return {
            'formula': formula,
            'is_tautology': True,  # filter_formulasで既にトートロジーがフィルタリングされている
            'worker_id': worker_id,
            'formula_hash': example_hash(formula)
        }
        
    except Exception as e:
        return {
            'formula': formula_data.get('formula', ''),
            'is_tautology': False,
            'worker_id': worker_id,
            'error': str(e),
            'formula_hash': ''
        }


class TautologyGenerator:
    """トートロジーな論理式を生成してJSONに格納するクラス"""
    
    def __init__(self, 
                 dataset_file_path: str = "tautology_data.json",
                 num_workers: int = None,
                 examples_per_file: int = 10000,
                 buffer_size: int = 1000,
                 check_duplicates: bool = True,
                 gcs_bucket: str = None,
                 gcs_prefix: str = ""):
        self.dataset_file_path = dataset_file_path
        # Default to CPU count, but allow override via num_workers parameter
        # Conservative limit of 8 to prevent memory issues
        self.num_workers = num_workers or min(mp.cpu_count(), 8)
        self.examples_per_file = examples_per_file
        self.buffer_size = min(buffer_size, examples_per_file)
        self.check_duplicates = check_duplicates
        
        # データ管理
        self.all_formulas = []
        self.current_file_index = 1
        self.formulas_in_current_file = 0
        self.buffer_formulas = 0
        self.total_generated = 0
        
        # 重複チェック用
        self.formula_hashes = set()
        self.global_hashes_file = "global_tautology_hashes.json"
        self.load_global_hashes()
        
        # GCS設定
        self.gcs_bucket = gcs_bucket
        self.gcs_prefix = gcs_prefix
    
    def clear_global_hashes(self):
        """グローバルハッシュファイルを削除してリセット"""
        if os.path.exists(self.global_hashes_file):
            os.remove(self.global_hashes_file)
            print(f"Cleared global hashes file: {self.global_hashes_file}")
        
        # メモリ内のハッシュもリセット
        self.formula_hashes = set()
        self.current_file_index = 1
        print("Reset global hash state")
    
    def load_global_hashes(self):
        """既存のグローバルハッシュを読み込み"""
        if os.path.exists(self.global_hashes_file):
            try:
                with open(self.global_hashes_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # シンプルな形式：ハッシュのリストのみ
                    if isinstance(data, list):
                        self.formula_hashes = set(data)
                    else:
                        # 旧形式との互換性
                        self.formula_hashes = set(data.get('formula_hashes', []))
                print(f"Loaded {len(self.formula_hashes)} existing formula hashes from {self.global_hashes_file}")
            except (FileNotFoundError, json.JSONDecodeError) as e:
                print(f"Could not load global hashes: {e}")
                self.formula_hashes = set()
        else:
            self.formula_hashes = set()
    
    def save_global_hashes(self):
        """グローバルハッシュを保存"""
        # シンプルな形式：ハッシュのリストのみ
        with open(self.global_hashes_file, 'w', encoding='utf-8') as f:
            json.dump(list(self.formula_hashes), f, ensure_ascii=False, indent=2)
    
    def get_current_filename(self) -> str:
        """現在のファイル名を取得"""
        base_name = os.path.basename(self.dataset_file_path).replace('.json', '')
        return f"{base_name}_{self.current_file_index:05d}.json"
    
    def clear_generated_data(self):
        """既存のgenerated_dataディレクトリをクリア"""
        generated_dir = "generated_data"
        if os.path.exists(generated_dir):
            import shutil
            shutil.rmtree(generated_dir)
            print(f"Cleared: {generated_dir}/")
        os.makedirs(generated_dir, exist_ok=True)
        
        if self.gcs_bucket:
            print(f"GCS upload: gs://{self.gcs_bucket}/{self.gcs_prefix}")
    
    def upload_file_to_gcs(self, local_file_path: str, gcs_filename: str) -> bool:
        """ローカルファイルをGCSバケットにアップロード"""
        if not self.gcs_bucket:
            return False
        
        try:
            # GCSパスを構築
            gcs_path = f"gs://{self.gcs_bucket}/{self.gcs_prefix}{gcs_filename}"
            
            # gcloudコマンドでアップロード
            result = subprocess.run([
                'gcloud', 'storage', 'cp', local_file_path, gcs_path
            ], capture_output=True, text=True, check=True)
            
            print(f"Uploaded: {gcs_filename}")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"Upload failed: {gcs_filename}")
            return False
        except Exception as e:
            print(f"Upload failed: {gcs_filename}")
            return False
    
    def generate_tautologies_parallel(self, gen, gen_params) -> List[Dict]:
        """並列でトートロジーを生成"""
        print(f"Starting parallel tautology generation with {self.num_workers} workers...")
        
        results = []
        successful_formulas = 0
        processed_count = 0
        skipped_duplicates = 0
        batch_size = self.num_workers * 4  # Process 4x worker count at a time
        
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            with tqdm(total=gen_params.count, desc="Generating tautologies", unit="formula", 
                     bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}') as pbar:
                while processed_count < gen_params.count:
                    # バッチの論理式を生成
                    batch_formulas = []
                    batch_count = min(batch_size, gen_params.count - processed_count)
                    
                    successful_count = 0
                    for i in range(batch_count):
                        # require_tautology=Trueでトートロジーのみを生成
                        goal_list = filter_formulas(gen, max_len=gen_params.max_len, require_tautology=True, limit=1)
                        if goal_list:
                            goal = goal_list[0]
                            # 重複チェックは保存時に統合して行うため、ここではスキップ
                            
                            batch_formulas.append({
                                "formula": goal,
                                "index": successful_count
                            })
                            successful_count += 1
                        else:
                            # 有効な論理式がない場合はプレースホルダー
                            batch_formulas.append({
                                "formula": "",
                                "index": successful_count
                            })
                            successful_count += 1
                    
                    # バッチを並列処理
                    worker_args = [
                        (formula, os.getpid()) 
                        for formula in batch_formulas
                    ]
                    
                    # バッチタスクを送信
                    future_to_index = {
                        executor.submit(process_single_formula_worker, args): formula["index"] 
                        for args, formula in zip(worker_args, batch_formulas)
                    }
                    
                    # 完了したタスクを処理
                    for future in as_completed(future_to_index):
                        index = future_to_index[future]
                        try:
                            result = future.result()
                            results.append(result)
                            
                            # 成功した論理式を収集
                            if result.get('is_tautology', False) and result.get('formula'):
                                self.add_formula_and_check_save(result)
                                successful_formulas += 1
                                        
                        except Exception as e:
                            results.append({
                                'formula': '',
                                'is_tautology': False,
                                'worker_id': os.getpid(),
                                'error': str(e)
                            })
                        
                        pbar.update(1)
                        processed_count += 1
                        
                        # プログレスバーを更新
                        if processed_count % 100 == 0:
                            pbar.set_postfix({
                                'file': f"{self.current_file_index:05d}",
                                'formulas': len(self.all_formulas),
                                'skipped': skipped_duplicates
                            })
                        
                        # 十分に処理したら終了
                        if processed_count >= gen_params.count:
                            break
                    
                    # バッチ処理完了
                    
                    # 十分に処理したら終了
                    if processed_count >= gen_params.count:
                        break
        
        # 結果を論理式でソートして順序を維持
        results.sort(key=lambda x: x.get('formula', ''))
        
        print(f"Completed: {successful_formulas}/{gen_params.count} tautologies generated")
        print(f"Duplicates skipped: {skipped_duplicates} ({skipped_duplicates/gen_params.count*100:.1f}%)")
        print(f"Global unique formulas: {len(self.formula_hashes)}")
        
        return results
    
    def add_formula_and_check_save(self, formula_data: Dict):
        """論理式を追加し、ファイル制限に達したら保存"""
        # 重複チェックは保存時に統合して行うため、ここでは単純に追加
        self.all_formulas.append(formula_data)
        self.formulas_in_current_file += 1
        self.buffer_formulas += 1
        
        # バッファサイズまたはファイルサイズのいずれかに達したら保存
        if (self.buffer_formulas >= self.buffer_size or 
            self.buffer_formulas >= self.examples_per_file):
            self.save_current_data()
    
    def save_current_data(self):
        """現在のデータを保存"""
        if not self.all_formulas:
            return
            
        # データを変換
        transformed_data = self.transform_to_output_format(self.all_formulas)
        num_formulas = len(transformed_data)
        
        # バッファの例数カウンターをリセット
        self.buffer_formulas = 0
        
        # グローバル統計を更新
        self.total_generated += len(self.all_formulas)
        
        # ローカルファイルに保存
        local_file_path = self._save_to_local_file(transformed_data)
        
        # バッファをリセット
        self.all_formulas = []
        self.formulas_in_current_file = 0
    
    def transform_to_output_format(self, formulas: List[Dict]) -> List[str]:
        """出力形式に変換 - 論理式の文字列のみを返す"""
        transformed_data = []
        seen_hashes = set()  # 重複チェック用
        
        for i, formula_data in enumerate(formulas):
            formula = formula_data.get('formula', '')
            if not formula:
                continue
                
            formula_hash_val = formula_data.get('formula_hash', example_hash(formula))
            
            # 重複チェック
            if self.check_duplicates:
                if formula_hash_val in seen_hashes:
                    continue
                seen_hashes.add(formula_hash_val)
            
            transformed_data.append(formula)
        
        return transformed_data
    
    def _save_to_local_file(self, transformed_data: List[str]) -> str:
        """ローカルファイルに保存してファイルパスを返す"""
        filename = self.get_current_filename()
        num_formulas = len(transformed_data)
        
        # 現在のファイルが存在し、スペースがあるかチェック
        local_file_path = os.path.join("generated_data", filename)
        if os.path.exists(local_file_path):
            try:
                with open(local_file_path, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                if len(existing_data) + num_formulas > self.examples_per_file:
                    # 現在のファイルが満杯になるので、新しいファイルを作成
                    self.current_file_index += 1
                    filename = self.get_current_filename()
                    local_file_path = os.path.join("generated_data", filename)
                    existing_data = []
            except (FileNotFoundError, json.JSONDecodeError):
                existing_data = []
        else:
            existing_data = []
        
        # 統合重複チェック（グローバル + ファイル内）
        filtered_data = []
        duplicates_removed = 0
        
        for formula in transformed_data:
            formula_hash = example_hash(formula)
            
            # グローバル重複チェック（既にadd_formula_and_check_saveでチェック済みだが、念のため）
            if self.check_duplicates and formula_hash in self.formula_hashes:
                duplicates_removed += 1
                continue
            
            # ファイル内重複チェック
            if any(example_hash(existing_formula) == formula_hash for existing_formula in existing_data):
                duplicates_removed += 1
                continue
            
            # 重複なし - データを追加
            filtered_data.append(formula)
            # グローバルハッシュに追加（まだ追加されていない場合）
            if self.check_duplicates:
                self.formula_hashes.add(formula_hash)
        
        if duplicates_removed > 0:
            duplicate_rate = (duplicates_removed / len(transformed_data)) * 100
            print(f"  📊 File {self.current_file_index:05d}: Removed {duplicates_removed}/{len(transformed_data)} duplicates ({duplicate_rate:.1f}%)")
        
        transformed_data = filtered_data
        num_formulas = len(transformed_data)
        
        # 新しいデータを追加
        existing_data.extend(transformed_data)
        
        # ファイルに書き込み
        with open(local_file_path, 'w', encoding='utf-8') as f:
            json.dump(existing_data, f, ensure_ascii=False, indent=2)
        
        # 追跡を更新
        self.formulas_in_current_file = len(existing_data)
        
        # グローバルハッシュを保存
        self.save_global_hashes()
        
        # GCSにアップロード
        if self.gcs_bucket:
            self.upload_file_to_gcs(local_file_path, filename)
        
        # ファイル保存時の統計を表示
        if len(existing_data) == num_formulas:
            print(f"Created: {filename} ({num_formulas} formulas)")
        else:
            print(f"Appended: {filename} (+{num_formulas}, total: {len(existing_data)})")
        
        return local_file_path
    
    def save_data(self):
        """収集したデータを保存"""
        # 残りのデータを保存
        if self.all_formulas:
            self.save_current_data()
    
    def get_stats(self) -> Dict[str, int]:
        """収集したデータの統計を取得"""
        total_formulas = self.total_generated + len(self.all_formulas)
        unique_formulas = len(self.formula_hashes)
        
        return {
            "total_formulas": total_formulas,
            "unique_formulas": unique_formulas,
            "files_created": self.current_file_index
        }


def main() -> None:
    # パラメータを初期化
    gen_params = get_generation_params()
    train_params = get_training_params()
    system_params = get_system_params()
    
    parser = argparse.ArgumentParser(description="Generate tautology formulas and save to JSON")
    parser.add_argument("--count", type=int, default=gen_params.count, help="number of formulas to generate")
    parser.add_argument("--difficulty", type=float, default=gen_params.difficulty, help="formula generation difficulty")
    parser.add_argument("--seed", type=int, default=gen_params.seed, help="random seed")
    parser.add_argument("--max_len", type=int, default=gen_params.max_len, help="maximum formula string length")
    parser.add_argument("--dataset_file", type=str, default="tautology_data", help="base name for output files")
    parser.add_argument("--workers", type=int, default=None, 
                       help="number of parallel workers (default: min(cpu_count, 8))")
    parser.add_argument("--examples_per_file", type=int, default=10000,
                       help="number of examples per output file (default: 10000)")
    parser.add_argument("--buffer_size", type=int, default=1000,
                       help="buffer size for writing data (default: 1000)")
    parser.add_argument("--gcs_bucket", type=str, default=None,
                       help="GCS bucket name for direct upload (e.g., fof-data-20251010-milano)")
    parser.add_argument("--gcs_prefix", type=str, default="tautology/",
                       help="GCS prefix for uploaded files (e.g., tautology/)")
    parser.add_argument("--keep_global_hashes", action="store_true",
                       help="Keep existing global hashes file (continue from previous run)")
    args = parser.parse_args()

    # パラメータを更新
    default_params.update_generation_params(
        count=args.count,
        difficulty=args.difficulty,
        seed=args.seed,
        max_len=args.max_len
    )
    default_params.update_training_params(
        dataset_file=args.dataset_file
    )
    
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    
    # システムパラメータを更新
    default_params.update_system_params(
        root_dir=root_dir,
        pyprover_dir=os.path.join(root_dir, "pyprover")
    )

    # ジェネレーターを構築
    gen = FormulaGenerator(
        variables=gen_params.variables, 
        allow_const=gen_params.allow_const, 
        difficulty=gen_params.difficulty, 
        seed=gen_params.seed
    )
    
    # トートロジー生成器を初期化
    tautology_generator = TautologyGenerator(
        dataset_file_path=args.dataset_file,
        num_workers=args.workers,
        examples_per_file=args.examples_per_file,
        buffer_size=args.buffer_size,
        check_duplicates=True,
        gcs_bucket=args.gcs_bucket,
        gcs_prefix=args.gcs_prefix
    )
    
    # グローバルハッシュをクリア（--keep_global_hashesが指定されていない場合）
    if not args.keep_global_hashes:
        tautology_generator.clear_global_hashes()

    print(f"Starting tautology generation: {gen_params.count} formulas, {tautology_generator.num_workers} workers")
    if args.gcs_bucket:
        print(f"Output: gs://{args.gcs_bucket}/{args.gcs_prefix}{args.dataset_file}_XXXXX.json")
    else:
        print(f"Output: generated_data/{args.dataset_file}_XXXXX.json")
    
    # 既存のgenerated_dataをクリア
    tautology_generator.clear_generated_data()
    
    start_time = time.time()
    
    try:
        # 論理式を並列処理で生成
        results = tautology_generator.generate_tautologies_parallel(gen, gen_params)
        
        # 構造化された形式でデータを保存
        tautology_generator.save_data()
        stats = tautology_generator.get_stats()
        
        print(f"\nCompleted: {stats['total_formulas']} formulas, {stats['unique_formulas']} unique formulas")
        print(f"Files created: {stats['files_created']}")
        if args.gcs_bucket:
            print(f"Saved to: gs://{args.gcs_bucket}/{args.gcs_prefix}")
        else:
            print(f"Saved to: generated_data/")

    finally:
        # 終了前にグローバルハッシュを保存
        tautology_generator.save_global_hashes()
        
        # 総時間を計算
        total_time = time.time() - start_time
        print(f"Time: {total_time:.1f}s ({total_time/60:.1f}min), {total_time/gen_params.count:.2f}s/formula")


if __name__ == "__main__":
    main()
