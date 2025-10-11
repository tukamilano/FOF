from __future__ import annotations

import argparse
import os
import sys
import json
import hashlib
import time
import multiprocessing as mp
from typing import List, Tuple, Dict, Any, Optional
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import subprocess

# プロジェクトルートをパスに追加
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, project_root)

from src.core.generate_prop import FormulaGenerator, filter_formulas
from src.core.state_encoder import encode_prover_state, parse_tactic_string
from src.core.parameter import (
    default_params, get_generation_params, get_training_params, 
    get_system_params, DeviceType, DataFilterType
)
from src.core.utils import pushd, import_pyprover


def apply_tactic_from_label(prover, label: str) -> bool:
    """戦略を適用し、成功したかどうかを返す"""
    # 証明が既に完了している場合は失敗
    if prover.goal is None:
        return False
        
    if label == "assumption":
        return not prover.assumption()
    if label == "intro":
        return not prover.intro()
    if label == "split":
        return not prover.split()
    if label == "left":
        return not prover.left()
    if label == "right":
        return not prover.right()
    if label == "add_dn":
        return not prover.add_dn()
    
    parts = label.split()
    if parts[0] == "apply" and len(parts) == 2 and parts[1].isdigit():
        idx = int(parts[1])
        if idx >= len(prover.variables):
            return False
        return not prover.apply(idx)
    
    if parts[0] == "destruct" and len(parts) == 2 and parts[1].isdigit():
        idx = int(parts[1])
        if idx >= len(prover.variables):
            return False
        return not prover.destruct(idx)
    
    if parts[0] == "specialize" and len(parts) == 3 and parts[1].isdigit() and parts[2].isdigit():
        func_idx = int(parts[1])
        domain_idx = int(parts[2])
        if func_idx >= len(prover.variables) or domain_idx >= len(prover.variables):
            return False
        return not prover.specialize(func_idx, domain_idx)
    
    return False


def record_hash(premises: List[str], goal: str, tactic: str) -> str:
    """データレコードの重複チェック用ハッシュを生成"""
    record_str = f"{'|'.join(premises)}|{goal}|{tactic}"
    return hashlib.md5(record_str.encode()).hexdigest()


def example_hash(original_goal: str) -> str:
    """Example全体の重複チェック用ハッシュを生成（元の目標式のみ）"""
    return hashlib.md5(original_goal.encode()).hexdigest()


def group_steps_into_proofs(steps: List[Dict]) -> List[List[Dict]]:
    """Group individual steps into complete proof sequences."""
    proofs = []
    current_proof = []
    
    for step in steps:
        if not step.get('premises', []):
            if current_proof:
                proofs.append(current_proof)
            current_proof = [step]
        else:
            current_proof.append(step)
    
    if current_proof:
        proofs.append(current_proof)
    
    return proofs


def create_tactic_sequence(steps: List[Dict]) -> List[str]:
    """Create a sequence of tactic names from steps."""
    sequence = []
    for step in steps:
        tactic = step.get('tactic', {})
        main = tactic.get('main', '')
        arg1 = tactic.get('arg1')
        arg2 = tactic.get('arg2')
        
        if main:
            if arg1 and arg2:
                sequence.append(f"{main} {arg1} {arg2}")
            elif arg1:
                sequence.append(f"{main} {arg1}")
            else:
                sequence.append(main)
    
    return sequence


def transform_to_new_format(steps: List[Dict]) -> List[Dict]:
    """Transform flat steps to new structured format."""
    proof_groups = group_steps_into_proofs(steps)
    examples = []
    seen_example_hashes = set()  # 重複チェック用
    
    for i, proof_steps in enumerate(proof_groups):
        if not proof_steps:
            continue
            
        original_goal = proof_steps[0]['goal']
        is_proved = proof_steps[-1].get('is_proved', False)
        
        # Example重複チェック
        example_hash_val = example_hash(original_goal)
        if example_hash_val in seen_example_hashes:
            # 重複をスキップ（ログなし）
            continue
        seen_example_hashes.add(example_hash_val)
        
        # Transform steps
        transformed_steps = []
        for j, step in enumerate(proof_steps):
            transformed_steps.append({
                "step_index": j,
                "premises": step.get('premises', []),
                "goal": step['goal'],
                "tactic": step['tactic'],
                "tactic_apply": step['tactic_apply'],
                "state_hash": step.get('record_hash', '')
            })
        
        # Create summary if proof was completed
        summary = None
        if is_proved:
            tactic_sequence = create_tactic_sequence(proof_steps)
            summary = {
                "tactic_sequence": tactic_sequence,
                "total_steps": len(proof_steps)
            }
        
        example = {
            "example_hash": example_hash_val,  # 計算済みのハッシュを使用
            "meta": {
                "goal_original": original_goal,
                "is_proved": is_proved
            },
            "steps": transformed_steps,
            "summary": summary
        }
        
        examples.append(example)
    
    return examples




def process_single_formula_worker(args: Tuple) -> Dict[str, Any]:
    """Worker function for processing a single formula in parallel.
    
    Args:
        args: Tuple containing (formula_data, max_depth, pyprover_dir, check_step_duplicates)
    
    Returns:
        Dictionary containing the processing result
    """
    formula_data, max_depth, pyprover_dir, check_step_duplicates = args
    
    try:
        # Import pyprover in the worker process
        proposition_mod, prover_mod = import_pyprover(pyprover_dir)
        PropParseTree = proposition_mod.PropParseTree
        prop_parser = proposition_mod.parser
        Prover = prover_mod.Prover
        
        # Parse the formula
        parse_tree = PropParseTree()
        with pushd(pyprover_dir):
            goal_node = parse_tree.transform(prop_parser.parse(formula_data['goal']))
        prover = Prover(goal_node)
        
        # Process the formula
        result = process_formula_data(prover, max_depth, check_step_duplicates)
        result['formula'] = formula_data['goal']
        result['worker_id'] = os.getpid()
        
        return result
        
    except Exception as e:
        return {
            'formula': formula_data.get('goal', ''),
            'proof_found': False,
            'proof_path': [],
            'total_steps': 0,
            'time_taken': 0.0,
            'error': str(e),
            'worker_id': os.getpid()
        }


def process_formula_data(prover, max_depth: int, check_step_duplicates: bool = False) -> Dict[str, Any]:
    """Process a single formula and collect detailed proof data.
    
    This collects detailed step-by-step data like auto_data_collector.py
    """
    initial_state = encode_prover_state(prover)
    start_time = time.time()
    collected_steps = []
    record_hashes = set()
    
    # Use auto_classical() to find proof
    prover_copy_for_auto = deepcopy(prover)
    proof_path = prover_copy_for_auto.auto_classical(depth_limit=max_depth)
    
    if proof_path:
        # 証明が見つかった場合、各ステップを記録
        current_prover = deepcopy(prover)  # 元のproverから新しくコピー
        for i, tactic in enumerate(proof_path):
            # 証明が完了している場合はループを終了
            if current_prover.goal is None:
                break
                
            current_state = encode_prover_state(current_prover)
            
            # ステップハッシュを常に計算
            record_hash_val = record_hash(current_state["premises"], current_state["goal"], tactic)
            
            # ステップ重複チェック（オプション）
            should_add_step = True
            if check_step_duplicates:
                if record_hash_val in record_hashes:
                    should_add_step = False
                else:
                    record_hashes.add(record_hash_val)
            
            # 戦略を適用
            success = apply_tactic_from_label(current_prover, tactic)
            
            # データを記録（このexampleが解けたので、すべてのレコードでis_proved=true）
            if should_add_step:
                record = {
                    "premises": current_state["premises"],
                    "goal": current_state["goal"],
                    "tactic": parse_tactic_string(tactic),  # 構造化されたtactic形式に変換
                    "tactic_apply": success,
                    "is_proved": True,  # このexampleが解けたので常にtrue
                    "record_hash": record_hash_val  # 常にハッシュを保存
                }
                collected_steps.append(record)
        
        return {
            "initial_state": initial_state,
            "proof_found": True,
            "proof_path": proof_path,
            "total_steps": len(proof_path),
            "time_taken": time.time() - start_time,
            "collected_steps": collected_steps
        }
    
    return {
        "initial_state": initial_state,
        "proof_found": False,
        "proof_path": [],
        "total_steps": 0,
        "time_taken": time.time() - start_time,
        "collected_steps": []
    }


class ParallelDataCollector:
    """Parallel version of AutoDataCollector using multiprocessing"""
    
    def __init__(self, 
                 dataset_file_path: str = "training_data.json",
                 max_depth: int = 8,
                 num_workers: int = None,
                 examples_per_file: int = 10000,
                 buffer_size: int = 1000,
                 gcs_bucket: str = None,
                 gcs_prefix: str = "",
                 check_example_duplicates: bool = True,
                 check_step_duplicates: bool = False):
        self.dataset_file_path = dataset_file_path
        self.max_depth = max_depth
        # Default to CPU count, but allow override via num_workers parameter
        # Conservative limit of 8 to prevent memory issues with proof search
        self.num_workers = num_workers or min(mp.cpu_count(), 8)
        self.examples_per_file = examples_per_file
        # バッファサイズを例数ベースで計算（平均ステップ数を考慮）
        self.buffer_size = min(buffer_size, examples_per_file)  # Buffer size can't exceed file size
        self.avg_steps_per_example = 4.0  # 初期推定値
        self.buffer_steps = int(self.buffer_size * self.avg_steps_per_example)  # ステップ数ベースのバッファ
        self.all_collected_steps = []  # Store all collected steps for transformation
        self.current_file_index = 1
        self.steps_in_current_file = 0
        self.examples_in_current_file = 0  # Track examples in current file
        self.buffer_examples = 0  # Track examples in current buffer
        self.total_collected_steps = 0  # Track total steps across all files
        self.total_successful_tactics = 0  # Track total successful tactics
        self.total_proved_records = 0  # Track total proved records
        self.global_example_counter = 0  # Global counter for unique example IDs across all files
        self.example_hashes = set()  # Example重複排除用（グローバル）
        self.check_example_duplicates = check_example_duplicates
        self.check_step_duplicates = check_step_duplicates
        self.file_duplicate_stats = []  # ファイルごとの重複統計
        
        # GCS settings
        self.gcs_bucket = gcs_bucket
        self.gcs_prefix = gcs_prefix
        
        # グローバル重複チェック用の永続化ファイル
        self.global_hashes_file = "global_example_hashes.json"
        self.load_global_hashes()
    
    def clear_global_hashes(self):
        """グローバルハッシュファイルを削除してリセット"""
        if os.path.exists(self.global_hashes_file):
            os.remove(self.global_hashes_file)
            print(f"Cleared global hashes file: {self.global_hashes_file}")
        
        # メモリ内のハッシュもリセット
        self.example_hashes = set()
        self.global_example_counter = 0
        self.current_file_index = 1
        print("Reset global hash state")
    
    def load_global_hashes(self):
        """既存のグローバルハッシュを読み込み"""
        if os.path.exists(self.global_hashes_file):
            try:
                with open(self.global_hashes_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.example_hashes = set(data.get('example_hashes', []))
                    self.global_example_counter = data.get('global_example_counter', 0)
                    self.current_file_index = data.get('current_file_index', 1)
                print(f"Loaded {len(self.example_hashes)} existing example hashes from {self.global_hashes_file}")
            except (FileNotFoundError, json.JSONDecodeError) as e:
                print(f"Could not load global hashes: {e}")
                self.example_hashes = set()
                self.global_example_counter = 0
                self.current_file_index = 1
        else:
            self.example_hashes = set()
            self.global_example_counter = 0
            self.current_file_index = 1
    
    def save_global_hashes(self):
        """グローバルハッシュを保存"""
        data = {
            'example_hashes': list(self.example_hashes),
            'global_example_counter': self.global_example_counter,
            'current_file_index': self.current_file_index
        }
        with open(self.global_hashes_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def load_existing_file_hashes(self, filename: str) -> set:
        """既存ファイルからexample_hashを読み込み"""
        if not os.path.exists(filename):
            return set()
        
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
                hashes = set()
                for example in data:
                    if 'example_hash' in example:
                        hashes.add(example['example_hash'])
                return hashes
        except (FileNotFoundError, json.JSONDecodeError):
            return set()
        
    def collect_data_parallel_streaming(self, gen, gen_params, pyprover_dir: str) -> List[Dict]:
        """Collect data using streaming approach - generate and process formulas in batches"""
        print(f"Starting parallel data collection with {self.num_workers} workers...")
        
        results = []
        successful_proofs = 0
        processed_count = 0
        skipped_duplicates = 0  # 重複スキップのカウンター
        batch_size = self.num_workers * 4  # Process 4x worker count at a time
        
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            with tqdm(total=gen_params.count, desc="Processing formulas", unit="formula", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}') as pbar:
                while processed_count < gen_params.count:
                    # Generate a batch of formulas
                    batch_formulas = []
                    batch_count = min(batch_size, gen_params.count - processed_count)
                    
                    successful_count = 0
                    for i in range(batch_count):
                        goal_list = filter_formulas(gen, max_len=gen_params.max_len, require_tautology=True, limit=1)
                        if goal_list:
                            goal = goal_list[0]
                            # Example重複チェック（メインプロセスで実行）
                            if self.check_example_duplicates:
                                example_hash_val = example_hash(goal)
                                if example_hash_val in self.example_hashes:
                                    # 重複している場合はスキップ
                                    processed_count += 1  # スキップした場合もカウントを増加
                                    skipped_duplicates += 1
                                    continue
                                self.example_hashes.add(example_hash_val)
                            
                            batch_formulas.append({
                                "goal": goal,
                                "index": successful_count
                            })
                            successful_count += 1
                        else:
                            # If no valid formula, create a placeholder
                            batch_formulas.append({
                                "goal": "",
                                "index": successful_count
                            })
                            successful_count += 1
                    
                    # Process the batch in parallel
                    worker_args = [
                        (formula, self.max_depth, pyprover_dir, self.check_step_duplicates) 
                        for formula in batch_formulas
                    ]
                    
                    # Submit batch tasks
                    future_to_index = {
                        executor.submit(process_single_formula_worker, args): formula["index"] 
                        for args, formula in zip(worker_args, batch_formulas)
                    }
                    
                    # Process completed tasks from this batch
                    for future in as_completed(future_to_index):
                        index = future_to_index[future]
                        try:
                            result = future.result()
                            results.append(result)
                            
                            # Collect detailed steps from successful proofs
                            if result.get('proof_found', False) and 'collected_steps' in result:
                                self.add_steps_and_check_save(result['collected_steps'])
                                successful_proofs += 1
                                        
                        except Exception as e:
                            results.append({
                                'formula': '',
                                'proof_found': False,
                                'proof_path': [],
                                'total_steps': 0,
                                'time_taken': 0.0,
                                'error': str(e),
                                'collected_steps': []
                            })
                        
                        pbar.update(1)
                        processed_count += 1
                        
                        # Update progress bar with current file info
                        if processed_count % 100 == 0:  # Update every 100 processed
                            pbar.set_postfix({
                                'file': f"{self.current_file_index:05d}",
                                'steps': len(self.all_collected_steps),
                                'skipped': skipped_duplicates
                            })
                        
                        # Break if we've processed enough
                        if processed_count >= gen_params.count:
                            break
                    
                    # Update global example counter after processing batch
                    self.global_example_counter += successful_count
                    
                    # Break if we've processed enough
                    if processed_count >= gen_params.count:
                        break
        
        # Sort results by formula to maintain order
        results.sort(key=lambda x: x.get('formula', ''))
        
        print(f"Completed: {successful_proofs}/{gen_params.count} proofs, {len(self.all_collected_steps)} steps")
        print(f"Duplicates skipped: {skipped_duplicates} ({skipped_duplicates/gen_params.count*100:.1f}%)")
        print(f"Global unique examples: {len(self.example_hashes)}")
        
        # ファイルごとの重複統計を表示
        if self.file_duplicate_stats:
            print(f"\n📈 File-by-file duplicate statistics:")
            for stat in self.file_duplicate_stats:
                print(f"  File {stat['file_index']:05d}: {stat['duplicates_removed']}/{stat['total_processed']} duplicates ({stat['duplicate_rate']:.1f}%) - Efficiency: {stat['efficiency']:.1f}%")
            
            # 平均効率を計算
            avg_efficiency = sum(s['efficiency'] for s in self.file_duplicate_stats) / len(self.file_duplicate_stats)
            print(f"  📊 Average efficiency: {avg_efficiency:.1f}%")
        
        return results
    
    def get_current_filename(self) -> str:
        """Get the current filename based on file index"""
        # Create generated_data directory if it doesn't exist
        generated_dir = "generated_data"
        os.makedirs(generated_dir, exist_ok=True)
        
        base_name = os.path.basename(self.dataset_file_path).replace('.json', '')
        return os.path.join(generated_dir, f"{base_name}_{self.current_file_index:05d}.json")
    
    def upload_file_to_gcs(self, local_file_path: str, gcs_filename: str) -> bool:
        """Upload a local file to GCS bucket using gcloud command"""
        if not self.gcs_bucket:
            return False
        
        try:
            # Construct GCS path
            gcs_path = f"gs://{self.gcs_bucket}/{self.gcs_prefix}{gcs_filename}"
            
            # Use gcloud command to upload
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
    
    def clear_generated_data(self):
        """Clear existing generated_data directory"""
        generated_dir = "generated_data"
        if os.path.exists(generated_dir):
            import shutil
            shutil.rmtree(generated_dir)
            print(f"Cleared: {generated_dir}/")
        os.makedirs(generated_dir, exist_ok=True)
        
        if self.gcs_bucket:
            print(f"GCS upload: gs://{self.gcs_bucket}/{self.gcs_prefix}")
    
    def save_current_data(self):
        """Save current collected data to local file and optionally upload to GCS"""
        if not self.all_collected_steps:
            return
            
        # Transform to new format
        transformed_data = transform_to_new_format(self.all_collected_steps)
        num_examples = len(transformed_data)
        
        # 平均ステップ数は固定値を使用（シンプル化）
        
        # バッファの例数カウンターをリセット
        self.buffer_examples = 0
        
        # Update global example counter
        self.global_example_counter += num_examples
        
        # Update global statistics
        self.total_collected_steps += len(self.all_collected_steps)
        self.total_successful_tactics += sum(1 for record in self.all_collected_steps if record["tactic_apply"])
        self.total_proved_records += sum(1 for record in self.all_collected_steps if record["is_proved"])
        
        # Always save to local file first
        local_file_path = self._save_to_local_file(transformed_data)
        
        # Upload to GCS if configured
        if self.gcs_bucket and local_file_path:
            gcs_filename = f"training_data_{self.current_file_index:05d}.json"
            self.upload_file_to_gcs(local_file_path, gcs_filename)
        
        # Reset buffer
        self.all_collected_steps = []
        self.steps_in_current_file = 0
    
    def _save_to_local_file(self, transformed_data: List[Dict]) -> str:
        """Save data to local file and return the file path"""
        filename = self.get_current_filename()
        num_examples = len(transformed_data)
        
        # Check if current file exists and has space
        if os.path.exists(filename):
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                if len(existing_data) + num_examples > self.examples_per_file:
                    # Current file would be too full, create new file
                    self.current_file_index += 1
                    filename = self.get_current_filename()
                    existing_data = []
            except (FileNotFoundError, json.JSONDecodeError):
                existing_data = []
        else:
            existing_data = []
        
        # グローバル重複チェック（既存の全ファイルとの重複をチェック）
        if self.check_example_duplicates:
            # 現在のファイルの既存ハッシュを取得
            current_file_hashes = set()
            for ex in existing_data:
                if 'example_hash' in ex:
                    current_file_hashes.add(ex['example_hash'])
            
            # グローバルハッシュと現在のファイルハッシュをマージ
            all_existing_hashes = self.example_hashes | current_file_hashes
            
            # 重複を除去
            filtered_data = []
            duplicates_removed = 0
            for ex in transformed_data:
                if 'example_hash' in ex and ex['example_hash'] in all_existing_hashes:
                    # 重複をスキップ
                    duplicates_removed += 1
                    # 個別ログを削除（重複が多いため）
                else:
                    filtered_data.append(ex)
                    # グローバルハッシュに追加
                    if 'example_hash' in ex:
                        self.example_hashes.add(ex['example_hash'])
            
            if duplicates_removed > 0:
                duplicate_rate = (duplicates_removed / len(transformed_data)) * 100
                print(f"  📊 File {self.current_file_index:05d}: Removed {duplicates_removed}/{len(transformed_data)} duplicates ({duplicate_rate:.1f}%)")
                
                # 統計を記録
                self.file_duplicate_stats.append({
                    'file_index': self.current_file_index,
                    'duplicates_removed': duplicates_removed,
                    'total_processed': len(transformed_data) + duplicates_removed,
                    'duplicate_rate': duplicate_rate,
                    'efficiency': len(transformed_data) / (len(transformed_data) + duplicates_removed) * 100
                })
            else:
                # 重複なしの場合も記録
                self.file_duplicate_stats.append({
                    'file_index': self.current_file_index,
                    'duplicates_removed': 0,
                    'total_processed': len(transformed_data),
                    'duplicate_rate': 0.0,
                    'efficiency': 100.0
                })
            
            transformed_data = filtered_data
            num_examples = len(transformed_data)
        
        # Add new data
        existing_data.extend(transformed_data)
        
        # Write to file
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(existing_data, f, ensure_ascii=False, indent=2)
        
        # Update tracking
        self.examples_in_current_file = len(existing_data)
        
        # グローバルハッシュを保存
        self.save_global_hashes()
        
        # ファイル保存時の統計を表示
        total_processed = len(transformed_data) + duplicates_removed if 'duplicates_removed' in locals() else len(transformed_data)
        if 'duplicates_removed' in locals() and duplicates_removed > 0:
            print(f"  ✅ File {self.current_file_index:05d} saved: {len(existing_data)} examples (efficiency: {len(transformed_data)/total_processed*100:.1f}%)")
        else:
            print(f"  ✅ File {self.current_file_index:05d} saved: {len(existing_data)} examples (no duplicates)")
        
        if len(existing_data) == num_examples:
            print(f"Created: {filename} ({num_examples} examples)")
        else:
            print(f"Appended: {filename} (+{num_examples}, total: {len(existing_data)})")
        
        return filename
    
    def add_steps_and_check_save(self, new_steps: List[Dict]):
        """Add new steps and save if we've reached the file limit"""
        self.all_collected_steps.extend(new_steps)
        self.steps_in_current_file += len(new_steps)
        
        # 新しいステップから例数を推定して更新
        new_examples = len(new_steps) / max(self.avg_steps_per_example, 1.0)
        self.buffer_examples += new_examples
        
        # バッファサイズまたはファイルサイズのいずれかに達したら保存
        if (self.buffer_examples >= self.buffer_size or 
            self.buffer_examples >= self.examples_per_file):
            self.save_current_data()
    
    def save_data(self):
        """Save collected data in the structured format like auto_data_collector.py"""
        # Save any remaining data
        if self.all_collected_steps:
            self.save_current_data()
    
    def get_stats(self) -> Dict[str, int]:
        """Get statistics about collected data"""
        # Use global statistics that track all saved data
        total_records = self.total_collected_steps + len(self.all_collected_steps)
        successful_tactics = self.total_successful_tactics + sum(1 for record in self.all_collected_steps if record["tactic_apply"])
        proved_records = self.total_proved_records + sum(1 for record in self.all_collected_steps if record["is_proved"])
        unique_examples = len(self.example_hashes)
        
        # Count unique tactics used (this is approximate since we don't track all tactics)
        unique_tactics = set()
        for record in self.all_collected_steps:
            tactic = record.get("tactic", {})
            main = tactic.get("main", "")
            arg1 = tactic.get("arg1")
            if main:
                if arg1:
                    unique_tactics.add(f"{main} {arg1}")
                else:
                    unique_tactics.add(main)
        
        return {
            "total_records": total_records,
            "successful_tactics": successful_tactics,
            "proved_records": proved_records,
            "unique_examples": unique_examples,
            "unique_tactics": len(unique_tactics)
        }


def main() -> None:
    # パラメータを初期化
    gen_params = get_generation_params()
    train_params = get_training_params()
    system_params = get_system_params()
    
    parser = argparse.ArgumentParser(description="Parallel auto_classical() based proof discovery and data collection")
    parser.add_argument("--count", type=int, default=gen_params.count, help="number of formulas to process")
    parser.add_argument("--difficulty", type=float, default=gen_params.difficulty, help="formula generation difficulty")
    parser.add_argument("--seed", type=int, default=gen_params.seed, help="random seed")
    parser.add_argument("--max_depth", type=int, default=gen_params.max_depth, help="maximum auto_classical search depth")
    parser.add_argument("--dataset_file", type=str, default=train_params.dataset_file, help="base name for output files (e.g., training_data)")
    parser.add_argument("--workers", type=int, default=None, 
                       help="number of parallel workers (default: min(cpu_count, 8). Use more for powerful systems, fewer for memory-constrained systems)")
    parser.add_argument("--examples_per_file", type=int, default=10000,
                       help="number of examples per output file (default: 10000)")
    parser.add_argument("--buffer_size", type=int, default=1000,
                       help="buffer size for writing data (default: 1000, smaller = more frequent writes)")
    parser.add_argument("--gcs_bucket", type=str, default=None,
                       help="GCS bucket name for direct upload (e.g., fof-data-20251009-milano)")
    parser.add_argument("--gcs_prefix", type=str, default="",
                       help="GCS prefix for uploaded files (e.g., training_data/)")
    parser.add_argument("--keep_global_hashes", action="store_true",
                       help="Keep existing global hashes file (continue from previous run)")
    args = parser.parse_args()

    # パラメータを更新
    default_params.update_generation_params(
        count=args.count,
        difficulty=args.difficulty,
        seed=args.seed,
        max_depth=args.max_depth
    )
    default_params.update_training_params(
        dataset_file=args.dataset_file
    )
    
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    pyprover_dir = os.path.join(root_dir, "pyprover")
    
    # システムパラメータを更新
    default_params.update_system_params(
        root_dir=root_dir,
        pyprover_dir=pyprover_dir
    )

    # Import pyprover
    proposition_mod, prover_mod = import_pyprover(pyprover_dir)
    PropParseTree = proposition_mod.PropParseTree
    prop_parser = proposition_mod.parser
    Prover = prover_mod.Prover

    # Build generator
    gen = FormulaGenerator(
        variables=gen_params.variables, 
        allow_const=gen_params.allow_const, 
        difficulty=gen_params.difficulty, 
        seed=gen_params.seed
    )
    
    # Initialize parallel data collector
    data_collector = ParallelDataCollector(
        dataset_file_path=train_params.dataset_file,
        max_depth=gen_params.max_depth,
        num_workers=args.workers,
        examples_per_file=args.examples_per_file,
        buffer_size=args.buffer_size,
        gcs_bucket=args.gcs_bucket,
        gcs_prefix=args.gcs_prefix,
        check_example_duplicates=True,  # Example重複チェックを有効化
        check_step_duplicates=False    # ステップ重複チェックを無効化
    )
    
    # Clear global hashes by default (unless --keep_global_hashes is specified)
    if not args.keep_global_hashes:
        data_collector.clear_global_hashes()

    print(f"Starting data collection: {gen_params.count} formulas, {data_collector.num_workers} workers")
    
    base_name = os.path.basename(train_params.dataset_file).replace('.json', '')
    if args.gcs_bucket:
        print(f"Output: gs://{args.gcs_bucket}/{args.gcs_prefix}{base_name}_XXXXX.json")
    else:
        print(f"Output: generated_data/{base_name}_XXXXX.json")
    
    # Clear existing generated data
    data_collector.clear_generated_data()
    
    start_time = time.time()
    
    try:
        # Process formulas using streaming approach
        results = data_collector.collect_data_parallel_streaming(gen, gen_params, pyprover_dir)
        
        # Save data in structured format
        data_collector.save_data()
        stats = data_collector.get_stats()
        
        print(f"\nCompleted: {stats['total_records']} records, {stats['unique_examples']} unique examples")
        if args.gcs_bucket:
            print(f"Saved to: gs://{args.gcs_bucket}/{args.gcs_prefix}")
        else:
            print(f"Saved to: generated_data/")

    finally:
        # Save global hashes before exit
        data_collector.save_global_hashes()
        
        # Calculate total time
        total_time = time.time() - start_time
        print(f"Time: {total_time:.1f}s ({total_time/60:.1f}min), {total_time/gen_params.count:.2f}s/formula")


if __name__ == "__main__":
    main()
