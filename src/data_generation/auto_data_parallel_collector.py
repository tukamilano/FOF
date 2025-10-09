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
        
        if main:
            if arg1:
                sequence.append(f"{main} {arg1}")
            else:
                sequence.append(main)
    
    return sequence


def transform_to_new_format(steps: List[Dict]) -> List[Dict]:
    """Transform flat steps to new structured format."""
    proof_groups = group_steps_into_proofs(steps)
    examples = []
    
    for i, proof_steps in enumerate(proof_groups):
        if not proof_steps:
            continue
            
        example_id = f"ex_{i+1:04d}"
        original_goal = proof_steps[0]['goal']
        is_proved = proof_steps[-1].get('is_proved', False)
        
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
            "example_id": example_id,
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
        args: Tuple containing (formula_data, example_id, max_depth, pyprover_dir)
    
    Returns:
        Dictionary containing the processing result
    """
    formula_data, example_id, max_depth, pyprover_dir = args
    
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
        result = process_formula_data(prover, example_id, max_depth)
        result['formula'] = formula_data['goal']
        result['worker_id'] = os.getpid()
        
        return result
        
    except Exception as e:
        return {
            'example_id': example_id,
            'formula': formula_data.get('goal', ''),
            'proof_found': False,
            'proof_path': [],
            'total_steps': 0,
            'time_taken': 0.0,
            'error': str(e),
            'worker_id': os.getpid()
        }


def process_formula_data(prover, example_id: int, max_depth: int) -> Dict[str, Any]:
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
            
            # データレコードのハッシュを計算
            record_hash_val = record_hash(current_state["premises"], current_state["goal"], tactic)
            
            # 重複チェック
            if record_hash_val not in record_hashes:
                record_hashes.add(record_hash_val)
                
                # 戦略を適用
                success = apply_tactic_from_label(current_prover, tactic)
                
                # データを記録（このexampleが解けたので、すべてのレコードでis_proved=true）
                record = {
                    "premises": current_state["premises"],
                    "goal": current_state["goal"],
                    "tactic": parse_tactic_string(tactic),  # 構造化されたtactic形式に変換
                    "tactic_apply": success,
                    "is_proved": True,  # このexampleが解けたので常にtrue
                    "record_hash": record_hash_val
                }
                
                collected_steps.append(record)
        
        return {
            "example_id": example_id,
            "initial_state": initial_state,
            "proof_found": True,
            "proof_path": proof_path,
            "total_steps": len(proof_path),
            "time_taken": time.time() - start_time,
            "collected_steps": collected_steps
        }
    
    return {
        "example_id": example_id,
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
                 buffer_size: int = 1000):
        self.dataset_file_path = dataset_file_path
        self.max_depth = max_depth
        # Default to CPU count, but allow override via num_workers parameter
        # Conservative limit of 8 to prevent memory issues with proof search
        self.num_workers = num_workers or min(mp.cpu_count(), 8)
        self.examples_per_file = examples_per_file
        self.buffer_size = min(buffer_size, examples_per_file)  # Buffer size can't exceed file size
        self.all_collected_steps = []  # Store all collected steps for transformation
        self.current_file_index = 1
        self.steps_in_current_file = 0
        self.examples_in_current_file = 0  # Track examples in current file
        self.total_collected_steps = 0  # Track total steps across all files
        self.total_successful_tactics = 0  # Track total successful tactics
        self.total_proved_records = 0  # Track total proved records
        
    def collect_data_parallel_streaming(self, gen, gen_params, pyprover_dir: str) -> List[Dict]:
        """Collect data using streaming approach - generate and process formulas in batches"""
        print(f"Starting streaming parallel data collection with {self.num_workers} workers...")
        
        results = []
        successful_proofs = 0
        processed_count = 0
        batch_size = self.num_workers * 4  # Process 4x worker count at a time
        
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            with tqdm(total=gen_params.count, desc="Processing formulas", unit="formula", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}') as pbar:
                while processed_count < gen_params.count:
                    # Generate a batch of formulas
                    batch_formulas = []
                    batch_count = min(batch_size, gen_params.count - processed_count)
                    
                    for i in range(batch_count):
                        goal_list = filter_formulas(gen, max_len=gen_params.max_len, require_tautology=True, limit=1)
                        if goal_list:
                            batch_formulas.append({
                                "goal": goal_list[0],
                                "index": processed_count + i
                            })
                        else:
                            # If no valid formula, create a placeholder
                            batch_formulas.append({
                                "goal": "",
                                "index": processed_count + i
                            })
                    
                    # Process the batch in parallel
                    worker_args = [
                        (formula, formula["index"], self.max_depth, pyprover_dir) 
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
                                'example_id': index,
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
                                'steps': len(self.all_collected_steps)
                            })
                        
                        # Break if we've processed enough
                        if processed_count >= gen_params.count:
                            break
                    
                    # Break if we've processed enough
                    if processed_count >= gen_params.count:
                        break
        
        # Sort results by example_id to maintain order
        results.sort(key=lambda x: x['example_id'])
        
        print(f"\nStreaming parallel processing completed!")
        print(f"Successful proofs: {successful_proofs}/{gen_params.count}")
        print(f"Total collected steps: {len(self.all_collected_steps)}")
        
        return results
    
    def get_current_filename(self) -> str:
        """Get the current filename based on file index"""
        # Create generated_data directory if it doesn't exist
        generated_dir = "generated_data"
        os.makedirs(generated_dir, exist_ok=True)
        
        base_name = os.path.basename(self.dataset_file_path).replace('.json', '')
        return os.path.join(generated_dir, f"{base_name}_{self.current_file_index:05d}.json")
    
    def clear_generated_data(self):
        """Clear existing generated_data directory"""
        generated_dir = "generated_data"
        if os.path.exists(generated_dir):
            import shutil
            shutil.rmtree(generated_dir)
            print(f"✓ Cleared existing {generated_dir}/ directory")
        os.makedirs(generated_dir, exist_ok=True)
    
    def save_current_data(self):
        """Save current collected data to a file"""
        if not self.all_collected_steps:
            return
            
        # Transform to new format
        transformed_data = transform_to_new_format(self.all_collected_steps)
        num_examples = len(transformed_data)
        
        filename = self.get_current_filename()
        
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
        
        # Add new data
        existing_data.extend(transformed_data)
        
        # Write to file
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(existing_data, f, ensure_ascii=False, indent=2)
        
        # Update tracking
        self.examples_in_current_file = len(existing_data)
        
        # Update global statistics
        self.total_collected_steps += len(self.all_collected_steps)
        self.total_successful_tactics += sum(1 for record in self.all_collected_steps if record["tactic_apply"])
        self.total_proved_records += sum(1 for record in self.all_collected_steps if record["is_proved"])
        
        if len(existing_data) == num_examples:
            print(f"✓ Created new file with {num_examples} examples: {filename}")
        else:
            print(f"✓ Appended {num_examples} examples to {filename} (total: {len(existing_data)})")
        
        # Reset buffer
        self.all_collected_steps = []
        self.steps_in_current_file = 0
    
    def add_steps_and_check_save(self, new_steps: List[Dict]):
        """Add new steps and save if we've reached the file limit"""
        self.all_collected_steps.extend(new_steps)
        self.steps_in_current_file += len(new_steps)
        
        # Check if we need to save current file
        # Use configurable buffer size to avoid large writes
        if len(self.all_collected_steps) >= self.buffer_size:
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
    parser.add_argument("--dataset_file", type=str, default=train_params.dataset_file, help="output dataset file")
    parser.add_argument("--workers", type=int, default=None, 
                       help="number of parallel workers (default: min(cpu_count, 8). Use more for powerful systems, fewer for memory-constrained systems)")
    parser.add_argument("--examples_per_file", type=int, default=10000,
                       help="number of examples per output file (default: 10000)")
    parser.add_argument("--buffer_size", type=int, default=1000,
                       help="buffer size for writing data (default: 1000, smaller = more frequent writes)")
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
        buffer_size=args.buffer_size
    )

    print(f"Starting parallel auto_classical() data collection for {gen_params.count} formulas...")
    print(f"Max depth: {gen_params.max_depth}")
    print(f"Workers: {data_collector.num_workers}")
    print(f"Examples per file: {args.examples_per_file}")
    print(f"Buffer size: {args.buffer_size}")
    print(f"Output files: generated_data/{os.path.basename(train_params.dataset_file).replace('.json', '')}_XXXXX.json")
    
    # Clear existing generated data
    data_collector.clear_generated_data()
    
    start_time = time.time()
    
    try:
        # Process formulas using streaming approach
        results = data_collector.collect_data_parallel_streaming(gen, gen_params, pyprover_dir)
        
        # Save data in structured format
        data_collector.save_data()
        stats = data_collector.get_stats()
        
        print(f"\nData collection completed!")
        print(f"Total records: {stats['total_records']}")
        print(f"Successful tactics: {stats['successful_tactics']}")
        print(f"Proved records: {stats['proved_records']}")
        print(f"Unique tactics used: {stats['unique_tactics']}")
        print(f"Data saved to: generated_data/ directory")

    finally:
        # Calculate total time
        total_time = time.time() - start_time
        
        print(f"\nParallel processing completed!")
        print(f"Total time: {total_time:.2f}s ({total_time/60:.1f} minutes)")
        print(f"Average time per formula: {total_time/gen_params.count:.2f}s")


if __name__ == "__main__":
    main()
