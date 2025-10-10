from __future__ import annotations

import argparse
import os
import sys
import json
import hashlib
import time
from typing import List, Tuple, Dict, Any, Optional
from copy import deepcopy

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


def transform_to_new_format(steps: List[Dict], start_example_id: int = 1) -> List[Dict]:
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
            print(f"  ⏭ Skipping duplicate example in transform: {original_goal[:50]}... (hash: {example_hash_val[:8]}...)")
            continue
        seen_example_hashes.add(example_hash_val)
        
        example_id = f"ex_{start_example_id + len(examples):04d}"
        
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


class AutoDataCollector:
    """auto_classical()で証明を探索し、データを収集するクラス"""
    
    def __init__(self, 
                 dataset_file_path: str = "training_data.json",
                 max_depth: int = 8,
                 check_example_duplicates: bool = True,
                 check_step_duplicates: bool = False):
        self.dataset_file_path = dataset_file_path
        self.max_depth = max_depth
        self.collected_data = []
        self.record_hashes = set()  # ステップ重複排除用
        self.example_hashes = set()  # Example重複排除用
        self.check_example_duplicates = check_example_duplicates
        self.check_step_duplicates = check_step_duplicates
        
    def collect_data_for_formula(self, prover, example_id: int) -> Dict[str, Any]:
        """単一の式に対してauto_classical()で証明を探索し、データを収集"""
        initial_state = encode_prover_state(prover)
        start_time = time.time()
        
        # Example重複チェック（元の目標式でチェック）
        if self.check_example_duplicates:
            original_goal = initial_state["goal"]
            example_hash_val = example_hash(original_goal)
            if example_hash_val in self.example_hashes:
                print(f"  ⏭ Skipping duplicate example: {original_goal[:50]}... (hash: {example_hash_val[:8]}...)")
                return {
                    "example_id": example_id,
                    "initial_state": initial_state,
                    "proof_found": False,
                    "proof_path": [],
                    "total_steps": 0,
                    "time_taken": time.time() - start_time,
                    "skipped": True,
                    "reason": "duplicate_example"
                }
            self.example_hashes.add(example_hash_val)
            print(f"  ✓ Added new example: {original_goal[:50]}... (hash: {example_hash_val[:8]}...)")
        
        # pyproverのauto_classical()メソッドで証明を試行
        prover_copy_for_auto = deepcopy(prover)
        proof_path = prover_copy_for_auto.auto_classical(depth_limit=self.max_depth)
        
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
                if self.check_step_duplicates:
                    if record_hash_val in self.record_hashes:
                        should_add_step = False
                    else:
                        self.record_hashes.add(record_hash_val)
                
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
                    self.collected_data.append(record)
            
            return {
                "example_id": example_id,
                "initial_state": initial_state,
                "proof_found": True,
                "proof_path": proof_path,
                "total_steps": len(proof_path),
                "time_taken": time.time() - start_time
            }
        
        # 証明が見つからなかった場合
        return {
            "example_id": example_id,
            "initial_state": initial_state,
            "proof_found": False,
            "proof_path": [],
            "total_steps": 0,
            "time_taken": time.time() - start_time
        }
    
    
    def save_data(self):
        """収集したデータをJSONファイルに保存（新しい形式）"""
        # Transform to new format
        transformed_data = transform_to_new_format(self.collected_data)
        
        with open(self.dataset_file_path, 'w', encoding='utf-8') as f:
            json.dump(transformed_data, f, ensure_ascii=False, indent=2)
    
    def get_stats(self) -> Dict[str, int]:
        """収集したデータの統計情報を取得"""
        total_records = len(self.collected_data)
        successful_tactics = sum(1 for record in self.collected_data if record["tactic_apply"])
        proved_records = sum(1 for record in self.collected_data if record["is_proved"])
        unique_examples = len(self.example_hashes)
        
        return {
            "total_records": total_records,
            "successful_tactics": successful_tactics,
            "proved_records": proved_records,
            "unique_examples": unique_examples
        }


def main() -> None:
    # パラメータを初期化
    gen_params = get_generation_params()
    train_params = get_training_params()
    system_params = get_system_params()
    
    parser = argparse.ArgumentParser(description="auto_classical() based proof discovery and data collection")
    parser.add_argument("--count", type=int, default=gen_params.count, help="number of formulas to process")
    parser.add_argument("--difficulty", type=float, default=gen_params.difficulty, help="formula generation difficulty")
    parser.add_argument("--seed", type=int, default=gen_params.seed, help="random seed")
    parser.add_argument("--max_depth", type=int, default=gen_params.max_depth, help="maximum auto_classical search depth")
    parser.add_argument("--dataset_file", type=str, default=train_params.dataset_file, help="output dataset file")
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
    
    # Initialize data collector
    data_collector = AutoDataCollector(
        dataset_file_path=train_params.dataset_file,
        max_depth=gen_params.max_depth,
        check_example_duplicates=True,  # Example重複チェックを有効化
        check_step_duplicates=False    # ステップ重複チェックを無効化
    )

    print(f"Starting auto_classical() data collection for {gen_params.count} formulas...")
    print(f"Max depth: {gen_params.max_depth}")
    print(f"Output file: {train_params.dataset_file}")
    
    try:
        successful_count = 0
        for i in range(gen_params.count):
            print(f"\nProcessing formula {i+1}/{gen_params.count}...")
            
            # Generate formula
            goal_list = filter_formulas(gen, max_len=gen_params.max_len, require_tautology=True, limit=1)
            if not goal_list:
                print(f"Warning: No valid formulas generated for example {i+1}, skipping...")
                continue
            
            seed_goal = goal_list[0]
            print(f"Goal: {seed_goal}")

            # Parse and create prover
            parse_tree = PropParseTree()
            with pushd(pyprover_dir):
                goal_node = parse_tree.transform(prop_parser.parse(seed_goal))
            prover = Prover(goal_node)

            # Collect data using BFS
            result = data_collector.collect_data_for_formula(prover, successful_count)
            
            if result.get("skipped", False):
                print(f"  ⏭ Skipped duplicate example (time: {result['time_taken']:.2f}s)")
            elif result["proof_found"]:
                print(f"  ✓ Proof found in {result['total_steps']} steps ({result['time_taken']:.2f}s)")
                print(f"  Proof path: {' -> '.join(result['proof_path'])}")
                successful_count += 1
            else:
                print(f"  ✗ No proof found within limits (time: {result['time_taken']:.2f}s)")

    finally:
        # Save collected data
        data_collector.save_data()
        stats = data_collector.get_stats()
        
        print(f"\nData collection completed!")
        print(f"Total records: {stats['total_records']}")
        print(f"Successful tactics: {stats['successful_tactics']}")
        print(f"Proved records: {stats['proved_records']}")
        print(f"Unique examples: {stats['unique_examples']}")
        print(f"Data saved to: {args.dataset_file}")
        
        # 重複チェック統計を表示
        if data_collector.check_example_duplicates:
            print(f"\nExample重複チェック統計:")
            print(f"  処理した例数: {gen_params.count}")
            print(f"  ユニークな例数: {stats['unique_examples']}")
            print(f"  スキップされた重複例数: {gen_params.count - stats['unique_examples']}")


if __name__ == "__main__":
    main()
