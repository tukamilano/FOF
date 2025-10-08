from __future__ import annotations

import argparse
import contextlib
import os
import sys
import json
import hashlib
import time
from typing import List, Tuple, Dict, Any, Optional
from copy import deepcopy

from generate_prop import FormulaGenerator, filter_formulas
from training_data_collector import TrainingDataCollector
from state_encoder import encode_prover_state, parse_tactic_string
from parameter import (
    default_params, get_generation_params, get_training_params, 
    get_system_params, DeviceType, DataFilterType
)


@contextlib.contextmanager
def pushd(path: str):
    prev = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def import_pyprover(pyprover_dir: str):
    with pushd(pyprover_dir):
        if pyprover_dir not in sys.path:
            sys.path.insert(0, pyprover_dir)
        # Local imports after chdir so that proposition.py can open its grammar
        import proposition as proposition_mod  # type: ignore
        import prover as prover_mod  # type: ignore
    return proposition_mod, prover_mod


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


class AutoDataCollector:
    """auto_classical()で証明を探索し、データを収集するクラス"""
    
    def __init__(self, 
                 dataset_file_path: str = "training_data.json",
                 max_depth: int = 8,
                 filter_tactic_success_only: bool = False):
        self.dataset_file_path = dataset_file_path
        self.max_depth = max_depth
        self.filter_tactic_success_only = filter_tactic_success_only
        self.collected_data = []
        self.record_hashes = set()  # 重複排除用
        
    def collect_data_for_formula(self, prover, example_id: int) -> Dict[str, Any]:
        """単一の式に対してauto_classical()で証明を探索し、データを収集"""
        initial_state = encode_prover_state(prover)
        start_time = time.time()
        
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
                
                # データレコードのハッシュを計算
                record_hash_val = record_hash(current_state["premises"], current_state["goal"], tactic)
                
                # 重複チェック
                if record_hash_val not in self.record_hashes:
                    self.record_hashes.add(record_hash_val)
                    
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
                    
                    # フィルタリング条件をチェック
                    should_add = True
                    if self.filter_tactic_success_only:
                        # 成功したテクティクのみ
                        should_add = success
                    
                    if should_add:
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
        """収集したデータをJSONファイルに保存"""
        with open(self.dataset_file_path, 'w', encoding='utf-8') as f:
            json.dump(self.collected_data, f, ensure_ascii=False, indent=2)
    
    def get_stats(self) -> Dict[str, int]:
        """収集したデータの統計情報を取得"""
        total_records = len(self.collected_data)
        successful_tactics = sum(1 for record in self.collected_data if record["tactic_apply"])
        proved_records = sum(1 for record in self.collected_data if record["is_proved"])
        
        return {
            "total_records": total_records,
            "successful_tactics": successful_tactics,
            "proved_records": proved_records
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
    parser.add_argument("--filter_tactic_success_only", action="store_true", help="only store successful tactics")
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
    
    # フィルタリング設定を更新
    if args.filter_tactic_success_only:
        default_params.update_training_params(filter_type=DataFilterType.TACTIC_SUCCESS_ONLY)

    root_dir = os.path.dirname(__file__)
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
        filter_tactic_success_only=args.filter_tactic_success_only
    )

    print(f"Starting auto_classical() data collection for {gen_params.count} formulas...")
    print(f"Max depth: {gen_params.max_depth}")
    print(f"Output file: {train_params.dataset_file}")
    
    try:
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
            result = data_collector.collect_data_for_formula(prover, i)
            
            if result["proof_found"]:
                print(f"  ✓ Proof found in {result['total_steps']} steps ({result['time_taken']:.2f}s)")
                print(f"  Proof path: {' -> '.join(result['proof_path'])}")
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
        print(f"Data saved to: {args.dataset_file}")


if __name__ == "__main__":
    main()
