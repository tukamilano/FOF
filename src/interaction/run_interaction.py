from __future__ import annotations

import argparse
import os
import sys
import json
import hashlib
from typing import List, Tuple, Dict, Any

# プロジェクトルートをパスに追加
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, project_root)

import torch

from src.core.transformer_classifier import (
    load_tokens_and_labels_from_token_py,
    CharTokenizer,
    TransformerClassifier,
    build_hierarchical_label_mappings,
)
from src.core.generate_prop import FormulaGenerator, filter_formulas
from src.core.state_encoder import encode_prover_state, format_tactic_string, parse_tactic_string
from src.core.parameter import (
    default_params, get_model_params, get_training_params, 
    get_generation_params, get_system_params, DeviceType, DataFilterType
)
from src.core.utils import pushd, import_pyprover


def apply_tactic_from_label(prover, label) -> bool:
    # Returns True if tactic succeeded, False if failed
    # label can be either a string or a structured tactic dict
    
    # Convert structured tactic to string if needed
    if isinstance(label, dict):
        tactic_str = format_tactic_string(label)
    else:
        tactic_str = label
    
    if tactic_str == "assumption":
        return not prover.assumption()
    if tactic_str == "intro":
        return not prover.intro()
    if tactic_str == "split":
        return not prover.split()
    if tactic_str == "left":
        return not prover.left()
    if tactic_str == "right":
        return not prover.right()
    if tactic_str == "add_dn":
        return not prover.add_dn()
    
    parts = tactic_str.split()
    if parts[0] == "apply" and len(parts) == 2 and parts[1].isdigit():
        idx = int(parts[1])
        if idx >= len(prover.variables):
            return False  # Index out of range, tactic fails
        return not prover.apply(idx)
    if parts[0] == "destruct" and len(parts) == 2 and parts[1].isdigit():
        idx = int(parts[1])
        if idx >= len(prover.variables):
            return False  # Index out of range, tactic fails
        return not prover.destruct(idx)
    if parts[0] == "specialize" and len(parts) == 3 and parts[1].isdigit() and parts[2].isdigit():
        func_idx = int(parts[1])
        domain_idx = int(parts[2])
        if func_idx >= len(prover.variables) or domain_idx >= len(prover.variables):
            return False  # Index out of range, tactic fails
        return not prover.specialize(func_idx, domain_idx)
    return False


def record_hash(premises: List[str], goal: str, tactic: str) -> str:
    """データレコードの重複チェック用ハッシュを生成"""
    record_str = f"{'|'.join(premises)}|{goal}|{tactic}"
    return hashlib.md5(record_str.encode()).hexdigest()


class StructuredDataCollector:
    """run_interaction.py用の新しい構造化データコレクター"""
    
    def __init__(self, dataset_file_path: str = "training_data.json", filter_successful_only: bool = False):
        self.dataset_file_path = dataset_file_path
        self.filter_successful_only = filter_successful_only
        self.examples = []
        self.current_example_steps = []
        self.example_id = 0
        self.original_goal = ""
        
    def start_example(self, example_id: int, initial_premises: List[str], initial_goal: str):
        """新しい例の開始"""
        self.example_id = example_id
        self.original_goal = initial_goal
        self.current_example_steps = []
        
    def add_tactic_application(self, step: int, premises: List[str], goal: str, tactic: str, tactic_apply: bool):
        """戦略適用の記録"""
        # 戦略文字列を構造化形式に変換
        structured_tactic = parse_tactic_string(tactic)
        
        # レコードハッシュを計算
        record_hash_val = record_hash(premises, goal, tactic)
        
        step_data = {
            "step_index": step,
            "premises": premises,
            "goal": goal,
            "tactic": structured_tactic,
            "tactic_apply": tactic_apply,
            "state_hash": record_hash_val
        }
        
        self.current_example_steps.append(step_data)
        
    def update_last_tactic_apply(self, success: bool):
        """最後の戦略適用結果を更新"""
        if self.current_example_steps:
            self.current_example_steps[-1]["tactic_apply"] = success
            
    def finish_example(self, is_proved: bool):
        """例の終了処理"""
        if self.current_example_steps:
            # フィルタリング適用
            if self.filter_successful_only:
                # 成功した戦略のみをフィルタリング
                filtered_steps = [step for step in self.current_example_steps if step["tactic_apply"]]
                # 証明が完了していない場合は例全体をスキップ
                if not is_proved or not filtered_steps:
                    self.current_example_steps = []
                    return
                self.current_example_steps = filtered_steps
            
            # 戦略シーケンスを作成
            tactic_sequence = []
            for step in self.current_example_steps:
                tactic = step.get('tactic', {})
                main = tactic.get('main', '')
                arg1 = tactic.get('arg1')
                arg2 = tactic.get('arg2')
                
                if main:
                    if arg1 and arg2:
                        tactic_sequence.append(f"{main} {arg1} {arg2}")
                    elif arg1:
                        tactic_sequence.append(f"{main} {arg1}")
                    else:
                        tactic_sequence.append(main)
            
            # 例のサマリーを作成
            summary = None
            if is_proved:
                summary = {
                    "tactic_sequence": tactic_sequence,
                    "total_steps": len(self.current_example_steps)
                }
            
            # 例を作成
            example = {
                "example_id": f"ex_{self.example_id + 1:04d}",
                "meta": {
                    "goal_original": self.original_goal,
                    "is_proved": is_proved
                },
                "steps": self.current_example_steps,
                "summary": summary
            }
            
            self.examples.append(example)
            self.current_example_steps = []
    
    def save_data(self):
        """収集したデータを新しい形式でJSONファイルに保存"""
        with open(self.dataset_file_path, 'w', encoding='utf-8') as f:
            json.dump(self.examples, f, ensure_ascii=False, indent=2)
    
    def get_stats(self) -> Dict[str, int]:
        """収集したデータの統計情報を取得"""
        total_records = sum(len(example["steps"]) for example in self.examples)
        successful_tactics = sum(
            sum(1 for step in example["steps"] if step["tactic_apply"]) 
            for example in self.examples
        )
        proved_records = sum(
            sum(1 for step in example["steps"] if example["meta"]["is_proved"]) 
            for example in self.examples
        )
        
        return {
            "total_records": total_records,
            "successful_tactics": successful_tactics,
            "proved_records": proved_records
        }


def main() -> None:
    # パラメータを初期化
    gen_params = get_generation_params()
    train_params = get_training_params()
    model_params = get_model_params()
    system_params = get_system_params()
    
    parser = argparse.ArgumentParser(description="Generate formulas, run Transformer, apply tactic in pyprover")
    parser.add_argument("--count", type=int, default=gen_params.count, help="number of interactions to run")
    parser.add_argument("--difficulty", type=float, default=gen_params.difficulty)
    parser.add_argument("--seed", type=int, default=gen_params.seed)
    parser.add_argument("--device", type=str, default=system_params.device.value, choices=[e.value for e in DeviceType])
    parser.add_argument("--selftest", action="store_true", help="run intro/apply deterministic checks and exit")
    parser.add_argument("--max_steps", type=int, default=gen_params.max_steps, help="max number of tactic steps (attempts) per example, including both successful and failed attempts")
    parser.add_argument("--collect_data", action="store_true", default=train_params.collect_data, help="collect training data in JSON format")
    parser.add_argument("--work_file", type=str, default=train_params.work_file, help="temporary work file for data collection")
    parser.add_argument("--dataset_file", type=str, default=train_params.dataset_file, help="dataset file for collected data")
    parser.add_argument("--filter_successful_only", action="store_true", help="only store records where both tactic_apply and is_proved are true")
    args = parser.parse_args()

    # パラメータを更新
    default_params.update_generation_params(
        count=args.count,
        difficulty=args.difficulty,
        seed=args.seed,
        max_steps=args.max_steps
    )
    default_params.update_training_params(
        collect_data=args.collect_data,
        work_file=args.work_file,
        dataset_file=args.dataset_file
    )
    default_params.update_system_params(
        device=DeviceType(args.device)
    )
    
    # フィルタリング設定を更新
    if args.filter_successful_only:
        default_params.update_training_params(filter_type=DataFilterType.SUCCESSFUL_ONLY)

    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    token_py_path = os.path.join(root_dir, "src", "core", "fof_tokens.py")
    pyprover_dir = os.path.join(root_dir, "pyprover")
    
    # システムパラメータを更新
    default_params.update_system_params(
        root_dir=root_dir,
        token_py_path=token_py_path,
        pyprover_dir=pyprover_dir
    )

    base_tokens, label_names = load_tokens_and_labels_from_token_py(token_py_path)

    # モデルパラメータを更新
    default_params.update_model_params(
        vocab_size=len(base_tokens) + 5,  # base_tokens + special tokens
        pad_id=0  # 特殊トークンの最初がPAD
    )

    # Build tokenizer and model
    tokenizer = CharTokenizer(base_tokens=base_tokens)
    
    # 階層分類用のラベルマッピングを取得
    from src.core.parameter import get_hierarchical_labels
    hierarchical_labels = get_hierarchical_labels()
    main_to_id, arg1_to_id, arg2_to_id, id_to_main, id_to_arg1, id_to_arg2 = build_hierarchical_label_mappings(
        hierarchical_labels.main_tactics,
        hierarchical_labels.arg1_values,
        hierarchical_labels.arg2_values
    )
    
    model = TransformerClassifier(
        vocab_size=tokenizer.vocab_size,
        pad_id=tokenizer.pad_id,
        max_seq_len=model_params.max_seq_len,
        d_model=model_params.d_model,
        nhead=model_params.nhead,
        num_layers=model_params.num_layers,
        dim_feedforward=model_params.dim_feedforward,
        dropout=model_params.dropout,
        num_main_classes=len(id_to_main),
        num_arg1_classes=len(id_to_arg1),
        num_arg2_classes=len(id_to_arg2),
    )

    device = torch.device(default_params.get_device())
    model.to(device)
    model.eval()

    # Import pyprover robustly
    proposition_mod, prover_mod = import_pyprover(pyprover_dir)
    PropParseTree = proposition_mod.PropParseTree
    prop_parser = proposition_mod.parser
    Prover = prover_mod.Prover

    # Build generator
    variables = [t for t in gen_params.variables if t in base_tokens] or gen_params.variables
    gen = FormulaGenerator(
        variables=variables, 
        allow_const=gen_params.allow_const, 
        difficulty=gen_params.difficulty, 
        seed=gen_params.seed
    )
    
    # Initialize data collector if needed
    data_collector = None
    if train_params.collect_data:
        # Clear existing dataset file at start
        if os.path.exists(train_params.dataset_file):
            os.remove(train_params.dataset_file)
        
        # フィルタリングフラグを取得
        filter_flags = default_params.get_filter_flags()
        data_collector = StructuredDataCollector(
            dataset_file_path=train_params.dataset_file,
            filter_successful_only=filter_flags.get("filter_successful_only", False)
        )

    if args.selftest:
        # Deterministic intro test: goal (a → a), no premises
        parse_tree = PropParseTree()
        with pushd(pyprover_dir):
            goal_intro = parse_tree.transform(prop_parser.parse("(a → a)"))
        prover_intro = Prover(goal_intro)
        ok_intro = not prover_intro.intro()
        # After intro, goal becomes 'a' and variables contains 'a'; assumption should then solve
        solved_after_assumption = False
        if ok_intro:
            solved_after_assumption = not prover_intro.assumption() and prover_intro.goal is None
        print("[selftest:intro] intro ok:", ok_intro, " then assumption solved:", solved_after_assumption)

        # Deterministic apply test: premise (a → b), goal b
        with pushd(pyprover_dir):
            goal_apply = parse_tree.transform(prop_parser.parse("b"))
            prem_apply = parse_tree.transform(prop_parser.parse("(a → b)"))
        prover_apply = Prover(goal_apply)
        prover_apply.variables = [prem_apply]
        ok_apply = not prover_apply.apply(0)
        print("[selftest:apply] apply ok:", ok_apply, " new goal:", prover_apply.goal)
        return

    try:
        for i in range(gen_params.count):
            # Generate an initial state for the prover (we still use generator to seed it)
            goal_list = filter_formulas(gen, max_len=gen_params.max_len, require_tautology=True, limit=1)
            if not goal_list:
                print(f"Warning: No valid formulas generated for example {i+1}, skipping...")
                continue
            seed_goal = goal_list[0]

            # Pyprover parse and create state
            parse_tree = PropParseTree()
            with pushd(pyprover_dir):
                goal_node = parse_tree.transform(prop_parser.parse(seed_goal))
            prover = Prover(goal_node)

            if not train_params.collect_data:
                print(f"Example {i+1}")
                print(f"  Goal    : {seed_goal}")

            # Start data collection for this example
            if data_collector:
                initial_state = encode_prover_state(prover)
                data_collector.start_example(
                    example_id=i,
                    initial_premises=initial_state["premises"],
                    initial_goal=initial_state["goal"]
                )

            step = 0
            solved = prover.goal is None
            while not solved:
                # Extract current state for tokenization
                current_state = encode_prover_state(prover)
                premises, goal = current_state["premises"], current_state["goal"]
                
                # Maintain a banned set to avoid repeating failed predictions in this step
                banned: set[str] = set()
                applied = False

                while not applied and len(banned) < len(label_names) and step < gen_params.max_steps:
                    ids, mask, seg = tokenizer.encode(goal, premises, model_params.max_seq_len)
                    with torch.no_grad():
                        outputs = model(
                            ids.unsqueeze(0).to(device),
                            mask.unsqueeze(0).to(device),
                            seg.unsqueeze(0).to(device),
                        )
                        
                        # 階層分類モデル
                        if isinstance(outputs, tuple):
                            # 階層分類モデル
                            main_logits, arg1_logits, arg2_logits = outputs
                            
                            # 禁止されたタクティクをマスキング
                            if banned:
                                # 禁止されたタクティクの主タクティクをマスキング
                                for banned_tactic in banned:
                                    if banned_tactic in ['assumption', 'intro', 'split', 'left', 'right', 'add_dn']:
                                        # 引数なしのタクティク
                                        if banned_tactic in main_to_id:
                                            main_logits[0, main_to_id[banned_tactic]] = float('-inf')
                                    elif banned_tactic.startswith('apply ') or banned_tactic.startswith('destruct '):
                                        # apply/destruct タクティク
                                        if 'apply' in main_to_id:
                                            main_logits[0, main_to_id['apply']] = float('-inf')
                                        if 'destruct' in main_to_id:
                                            main_logits[0, main_to_id['destruct']] = float('-inf')
                                    elif banned_tactic.startswith('specialize '):
                                        # specialize タクティク
                                        if 'specialize' in main_to_id:
                                            main_logits[0, main_to_id['specialize']] = float('-inf')
                            
                            # 主タクティクを予測
                            main_pred_id = int(torch.argmax(main_logits, dim=-1).item())
                            main_tactic = id_to_main[main_pred_id]
                            
                            # 引数を予測
                            arg1_pred_id = int(torch.argmax(arg1_logits, dim=-1).item())
                            arg1_value = id_to_arg1[arg1_pred_id]
                            arg2_pred_id = int(torch.argmax(arg2_logits, dim=-1).item())
                            arg2_value = id_to_arg2[arg2_pred_id]
                            
                            # タクティク文字列を構築
                            if main_tactic in ['assumption', 'intro', 'split', 'left', 'right', 'add_dn']:
                                pred_label = main_tactic
                            elif main_tactic in ['apply', 'destruct']:
                                pred_label = f"{main_tactic} {arg1_value}"
                            elif main_tactic == 'specialize':
                                pred_label = f"{main_tactic} {arg1_value} {arg2_value}"
                            else:
                                pred_label = main_tactic
                                
                        else:
                            # 階層分類モデル
                            main_logits, arg1_logits, arg2_logits = outputs
                            if banned:
                                mask_vec = torch.zeros_like(main_logits)
                                for b in banned:
                                    if b in main_to_id:
                                        idx = main_to_id[b]
                                        if idx < main_logits.size(-1):
                                            mask_vec[0, idx] = float('-inf')
                                main_logits = main_logits + mask_vec
                            pred_id = int(torch.argmax(main_logits, dim=-1).item())
                            pred_label = id_to_main[pred_id]

                    # Record tactic application for data collection (BEFORE applying tactic)
                    if data_collector:
                        data_collector.add_tactic_application(
                            step=step,
                            premises=premises,
                            goal=goal,
                            tactic=pred_label,
                            tactic_apply=False  # Will be updated after applying tactic
                        )
                    
                    ok = apply_tactic_from_label(prover, pred_label)
                    
                    # Update tactic_apply result after applying tactic
                    if data_collector:
                        data_collector.update_last_tactic_apply(ok)
                    
                    step += 1
                    
                    if not train_params.collect_data:
                        print(f"  Step {step}: {pred_label} -> {'applied' if ok else 'failed'}")
                        current_premises_all = " | ".join([str(v) for v in getattr(prover, "variables", [])])
                        print(f"    premises = {current_premises_all}")
                        print(f"    goal     = {prover.goal}")

                    if ok:
                        applied = True
                    else:
                        banned.add(pred_label)

                if not applied:
                    if not train_params.collect_data:
                        print("  Stuck: no applicable tactic predicted")
                    break

                solved = prover.goal is None

            # Finish data collection for this example
            if data_collector:
                data_collector.finish_example(is_proved=solved)

            if not train_params.collect_data:
                if solved:
                    print("  Result  : goal solved")
                else:
                    print(f"  Result  : not solved within {gen_params.max_steps} step limit")
                print()
            
    finally:
        if data_collector:
            data_collector.save_data()
            stats = data_collector.get_stats()
            print(f"Data collection completed. "
                  f"Records: {stats['total_records']}, "
                  f"Successful tactics: {stats['successful_tactics']}, "
                  f"Proved records: {stats['proved_records']}")


if __name__ == "__main__":
    main()


