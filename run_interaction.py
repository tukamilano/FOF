from __future__ import annotations

import argparse
import os
import sys
from typing import List, Tuple

import torch

from transformer_classifier import (
    load_tokens_and_labels_from_token_py,
    CharTokenizer,
    TransformerClassifier,
    build_hierarchical_label_mappings,
)
from generate_prop import FormulaGenerator, filter_formulas
from training_data_collector import TrainingDataCollector
from state_encoder import encode_prover_state, format_tactic_string
from parameter import (
    default_params, get_model_params, get_training_params, 
    get_generation_params, get_system_params, DeviceType, DataFilterType
)
from utils import pushd, import_pyprover


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


def extract_inputs_from_prover(prover) -> Tuple[List[str], str]:
    """proverの状態をエンコード（制限なし）"""
    state = encode_prover_state(prover)
    return state["premises"], state["goal"]


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

    root_dir = os.path.dirname(__file__)
    token_py_path = os.path.join(root_dir, "fof_tokens.py")
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
    from parameter import get_hierarchical_labels
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
        
        filter_flags = default_params.get_filter_flags()
        data_collector = TrainingDataCollector(
            work_file_path=train_params.work_file,
            dataset_file_path=train_params.dataset_file,
            filter_successful_only=filter_flags["filter_successful_only"]
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
                initial_state = encode_prover_state(prover)  # 完全なデータを保存（制限なし）
                data_collector.start_example(
                    example_id=i,
                    initial_premises=initial_state["premises"],
                    initial_goal=initial_state["goal"]
                )

            step = 0
            solved = prover.goal is None
            while not solved:
                # Extract inputs from current state (no length restrictions)
                premises, goal = extract_inputs_from_prover(prover)

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
                        current_state = encode_prover_state(prover)  # 完全なデータを保存（制限なし）
                        # 戦略適用前の状態を記録（tactic_applyは後で更新）
                        data_collector.add_tactic_application(
                            step=step + 1,  # stepを先にインクリメント
                            premises=current_state["premises"],
                            goal=current_state["goal"],
                            tactic=pred_label,
                            tactic_apply=False  # 仮の値、後で更新
                        )
                    
                    ok = apply_tactic_from_label(prover, pred_label)
                    
                    # Update tactic_apply result after applying tactic
                    if data_collector:
                        data_collector.update_last_tactic_apply(ok)
                    
                    # Count each tactic attempt as a step (both successful and failed)
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
        # Cleanup data collector
        if data_collector:
            data_collector.cleanup()
            stats = data_collector.get_dataset_stats()
            if stats.get('filtered_mode') == "successful_only":
                print(f"Data collection completed (successful tactics + proved only). "
                      f"Examples: {stats['total_examples']} (proved: {stats['proved_examples']}, failed: {stats['failed_examples']}), "
                      f"Records: {stats['total_records']}, "
                      f"Successful tactics: {stats['successful_tactics']}")
            elif stats.get('filtered_mode') == "tactic_success_only":
                print(f"Data collection completed (successful tactics only). "
                      f"Examples: {stats['total_examples']} (proved: {stats['proved_examples']}, failed: {stats['failed_examples']}), "
                      f"Records: {stats['total_records']}, "
                      f"Successful tactics: {stats['successful_tactics']}")
            else:
                print(f"Data collection completed. "
                      f"Examples: {stats['total_examples']} (proved: {stats['proved_examples']}, failed: {stats['failed_examples']}), "
                      f"Records: {stats['total_records']}")


if __name__ == "__main__":
    main()


