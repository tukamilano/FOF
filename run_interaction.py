from __future__ import annotations

import argparse
import contextlib
import os
import sys
from typing import List, Tuple

import torch

from transformer_classifier import (
    load_tokens_and_labels_from_token_py,
    CharTokenizer,
    TransformerClassifier,
    build_label_mappings,
)
from generate_prop import FormulaGenerator, filter_formulas
from training_data_collector import TrainingDataCollector
from state_encoder import encode_prover_state, encode_prover_state_for_transformer


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
    # Returns True if tactic succeeded, False if failed
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


def extract_inputs_from_prover(prover, max_len: int) -> Tuple[str, str, str, str]:
    """既存のTransformer用の形式でエンコード（後方互換性のため）"""
    return encode_prover_state_for_transformer(prover, max_len)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate formulas, run Transformer, apply tactic in pyprover")
    parser.add_argument("--count", type=int, default=3, help="number of interactions to run")
    parser.add_argument("--difficulty", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--selftest", action="store_true", help="run intro/apply deterministic checks and exit")
    parser.add_argument("--max_steps", type=int, default=5, help="max number of tactic steps (attempts) per example, including both successful and failed attempts")
    parser.add_argument("--collect_data", action="store_true", help="collect training data in JSON format")
    parser.add_argument("--work_file", type=str, default="temp_work.json", help="temporary work file for data collection")
    parser.add_argument("--dataset_file", type=str, default="training_data.json", help="dataset file for collected data")
    args = parser.parse_args()

    root_dir = os.path.dirname(__file__)
    token_py_path = os.path.join(root_dir, "fof_tokens.py")
    pyprover_dir = os.path.join(root_dir, "pyprover")

    base_tokens, label_names = load_tokens_and_labels_from_token_py(token_py_path)
    label_to_id, id_to_label = build_label_mappings(label_names)

    # Build tokenizer and model
    tokenizer = CharTokenizer(base_tokens=base_tokens, max_sentence_length=50)
    max_seq_len = 1 + 4 * (tokenizer.max_sentence_length + 1)
    model = TransformerClassifier(
        vocab_size=tokenizer.vocab_size,
        num_classes=len(label_names),
        pad_id=tokenizer.pad_id,
        max_seq_len=max_seq_len,
        d_model=128,
        nhead=4,
        num_layers=2,
        dim_feedforward=256,
        dropout=0.1,
    )

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    model.to(device)
    model.eval()

    # Import pyprover robustly
    proposition_mod, prover_mod = import_pyprover(pyprover_dir)
    PropParseTree = proposition_mod.PropParseTree
    prop_parser = proposition_mod.parser
    Prover = prover_mod.Prover

    # Build generator
    variables = [t for t in ["a", "b", "c"] if t in base_tokens] or ["a", "b", "c"]
    gen = FormulaGenerator(variables=variables, allow_const=False, difficulty=args.difficulty, seed=args.seed)
    
    # Initialize data collector if needed
    data_collector = None
    if args.collect_data:
        # Clear existing dataset file at start
        if os.path.exists(args.dataset_file):
            os.remove(args.dataset_file)
        
        data_collector = TrainingDataCollector(
            work_file_path=args.work_file,
            dataset_file_path=args.dataset_file
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
        for i in range(args.count):
            # Generate an initial state for the prover (we still use generator to seed it)
            goal_list = filter_formulas(gen, max_len=50, require_tautology=True, limit=1)
            seed_goal = goal_list[0]

            # Pyprover parse and create state
            parse_tree = PropParseTree()
            with pushd(pyprover_dir):
                goal_node = parse_tree.transform(prop_parser.parse(seed_goal))
            prover = Prover(goal_node)

            if not args.collect_data:
                print(f"Example {i+1}")
                print(f"  Goal    : {seed_goal}")

            # Start data collection for this example
            if data_collector:
                initial_state = encode_prover_state(prover, max_len=None)  # 完全なデータを保存
                data_collector.start_example(
                    example_id=i,
                    initial_premises=initial_state["premises"],
                    initial_goal=initial_state["goal"]
                )

            step = 0
            solved = prover.goal is None
            while not solved:
                # Extract inputs from current state
                p1, p2, p3, goal = extract_inputs_from_prover(prover, tokenizer.max_sentence_length)

                # Maintain a banned set to avoid repeating failed predictions in this step
                banned: set[str] = set()
                applied = False

                while not applied and len(banned) < len(label_names) and step < args.max_steps:
                    ids, mask, seg = tokenizer.encode_four_fixed_blocks(p1, p2, p3, goal)
                    with torch.no_grad():
                        logits = model(
                            ids.unsqueeze(0).to(device),
                            mask.unsqueeze(0).to(device),
                            seg.unsqueeze(0).to(device),
                        )[0]
                        if banned:
                            mask_vec = torch.zeros_like(logits)
                            for b in banned:
                                mask_vec[label_to_id[b]] = float('-inf')
                            logits = logits + mask_vec
                        pred_id = int(torch.argmax(logits, dim=-1).item())
                        pred_label = id_to_label[pred_id]

                    # Record tactic application for data collection (BEFORE applying tactic)
                    if data_collector:
                        current_state = encode_prover_state(prover, max_len=None)  # 完全なデータを保存
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
                    
                    if not args.collect_data:
                        print(f"  Step {step}: {pred_label} -> {'applied' if ok else 'failed'}")
                        current_premises_all = " | ".join([str(v) for v in getattr(prover, "variables", [])])
                        print(f"    premises = {current_premises_all}")
                        print(f"    goal     = {prover.goal}")

                    if ok:
                        applied = True
                    else:
                        banned.add(pred_label)

                if not applied:
                    if not args.collect_data:
                        print("  Stuck: no applicable tactic predicted")
                    break

                solved = prover.goal is None

            # Finish data collection for this example
            if data_collector:
                data_collector.finish_example(is_proved=solved)

            if not args.collect_data:
                if solved:
                    print("  Result  : goal solved")
                else:
                    print(f"  Result  : not solved within {args.max_steps} step limit")
                print()
            
    finally:
        # Cleanup data collector
        if data_collector:
            data_collector.cleanup()
            stats = data_collector.get_dataset_stats()
            print(f"Data collection completed. Total records: {stats['total_records']}, "
                  f"Proved examples: {stats['proved_examples']}, "
                  f"Failed examples: {stats['failed_examples']}")


if __name__ == "__main__":
    main()


