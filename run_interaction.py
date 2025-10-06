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
        return not prover.apply(int(parts[1]))
    if parts[0] == "destruct" and len(parts) == 2 and parts[1].isdigit():
        return not prover.destruct(int(parts[1]))
    if parts[0] == "specialize" and len(parts) == 3 and parts[1].isdigit() and parts[2].isdigit():
        return not prover.specialize(int(parts[1]), int(parts[2]))
    return False


def extract_inputs_from_prover(prover, max_len: int) -> Tuple[str, str, str, str]:
    # Take first three premises; ignore extras. If fewer than 3, pad with empty strings.
    vars_as_str: List[str] = [str(v) for v in getattr(prover, "variables", [])]
    p1_full = vars_as_str[0] if len(vars_as_str) > 0 else ""
    p2_full = vars_as_str[1] if len(vars_as_str) > 1 else ""
    p3_full = vars_as_str[2] if len(vars_as_str) > 2 else ""
    goal_full = str(getattr(prover, "goal", "")) if getattr(prover, "goal", None) is not None else ""
    # Truncate to tokenizer's max sentence length
    p1 = p1_full[:max_len]
    p2 = p2_full[:max_len]
    p3 = p3_full[:max_len]
    goal_str = goal_full[:max_len]
    return p1, p2, p3, goal_str


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate formulas, run Transformer, apply tactic in pyprover")
    parser.add_argument("--count", type=int, default=3, help="number of interactions to run")
    parser.add_argument("--difficulty", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--selftest", action="store_true", help="run intro/apply deterministic checks and exit")
    parser.add_argument("--max_steps", type=int, default=5, help="max number of tactic applications to attempt per example")
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

    for i in range(args.count):
        # Generate an initial state for the prover (we still use generator to seed it)
        premises = filter_formulas(gen, max_len=50, require_tautology=False, limit=3)
        goal_list = filter_formulas(gen, max_len=50, require_tautology=False, limit=1)
        seed_p1, seed_p2, seed_p3 = premises
        seed_goal = goal_list[0]

        # Pyprover parse and create state
        parse_tree = PropParseTree()
        with pushd(pyprover_dir):
            goal_node = parse_tree.transform(prop_parser.parse(seed_goal))
            p_nodes = [parse_tree.transform(prop_parser.parse(s)) for s in [seed_p1, seed_p2, seed_p3]]
        prover = Prover(goal_node)
        prover.variables = p_nodes[:]  # seed premises as assumptions

        print(f"Example {i+1}")
        print(f"  Premises: {seed_p1} | {seed_p2} | {seed_p3}")
        print(f"  Goal    : {seed_goal}")

        step = 0
        solved = prover.goal is None
        while step < args.max_steps and not solved:
            # Extract inputs from current state
            p1, p2, p3, goal = extract_inputs_from_prover(prover, tokenizer.max_sentence_length)

            # Maintain a banned set to avoid repeating failed predictions in this step
            banned: set[str] = set()
            applied = False

            while not applied and len(banned) < len(label_names):
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

                ok = apply_tactic_from_label(prover, pred_label)
                print(f"  Step {step+1}: {pred_label} -> {'applied' if ok else 'failed'}")
                current_premises_all = " | ".join([str(v) for v in getattr(prover, "variables", [])])
                print(f"    premises = {current_premises_all}")
                print(f"    goal     = {prover.goal}")

                if ok:
                    applied = True
                    step += 1
                else:
                    banned.add(pred_label)

            if not applied:
                print("  Stuck: no applicable tactic predicted")
                break

            solved = prover.goal is None

        if solved:
            print("  Result  : goal solved")
        else:
            print("  Result  : not solved within step limit")
        print()


if __name__ == "__main__":
    main()


