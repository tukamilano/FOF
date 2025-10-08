from __future__ import annotations

import argparse
import contextlib
import os
import sys
import time
from typing import List, Tuple, Dict

import torch

from transformer_classifier import (
    load_tokens_and_labels_from_token_py,
    CharTokenizer,
    TransformerClassifier,
    build_label_mappings,
)
from generate_prop import FormulaGenerator, filter_formulas
from parameter import (
    default_params, get_model_params, get_training_params, 
    get_generation_params, get_system_params, DeviceType, DataFilterType
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


class TimingProfiler:
    def __init__(self):
        self.timings: Dict[str, List[float]] = {
            'total_iteration': [],
            'transformer_inference': [],
            'tactic_application': [],
            'state_extraction': [],
            'tokenization': [],
            'other_processing': []
        }
    
    def add_timing(self, category: str, duration: float):
        if category in self.timings:
            self.timings[category].append(duration)
    
    def get_statistics(self) -> Dict[str, Dict[str, float]]:
        stats = {}
        for category, times in self.timings.items():
            if times:
                stats[category] = {
                    'mean': sum(times) / len(times),
                    'min': min(times),
                    'max': max(times),
                    'total': sum(times),
                    'count': len(times)
                }
            else:
                stats[category] = {
                    'mean': 0.0,
                    'min': 0.0,
                    'max': 0.0,
                    'total': 0.0,
                    'count': 0
                }
        return stats
    
    def print_summary(self):
        stats = self.get_statistics()
        print("\n" + "="*60)
        print("TIMING ANALYSIS SUMMARY")
        print("="*60)
        
        # Calculate percentages relative to total iteration time
        total_mean = stats['total_iteration']['mean']
        if total_mean > 0:
            print(f"\nAverage time per iteration: {total_mean:.4f} seconds")
            print(f"Total iterations measured: {stats['total_iteration']['count']}")
            print("\nBreakdown by component:")
            print("-" * 40)
            
            for category in ['transformer_inference', 'tactic_application', 'state_extraction', 'tokenization']:
                if stats[category]['count'] > 0:
                    mean_time = stats[category]['mean']
                    percentage = (mean_time / total_mean) * 100
                    print(f"{category:20s}: {mean_time:.4f}s ({percentage:5.1f}%)")
            
            # Transformer inference percentage
            transformer_percentage = (stats['transformer_inference']['mean'] / total_mean) * 100
            print(f"\nTransformer inference accounts for {transformer_percentage:.1f}% of total iteration time")
        else:
            print("No timing data available")


def evaluate_timing(
    model: TransformerClassifier,
    tokenizer: CharTokenizer,
    device: torch.device,
    prover,
    label_to_id: Dict[str, int],
    id_to_label: Dict[int, str],
    max_interactions: int = 5,
    profiler: TimingProfiler = None
) -> Tuple[bool, List[Tuple[str, bool]], int]:
    """Run one iteration of the transformer + pyprover loop and measure timing
    
    Args:
        max_interactions: Maximum number of tactic steps (attempts) per example.
                         This includes both successful and failed tactic attempts.
                         Each tactic attempt counts as one step.
    
    Returns:
        Tuple of (solved, tactic_sequence, steps) where:
        - solved: whether the goal was solved
        - tactic_sequence: list of (tactic_name, success) tuples
        - steps: number of tactic steps completed (both successful and failed)
    """
    
    step = 0
    solved = prover.goal is None
    tactic_sequence = []  # Record all attempted tactics with their success status
    
    while not solved:
        iteration_start = time.time()
        
        # State extraction timing
        extraction_start = time.time()
        p1, p2, p3, goal = extract_inputs_from_prover(prover, tokenizer.max_sentence_length)
        extraction_time = time.time() - extraction_start
        
        # Maintain a banned set to avoid repeating failed predictions in this step
        banned: set[str] = set()
        applied = False
        
        # Initialize cumulative times for this iteration
        total_tokenization_time = 0
        total_inference_time = 0
        total_tactic_time = 0
        
        while not applied and len(banned) < len(label_to_id) and step < max_interactions:
            # Tokenization timing
            tokenization_start = time.time()
            ids, mask, seg = tokenizer.encode_four_fixed_blocks(p1, p2, p3, goal)
            tokenization_time = time.time() - tokenization_start
            total_tokenization_time += tokenization_time
            
            # Transformer inference timing
            inference_start = time.time()
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
            inference_time = time.time() - inference_start
            total_inference_time += inference_time
            
            # Tactic application timing
            tactic_start = time.time()
            ok = apply_tactic_from_label(prover, pred_label)
            tactic_time = time.time() - tactic_start
            total_tactic_time += tactic_time
            
            # Record this tactic attempt (both successful and failed)
            tactic_sequence.append((pred_label, ok))
            
            # Count each tactic attempt as a step (both successful and failed)
            step += 1
            
            if profiler:
                # Record individual tactic timing
                profiler.add_timing('transformer_inference', inference_time)
                profiler.add_timing('tactic_application', tactic_time)
                profiler.add_timing('state_extraction', extraction_time)
                profiler.add_timing('tokenization', tokenization_time)
                
                # Record total time for this tactic attempt
                tactic_total_time = inference_time + tactic_time + extraction_time + tokenization_time
                profiler.add_timing('total_iteration', tactic_total_time)
            
            if ok:
                applied = True
            else:
                banned.add(pred_label)
        
        # Note: other_processing is not recorded per iteration as it's mainly loop control overhead
        
        if not applied:
            break
        
        solved = prover.goal is None
    
    return solved, tactic_sequence, step


def main() -> None:
    # パラメータを初期化
    gen_params = get_generation_params()
    train_params = get_training_params()
    model_params = get_model_params()
    system_params = get_system_params()
    
    parser = argparse.ArgumentParser(description="Evaluate timing of transformer + pyprover loop")
    parser.add_argument("--count", type=int, default=gen_params.count, help="number of examples to test")
    parser.add_argument("--difficulty", type=float, default=gen_params.difficulty)
    parser.add_argument("--seed", type=int, default=gen_params.seed)
    parser.add_argument("--device", type=str, default=system_params.device.value, choices=[e.value for e in DeviceType])
    parser.add_argument("--max_interactions", type=int, default=train_params.max_interactions, help="max number of tactic steps (attempts) per example, including both successful and failed attempts")
    parser.add_argument("--warmup", type=int, default=train_params.warmup_iterations, help="number of warmup iterations to exclude from timing")
    args = parser.parse_args()

    # パラメータを更新
    default_params.update_generation_params(
        count=args.count,
        difficulty=args.difficulty,
        seed=args.seed
    )
    default_params.update_training_params(
        max_interactions=args.max_interactions,
        warmup_iterations=args.warmup
    )
    default_params.update_system_params(
        device=DeviceType(args.device)
    )

    root_dir = os.path.dirname(__file__)
    token_py_path = os.path.join(root_dir, "fof_tokens.py")
    pyprover_dir = os.path.join(root_dir, "pyprover")
    
    # システムパラメータを更新
    default_params.update_system_params(
        root_dir=root_dir,
        token_py_path=token_py_path,
        pyprover_dir=pyprover_dir
    )

    # Load tokens and labels
    base_tokens, label_names = load_tokens_and_labels_from_token_py(token_py_path)
    label_to_id, id_to_label = build_label_mappings(label_names)

    # モデルパラメータを更新
    default_params.update_model_params(
        vocab_size=len(base_tokens) + 5,  # base_tokens + special tokens
        num_classes=len(label_names),
        pad_id=0  # 特殊トークンの最初がPAD
    )

    # Build tokenizer and model
    tokenizer = CharTokenizer(base_tokens=base_tokens, max_sentence_length=model_params.max_sentence_length)
    max_seq_len = 1 + 4 * (tokenizer.max_sentence_length + 1)
    model = TransformerClassifier(
        vocab_size=tokenizer.vocab_size,
        num_classes=len(label_names),
        pad_id=tokenizer.pad_id,
        max_seq_len=max_seq_len,
        d_model=model_params.d_model,
        nhead=model_params.nhead,
        num_layers=model_params.num_layers,
        dim_feedforward=model_params.dim_feedforward,
        dropout=model_params.dropout,
    )

    device = torch.device(default_params.get_device())
    if device.type == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available or not properly configured, falling back to CPU")
        device = torch.device("cpu")
    
    print(f"Using device: {device}")
    model.to(device)
    model.eval()

    # Import pyprover
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

    # Initialize profiler
    profiler = TimingProfiler()

    print(f"Evaluating timing for {gen_params.count} examples with {train_params.warmup_iterations} warmup iterations")
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"CUDA device: {torch.cuda.get_device_name(device)}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(device).total_memory / 1024**3:.1f} GB")
    print(f"Max tactic steps per example: {train_params.max_interactions}")
    print("-" * 60)

    solved_count = 0
    total_examples = 0

    for i in range(gen_params.count + train_params.warmup_iterations):
        # Generate an initial state for the prover
        premises = filter_formulas(gen, max_len=gen_params.max_len, require_tautology=False, limit=3)
        goal_list = filter_formulas(gen, max_len=gen_params.max_len, require_tautology=False, limit=1)
        seed_p1, seed_p2, seed_p3 = premises
        seed_goal = goal_list[0]

        # Pyprover parse and create state
        parse_tree = PropParseTree()
        with pushd(pyprover_dir):
            goal_node = parse_tree.transform(prop_parser.parse(seed_goal))
            p_nodes = [parse_tree.transform(prop_parser.parse(s)) for s in [seed_p1, seed_p2, seed_p3]]
        prover = Prover(goal_node)
        prover.variables = p_nodes[:]

        is_warmup = i < train_params.warmup_iterations
        if not is_warmup:
            total_examples += 1
            print(f"Example {total_examples}")
            print(f"  Premises: {seed_p1} | {seed_p2} | {seed_p3}")
            print(f"  Goal    : {seed_goal}")

        # Run timing evaluation
        solved, tactic_sequence, iterations = evaluate_timing(
            model, tokenizer, device, prover, 
            label_to_id, id_to_label, train_params.max_interactions, 
            profiler if not is_warmup else None
        )

        if not is_warmup:
            # Display step and tactic information
            print(f"  Steps completed: {iterations}/{train_params.max_interactions}")
            
            # Display tactic sequence
            if tactic_sequence:
                print(f"  Tactic sequence ({len(tactic_sequence)} steps):")
                for j, (tactic, success) in enumerate(tactic_sequence):
                    status = "✓" if success else "✗"
                    print(f"    Step {j+1}: {tactic} {status}")
            else:
                print("  No tactics attempted")
            
            if solved:
                solved_count += 1
                print("  Result  : goal solved")
            else:
                print(f"  Result  : not solved within {args.max_interactions} step limit")
            print()

    # Print timing analysis
    profiler.print_summary()
    
    print(f"\nOverall Results:")
    print(f"  Examples tested: {total_examples}")
    print(f"  Examples solved: {solved_count}")
    print(f"  Success rate: {solved_count/total_examples*100:.1f}%")
    
    # Calculate tactic success statistics
    if profiler.timings['transformer_inference']:
        total_tactics = len(profiler.timings['transformer_inference'])
        print(f"  Total tactics attempted: {total_tactics}")
        print(f"  Average tactics per example: {total_tactics/total_examples:.1f}")


if __name__ == "__main__":
    main()
