"""Parallel self improvement data collector.

This module provides a parallelised variant of
``self_improvement_data_collector.collect_self_improvement_data``.
It follows the logic of the sequential implementation but distributes
the proof search for individual tautologies across multiple worker
processes. The workers lazily load the required model and pyprover
modules on first use which keeps the startup overhead small while
ensuring each process owns its resources.
"""
from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import torch
from tqdm import tqdm

# プロジェクトルートをパスに追加
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.core.transformer_classifier import CharTokenizer, load_tokens_and_labels_from_token_py
from src.core.state_encoder import encode_prover_state
from src.interaction import self_improvement_data_collector as base_collector


# --- Worker global state -------------------------------------------------
_WORKER_MODEL: Optional[torch.nn.Module] = None
_WORKER_LABEL_MAPPINGS: Optional[Dict[str, Any]] = None
_WORKER_TOKENIZER: Optional[CharTokenizer] = None
_WORKER_DEVICE: Optional[torch.device] = None
_WORKER_MAX_SEQ_LEN: Optional[int] = None
_WORKER_MODEL_PATH: Optional[str] = None


def _ensure_worker_initialized(config: Dict[str, Any]) -> None:
    """Initialise heavy resources inside a worker process on demand."""

    global _WORKER_MODEL, _WORKER_LABEL_MAPPINGS, _WORKER_TOKENIZER
    global _WORKER_DEVICE, _WORKER_MAX_SEQ_LEN, _WORKER_MODEL_PATH

    model_path = config["model_path"]
    device_str = config["device"]
    max_seq_len = config["max_seq_len"]

    if _WORKER_MODEL is not None and _WORKER_MODEL_PATH == model_path:
        # Already initialised for this model path.
        return

    # Ensure the base collector initialises global constants (pyprover, tokens, ...)
    base_collector.initialize_global_constants()

    device = torch.device(device_str)
    model, label_mappings = base_collector.load_hierarchical_model(model_path, device)
    model.eval()

    # CharTokenizer requires base tokens.  ``initialize_global_constants`` already populates
    # ``BASE_TOKENS`` but we keep a fallback in case the sequential collector is updated.
    if base_collector.BASE_TOKENS is None:
        token_py_path = os.path.join(project_root, "src", "core", "fof_tokens.py")
        base_tokens, _ = load_tokens_and_labels_from_token_py(token_py_path)
    else:
        base_tokens = base_collector.BASE_TOKENS

    tokenizer = CharTokenizer(base_tokens=base_tokens)

    _WORKER_MODEL = model
    _WORKER_LABEL_MAPPINGS = label_mappings
    _WORKER_TOKENIZER = tokenizer
    _WORKER_DEVICE = device
    _WORKER_MAX_SEQ_LEN = max_seq_len
    _WORKER_MODEL_PATH = model_path


def _process_tautology_worker(args: Tuple[Any, ...]) -> Dict[str, Any]:
    """Process a single tautology inside a worker process."""

    (
        index,
        goal_str,
        max_steps,
        probability_threshold,
        temperature,
        config,
    ) = args

    try:
        _ensure_worker_initialized(config)
        assert _WORKER_MODEL is not None
        assert _WORKER_TOKENIZER is not None
        assert _WORKER_LABEL_MAPPINGS is not None
        assert _WORKER_DEVICE is not None
        assert _WORKER_MAX_SEQ_LEN is not None

        if not goal_str:
            return {
                "index": index,
                "goal": goal_str,
                "solved": False,
                "successful_tactics": [],
                "steps": 0,
                "error": "Empty goal string",
            }

        # Parse tautology and create prover instance.
        parse_tree = base_collector.PYPROVER_MODULES["PropParseTree"]()
        goal_node = parse_tree.transform(
            base_collector.PYPROVER_MODULES["prop_parser"].parse(goal_str)
        )
        prover = base_collector.PYPROVER_MODULES["Prover"](goal_node)

        example_successful_tactics: List[Dict[str, Any]] = []
        failed_tactics = set()
        step = 0
        solved = prover.goal is None
        example_terminated = False

        while not solved and step < max_steps and not example_terminated:
            current_state = encode_prover_state(prover)
            current_premises = current_state["premises"]
            current_goal = current_state["goal"]

            tactic_combinations = base_collector.generate_tactic_combinations(
                _WORKER_MODEL,
                _WORKER_TOKENIZER,
                current_premises,
                current_goal,
                _WORKER_LABEL_MAPPINGS,
                _WORKER_DEVICE,
                _WORKER_MAX_SEQ_LEN,
                probability_threshold,
            )

            success = False
            max_attempts = len(tactic_combinations)
            attempts = 0
            available_tactics = [
                tactic for tactic, _ in tactic_combinations if tactic not in failed_tactics
            ]

            while (
                not success
                and attempts < max_attempts
                and not example_terminated
                and available_tactics
            ):
                selected_tactic, _ = base_collector.select_tactic_probabilistically(
                    tactic_combinations, temperature, failed_tactics
                )

                if not selected_tactic:
                    example_terminated = True
                    break

                success = base_collector.apply_tactic_from_label(prover, selected_tactic)
                attempts += 1

                if success:
                    tactic_dict = base_collector.parse_tactic_string_cached(selected_tactic)
                    state_tactic_hash = base_collector.create_state_hash(
                        current_premises, current_goal, selected_tactic
                    )
                    example_successful_tactics.append(
                        {
                            "step_index": step,
                            "premises": list(current_premises or []),
                            "goal": current_goal,
                            "tactic": tactic_dict,
                            "tactic_apply": True,
                            "state_tactic_hash": state_tactic_hash,
                        }
                    )
                else:
                    failed_tactics.add(selected_tactic)
                    available_tactics = [
                        tactic
                        for tactic, _ in tactic_combinations
                        if tactic not in failed_tactics
                    ]

                    if not available_tactics:
                        example_terminated = True
                        break

            step += 1
            solved = prover.goal is None

        return {
            "index": index,
            "goal": goal_str,
            "solved": solved,
            "successful_tactics": example_successful_tactics if solved else [],
            "steps": step,
            "terminated": example_terminated,
            "error": None,
        }

    except Exception as exc:  # pragma: no cover - defensive logging in workers
        return {
            "index": index,
            "goal": goal_str,
            "solved": False,
            "successful_tactics": [],
            "steps": 0,
            "terminated": False,
            "error": str(exc),
        }


def collect_self_improvement_data_parallel(
    model_path: str,
    num_examples: int,
    max_steps: int,
    probability_threshold: float,
    temperature: float,
    device: torch.device,
    generated_data_dir: str,
    max_seq_len: int,
    num_workers: int,
) -> List[Dict[str, Any]]:
    """Collect self improvement data using multiple worker processes."""

    if num_workers <= 1:
        # Fallback to the original sequential implementation.
        base_collector.initialize_global_constants()
        if base_collector.BASE_TOKENS is None:
            token_py_path = os.path.join(project_root, "src", "core", "fof_tokens.py")
            base_tokens, _ = load_tokens_and_labels_from_token_py(token_py_path)
        else:
            base_tokens = base_collector.BASE_TOKENS

        tokenizer = CharTokenizer(base_tokens=base_tokens)
        model, label_mappings = base_collector.load_hierarchical_model(model_path, device)
        return base_collector.collect_self_improvement_data(
            model=model,
            tokenizer=tokenizer,
            label_mappings=label_mappings,
            device=device,
            max_seq_len=max_seq_len,
            num_examples=num_examples,
            max_steps=max_steps,
            probability_threshold=probability_threshold,
            verbose=False,
            generated_data_dir=generated_data_dir,
            temperature=temperature,
        )

    base_collector.initialize_global_constants()
    tautologies = base_collector.load_tautologies_from_generated_data(
        generated_data_dir=generated_data_dir,
        max_examples=num_examples,
    )

    if not tautologies:
        print("Failed to load tautologies from generated_data directory!")
        return []

    print(f"Loaded {len(tautologies)} tautologies from generated_data directory")
    print("First 5 tautologies:")
    for i in range(min(5, len(tautologies))):
        print(f"  {i + 1}: {tautologies[i]}")

    config = {
        "model_path": model_path,
        "device": str(device),
        "max_seq_len": max_seq_len,
    }

    tasks = [
        (index, goal, max_steps, probability_threshold, temperature, config)
        for index, goal in enumerate(tautologies)
    ]

    results: List[Optional[Dict[str, Any]]] = [None] * len(tasks)

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(_process_tautology_worker, task) for task in tasks]

        with tqdm(
            total=len(futures),
            desc="Processing tautologies",
            unit="formula",
        ) as progress:
            for future in as_completed(futures):
                result = future.result()
                results[result["index"]] = result
                progress.update(1)

    successful_tactics: List[Dict[str, Any]] = []
    solved_count = 0

    for result in results:
        if result is None:
            continue
        if result.get("error"):
            print(
                f"Warning: failed to process tautology {result['index'] + 1}: {result['error']}"
            )
            continue
        if result.get("solved"):
            solved_count += 1
            successful_tactics.extend(result.get("successful_tactics", []))

    print("\nSelf improvement data collection completed:")
    print(f"  Solved examples: {solved_count}/{len(tautologies)}")
    print(f"  Successful tactics collected: {len(successful_tactics)}")

    return successful_tactics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect self improvement data from solved examples in parallel"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/pretrained_model.pth",
        help="model path",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=100,
        help="number of examples to process",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=30,
        help="max steps per example",
    )
    parser.add_argument(
        "--probability_threshold",
        type=float,
        default=0.001,
        help="probability threshold for tactic generation",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="device",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=256,
        help="maximum sequence length",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="self_improvement_data",
        help="output directory for collected data",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1000,
        help="batch size for saving data",
    )
    parser.add_argument(
        "--generated_data_dir",
        type=str,
        default="generated_data",
        help="directory containing generated tautology data",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="temperature for probabilistic tactic selection",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=os.cpu_count() or 1,
        help="number of worker processes",
    )

    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    if not os.path.exists(args.model_path):
        print(f"Model file not found: {args.model_path}")
        print("Please provide a valid model path.")
        return

    # Determine max sequence length from checkpoint if available.
    try:
        checkpoint = torch.load(args.model_path, map_location="cpu")
        max_seq_len = checkpoint.get("max_seq_len", args.max_seq_len)
    except Exception:
        max_seq_len = args.max_seq_len

    print(f"Using device: {device}")
    print(f"Starting parallel self improvement data collection with {args.num_workers} workers...")
    print(f"  Examples to process: {args.count}")
    print(f"  Max steps per example: {args.max_steps}")
    print(f"  Probability threshold: {args.probability_threshold}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Generated data directory: {args.generated_data_dir}")
    print(f"  Output directory: {args.output_dir}")

    base_collector.clear_self_improvement_data(args.output_dir)

    successful_tactics = collect_self_improvement_data_parallel(
        model_path=args.model_path,
        num_examples=args.count,
        max_steps=args.max_steps,
        probability_threshold=args.probability_threshold,
        temperature=args.temperature,
        device=device,
        generated_data_dir=args.generated_data_dir,
        max_seq_len=max_seq_len,
        num_workers=max(args.num_workers, 1),
    )

    if successful_tactics:
        base_collector.save_self_improvement_data(
            data=successful_tactics,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
        )
    else:
        print("No successful tactics collected. Please check your model and parameters.")


if __name__ == "__main__":
    main()

