"""
Parallel self improvement data collector with beam search.

This module provides a parallelised variant of self improvement data collection
using beam search for more diverse and effective tactic exploration.
It follows the logic of the sequential implementation but distributes
the proof search for individual tautologies across multiple worker
processes using beam search instead of greedy or probabilistic selection.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple
import multiprocessing
from dataclasses import dataclass
import copy

import torch
import numpy as np

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

try:
    from google.cloud import storage
    GCS_AVAILABLE = True
except ImportError:
    GCS_AVAILABLE = False

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.core.transformer_classifier import CharTokenizer, load_tokens_and_labels_from_token_py
from src.core.state_encoder import encode_prover_state, format_tactic_string
try:
    from src.interaction import self_improvement_data_collector as base_collector
    BASE_COLLECTOR_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import base_collector: {e}")
    BASE_COLLECTOR_AVAILABLE = False
    base_collector = None


@dataclass
class BeamState:
    """ビームサーチの状態 表すクラス"""
    prover: Any  # Proverオブジェクト
    tactic_sequence: List[str]  # これま with/at  適用didタクティクのシーケンス
    probability: float  # 現在の状態の確率
    step: int  # 現在のステップ数
    solved: bool  # 解決済みかどうか
    confidence_scores: List[float]  # 各ステップの確信度スコア
    
    def __lt__(self, other):
        """優先度キュー用の比較関数（確率の高い順）"""
        return self.probability > other.probability


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

    if not BASE_COLLECTOR_AVAILABLE:
        raise ImportError("base_collector is not available. Please ensure all dependencies are installed.")

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


def calculate_tactic_probability(
    main_tactic: str,
    arg1_value: str,
    arg2_value: str,
    main_confidence: float,
    arg1_confidence: float,
    arg2_confidence: float
) -> float:
    """
    タクティクの種類 応じて適切な確率 計算
    
    Args:
        main_tactic: メインタクティク
        arg1_value: 第1引数
        arg2_value: 第2引数
        main_confidence: メインタクティクの確信度
        arg1_confidence: 第1引数の確信度
        arg2_confidence: 第2引数の確信度
    
    Returns:
        計算was done確率
    """
    # 引数不要なタクティク
    if main_tactic in ['assumption', 'intro', 'split', 'left', 'right', 'add_dn']:
        return main_confidence
    
    # 引数1のタクティク
    elif main_tactic in ['apply', 'destruct']:
        return main_confidence * arg1_confidence
    
    # 引数2のタクティク
    elif main_tactic == 'specialize':
        return main_confidence * arg1_confidence * arg2_confidence
    
    # Other tactics（引数不要 as 扱う）
    else:
        return main_confidence


def generate_tactic_candidates(
    model: torch.nn.Module,
    tokenizer: CharTokenizer,
    premises: List[str],
    goal: str,
    label_mappings: Dict[str, Any],
    device: torch.device,
    max_seq_len: int = 256,
    top_k: int = 10
) -> List[Tuple[str, float]]:
    """
    現在の状態 from 可能なタクティク候補 Generationし、上位k 返す
    
    Returns:
        [(tactic_string, probability), ...] のリスト（確率の高い順）
    """
    # Encode input
    input_ids, attention_mask, segment_ids = tokenizer.encode(goal, premises, max_seq_len)
    input_ids = input_ids.unsqueeze(0).to(device)
    attention_mask = attention_mask.unsqueeze(0).to(device)
    segment_ids = segment_ids.unsqueeze(0).to(device)
    
    with torch.no_grad():
        main_logits, arg1_logits, arg2_logits = model(input_ids, attention_mask)
        
        # softmax with/at 確率 変換
        main_probs = torch.softmax(main_logits, dim=-1)
        arg1_probs = torch.softmax(arg1_logits, dim=-1)
        arg2_probs = torch.softmax(arg2_logits, dim=-1)
        
        tactic_candidates = []
        
        # allの可能な組み合わせ Generation
        for main_id, main_tactic in enumerate(label_mappings['id_to_main']):
            main_confidence = main_probs[0, main_id].item()
            
            # 引数 不要なタクティクの場合
            if main_tactic in ['assumption', 'intro', 'split', 'left', 'right', 'add_dn']:
                tactic_string = main_tactic
                probability = calculate_tactic_probability(
                    main_tactic, "", "",
                    main_confidence, 0.0, 0.0
                )
                tactic_candidates.append((tactic_string, probability))
            
            # 引数1のタクティクの場合
            elif main_tactic in ['apply', 'destruct']:
                for arg1_id, arg1_value in enumerate(label_mappings['id_to_arg1']):
                    arg1_confidence = arg1_probs[0, arg1_id].item()
                    tactic_string = f"{main_tactic} {arg1_value}"
                    probability = calculate_tactic_probability(
                        main_tactic, arg1_value, "",
                        main_confidence, arg1_confidence, 0.0
                    )
                    tactic_candidates.append((tactic_string, probability))
            
            # 引数2のタクティクの場合
            elif main_tactic == 'specialize':
                for arg1_id, arg1_value in enumerate(label_mappings['id_to_arg1']):
                    arg1_confidence = arg1_probs[0, arg1_id].item()
                    for arg2_id, arg2_value in enumerate(label_mappings['id_to_arg2']):
                        arg2_confidence = arg2_probs[0, arg2_id].item()
                        tactic_string = f"{main_tactic} {arg1_value} {arg2_value}"
                        probability = calculate_tactic_probability(
                            main_tactic, arg1_value, arg2_value,
                            main_confidence, arg1_confidence, arg2_confidence
                        )
                        tactic_candidates.append((tactic_string, probability))
            
            # Other tactics（引数不要 as 扱う）
            else:
                tactic_string = main_tactic
                probability = calculate_tactic_probability(
                    main_tactic, "", "",
                    main_confidence, 0.0, 0.0
                )
                tactic_candidates.append((tactic_string, probability))
        
        # 確率の高い順 ソートand上位k 返す
        tactic_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # 確率の閾値 with/at 早期カット（高速化）
        min_probability = 1e-8
        filtered_candidates = [
            (tactic, prob) for tactic, prob in tactic_candidates 
            if prob >= min_probability
        ]
        
        return filtered_candidates[:top_k]


def apply_tactic_from_label(prover, label) -> bool:
    """タクティク 適用"""
    if isinstance(label, dict):
        tactic_str = format_tactic_string(label)
    else:
        tactic_str = label
    
    try:
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
    except Exception as e:
        # pyproverのエラー キャッチandFalse 返す
        return False


def beam_search_inference(
    model: torch.nn.Module,
    tokenizer: CharTokenizer,
    label_mappings: Dict[str, Any],
    device: torch.device,
    prover: Any,
    max_steps: int = 30,
    beam_width: int = 5,
    top_k: int = 10,
    max_seq_len: int = 256,
    verbose: bool = False
) -> Tuple[bool, int, List[str], List[float]]:
    """
    ビームサーチ 使用and推論 実行
    
    Args:
        model: 推論 使用do/performModel
        tokenizer: トークナイザー
        label_mappings: ラベルマッピング
        device: Device
        prover: プロバーオブジェクト
        max_steps: 最大ステップ数
        beam_width: ビーム幅
        top_k: 各ステップ with/at 考慮do/performタクティク候補数
        max_seq_len: Maximum sequence length
        verbose: 詳細出力フラグ
    
    Returns:
        (solved, steps, tactic_sequence, confidence_scores)
    """
    # 初期状態 作成
    initial_state = encode_prover_state(prover)
    initial_premises = initial_state["premises"]
    initial_goal = initial_state["goal"]
    
    # 初期ビーム状態 作成
    beam = [BeamState(
        prover=prover,
        tactic_sequence=[],
        probability=1.0,
        step=0,
        solved=prover.goal is None,
        confidence_scores=[]
    )]
    
    if verbose:
        print(f"Initial goal: {initial_goal}")
        print(f"Initial premises: {initial_premises}")
    
    for step in range(max_steps):
        if verbose:
            print(f"\nStep {step + 1}: Beam width = {len(beam)}")
        
        # 解決済みの状態 exists/hasかチェック
        solved_states = [state for state in beam if state.solved]
        if solved_states:
            # 最も確率の高い解決済み状態 返す
            best_solved = max(solved_states, key=lambda s: s.probability)
            return True, best_solved.step, best_solved.tactic_sequence, best_solved.confidence_scores
        
        # 新しい候補状態 Generation
        new_candidates = []
        
        for state in beam:
            if state.solved:
                continue
                
            # 現在の状態 get
            current_state = encode_prover_state(state.prover)
            current_premises = current_state["premises"]
            current_goal = current_state["goal"]
            
            # タクティク候補 Generation
            tactic_candidates = generate_tactic_candidates(
                model, tokenizer, current_premises, current_goal,
                label_mappings, device, max_seq_len, top_k
            )
            
            if verbose:
                print(f"  Generated {len(tactic_candidates)} candidates for current state")
                print(f"    Top 3: {[(t, f'{p:.3f}') for t, p in tactic_candidates[:3]]}")
            
            # 各タクティク候補 試す
            for tactic_str, tactic_prob in tactic_candidates:
                # プロバーのコピー 作成
                new_prover = copy.deepcopy(state.prover)
                
                # タクティク 適用
                success = apply_tactic_from_label(new_prover, tactic_str)
                
                if success:
                    # 新しい状態 作成
                    new_sequence = state.tactic_sequence + [tactic_str]
                    new_probability = state.probability * tactic_prob
                    new_confidence_scores = state.confidence_scores + [tactic_prob]
                    new_solved = new_prover.goal is None
                    
                    new_state = BeamState(
                        prover=new_prover,
                        tactic_sequence=new_sequence,
                        probability=new_probability,
                        step=state.step + 1,
                        solved=new_solved,
                        confidence_scores=new_confidence_scores
                    )
                    
                    new_candidates.append(new_state)
                    
                    if verbose:
                        print(f"    Applied {tactic_str} (prob: {tactic_prob:.3f}) - Success")
                else:
                    if verbose:
                        print(f"    Applied {tactic_str} (prob: {tactic_prob:.3f}) - Failed")
        
        if not new_candidates:
            # 新しい候補 no/not場合は失敗
            break
        
        # ビーム幅分の最良の候補 選択
        new_candidates.sort(key=lambda x: x.probability, reverse=True)
        beam = new_candidates[:beam_width]
        
        if verbose:
            print(f"  Selected top {len(beam)} candidates for next step")
            for i, state in enumerate(beam):
                print(f"    {i+1}: prob={state.probability:.6f}, steps={state.step}, solved={state.solved}")
    
    # 最終的 最も確率の高い状態 返す
    if beam:
        best_state = max(beam, key=lambda s: s.probability)
        return best_state.solved, best_state.step, best_state.tactic_sequence, best_state.confidence_scores
    else:
        return False, max_steps, [], []


def _process_tautology_worker_beam_search(args: Tuple[Any, ...]) -> Dict[str, Any]:
    """Process a single tautology inside a worker process using beam search."""

    (
        index,
        goal_str,
        max_steps,
        beam_width,
        top_k,
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

        # ビームサーチ with/at 推論 実行
        solved, steps, tactic_sequence, confidences = beam_search_inference(
            _WORKER_MODEL,
            _WORKER_TOKENIZER,
            _WORKER_LABEL_MAPPINGS,
            _WORKER_DEVICE,
            prover,
            max_steps=max_steps,
            beam_width=beam_width,
            top_k=top_k,
            max_seq_len=_WORKER_MAX_SEQ_LEN,
            verbose=False
        )

        # 成功didタクティクのデータ 構築
        example_successful_tactics: List[Dict[str, Any]] = []
        if solved and tactic_sequence:
            # 各ステップの状態 再構築andタクティクデータ 作成
            temp_prover = base_collector.PYPROVER_MODULES["Prover"](goal_node)
            
            for step_idx, (tactic_str, confidence) in enumerate(zip(tactic_sequence, confidences)):
                current_state = encode_prover_state(temp_prover)
                current_premises = current_state["premises"]
                current_goal = current_state["goal"]
                
                # タクティク 適用
                success = apply_tactic_from_label(temp_prover, tactic_str)
                if not success:
                    break
                
                # タクティクデータ 作成
                tactic_dict = base_collector.parse_tactic_string_cached(tactic_str)
                state_tactic_hash = base_collector.create_state_hash(
                    current_premises, current_goal, tactic_str
                )
                example_successful_tactics.append(
                    {
                        "step_index": step_idx,
                        "premises": list(current_premises or []),
                        "goal": current_goal,
                        "tactic": tactic_dict,
                        "tactic_apply": True,
                        "state_tactic_hash": state_tactic_hash,
                    }
                )

        return {
            "index": index,
            "goal": goal_str,
            "solved": solved,
            "successful_tactics": example_successful_tactics,
            "steps": steps,
            "terminated": False,
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


class StreamingBeamSearchCollector:
    """Streaming self improvement data collector with beam search and real-time GCS upload"""
    
    def __init__(self, 
                 model_path: str,
                 max_steps: int,
                 beam_width: int,
                 top_k: int,
                 device: torch.device,
                 generated_data_dir: str,
                 max_seq_len: int,
                 num_workers: int,
                 output_dir: str = "self_improvement_data_beam",
                 batch_size: int = 1000,
                 gcs_bucket: str = None,
                 gcs_prefix: str = ""):
        self.model_path = model_path
        self.max_steps = max_steps
        self.beam_width = beam_width
        self.top_k = top_k
        self.device = device
        self.generated_data_dir = generated_data_dir
        self.max_seq_len = max_seq_len
        self.num_workers = num_workers
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.gcs_bucket = gcs_bucket
        self.gcs_prefix = gcs_prefix
        
        # Buffer management
        self.all_successful_tactics = []
        self.current_file_index = 0
        self.total_solved = 0
        self.total_tactics = 0
        
        # Initialize global constants
        base_collector.initialize_global_constants()
        
        # Load tautologies
        self.tautologies = base_collector.load_tautologies_from_generated_data(
            generated_data_dir=generated_data_dir,
            max_examples=None  # Load all available
        )
        
        if not self.tautologies:
            raise ValueError("Failed to load tautologies from generated_data directory!")
        
        print(f"Loaded {len(self.tautologies)} tautologies from generated_data directory")
        print(f"Beam search parameters: beam_width={beam_width}, top_k={top_k}")
        
        # Clear output directory
        base_collector.clear_self_improvement_data(output_dir)
        
        if gcs_bucket:
            print(f"GCS upload: gs://{gcs_bucket}/{gcs_prefix}")
    
    def add_tactics_and_check_save(self, tactics: List[Dict[str, Any]]):
        """Add tactics to buffer and save if buffer is full"""
        if not tactics:
            return
            
        self.all_successful_tactics.extend(tactics)
        self.total_tactics += len(tactics)
        
        # Save if buffer is full
        if len(self.all_successful_tactics) >= self.batch_size:
            self.save_current_data()
    
    def save_current_data(self):
        """Save current buffer to file and upload to GCS"""
        if not self.all_successful_tactics:
            return
            
        # Save to local file
        filename = f"training_data_beam_{self.current_file_index:05d}.json"
        filepath = os.path.join(self.output_dir, filename)
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.all_successful_tactics, f, ensure_ascii=False, indent=2)
        
        print(f"Saved {len(self.all_successful_tactics)} tactics to {filepath}")
        
        # Upload to GCS if configured
        if self.gcs_bucket:
            if upload_to_gcs(filepath, self.gcs_bucket, self.gcs_prefix):
                print(f"✅ Uploaded {filename} to GCS")
            else:
                print(f"❌ Failed to upload {filename} to GCS")
        
        # Reset buffer
        self.all_successful_tactics = []
        self.current_file_index += 1
    
    def collect_data_streaming(self, num_examples: int) -> List[Dict[str, Any]]:
        """Collect data using streaming approach with beam search"""
        if self.num_workers <= 1:
            # Fallback to sequential implementation
            return self._collect_sequential(num_examples)
        
        config = {
            "model_path": self.model_path,
            "device": str(self.device),
            "max_seq_len": self.max_seq_len,
        }
        
        # Limit examples to available tautologies
        if num_examples is None:
            examples_to_process = len(self.tautologies)
        else:
            examples_to_process = min(num_examples, len(self.tautologies))
        tasks = [
            (index, goal, self.max_steps, self.beam_width, self.top_k, config)
            for index, goal in enumerate(self.tautologies[:examples_to_process])
        ]
        
        print(f"Starting streaming parallel beam search data collection with {self.num_workers} workers...")
        print(f"Processing {examples_to_process} examples...")
        
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [executor.submit(_process_tautology_worker_beam_search, task) for task in tasks]
            
            if TQDM_AVAILABLE:
                progress_bar = tqdm(
                    total=len(futures),
                    desc="Processing tautologies (beam search)",
                    unit="formula",
                )
            else:
                progress_bar = None
            
            try:
                for future in as_completed(futures):
                    result = future.result()
                    
                    if result.get("error"):
                        print(f"Warning: failed to process tautology {result['index'] + 1}: {result['error']}")
                        if progress_bar:
                            progress_bar.update(1)
                        continue
                    
                    if result.get("solved"):
                        self.total_solved += 1
                        tactics = result.get("successful_tactics", [])
                        self.add_tactics_and_check_save(tactics)
                    
                    if progress_bar:
                        progress_bar.update(1)
                    elif (result['index'] + 1) % 10 == 0 or result['index'] == len(futures) - 1:
                        print(f"Processed {result['index'] + 1}/{len(futures)} examples...")
            
            finally:
                if progress_bar:
                    progress_bar.close()
        
        # Save any remaining data
        if self.all_successful_tactics:
            self.save_current_data()
        
        print(f"\nBeam search data collection completed:")
        print(f"  Solved examples: {self.total_solved}/{examples_to_process}")
        print(f"  Total tactics collected: {self.total_tactics}")
        print(f"  Beam search parameters: beam_width={self.beam_width}, top_k={self.top_k}")
        
        return []  # Data is saved incrementally, not returned
    
    def _collect_sequential(self, num_examples: int) -> List[Dict[str, Any]]:
        """Fallback to sequential implementation with beam search"""
        base_collector.initialize_global_constants()
        if base_collector.BASE_TOKENS is None:
            token_py_path = os.path.join(project_root, "src", "core", "fof_tokens.py")
            base_tokens, _ = load_tokens_and_labels_from_token_py(token_py_path)
        else:
            base_tokens = base_collector.BASE_TOKENS

        tokenizer = CharTokenizer(base_tokens=base_tokens)
        model, label_mappings = base_collector.load_hierarchical_model(self.model_path, self.device)
        
        # Handle None case for sequential collection
        if num_examples is None:
            num_examples = len(self.tautologies)
        
        # Use beam search for sequential collection
        successful_tactics = []
        solved_count = 0
        
        for i, goal_str in enumerate(self.tautologies[:num_examples]):
            if not goal_str:
                continue
                
            try:
                # Parse tautology and create prover instance
                parse_tree = base_collector.PYPROVER_MODULES["PropParseTree"]()
                goal_node = parse_tree.transform(
                    base_collector.PYPROVER_MODULES["prop_parser"].parse(goal_str)
                )
                prover = base_collector.PYPROVER_MODULES["Prover"](goal_node)
                
                # ビームサーチ with/at 推論 実行
                solved, steps, tactic_sequence, confidences = beam_search_inference(
                    model, tokenizer, label_mappings, self.device, prover,
                    max_steps=self.max_steps,
                    beam_width=self.beam_width,
                    top_k=self.top_k,
                    max_seq_len=self.max_seq_len,
                    verbose=False
                )
                
                if solved and tactic_sequence:
                    solved_count += 1
                    # 各ステップの状態 再構築andタクティクデータ 作成
                    temp_prover = base_collector.PYPROVER_MODULES["Prover"](goal_node)
                    
                    for step_idx, (tactic_str, confidence) in enumerate(zip(tactic_sequence, confidences)):
                        current_state = encode_prover_state(temp_prover)
                        current_premises = current_state["premises"]
                        current_goal = current_state["goal"]
                        
                        # タクティク 適用
                        success = apply_tactic_from_label(temp_prover, tactic_str)
                        if not success:
                            break
                        
                        # タクティクデータ 作成
                        tactic_dict = base_collector.parse_tactic_string_cached(tactic_str)
                        state_tactic_hash = base_collector.create_state_hash(
                            current_premises, current_goal, tactic_str
                        )
                        successful_tactics.append(
                            {
                                "step_index": step_idx,
                                "premises": list(current_premises or []),
                                "goal": current_goal,
                                "tactic": tactic_dict,
                                "tactic_apply": True,
                                "state_tactic_hash": state_tactic_hash,
                            }
                        )
                
                if (i + 1) % 10 == 0:
                    print(f"Processed {i + 1}/{num_examples} examples...")
                    
            except Exception as e:
                print(f"Warning: Failed to process tautology {i+1}: {e}")
                continue
        
        print(f"\nSequential beam search data collection completed:")
        print(f"  Solved examples: {solved_count}/{num_examples}")
        print(f"  Total tactics collected: {len(successful_tactics)}")
        
        return successful_tactics


def upload_to_gcs(local_file_path: str, gcs_bucket: str, gcs_prefix: str) -> bool:
    """Upload a local file to Google Cloud Storage.
    
    Args:
        local_file_path: Path to the local file to upload
        gcs_bucket: GCS bucket name
        gcs_prefix: GCS prefix for the uploaded file
        
    Returns:
        True if upload successful, False otherwise
    """
    if not GCS_AVAILABLE:
        print("❌ Google Cloud Storage not available. Please install google-cloud-storage.")
        return False
        
    try:
        client = storage.Client()
        bucket = client.bucket(gcs_bucket)
        
        # Create the blob name by combining prefix and filename
        filename = os.path.basename(local_file_path)
        blob_name = f"{gcs_prefix.rstrip('/')}/{filename}" if gcs_prefix else filename
        
        blob = bucket.blob(blob_name)
        
        print(f"Uploading {local_file_path} to gs://{gcs_bucket}/{blob_name}")
        blob.upload_from_filename(local_file_path)
        
        print(f"✅ Successfully uploaded to gs://{gcs_bucket}/{blob_name}")
        return True
        
    except Exception as e:
        print(f"❌ Error uploading {local_file_path} to GCS: {e}")
        return False


def main() -> None:
    # Set multiprocessing start method to 'spawn' to avoid CUDA issues
    multiprocessing.set_start_method('spawn', force=True)
    
    parser = argparse.ArgumentParser(
        description="Collect self improvement data using beam search in parallel"
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
        default=None,
        help="number of examples to process (default: all available)",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=30,
        help="max steps per example",
    )
    parser.add_argument(
        "--beam_width",
        type=int,
        default=5,
        help="beam width for beam search",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="number of tactic candidates to consider at each step",
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
        default="self_improvement_data_beam",
        help="output directory for collected data",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=10000,
        help="batch size for saving data",
    )
    parser.add_argument(
        "--generated_data_dir",
        type=str,
        default="generated_data",
        help="directory containing generated tautology data",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=os.cpu_count() or 1,
        help="number of worker processes",
    )
    parser.add_argument(
        "--gcs_bucket",
        type=str,
        default=None,
        help="GCS bucket name for direct upload (e.g., fof-data-20251010-milano)",
    )
    parser.add_argument(
        "--gcs_prefix",
        type=str,
        default="",
        help="GCS prefix for uploaded files (e.g., generated_data_RL1_beam/)",
    )

    args = parser.parse_args()

    if not BASE_COLLECTOR_AVAILABLE:
        print("❌ base_collector is not available. Please ensure all dependencies are installed.")
        print("   Try: pip install tqdm")
        return

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
    print(f"Starting parallel beam search data collection with {args.num_workers} workers...")
    print(f"  Examples to process: {args.count}")
    print(f"  Max steps per example: {args.max_steps}")
    print(f"  Beam search parameters: beam_width={args.beam_width}, top_k={args.top_k}")
    print(f"  Generated data directory: {args.generated_data_dir}")
    print(f"  Output directory: {args.output_dir}")
    if args.gcs_bucket:
        print(f"  GCS bucket: {args.gcs_bucket}")
        print(f"  GCS prefix: {args.gcs_prefix}")

    # Initialize streaming collector
    collector = StreamingBeamSearchCollector(
        model_path=args.model_path,
        max_steps=args.max_steps,
        beam_width=args.beam_width,
        top_k=args.top_k,
        device=device,
        generated_data_dir=args.generated_data_dir,
        max_seq_len=max_seq_len,
        num_workers=max(args.num_workers, 1),
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        gcs_bucket=args.gcs_bucket,
        gcs_prefix=args.gcs_prefix
    )

    # Collect data with streaming approach
    collector.collect_data_streaming(num_examples=args.count)


if __name__ == "__main__":
    main()
