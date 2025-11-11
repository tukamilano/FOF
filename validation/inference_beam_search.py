"""
階層分類対応のビームサーチ推論スクリプト
"""
from __future__ import annotations

import argparse
import os
import sys
import json
import time
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
import heapq
import copy

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


# Add project root to path
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_root)

import torch
import numpy as np

from src.core.transformer_classifier import (
    load_tokens_and_labels_from_token_py,
    CharTokenizer,
    TransformerClassifier,
)
from src.core.state_encoder import encode_prover_state, format_tactic_string


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


def load_hierarchical_model(model_path: str, device: torch.device) -> Tuple[TransformerClassifier, Dict[str, Any]]:
    """階層分類Model 読み込み"""
    checkpoint = torch.load(model_path, map_location=device)
    
    # 新しい形式のチェックポイントかどうか 判定
    if 'model_params' in checkpoint:
        # 新しい形式のチェックポイント
        model_params = checkpoint['model_params']
        vocab_size = checkpoint.get('vocab_size', model_params['vocab_size'])
        pad_id = checkpoint.get('pad_id', model_params['pad_id'])
        max_seq_len = checkpoint.get('max_seq_len', model_params['max_seq_len'])
        
        # クラス数 チェックポイント from get
        num_main_classes = len(checkpoint['id_to_main'])
        num_arg1_classes = len(checkpoint['id_to_arg1'])
        num_arg2_classes = len(checkpoint['id_to_arg2'])
        
        # ラベルマッピング get
        label_mappings = {
            'main_to_id': checkpoint['main_to_id'],
            'arg1_to_id': checkpoint['arg1_to_id'],
            'arg2_to_id': checkpoint['arg2_to_id'],
            'id_to_main': checkpoint['id_to_main'],
            'id_to_arg1': checkpoint['id_to_arg1'],
            'id_to_arg2': checkpoint['id_to_arg2'],
        }
        
        model = TransformerClassifier(
            vocab_size=vocab_size,
            pad_id=pad_id,
            max_seq_len=max_seq_len,
            d_model=model_params['d_model'],
            nhead=model_params['nhead'],
            num_layers=model_params['num_layers'],
            dim_feedforward=model_params['dim_feedforward'],
            dropout=model_params['dropout'],
            num_main_classes=num_main_classes,
            num_arg1_classes=num_arg1_classes,
            num_arg2_classes=num_arg2_classes,
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # 古い形式のチェックポイント（重みonly）
        print("Loading old format checkpoint (weights only)")
        
        # デフォルトのModelパラメータ get
        from src.core.parameter import get_model_params
        model_params = get_model_params()
        
        # チェックポイント from vocab_size get（embedding層のサイズ from ）
        vocab_size = checkpoint['embedding.weight'].shape[0]
        print(f"Detected vocab_size from checkpoint: {vocab_size}")
        
        # デフォルトのラベルマッピング get
        from src.core.parameter import get_hierarchical_labels
        from src.core.transformer_classifier import build_hierarchical_label_mappings
        hierarchical_labels = get_hierarchical_labels()
        main_to_id, arg1_to_id, arg2_to_id, id_to_main, id_to_arg1, id_to_arg2 = build_hierarchical_label_mappings(
            hierarchical_labels.main_tactics,
            hierarchical_labels.arg1_values,
            hierarchical_labels.arg2_values
        )
        
        label_mappings = {
            'main_to_id': main_to_id,
            'arg1_to_id': arg1_to_id,
            'arg2_to_id': arg2_to_id,
            'id_to_main': id_to_main,
            'id_to_arg1': id_to_arg1,
            'id_to_arg2': id_to_arg2,
        }
        
        # Create model（チェックポイント from getdidvocab_size 使用）
        model = TransformerClassifier(
            vocab_size=vocab_size,
            pad_id=model_params.pad_id,
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
        
        # 重み 読み込み
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    return model, label_mappings


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
    model: TransformerClassifier,
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
    model: TransformerClassifier,
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
        
        # 早期終了: 確率 非常 低い状態 除外
        if beam and max(state.probability for state in beam) < 1e-6:
            break
        
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
        
        # 重複排除（同じ状態の重複 避ける）
        seen_states = set()
        unique_candidates = []
        for candidate in new_candidates:
            # 状態のハッシュ 作成（簡易版）
            state_hash = hash(str(candidate.prover.goal) + str(candidate.tactic_sequence))
            if state_hash not in seen_states:
                seen_states.add(state_hash)
                unique_candidates.append(candidate)
                if len(unique_candidates) >= beam_width:
                    break
        
        beam = unique_candidates
        
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


def load_validation_data(validation_file: str, num_examples: int = None) -> List[str]:
    """
    バリデーションデータ 読み込み
    
    Args:
        validation_file: バリデーションファイルのパス
        num_examples: 読み込む例の数（Noneの場合はall）
    
    Returns:
        論理式のリスト
    """
    try:
        with open(validation_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            tautologies = data
        else:
            print(f"Warning: Unexpected data format in {validation_file}")
            return []
        
        if num_examples is not None:
            tautologies = tautologies[:num_examples]
        
        print(f"Loaded {len(tautologies)} tautologies from {validation_file}")
        print("First 5 tautologies:")
        for i in range(min(5, len(tautologies))):
            print(f"  {i+1}: {tautologies[i]}")
        
        return tautologies
        
    except FileNotFoundError:
        print(f"Error: Validation file not found: {validation_file}")
        return []
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in validation file: {e}")
        return []
    except Exception as e:
        print(f"Error loading validation data: {e}")
        return []


def main():
    parser = argparse.ArgumentParser(description="Run hierarchical tactic inference with beam search")
    parser.add_argument("--model_path", type=str, default="models/pretrained_model.pth", help="model path")
    parser.add_argument("--count", type=int, default=None, help="number of examples to run (default: all in validation file)")
    parser.add_argument("--max_steps", type=int, default=30, help="max steps per example")
    parser.add_argument("--beam_width", type=int, default=5, help="beam width for beam search")
    parser.add_argument("--top_k", type=int, default=10, help="number of tactic candidates to consider at each step")
    parser.add_argument("--device", type=str, default="auto", help="device")
    parser.add_argument("--verbose", action="store_true", help="verbose output")
    parser.add_argument("--max_seq_len", type=int, default=256, help="maximum sequence length")
    parser.add_argument("--validation_file", type=str, default="validation_tautology.json", help="validation file path")
    
    args = parser.parse_args()
    
    # Device setup
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    print(f"Beam search parameters: beam_width={args.beam_width}, top_k={args.top_k}")
    
    # Model 読み込みor初期化
    if not os.path.exists(args.model_path):
        print(f"Model file not found: {args.model_path}")
        print("Creating a randomly initialized model...")
        
        # 初期化was doneModel 作成
        from src.core.parameter import get_model_params, get_hierarchical_labels
        
        model_params = get_model_params()
        hierarchical_labels = get_hierarchical_labels()
        
        # ラベルマッピング 作成
        main_to_id = hierarchical_labels.get_main_to_id()
        arg1_to_id = hierarchical_labels.get_arg1_to_id()
        arg2_to_id = hierarchical_labels.get_arg2_to_id()
        id_to_main = hierarchical_labels.get_id_to_main()
        id_to_arg1 = hierarchical_labels.get_id_to_arg1()
        id_to_arg2 = hierarchical_labels.get_id_to_arg2()
        
        label_mappings = {
            'main_to_id': main_to_id,
            'arg1_to_id': arg1_to_id,
            'arg2_to_id': arg2_to_id,
            'id_to_main': id_to_main,
            'id_to_arg1': id_to_arg1,
            'id_to_arg2': id_to_arg2,
        }
        
        # Create tokenizer
        root_dir = os.path.dirname(os.path.dirname(__file__))
        token_py_path = os.path.join(root_dir, "src", "core", "fof_tokens.py")
        base_tokens, _ = load_tokens_and_labels_from_token_py(token_py_path)
        tokenizer = CharTokenizer(base_tokens=base_tokens)
        
        # Create model
        vocab_size = tokenizer.vocab_size
        model = TransformerClassifier(
            vocab_size=vocab_size,
            pad_id=0,
            max_seq_len=args.max_seq_len,
            d_model=model_params.d_model,
            nhead=model_params.nhead,
            num_layers=model_params.num_layers,
            dim_feedforward=model_params.dim_feedforward,
            dropout=model_params.dropout,
            num_main_classes=len(main_to_id),
            num_arg1_classes=len(arg1_to_id),
            num_arg2_classes=len(arg2_to_id),
        )
        model.to(device)
        model.eval()
        
        max_seq_len = args.max_seq_len if hasattr(args, 'max_seq_len') else 256
        
        print(f"Created randomly initialized model with {len(main_to_id)} main tactics, {len(arg1_to_id)} arg1 values, {len(arg2_to_id)} arg2 values")
    else:
        model, label_mappings = load_hierarchical_model(args.model_path, device)
        print(f"Loaded model from {args.model_path}")
        
        # Modelのmax_seq_len get
        checkpoint = torch.load(args.model_path, map_location=device)
        max_seq_len = checkpoint.get('max_seq_len', 256)
        
        # Create tokenizer
        root_dir = os.path.dirname(os.path.dirname(__file__))
        token_py_path = os.path.join(root_dir, "src", "core", "fof_tokens.py")
        base_tokens, _ = load_tokens_and_labels_from_token_py(token_py_path)
        tokenizer = CharTokenizer(base_tokens=base_tokens)
    
    # pyprover インポート
    pyprover_dir = os.path.join(root_dir, "pyprover")
    sys.path.insert(0, pyprover_dir)
    
    # ディレクトリ 変更and from インポート
    original_cwd = os.getcwd()
    os.chdir(pyprover_dir)
    try:
        import proposition as proposition_mod
        import prover as prover_mod
    finally:
        os.chdir(original_cwd)
    
    PropParseTree = proposition_mod.PropParseTree
    prop_parser = proposition_mod.parser
    Prover = prover_mod.Prover
    
    # バリデーションデータ 読み込み
    validation_file = os.path.join(os.path.dirname(__file__), args.validation_file)
    print(f"\nLoading validation data from {validation_file}...")
    
    tautologies = load_validation_data(validation_file, args.count)
    
    if not tautologies:
        print("Error: No validation data loaded!")
        return
    
    print(f"Running {len(tautologies)} examples with beam search (max_steps: {args.max_steps})...")
    
    solved_count = 0
    step_counts = []
    tactic_usage = {}
    confidence_scores = []
    
    if TQDM_AVAILABLE:
        progress_bar = tqdm(
            total=len(tautologies),
            desc="Processing examples",
            unit="example",
        )
    else:
        progress_bar = None
    
    try:
        for i, goal_str in enumerate(tautologies):
            if not goal_str:
                print(f"Warning: Empty formula for example {i+1}, skipping...")
                if progress_bar:
                    progress_bar.update(1)
                continue
            try:
                # パースandprover 作成
                parse_tree = PropParseTree()
                goal_node = parse_tree.transform(prop_parser.parse(goal_str))
                prover = Prover(goal_node)
            
                if args.verbose:
                    print(f"\nExample {i+1}:")
                    print(f"  Goal: {goal_str}")
                
                # ビームサーチ with/at 推論 実行
                solved, steps, tactic_sequence, confidences = beam_search_inference(
                    model, tokenizer, label_mappings, device, prover,
                    max_steps=args.max_steps,
                    beam_width=args.beam_width,
                    top_k=args.top_k,
                    max_seq_len=max_seq_len,
                    verbose=args.verbose
                )
                
                # 結果 記録
                step_counts.append(steps)
                confidence_scores.extend(confidences)
                
                # タクティク使用統計 更新
                for tactic in tactic_sequence:
                    tactic_usage[tactic] = tactic_usage.get(tactic, 0) + 1
                
                if solved:
                    solved_count += 1
                    if args.verbose:
                        print(f"  Result: SOLVED in {steps} steps")
                        print(f"  Tactic sequence: {tactic_sequence}")
                else:
                    if args.verbose:
                        print(f"  Result: FAILED after {steps} steps")
                        print(f"  Tactic sequence: {tactic_sequence}")
                
            except Exception as e:
                # パースエラーetc with/at 失敗did場合はスキップ
                print(f"Warning: Failed to process tautology {i+1}: {e}")
                step_counts.append(args.max_steps)  # 失敗 as 記録
            
            # Update progress bar
            if progress_bar:
                progress_bar.update(1)
            elif (i + 1) % 10 == 0 or i == len(tautologies) - 1:
                print(f"Processed {i + 1}/{len(tautologies)} examples...")
    
    finally:
        if progress_bar:
            progress_bar.close()
    
    # 最終結果 計算
    total_examples = len(step_counts)  # 実際 処理was done問題数
    success_rate = solved_count / total_examples if total_examples > 0 else 0.0
    avg_steps = sum(step_counts) / len(step_counts) if step_counts else 0
    avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
    
    print(f"\nResults: {solved_count}/{total_examples} examples solved ({success_rate*100:.1f}%)")
    print(f"Average steps: {avg_steps:.2f}")
    print(f"Average confidence: {avg_confidence:.3f}")
    print(f"Beam search parameters: beam_width={args.beam_width}, top_k={args.top_k}")


if __name__ == "__main__":
    main()
