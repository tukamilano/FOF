"""
階層分類対応の推論スクリプト
"""
from __future__ import annotations

import argparse
import os
import sys
import glob
import json
import time
from typing import List, Tuple, Dict, Any

from tqdm import tqdm


# Add project root to path
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_root)

import torch

from src.core.transformer_classifier import (
    load_tokens_and_labels_from_token_py,
    CharTokenizer,
    TransformerClassifier,
)
from src.core.state_encoder import encode_prover_state, format_tactic_string


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


def select_tactic_probabilistically(
    tactic_combinations: List[Tuple[str, float]], 
    temperature: float = 1.0,
    failed_tactics: set = None
) -> Tuple[str, float]:
    """
    temperature 使用andタクティク 確率的 選択
    
    Args:
        tactic_combinations: [(tactic_string, probability), ...] のリスト
        temperature: 温度パラメータ（高いほどランダム、低いほど確率 従う）
        failed_tactics: 失敗didタクティクのセット
    
    Returns:
        選択was doneタクティクとその調整後の確率
    """
    import numpy as np
    
    if failed_tactics is None:
        failed_tactics = set()
    
    # 失敗andいno/notタクティクonly フィルタリング
    available_tactics = [(tactic, prob) for tactic, prob in tactic_combinations
                        if tactic not in failed_tactics]
    
    if not available_tactics:
        # 利用可能なタクティク no/not場合は空 返す
        return "", 0.0
    
    # 確率 温度 with/at 調整
    tactics, probabilities = zip(*available_tactics)
    probabilities = np.array(probabilities)
    
    # 温度 適用（log確率 変換and from 温度 with/at 割る）
    log_probs = np.log(probabilities + 1e-8)  # 数値安定性のため小さな値Add
    scaled_log_probs = log_probs / temperature
    
    # softmax with/at 正規化
    exp_probs = np.exp(scaled_log_probs - np.max(scaled_log_probs))  # 数値安定性のため
    softmax_probs = exp_probs / np.sum(exp_probs)
    
    # 確率的 選択
    selected_idx = np.random.choice(len(tactics), p=softmax_probs)
    selected_tactic = tactics[selected_idx]
    selected_prob = softmax_probs[selected_idx]  # 調整後の確率 返す
    
    return selected_tactic, selected_prob


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


def generate_all_tactic_combinations(
    model: TransformerClassifier,
    tokenizer: CharTokenizer,
    premises: List[str],
    goal: str,
    label_mappings: Dict[str, Any],
    device: torch.device,
    max_seq_len: int = 256,
    temperature: float = 1.0
) -> List[Tuple[str, float]]:
    """
    allの可能なタクティクの組み合わせ Generationし、確率の高い順 ソート
    
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
        
        # temperature 適用andsoftmax with/at 確率 変換
        if temperature == 0.0:
            # temperature=0の場合は確定的（softmax with/at 確率 計算し、確率の高い順 試す）
            main_probs = torch.softmax(main_logits, dim=-1)
            arg1_probs = torch.softmax(arg1_logits, dim=-1)
            arg2_probs = torch.softmax(arg2_logits, dim=-1)
        else:
            # temperature>0の場合はsoftmax
            main_probs = torch.softmax(main_logits / temperature, dim=-1)
            arg1_probs = torch.softmax(arg1_logits / temperature, dim=-1)
            arg2_probs = torch.softmax(arg2_logits / temperature, dim=-1)
        
        tactic_combinations = []
        
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
                tactic_combinations.append((tactic_string, probability))
            
            # 引数1のタクティクの場合
            elif main_tactic in ['apply', 'destruct']:
                for arg1_id, arg1_value in enumerate(label_mappings['id_to_arg1']):
                    arg1_confidence = arg1_probs[0, arg1_id].item()
                    tactic_string = f"{main_tactic} {arg1_value}"
                    probability = calculate_tactic_probability(
                        main_tactic, arg1_value, "",
                        main_confidence, arg1_confidence, 0.0
                    )
                    tactic_combinations.append((tactic_string, probability))
            
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
                        tactic_combinations.append((tactic_string, probability))
            
            # Other tactics（引数不要 as 扱う）
            else:
                tactic_string = main_tactic
                probability = calculate_tactic_probability(
                    main_tactic, "", "",
                    main_confidence, 0.0, 0.0
                )
                tactic_combinations.append((tactic_string, probability))
        
        # 確率の高い順 ソート
        tactic_combinations.sort(key=lambda x: x[1], reverse=True)
        
        return tactic_combinations


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


def evaluate_inference_performance(
    model: TransformerClassifier,
    tokenizer: CharTokenizer,
    label_mappings: Dict[str, Any],
    device: torch.device,
    max_seq_len: int,
    num_examples: int = 100,
    max_steps: int = 30,
    seed: int = 42,
    temperature: float = 1.0
) -> Tuple[float, float]:
    """
    推論性能 評価do/perform関数（inference_hierarchical.pyのmain関数と同じ方法）
    
    Args:
        model: 評価do/performModel
        tokenizer: トークナイザー
        label_mappings: ラベルマッピング
        device: Device
        max_seq_len: Maximum sequence length
        num_examples: 評価do/perform例の数
        max_steps: 最大ステップ数
        seed: ランダムシード
        temperature: softmax計算の温度パラメータ
    
    Returns:
        (success_rate, avg_steps): 成功率と平均ステップ数
    """
    import random
    import numpy as np
    
    # シード 設定
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # バリデーションデータ 読み込み
    validation_file = os.path.join(os.path.dirname(__file__), "validation_tautology.json")
    tautologies = load_validation_data(validation_file, num_examples)
    
    if not tautologies:
        print("Warning: No validation data available for inference evaluation")
        return 0.0, 0.0
    
    # pyprover インポート
    root_dir = os.path.dirname(os.path.dirname(__file__))
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
    
    print(f"Running {len(tautologies)} examples (max_steps: {max_steps})...")
    
    solved_count = 0
    step_counts = []
    tactic_usage = {}
    confidence_scores = []
    
    for i, goal_str in enumerate(tautologies):
        if not goal_str:
            print(f"Warning: Empty formula for example {i+1}, skipping...")
            continue
        try:
            # パースandprover 作成
            parse_tree = PropParseTree()
            goal_node = parse_tree.transform(prop_parser.parse(goal_str))
            prover = Prover(goal_node)
            
            # 前提は空（トートロジーなの with/at 前提なし with/at 証明可能）
            premises = []
            
            # 推論ループ（確定的な順序適用）
            step = 0
            solved = prover.goal is None
            example_tactics = []
            example_confidences = []
            
            while not solved and step < max_steps:
                # 現在の状態 get
                current_state = encode_prover_state(prover)
                current_premises = current_state["premises"]
                current_goal = current_state["goal"]
                
                # allの可能なタクティクの組み合わせ Generation（確率の高い順）
                tactic_combinations = generate_all_tactic_combinations(
                    model, tokenizer, current_premises, current_goal,
                    label_mappings, device, max_seq_len, temperature
                )
                
                # 上位max_stepsのタクティク 順次適用
                success = False
                for tactic_str, probability in tactic_combinations[:max_steps]:
                    # タクティク 適用
                    success = apply_tactic_from_label(prover, tactic_str)
                    
                    # ログ用データ 記録
                    example_tactics.append(tactic_str)
                    example_confidences.append(probability)
                    tactic_usage[tactic_str] = tactic_usage.get(tactic_str, 0) + 1
                    
                    if success:
                        break
                
                step += 1
                solved = prover.goal is None
            
            # 例の結果 記録
            step_counts.append(step)
            confidence_scores.extend(example_confidences)
            
            if solved:
                solved_count += 1
            
        except Exception as e:
            # パースエラーetc with/at 失敗did場合はスキップ
            print(f"Warning: Failed to process tautology {i+1}: {e}")
            step_counts.append(max_steps)  # 失敗 as 記録
            continue
    
    # 最終結果 計算
    total_examples = len(step_counts)  # 実際 処理was done問題数
    success_rate = solved_count / total_examples if total_examples > 0 else 0.0
    avg_steps = sum(step_counts) / len(step_counts) if step_counts else 0
    avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
    
    print(f"Results: {solved_count}/{total_examples} examples solved ({success_rate*100:.1f}%)")
    print(f"Average steps: {avg_steps:.2f}")
    print(f"Average confidence: {avg_confidence:.3f}")
    
    return success_rate, avg_steps


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


def main():
    parser = argparse.ArgumentParser(description="Run hierarchical tactic inference")
    parser.add_argument("--model_path", type=str, default="models/pretrained_model.pth", help="model path")
    parser.add_argument("--count", type=int, default=None, help="number of examples to run (default: all in validation file)")
    parser.add_argument("--max_steps", type=int, default=30, help="max steps per example")
    parser.add_argument("--device", type=str, default="auto", help="device")
    parser.add_argument("--verbose", action="store_true", help="verbose output")
    parser.add_argument("--max_seq_len", type=int, default=256, help="maximum sequence length")
    parser.add_argument("--validation_file", type=str, default="validation_tautology.json", help="validation file path")
    parser.add_argument("--temperature", type=float, default=None, help="temperature for softmax calculation (default: None for deterministic, 1.0 for normal)")
    
    args = parser.parse_args()
    
    # Device setup
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # temperatureの設定（Noneの場合は確定的 ）
    if args.temperature is None:
        temperature = 0.0
        print("Using deterministic mode (temperature=0.0)")
    else:
        temperature = args.temperature
        print(f"Using temperature: {temperature}")
    
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
    
    
    print(f"Running {len(tautologies)} examples (max_steps: {args.max_steps})...")
    
    solved_count = 0
    step_counts = []
    tactic_usage = {}
    confidence_scores = []
    
    with tqdm(
        total=len(tautologies),
        desc="Processing examples",
        unit="example",
    ) as progress:
        for i, goal_str in enumerate(tautologies):
            if not goal_str:
                print(f"Warning: Empty formula for example {i+1}, skipping...")
                progress.update(1)
                continue
            try:
                # パースandprover 作成
                parse_tree = PropParseTree()
                goal_node = parse_tree.transform(prop_parser.parse(goal_str))
                prover = Prover(goal_node)
                
                # 前提は空（トートロジーなの with/at 前提なし with/at 証明可能）
                premises = []
                
                if args.verbose:
                    print(f"\nExample {i+1}:")
                    print(f"  Goal: {goal_str}")
                    print(f"  Premises: {premises}")
                
                # 推論ループ
                step = 0
                solved = prover.goal is None
                example_tactics = []
                example_confidences = []
                failed_tactics = set()
                
                while not solved and step < args.max_steps:
                    # 現在の状態 get
                    current_state = encode_prover_state(prover)
                    current_premises = current_state["premises"]
                    current_goal = current_state["goal"]
                    
                    # allの可能なタクティクの組み合わせ Generation（確率の高い順）
                    tactic_combinations = generate_all_tactic_combinations(
                        model, tokenizer, current_premises, current_goal,
                        label_mappings, device, max_seq_len, temperature
                    )
                    
                    if args.verbose:
                        print(f"  Step {step+1}: Generated {len(tactic_combinations)} tactic combinations")
                        print(f"    Top 5: {[(t, f'{p:.3f}') for t, p in tactic_combinations[:5]]}")
                    
                    # temperature 応じて選択方法 変更
                    if temperature == 0.0:
                        # 確定的：確率の高い順 順番 試す
                        success = False
                        for tactic_str, probability in tactic_combinations[:args.max_steps]:
                            # タクティク 適用
                            success = apply_tactic_from_label(prover, tactic_str)
                            
                            # ログ用データ 記録
                            example_tactics.append(tactic_str)
                            example_confidences.append(probability)
                            tactic_usage[tactic_str] = tactic_usage.get(tactic_str, 0) + 1
                            
                            if args.verbose:
                                print(f"    Trying {tactic_str} (prob: {probability:.3f}) - {'Success' if success else 'Failed'}")
                            
                            if success:
                                break
                    else:
                        # 確率的：確率的 選択and試す
                        success = False
                        max_attempts = min(len(tactic_combinations), args.max_steps)
                        attempts = 0
                        
                        while not success and attempts < max_attempts:
                            selected_tactic, selected_prob = select_tactic_probabilistically(
                                tactic_combinations, temperature, failed_tactics
                            )
                            
                            if not selected_tactic:
                                # 利用可能なタクティク no/not場合は終了
                                break
                            
                            # タクティク 適用
                            success = apply_tactic_from_label(prover, selected_tactic)
                            attempts += 1
                            
                            # ログ用データ 記録
                            example_tactics.append(selected_tactic)
                            example_confidences.append(selected_prob)
                            tactic_usage[selected_tactic] = tactic_usage.get(selected_tactic, 0) + 1
                            
                            if args.verbose:
                                print(f"    Trying {selected_tactic} (prob: {selected_prob:.3f}) - {'Success' if success else 'Failed'}")
                            
                            if not success:
                                # 失敗didタクティク 記録
                                failed_tactics.add(selected_tactic)
                    
                    step += 1
                    solved = prover.goal is None
                
                # 例の結果 記録
                step_counts.append(step)
                confidence_scores.extend(example_confidences)
                
                if solved:
                    solved_count += 1
                    if args.verbose:
                        print(f"  Result: SOLVED in {step} steps")
                else:
                    if args.verbose:
                        print(f"  Result: FAILED after {step} steps")
                
            except Exception as e:
                # パースエラーetc with/at 失敗did場合はスキップ
                print(f"Warning: Failed to process tautology {i+1}: {e}")
                step_counts.append(args.max_steps)  # 失敗 as 記録
            
            # Update progress bar
            progress.update(1)
    
    # 最終結果 計算
    total_examples = len(step_counts)  # 実際 処理was done問題数
    success_rate = solved_count / total_examples if total_examples > 0 else 0.0
    avg_steps = sum(step_counts) / len(step_counts) if step_counts else 0
    avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
    
    print(f"\nResults: {solved_count}/{total_examples} examples solved ({success_rate*100:.1f}%)")
    print(f"Average steps: {avg_steps:.2f}")
    print(f"Average confidence: {avg_confidence:.3f}")


if __name__ == "__main__":
    main()
