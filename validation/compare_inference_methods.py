"""
階層分類推論とauto_classicalの性能比較スクリプト
"""
from __future__ import annotations

import argparse
import os
import sys
import json
import time
from typing import List, Tuple, Dict, Any, Optional

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Install with: pip install wandb")

# プロジェクトルートをパスに追加
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
    """階層分類モデルを読み込み"""
    checkpoint = torch.load(model_path, map_location=device)
    
    # 新しい形式のチェックポイントかどうかを判定
    if 'model_params' in checkpoint:
        # 新しい形式のチェックポイント
        model_params = checkpoint['model_params']
        vocab_size = checkpoint.get('vocab_size', model_params['vocab_size'])
        pad_id = checkpoint.get('pad_id', model_params['pad_id'])
        max_seq_len = checkpoint.get('max_seq_len', model_params['max_seq_len'])
        
        # クラス数をチェックポイントから取得
        num_main_classes = len(checkpoint['id_to_main'])
        num_arg1_classes = len(checkpoint['id_to_arg1'])
        num_arg2_classes = len(checkpoint['id_to_arg2'])
        
        # ラベルマッピングを取得
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
        # 古い形式のチェックポイント（重みのみ）
        print("Loading old format checkpoint (weights only)")
        
        # デフォルトのモデルパラメータを取得
        from src.core.parameter import get_model_params
        model_params = get_model_params()
        
        # チェックポイントからvocab_sizeを取得（embedding層のサイズから）
        vocab_size = checkpoint['embedding.weight'].shape[0]
        print(f"Detected vocab_size from checkpoint: {vocab_size}")
        
        # デフォルトのラベルマッピングを取得
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
        
        # モデルを作成（チェックポイントから取得したvocab_sizeを使用）
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
        
        # 重みを読み込み
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
    タクティクの種類に応じて適切な確率を計算
    """
    # 引数不要なタクティク
    if main_tactic in ['assumption', 'intro', 'split', 'left', 'right', 'add_dn']:
        return main_confidence
    
    # 引数1つのタクティク
    elif main_tactic in ['apply', 'destruct']:
        return main_confidence * arg1_confidence
    
    # 引数2つのタクティク
    elif main_tactic == 'specialize':
        return main_confidence * arg1_confidence * arg2_confidence
    
    # その他のタクティク（引数不要として扱う）
    else:
        return main_confidence


def generate_all_tactic_combinations(
    model: TransformerClassifier,
    tokenizer: CharTokenizer,
    premises: List[str],
    goal: str,
    label_mappings: Dict[str, Any],
    device: torch.device,
    max_seq_len: int = 256
) -> List[Tuple[str, float]]:
    """
    すべての可能なタクティクの組み合わせを生成し、確率の高い順にソート
    """
    # 入力をエンコード
    input_ids, attention_mask, segment_ids = tokenizer.encode(goal, premises, max_seq_len)
    input_ids = input_ids.unsqueeze(0).to(device)
    attention_mask = attention_mask.unsqueeze(0).to(device)
    segment_ids = segment_ids.unsqueeze(0).to(device)
    
    with torch.no_grad():
        main_logits, arg1_logits, arg2_logits = model(input_ids, attention_mask)
        
        # softmaxで確率に変換
        main_probs = torch.softmax(main_logits, dim=-1)
        arg1_probs = torch.softmax(arg1_logits, dim=-1)
        arg2_probs = torch.softmax(arg2_logits, dim=-1)
        
        tactic_combinations = []
        
        # すべての可能な組み合わせを生成
        for main_id, main_tactic in enumerate(label_mappings['id_to_main']):
            main_confidence = main_probs[0, main_id].item()
            
            # 引数が不要なタクティクの場合
            if main_tactic in ['assumption', 'intro', 'split', 'left', 'right', 'add_dn']:
                tactic_string = main_tactic
                probability = calculate_tactic_probability(
                    main_tactic, "", "",
                    main_confidence, 0.0, 0.0
                )
                tactic_combinations.append((tactic_string, probability))
            
            # 引数1つのタクティクの場合
            elif main_tactic in ['apply', 'destruct']:
                for arg1_id, arg1_value in enumerate(label_mappings['id_to_arg1']):
                    arg1_confidence = arg1_probs[0, arg1_id].item()
                    tactic_string = f"{main_tactic} {arg1_value}"
                    probability = calculate_tactic_probability(
                        main_tactic, arg1_value, "",
                        main_confidence, arg1_confidence, 0.0
                    )
                    tactic_combinations.append((tactic_string, probability))
            
            # 引数2つのタクティクの場合
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
            
            # その他のタクティク（引数不要として扱う）
            else:
                tactic_string = main_tactic
                probability = calculate_tactic_probability(
                    main_tactic, "", "",
                    main_confidence, 0.0, 0.0
                )
                tactic_combinations.append((tactic_string, probability))
        
        # 確率の高い順にソート
        tactic_combinations.sort(key=lambda x: x[1], reverse=True)
        
        return tactic_combinations


def apply_tactic_from_label(prover, label) -> bool:
    """タクティクを適用"""
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
        # pyproverのエラーをキャッチしてFalseを返す
        return False


def test_auto_classical(goal_str: str, depth_limit: int = 10) -> Tuple[bool, List[str], float]:
    """
    auto_classicalで問題を解く
    
    Returns:
        (solved, tactics_used, time_taken)
    """
    # pyproverをインポート
    pyprover_dir = os.path.join(project_root, "pyprover")
    sys.path.insert(0, pyprover_dir)
    
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
    
    try:
        # パースしてproverを作成
        parse_tree = PropParseTree()
        goal_node = parse_tree.transform(prop_parser.parse(goal_str))
        prover = Prover(goal_node)
        
        # auto_classicalを実行
        start_time = time.time()
        tactics = prover.auto_classical(depth_limit)
        time_taken = time.time() - start_time
        
        # 解けたかどうかを判定
        solved = prover.goal is None
        
        return solved, tactics, time_taken
        
    except Exception as e:
        print(f"Error in auto_classical for '{goal_str}': {e}")
        return False, [], 0.0


def test_hierarchical_inference(
    goal_str: str,
    model: TransformerClassifier,
    tokenizer: CharTokenizer,
    label_mappings: Dict[str, Any],
    device: torch.device,
    max_seq_len: int,
    max_steps: int = 30
) -> Tuple[bool, List[str], float, float]:
    """
    階層分類推論で問題を解く
    
    Returns:
        (solved, tactics_used, time_taken, avg_confidence)
    """
    # pyproverをインポート
    pyprover_dir = os.path.join(project_root, "pyprover")
    sys.path.insert(0, pyprover_dir)
    
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
    
    try:
        # パースしてproverを作成
        parse_tree = PropParseTree()
        goal_node = parse_tree.transform(prop_parser.parse(goal_str))
        prover = Prover(goal_node)
        
        # 前提は空（トートロジーなので前提なしで証明可能）
        premises = []
        
        # 推論ループ
        step = 0
        solved = prover.goal is None
        tactics_used = []
        confidences = []
        
        start_time = time.time()
        
        while not solved and step < max_steps:
            # 現在の状態を取得
            current_state = encode_prover_state(prover)
            current_premises = current_state["premises"]
            current_goal = current_state["goal"]
            
            # すべての可能なタクティクの組み合わせを生成（確率の高い順）
            tactic_combinations = generate_all_tactic_combinations(
                model, tokenizer, current_premises, current_goal,
                label_mappings, device, max_seq_len
            )
            
            # 上位max_steps個のタクティクを順次適用
            success = False
            for tactic_str, probability in tactic_combinations[:max_steps]:
                # タクティクを適用
                success = apply_tactic_from_label(prover, tactic_str)
                
                # ログ用データを記録
                tactics_used.append(tactic_str)
                confidences.append(probability)
                
                if success:
                    break
            
            step += 1
            solved = prover.goal is None
        
        time_taken = time.time() - start_time
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        return solved, tactics_used, time_taken, avg_confidence
        
    except Exception as e:
        print(f"Error in hierarchical inference for '{goal_str}': {e}")
        return False, [], 0.0, 0.0


def load_validation_data(validation_file: str, num_examples: int = None) -> List[str]:
    """
    バリデーションデータを読み込み
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
    parser = argparse.ArgumentParser(description="Compare hierarchical inference with auto_classical")
    parser.add_argument("--model_path", type=str, default="models/pretrained_model.pth", help="model path")
    parser.add_argument("--validation_file", type=str, default="validation_tautology.json", help="validation file path")
    parser.add_argument("--count", type=int, default=None, help="number of examples to run (default: all in validation file)")
    parser.add_argument("--max_steps", type=int, default=30, help="max steps per example for hierarchical inference")
    parser.add_argument("--auto_depth", type=int, default=10, help="depth limit for auto_classical")
    parser.add_argument("--device", type=str, default="auto", help="device")
    parser.add_argument("--verbose", action="store_true", help="verbose output")
    parser.add_argument("--use_wandb", action="store_true", help="use wandb for logging")
    parser.add_argument("--wandb_project", type=str, default="fof-comparison", help="wandb project name")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="wandb run name")
    parser.add_argument("--max_seq_len", type=int, default=256, help="maximum sequence length")
    
    args = parser.parse_args()
    
    # デバイス設定
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # wandb初期化
    if args.use_wandb and WANDB_AVAILABLE:
        run_name = args.wandb_run_name or f"comparison_{int(time.time())}"
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config={
                "model_path": args.model_path,
                "validation_file": args.validation_file,
                "count": args.count,
                "max_steps": args.max_steps,
                "auto_depth": args.auto_depth,
                "device": str(device)
            }
        )
        print(f"Wandb initialized: {args.wandb_project}/{run_name}")
    elif args.use_wandb and not WANDB_AVAILABLE:
        print("Warning: wandb requested but not available. Continuing without logging.")
    
    # バリデーションデータを読み込み
    validation_file = os.path.join(os.path.dirname(__file__), args.validation_file)
    print(f"\nLoading validation data from {validation_file}...")
    
    tautologies = load_validation_data(validation_file, args.count)
    
    if not tautologies:
        print("Error: No validation data loaded!")
        return
    
    # モデルを読み込みまたは初期化
    if not os.path.exists(args.model_path):
        print(f"Model file not found: {args.model_path}")
        print("Creating a randomly initialized model...")
        
        # 初期化されたモデルを作成
        from src.core.parameter import get_model_params, get_hierarchical_labels
        
        model_params = get_model_params()
        hierarchical_labels = get_hierarchical_labels()
        
        # ラベルマッピングを作成
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
        
        # トークナイザーを作成
        root_dir = os.path.dirname(os.path.dirname(__file__))
        token_py_path = os.path.join(root_dir, "src", "core", "fof_tokens.py")
        base_tokens, _ = load_tokens_and_labels_from_token_py(token_py_path)
        tokenizer = CharTokenizer(base_tokens=base_tokens)
        
        # モデルを作成
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
        
        print(f"Created randomly initialized model with {len(main_to_id)} main tactics, {len(arg1_to_id)} arg1 values, {len(arg2_to_id)} arg2 values")
    else:
        model, label_mappings = load_hierarchical_model(args.model_path, device)
        print(f"Loaded model from {args.model_path}")
        
        # トークナイザーを作成
        root_dir = os.path.dirname(os.path.dirname(__file__))
        token_py_path = os.path.join(root_dir, "src", "core", "fof_tokens.py")
        base_tokens, _ = load_tokens_and_labels_from_token_py(token_py_path)
        tokenizer = CharTokenizer(base_tokens=base_tokens)
    
    print(f"\n{len(tautologies)}個の問題を比較中...")
    
    # 結果を格納する変数
    results = []
    auto_classical_solved = 0
    hierarchical_solved = 0
    both_solved = 0
    only_auto_solved = 0
    only_hierarchical_solved = 0
    neither_solved = 0
    
    auto_times = []
    hierarchical_times = []
    hierarchical_confidences = []
    
    for i, goal_str in enumerate(tautologies):
        if not goal_str:
            print(f"Warning: Empty formula for example {i+1}, skipping...")
            continue
        
        # auto_classicalでテスト
        auto_solved, auto_tactics, auto_time = test_auto_classical(goal_str, args.auto_depth)
        
        # 階層分類推論でテスト
        hierarchical_solved_result, hierarchical_tactics, hierarchical_time, hierarchical_confidence = test_hierarchical_inference(
            goal_str, model, tokenizer, label_mappings, device, args.max_seq_len, args.max_steps
        )
        
        # 結果を記録
        result = {
            'example_id': i + 1,
            'goal': goal_str,
            'auto_classical': {
                'solved': auto_solved,
                'tactics': auto_tactics,
                'time': auto_time
            },
            'hierarchical': {
                'solved': hierarchical_solved_result,
                'tactics': hierarchical_tactics,
                'time': hierarchical_time,
                'avg_confidence': hierarchical_confidence
            }
        }
        results.append(result)
        
        # 統計を更新
        if auto_solved:
            auto_classical_solved += 1
            auto_times.append(auto_time)
        
        if hierarchical_solved_result:
            hierarchical_solved += 1
            hierarchical_times.append(hierarchical_time)
            hierarchical_confidences.append(hierarchical_confidence)
        
        if auto_solved and hierarchical_solved_result:
            both_solved += 1
        elif auto_solved and not hierarchical_solved_result:
            only_auto_solved += 1
        elif not auto_solved and hierarchical_solved_result:
            only_hierarchical_solved += 1
        else:
            neither_solved += 1
        
        if args.verbose:
            auto_status = "✓" if auto_solved else "✗"
            hierarchical_status = "✓" if hierarchical_solved_result else "✗"
            print(f"{i+1:3d}: {auto_status} auto_classical  {hierarchical_status} 階層分類推論")
        else:
            # 進捗表示（10問ごと）
            if (i + 1) % 10 == 0 or i == len(tautologies) - 1:
                print(f"進捗: {i+1}/{len(tautologies)}")
    
    # 最終結果を計算
    total_examples = len(results)
    auto_success_rate = auto_classical_solved / total_examples if total_examples > 0 else 0.0
    hierarchical_success_rate = hierarchical_solved / total_examples if total_examples > 0 else 0.0
    
    avg_auto_time = sum(auto_times) / len(auto_times) if auto_times else 0.0
    avg_hierarchical_time = sum(hierarchical_times) / len(hierarchical_times) if hierarchical_times else 0.0
    avg_hierarchical_confidence = sum(hierarchical_confidences) / len(hierarchical_confidences) if hierarchical_confidences else 0.0
    
    # 階層分類推論で解けたがauto_classicalで解けなかった問題を収集
    hierarchical_only_examples = []
    # auto_classicalで解けたが階層分類推論で解けなかった問題を収集
    auto_only_examples = []
    for result in results:
        if not result['auto_classical']['solved'] and result['hierarchical']['solved']:
            hierarchical_only_examples.append(result)
        elif result['auto_classical']['solved'] and not result['hierarchical']['solved']:
            auto_only_examples.append(result)
    
    # 結果を表示
    print(f"\n{'='*50}")
    print(f"比較結果")
    print(f"{'='*50}")
    print(f"総問題数: {total_examples}")
    print(f"")
    print(f"auto_classical:     {auto_classical_solved:3d}/{total_examples} ({auto_success_rate*100:5.1f}%)")
    print(f"階層分類推論:       {hierarchical_solved:3d}/{total_examples} ({hierarchical_success_rate*100:5.1f}%)")
    print(f"")
    print(f"階層分類推論の優位性:")
    print(f"  auto_classicalで解けなかった問題を解けた数: {only_hierarchical_solved:3d}")
    print(f"  追加成功率: {only_hierarchical_solved/total_examples*100:5.1f}%")
    
    # 階層分類推論で解けたがauto_classicalで解けなかった問題を表示
    if hierarchical_only_examples:
        print(f"\n階層分類推論で解けたがauto_classicalで解けなかった問題:")
        print(f"{'='*50}")
        for i, result in enumerate(hierarchical_only_examples, 1):
            print(f"{i:2d}. {result['goal']}")
            print(f"    使用タクティク: {', '.join(result['hierarchical']['tactics'])}")
            print(f"    実行時間: {result['hierarchical']['time']:.3f}s")
            print()
    
    # auto_classicalで解けたが階層分類推論で解けなかった問題を表示
    if auto_only_examples:
        print(f"\nauto_classicalで解けたが階層分類推論で解けなかった問題:")
        print(f"{'='*50}")
        for i, result in enumerate(auto_only_examples, 1):
            print(f"{i:2d}. {result['goal']}")
            print(f"    auto_classical使用タクティク: {', '.join(result['auto_classical']['tactics'])}")
            print(f"    auto_classical実行時間: {result['auto_classical']['time']:.3f}s")
            print(f"    階層分類推論実行時間: {result['hierarchical']['time']:.3f}s")
            print(f"    階層分類推論平均信頼度: {result['hierarchical']['avg_confidence']:.3f}")
            print()
    
    # wandbに結果をログ
    if args.use_wandb and WANDB_AVAILABLE:
        wandb.log({
            "final/auto_classical_success_rate": auto_success_rate,
            "final/hierarchical_success_rate": hierarchical_success_rate,
            "final/auto_classical_solved": auto_classical_solved,
            "final/hierarchical_solved": hierarchical_solved,
            "final/both_solved": both_solved,
            "final/only_auto_solved": only_auto_solved,
            "final/only_hierarchical_solved": only_hierarchical_solved,
            "final/neither_solved": neither_solved,
            "final/avg_auto_time": avg_auto_time,
            "final/avg_hierarchical_time": avg_hierarchical_time,
            "final/avg_hierarchical_confidence": avg_hierarchical_confidence,
            "final/hierarchical_advantage": only_hierarchical_solved / total_examples,
        })
        
        # 各例の結果もログ
        for result in results:
            wandb.log({
                f"example_{result['example_id']}/auto_solved": 1 if result['auto_classical']['solved'] else 0,
                f"example_{result['example_id']}/hierarchical_solved": 1 if result['hierarchical']['solved'] else 0,
                f"example_{result['example_id']}/auto_time": result['auto_classical']['time'],
                f"example_{result['example_id']}/hierarchical_time": result['hierarchical']['time'],
                f"example_{result['example_id']}/hierarchical_confidence": result['hierarchical']['avg_confidence'],
            })
        
        wandb.finish()
        print("Wandb logging completed!")
    


if __name__ == "__main__":
    main()
