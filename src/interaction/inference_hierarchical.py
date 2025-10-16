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

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Install with: pip install wandb")

# プロジェクトルートをパスに追加
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
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
    
    Args:
        main_tactic: メインタクティク
        arg1_value: 第1引数
        arg2_value: 第2引数
        main_confidence: メインタクティクの確信度
        arg1_confidence: 第1引数の確信度
        arg2_confidence: 第2引数の確信度
    
    Returns:
        計算された確率
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
    
    Returns:
        [(tactic_string, probability), ...] のリスト（確率の高い順）
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


def evaluate_inference_performance(
    model: TransformerClassifier,
    tokenizer: CharTokenizer,
    label_mappings: Dict[str, Any],
    device: torch.device,
    max_seq_len: int,
    num_examples: int = 50,
    max_steps: int = 5,
    difficulty: float = 0.7,
    seed: int = None,
    max_depth: int = None
) -> Tuple[float, float]:
    """
    推論性能を評価（トートロジーを新規生成）
    
    Returns:
        (success_rate, avg_steps_when_solved)
    """
    import sys
    import random
    import time
    
    # pyproverをインポート
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
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
    
    # generate_propをインポート
    from src.core.generate_prop import FormulaGenerator, filter_formulas
    
    # トートロジーを新規生成
    print(f"Generating {num_examples} new tautologies for validation...")
    
    # 変数を取得（fof_tokens.pyから）
    from src.core.transformer_classifier import load_tokens_and_labels_from_token_py
    token_py_path = os.path.join(project_root, "src", "core", "fof_tokens.py")
    base_tokens, _ = load_tokens_and_labels_from_token_py(token_py_path)
    
    # 利用可能な変数を推論
    variables = [t for t in ["a", "b", "c"] if t in base_tokens]
    if not variables:
        variables = ["a", "b", "c"]
    
    # データ生成時と同じパラメータを取得
    from src.core.parameter import get_generation_params
    gen_params = get_generation_params()
    
    # フォーミュラジェネレーターを作成
    gen = FormulaGenerator(
        variables=variables,
        allow_const=gen_params.allow_const,  # データ生成時と同じ設定
        difficulty=difficulty,  # 引数で指定された難易度を使用
        max_depth=max_depth if max_depth is not None else gen_params.max_depth,  # 指定された最大深度またはデータ生成時と同じ最大深度
        seed=seed if seed is not None else int(time.time() * 1000) % 2**32  # 指定されたシードまたは現在時刻を使用
    )
    
    # トートロジーを生成
    tautologies = filter_formulas(
        gen=gen,
        max_len=100,  # 最大長を制限
        require_tautology=True,  # トートロジーのみ
        limit=num_examples
    )
    
    if not tautologies:
        print("Failed to generate tautologies for validation!")
        return 0.0, 0.0
    
    print(f"Generated {len(tautologies)} tautologies for validation")
    print("First 5 tautologies:")
    for i in range(min(5, len(tautologies))):
        print(f"  {i+1}: {tautologies[i]}")
    
    # 推論性能評価を実行
    solved_count = 0
    solved_steps = []
    
    for i, goal_str in enumerate(tautologies):
        try:
            # パースしてproverを作成
            parse_tree = PropParseTree()
            goal_node = parse_tree.transform(prop_parser.parse(goal_str))
            prover = Prover(goal_node)
            
            # 前提は空（トートロジーなので前提なしで証明可能）
            premises = []
            
            # 推論ループ（確定的な順序適用）
            step = 0
            solved = prover.goal is None
            
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
                    if success:
                        break
                
                step += 1
                solved = prover.goal is None
            
            if solved:
                solved_count += 1
                solved_steps.append(step)
                
        except Exception as e:
            # パースエラーなどで失敗した場合はスキップ
            print(f"Warning: Failed to process tautology {i+1}: {e}")
            continue
    
    # 統計を計算
    success_rate = solved_count / len(tautologies) if tautologies else 0.0
    avg_steps_when_solved = sum(solved_steps) / len(solved_steps) if solved_steps else 0.0
    
    return success_rate, avg_steps_when_solved


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


def main():
    parser = argparse.ArgumentParser(description="Run hierarchical tactic inference")
    parser.add_argument("--model_path", type=str, default="models/pretrained_model.pth", help="model path")
    parser.add_argument("--count", type=int, default=10, help="number of examples to run")
    parser.add_argument("--max_steps", type=int, default=30, help="max steps per example")
    parser.add_argument("--device", type=str, default="auto", help="device")
    parser.add_argument("--verbose", action="store_true", help="verbose output")
    parser.add_argument("--use_wandb", action="store_true", help="use wandb for logging")
    parser.add_argument("--wandb_project", type=str, default="fof-inference", help="wandb project name")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="wandb run name")
    parser.add_argument("--max_seq_len", type=int, default=256, help="maximum sequence length")
    parser.add_argument("--difficulty", type=float, default=0.5, help="difficulty level for formula generation")
    parser.add_argument("--max_depth", type=int, default=8, help="maximum depth for formula generation")
    parser.add_argument("--seed", type=int, default=None, help="random seed for formula generation")
    
    args = parser.parse_args()
    
    # デバイス設定
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # wandb初期化
    if args.use_wandb and WANDB_AVAILABLE:
        run_name = args.wandb_run_name or f"inference_{int(time.time())}"
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config={
                "model_path": args.model_path,
                "count": args.count,
                "max_steps": args.max_steps,
                "device": str(device)
            }
        )
        print(f"Wandb initialized: {args.wandb_project}/{run_name}")
    elif args.use_wandb and not WANDB_AVAILABLE:
        print("Warning: wandb requested but not available. Continuing without logging.")
    
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
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
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
        
        max_seq_len = args.max_seq_len if hasattr(args, 'max_seq_len') else 256
        
        print(f"Created randomly initialized model with {len(main_to_id)} main tactics, {len(arg1_to_id)} arg1 values, {len(arg2_to_id)} arg2 values")
    else:
        model, label_mappings = load_hierarchical_model(args.model_path, device)
        print(f"Loaded model from {args.model_path}")
        
        # モデルのmax_seq_lenを取得
        checkpoint = torch.load(args.model_path, map_location=device)
        max_seq_len = checkpoint.get('max_seq_len', 256)
        
        # トークナイザーを作成
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        token_py_path = os.path.join(root_dir, "src", "core", "fof_tokens.py")
        base_tokens, _ = load_tokens_and_labels_from_token_py(token_py_path)
        tokenizer = CharTokenizer(base_tokens=base_tokens)
    
    # pyproverをインポート
    pyprover_dir = os.path.join(root_dir, "pyprover")
    sys.path.insert(0, pyprover_dir)
    
    # ディレクトリを変更してからインポート
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
    
    # トートロジーを新規生成
    print(f"\nGenerating {args.count} new tautologies for inference...")
    
    # generate_tautology関数を直接定義
    def generate_tautology(gen_params, base_tokens, seed_offset=0):
        """run_interaction.pyと同じ方法でトートロジーを生成する関数"""
        from src.core.generate_prop import FormulaGenerator, filter_formulas
        
        # 変数を取得
        variables = [t for t in gen_params.variables if t in base_tokens] or gen_params.variables
        
        # 予測可能なシードを使用（基本シード + オフセット）
        dynamic_seed = gen_params.seed + seed_offset
        
        # フォーミュラジェネレーターを作成
        gen = FormulaGenerator(
            variables=variables, 
            allow_const=gen_params.allow_const, 
            difficulty=gen_params.difficulty, 
            seed=dynamic_seed
        )
        
        # トートロジーを生成
        goal_list = filter_formulas(gen, max_len=gen_params.max_len, require_tautology=True, limit=1)
        if not goal_list:
            # デバッグ情報を追加
            print(f"Debug: Failed to generate tautology for seed_offset={seed_offset}")
            print(f"  - difficulty={gen_params.difficulty}")
            print(f"  - max_len={gen_params.max_len}")
            print(f"  - variables={variables}")
            print(f"  - allow_const={gen_params.allow_const}")
            return None
        return goal_list[0]
    
    # 変数を取得（fof_tokens.pyから）
    token_py_path = os.path.join(project_root, "src", "core", "fof_tokens.py")
    base_tokens, _ = load_tokens_and_labels_from_token_py(token_py_path)
    
    # データ生成時と同じパラメータを取得
    from src.core.parameter import get_generation_params
    gen_params = get_generation_params()
    
    
    print(f"Running {args.count} examples (max_steps: {args.max_steps})...")
    
    solved_count = 0
    step_counts = []
    tactic_usage = {}
    confidence_scores = []
    
    for i in range(args.count):
        # run_interaction.pyと同じ関数を使用して論理式を生成
        goal_str = generate_tautology(gen_params, base_tokens, seed_offset=i)
        if not goal_str:
            print(f"Warning: No valid formulas generated for example {i+1}, skipping...")
            continue
        try:
            # パースしてproverを作成
            parse_tree = PropParseTree()
            goal_node = parse_tree.transform(prop_parser.parse(goal_str))
            prover = Prover(goal_node)
            
            # 前提は空（トートロジーなので前提なしで証明可能）
            premises = []
            
            if args.verbose:
                print(f"\nExample {i+1}:")
                print(f"  Goal: {goal_str}")
                print(f"  Premises: {premises}")
            
            # 推論ループ（確定的な順序適用）
            step = 0
            solved = prover.goal is None
            example_tactics = []
            example_confidences = []
            
            while not solved and step < args.max_steps:
                # 現在の状態を取得
                current_state = encode_prover_state(prover)
                current_premises = current_state["premises"]
                current_goal = current_state["goal"]
                
                # すべての可能なタクティクの組み合わせを生成（確率の高い順）
                tactic_combinations = generate_all_tactic_combinations(
                    model, tokenizer, current_premises, current_goal,
                    label_mappings, device, max_seq_len
                )
                
                if args.verbose:
                    print(f"  Step {step+1}: Generated {len(tactic_combinations)} tactic combinations")
                    print(f"    Top 5: {[(t, f'{p:.3f}') for t, p in tactic_combinations[:5]]}")
                
                # 上位max_steps個のタクティクを順次適用
                success = False
                for tactic_str, probability in tactic_combinations[:args.max_steps]:
                    # タクティクを適用
                    success = apply_tactic_from_label(prover, tactic_str)
                    
                    # ログ用データを記録
                    example_tactics.append(tactic_str)
                    example_confidences.append(probability)
                    tactic_usage[tactic_str] = tactic_usage.get(tactic_str, 0) + 1
                    
                    if args.verbose:
                        print(f"    Trying {tactic_str} (prob: {probability:.3f}) - {'Success' if success else 'Failed'}")
                    
                    if success:
                        break
                
                step += 1
                solved = prover.goal is None
            
            # 例の結果を記録
            step_counts.append(step)
            confidence_scores.extend(example_confidences)
            
            if solved:
                solved_count += 1
                if args.verbose:
                    print(f"  Result: SOLVED in {step} steps")
            else:
                if args.verbose:
                    print(f"  Result: FAILED after {step} steps")
            
            # wandbに例の結果をログ
            if args.use_wandb and WANDB_AVAILABLE:
                wandb.log({
                    f"example_{i+1}/solved": 1 if solved else 0,
                    f"example_{i+1}/steps": step,
                    f"example_{i+1}/avg_confidence": sum(example_confidences) / len(example_confidences) if example_confidences else 0,
                    f"example_{i+1}/goal_length": len(goal_str),
                    f"example_{i+1}/premises_count": len(premises)
                })
                
        except Exception as e:
            # パースエラーなどで失敗した場合はスキップ
            print(f"Warning: Failed to process tautology {i+1}: {e}")
            step_counts.append(args.max_steps)  # 失敗として記録
            continue
    
    # 最終結果を計算
    total_examples = len(step_counts)  # 実際に処理された問題数
    success_rate = solved_count / total_examples if total_examples > 0 else 0.0
    avg_steps = sum(step_counts) / len(step_counts) if step_counts else 0
    avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
    
    print(f"\nResults: {solved_count}/{total_examples} examples solved ({success_rate*100:.1f}%)")
    print(f"Average steps: {avg_steps:.2f}")
    print(f"Average confidence: {avg_confidence:.3f}")
    
    # wandbに最終結果をログ
    if args.use_wandb and WANDB_AVAILABLE:
        wandb.log({
            "final/success_rate": success_rate,
            "final/avg_steps": avg_steps,
            "final/avg_confidence": avg_confidence,
            "final/solved_count": solved_count,
            "final/total_examples": total_examples
        })
        
        # タクティク使用頻度をログ
        for tactic, count in tactic_usage.items():
            wandb.log({f"tactics/{tactic}": count})
        
        wandb.finish()
        print("Wandb logging completed!")


if __name__ == "__main__":
    main()
