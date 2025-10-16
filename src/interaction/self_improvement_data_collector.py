"""
Self Improvement Data Collector
解けたexampleの成功したタクティクのみを収集して学習データとして保存するスクリプト
"""
from __future__ import annotations

import argparse
import os
import sys
import json
import time
import hashlib
import heapq
from typing import List, Tuple, Dict, Any

# プロジェクトルートをパスに追加
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, project_root)

import torch

# グローバル定数（一度だけ初期化）
PYPROVER_MODULES = None
GENERATION_PARAMS = None
BASE_TOKENS = None
TACTIC_PARSER_CACHE = {}

from src.core.transformer_classifier import (
    load_tokens_and_labels_from_token_py,
    CharTokenizer,
    TransformerClassifier,
)
from src.core.state_encoder import encode_prover_state, format_tactic_string


def initialize_global_constants():
    """グローバル定数を一度だけ初期化"""
    global PYPROVER_MODULES, GENERATION_PARAMS, BASE_TOKENS
    
    if PYPROVER_MODULES is None:
        # pyproverモジュールを初期化
        pyprover_dir = os.path.join(project_root, "pyprover")
        sys.path.insert(0, pyprover_dir)
        
        original_cwd = os.getcwd()
        os.chdir(pyprover_dir)
        try:
            import proposition as proposition_mod
            import prover as prover_mod
        finally:
            os.chdir(original_cwd)
        
        PYPROVER_MODULES = {
            'PropParseTree': proposition_mod.PropParseTree,
            'prop_parser': proposition_mod.parser,
            'Prover': prover_mod.Prover
        }
    
    if GENERATION_PARAMS is None:
        from src.core.parameter import get_generation_params
        GENERATION_PARAMS = get_generation_params()
    
    if BASE_TOKENS is None:
        token_py_path = os.path.join(project_root, "src", "core", "fof_tokens.py")
        BASE_TOKENS, _ = load_tokens_and_labels_from_token_py(token_py_path)


def parse_tactic_string_cached(tactic_str: str) -> Dict[str, Any]:
    """タクティク文字列をパース（キャッシュ付き）"""
    if tactic_str not in TACTIC_PARSER_CACHE:
        parts = tactic_str.split()
        
        if len(parts) == 1:
            TACTIC_PARSER_CACHE[tactic_str] = {
                "main": parts[0],
                "arg1": None,
                "arg2": None
            }
        elif len(parts) == 2:
            TACTIC_PARSER_CACHE[tactic_str] = {
                "main": parts[0],
                "arg1": parts[1],
                "arg2": None
            }
        elif len(parts) == 3:
            TACTIC_PARSER_CACHE[tactic_str] = {
                "main": parts[0],
                "arg1": parts[1],
                "arg2": parts[2]
            }
        else:
            TACTIC_PARSER_CACHE[tactic_str] = {
                "main": tactic_str,
                "arg1": None,
                "arg2": None
            }
    
    return TACTIC_PARSER_CACHE[tactic_str]


def create_state_hash(premises: List[str], goal: str, tactic_str: str) -> str:
    """状態ハッシュを効率的に作成"""
    # 文字列結合を最適化
    state_tactic_str = f"{'|'.join(premises)}|{goal}|{tactic_str}"
    return hashlib.md5(state_tactic_str.encode()).hexdigest()


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


def generate_top_k_tactic_combinations(
    model: TransformerClassifier,
    tokenizer: CharTokenizer,
    premises: List[str],
    goal: str,
    label_mappings: Dict[str, Any],
    device: torch.device,
    max_seq_len: int = 256,
    top_k: int = 10,
    probability_threshold: float = 0.001
) -> List[Tuple[str, float]]:
    """
    上位k個のタクティク組み合わせを効率的に生成（ヒープソート使用）
    
    Args:
        top_k: 返すタクティクの最大数
        probability_threshold: 確率の閾値（これ以下の組み合わせは生成しない）
    
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
        
        # 最小ヒープ（確率の低い順）で上位k個を保持
        min_heap = []
        
        # 引数が不要なタクティクを処理
        for main_id, main_tactic in enumerate(label_mappings['id_to_main']):
            main_confidence = main_probs[0, main_id].item()
            
            # 確率閾値チェック
            if main_confidence < probability_threshold:
                continue
                
            if main_tactic in ['assumption', 'intro', 'split', 'left', 'right', 'add_dn']:
                tactic_string = main_tactic
                probability = calculate_tactic_probability(
                    main_tactic, "", "",
                    main_confidence, 0.0, 0.0
                )
                
                # ヒープに追加（確率の低い順で保持）
                if len(min_heap) < top_k:
                    heapq.heappush(min_heap, (probability, tactic_string))
                elif probability > min_heap[0][0]:
                    heapq.heapreplace(min_heap, (probability, tactic_string))
            
            # 引数1つのタクティクの場合
            elif main_tactic in ['apply', 'destruct']:
                for arg1_id, arg1_value in enumerate(label_mappings['id_to_arg1']):
                    arg1_confidence = arg1_probs[0, arg1_id].item()
                    
                    # 確率閾値チェック（main * arg1）
                    if main_confidence * arg1_confidence < probability_threshold:
                        continue
                        
                    tactic_string = f"{main_tactic} {arg1_value}"
                    probability = calculate_tactic_probability(
                        main_tactic, arg1_value, "",
                        main_confidence, arg1_confidence, 0.0
                    )
                    
                    # ヒープに追加
                    if len(min_heap) < top_k:
                        heapq.heappush(min_heap, (probability, tactic_string))
                    elif probability > min_heap[0][0]:
                        heapq.heapreplace(min_heap, (probability, tactic_string))
            
            # 引数2つのタクティクの場合
            elif main_tactic == 'specialize':
                for arg1_id, arg1_value in enumerate(label_mappings['id_to_arg1']):
                    arg1_confidence = arg1_probs[0, arg1_id].item()
                    
                    # 確率閾値チェック（main * arg1）
                    if main_confidence * arg1_confidence < probability_threshold:
                        continue
                        
                    for arg2_id, arg2_value in enumerate(label_mappings['id_to_arg2']):
                        arg2_confidence = arg2_probs[0, arg2_id].item()
                        
                        # 確率閾値チェック（main * arg1 * arg2）
                        if main_confidence * arg1_confidence * arg2_confidence < probability_threshold:
                            continue
                            
                        tactic_string = f"{main_tactic} {arg1_value} {arg2_value}"
                        probability = calculate_tactic_probability(
                            main_tactic, arg1_value, arg2_value,
                            main_confidence, arg1_confidence, arg2_confidence
                        )
                        
                        # ヒープに追加
                        if len(min_heap) < top_k:
                            heapq.heappush(min_heap, (probability, tactic_string))
                        elif probability > min_heap[0][0]:
                            heapq.heapreplace(min_heap, (probability, tactic_string))
            
            # その他のタクティク（引数不要として扱う）
            else:
                tactic_string = main_tactic
                probability = calculate_tactic_probability(
                    main_tactic, "", "",
                    main_confidence, 0.0, 0.0
                )
                
                # ヒープに追加
                if len(min_heap) < top_k:
                    heapq.heappush(min_heap, (probability, tactic_string))
                elif probability > min_heap[0][0]:
                    heapq.heapreplace(min_heap, (probability, tactic_string))
        
        # ヒープから確率の高い順に取り出してリストに変換
        tactic_combinations = []
        while min_heap:
            probability, tactic_string = heapq.heappop(min_heap)
            tactic_combinations.append((tactic_string, probability))
        
        # 確率の高い順にソート（ヒープから取り出した順序を逆転）
        tactic_combinations.reverse()
        
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




def collect_self_improvement_data(
    model: TransformerClassifier,
    tokenizer: CharTokenizer,
    label_mappings: Dict[str, Any],
    device: torch.device,
    max_seq_len: int,
    num_examples: int = 100,
    max_steps: int = 30,
    top_k: int = 20,
    probability_threshold: float = 0.001,
    difficulty: float = 0.5,
    seed: int = None,
    verbose: bool = False
) -> List[Dict[str, Any]]:
    """
    Self improvement用のデータを収集
    
    Returns:
        成功したタクティクのリスト（deduplicated_batch形式）
    """
    # グローバル定数を初期化
    initialize_global_constants()
    
    # フォーミュラジェネレーターを作成
    from src.core.generate_prop import FormulaGenerator, filter_formulas
    
    # 利用可能な変数を推論
    variables = [t for t in ["a", "b", "c"] if t in BASE_TOKENS]
    if not variables:
        variables = ["a", "b", "c"]
    
    # フォーミュラジェネレーターを作成（inference_hierarchical.pyと同じ方法）
    # 注意: 実際の生成はgenerate_tautology関数内で行うため、ここでは作成しない
    
    # inference_hierarchical.pyと同じ方法でトートロジーを生成
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
    
    # トートロジーを逐次生成
    tautologies = []
    for i in range(num_examples):
        goal_str = generate_tautology(GENERATION_PARAMS, BASE_TOKENS, seed_offset=i)
        if goal_str:
            tautologies.append(goal_str)
    
    if not tautologies:
        print("Failed to generate tautologies for self improvement data collection!")
        return []
    
    print(f"Generated {len(tautologies)} tautologies for self improvement data collection")
    print("First 5 tautologies:")
    for i in range(min(5, len(tautologies))):
        print(f"  {i+1}: {tautologies[i]}")
    
    # 成功したタクティクを収集
    successful_tactics = []
    solved_count = 0
    
    for i, goal_str in enumerate(tautologies):
        try:
            # パースしてproverを作成
            parse_tree = PYPROVER_MODULES['PropParseTree']()
            goal_node = parse_tree.transform(PYPROVER_MODULES['prop_parser'].parse(goal_str))
            prover = PYPROVER_MODULES['Prover'](goal_node)
            
            # 前提は空（トートロジーなので前提なしで証明可能）
            premises = []
            
            if verbose:
                print(f"\nExample {i+1}: {goal_str}")
            
            # このexampleの成功したタクティクを一時的に保存
            example_successful_tactics = []
            
            # 推論ループ
            step = 0
            solved = prover.goal is None
            example_terminated = False  # example全体の早期終了フラグ
            consecutive_failures = 0  # 連続失敗数をステップ間で保持
            
            while not solved and step < max_steps and not example_terminated:
                # 現在の状態を取得
                current_state = encode_prover_state(prover)
                current_premises = current_state["premises"]
                current_goal = current_state["goal"]
                
                # 上位k個のタクティク組み合わせを効率的に生成
                tactic_combinations = generate_top_k_tactic_combinations(
                    model, tokenizer, current_premises, current_goal,
                    label_mappings, device, max_seq_len, top_k, probability_threshold
                )
                
                if verbose:
                    print(f"  Step {step+1}: Generated {len(tactic_combinations)} tactic combinations")
                    print(f"    Top 5: {[(t, f'{p:.3f}') for t, p in tactic_combinations[:5]]}")
                
                # 上位top_k個のタクティクを順次適用
                success = False
                step_failures = 0  # このステップでの失敗数
                total_tactics_in_step = len(tactic_combinations)  # このステップで生成されたタクティク数
                
                for tactic_str, probability in tactic_combinations:
                    # タクティクを適用
                    success = apply_tactic_from_label(prover, tactic_str)
                    
                    if verbose:
                        print(f"    Trying {tactic_str} (prob: {probability:.3f}) - {'Success' if success else 'Failed'}")
                    
                    if success:
                        # 成功したタクティクを一時的に記録
                        tactic_dict = parse_tactic_string_cached(tactic_str)
                        state_tactic_hash = create_state_hash(current_premises, current_goal, tactic_str)
                        
                        example_successful_tactics.append({
                            "step_index": step,
                            "premises": current_premises.copy(),
                            "goal": current_goal,
                            "tactic": tactic_dict,
                            "tactic_apply": True,
                            "state_tactic_hash": state_tactic_hash
                        })
                        consecutive_failures = 0  # 成功したら連続失敗数をリセット
                        break
                    else:
                        step_failures += 1
                        # このステップで全てのタクティクが失敗した場合、早期終了
                        if step_failures >= total_tactics_in_step:
                            if verbose:
                                print(f"    Early termination: {step_failures} consecutive failures in this step")
                            example_terminated = True
                            break
                
                step += 1
                solved = prover.goal is None
            
            # 解けた場合のみ、このexampleの成功したタクティクを追加
            if solved:
                solved_count += 1
                successful_tactics.extend(example_successful_tactics)
                if verbose:
                    print(f"  Result: SOLVED in {step} steps")
            else:
                if example_terminated:
                    if verbose:
                        print(f"  Result: EARLY TERMINATED after {step} steps (k consecutive failures)")
                else:
                    if verbose:
                        print(f"  Result: FAILED after {step} steps")
                
        except Exception as e:
            # パースエラーなどで失敗した場合はスキップ
            if verbose:
                print(f"Warning: Failed to process tautology {i+1}: {e}")
            continue
    
    print(f"\nSelf improvement data collection completed:")
    print(f"  Solved examples: {solved_count}/{len(tautologies)}")
    print(f"  Successful tactics collected: {len(successful_tactics)}")
    
    return successful_tactics


def clear_self_improvement_data(output_dir: str = "self_improvement_data") -> None:
    """Self improvementデータディレクトリをクリア"""
    if os.path.exists(output_dir):
        import shutil
        shutil.rmtree(output_dir)
        print(f"Cleared existing data in {output_dir}")
    else:
        print(f"Directory {output_dir} does not exist, no need to clear")


def save_self_improvement_data(
    data: List[Dict[str, Any]],
    output_dir: str = "self_improvement_data",
    batch_size: int = 1000
) -> None:
    """Self improvementデータをファイルに保存"""
    # 出力ディレクトリを作成
    os.makedirs(output_dir, exist_ok=True)
    
    # バッチごとに分割して保存
    for i in range(0, len(data), batch_size):
        batch_data = data[i:i + batch_size]
        batch_num = i // batch_size
        
        filename = f"training_data_{batch_num:05d}.json"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(batch_data, f, ensure_ascii=False, indent=2)
        
        print(f"Saved {len(batch_data)} tactics to {filepath}")


def main():
    parser = argparse.ArgumentParser(description="Collect self improvement data from solved examples")
    parser.add_argument("--model_path", type=str, default="models/pretrained_model.pth", help="model path")
    parser.add_argument("--count", type=int, default=100, help="number of examples to process")
    parser.add_argument("--max_steps", type=int, default=30, help="max steps per example")
    parser.add_argument("--top_k", type=int, default=20, help="top k tactics to generate and try per step")
    parser.add_argument("--probability_threshold", type=float, default=0.001, help="probability threshold for tactic generation")
    parser.add_argument("--device", type=str, default="auto", help="device")
    parser.add_argument("--verbose", action="store_true", help="verbose output")
    parser.add_argument("--max_seq_len", type=int, default=256, help="maximum sequence length")
    parser.add_argument("--difficulty", type=float, default=0.5, help="difficulty level for formula generation")
    parser.add_argument("--seed", type=int, default=None, help="random seed for formula generation")
    parser.add_argument("--output_dir", type=str, default="self_improvement_data", help="output directory for collected data")
    parser.add_argument("--batch_size", type=int, default=1000, help="batch size for saving data")
    
    args = parser.parse_args()
    
    # デバイス設定
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # モデルを読み込み
    if not os.path.exists(args.model_path):
        print(f"Model file not found: {args.model_path}")
        print("Please provide a valid model path.")
        return
    
    model, label_mappings = load_hierarchical_model(args.model_path, device)
    print(f"Loaded model from {args.model_path}")
    
    # トークナイザーを作成
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    token_py_path = os.path.join(root_dir, "src", "core", "fof_tokens.py")
    base_tokens, _ = load_tokens_and_labels_from_token_py(token_py_path)
    tokenizer = CharTokenizer(base_tokens=base_tokens)
    
    # モデルのmax_seq_lenを取得
    checkpoint = torch.load(args.model_path, map_location=device)
    max_seq_len = checkpoint.get('max_seq_len', 256)
    
    print(f"Starting self improvement data collection...")
    print(f"  Examples to process: {args.count}")
    print(f"  Max steps per example: {args.max_steps}")
    print(f"  Top k tactics per step: {args.top_k}")
    print(f"  Probability threshold: {args.probability_threshold}")
    print(f"  Difficulty: {args.difficulty}")
    print(f"  Output directory: {args.output_dir}")
    
    # 既存のデータをクリア
    clear_self_improvement_data(args.output_dir)
    
    # Self improvementデータを収集
    successful_tactics = collect_self_improvement_data(
        model=model,
        tokenizer=tokenizer,
        label_mappings=label_mappings,
        device=device,
        max_seq_len=max_seq_len,
        num_examples=args.count,
        max_steps=args.max_steps,
        top_k=args.top_k,
        probability_threshold=args.probability_threshold,
        difficulty=args.difficulty,
        seed=args.seed,
        verbose=args.verbose
    )
    
    if successful_tactics:
        # データを保存
        save_self_improvement_data(
            data=successful_tactics,
            output_dir=args.output_dir,
            batch_size=args.batch_size
        )
    else:
        print("No successful tactics collected. Please check your model and parameters.")


if __name__ == "__main__":
    main()
