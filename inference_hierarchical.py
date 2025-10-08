"""
階層分類対応の推論スクリプト
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import List, Tuple, Dict, Any

import torch

from transformer_classifier import (
    load_tokens_and_labels_from_token_py,
    CharTokenizer,
    TransformerClassifier,
)
from state_encoder import encode_prover_state, format_tactic_string
from parameter import (
    default_params, get_model_params, get_generation_params, 
    get_system_params, DeviceType
)


def load_hierarchical_model(model_path: str, device: torch.device) -> Tuple[TransformerClassifier, Dict[str, Any]]:
    """階層分類モデルを読み込み"""
    checkpoint = torch.load(model_path, map_location=device)
    
    # モデルパラメータを取得
    model_params = checkpoint['model_params']
    
    # モデルを作成
    vocab_size = checkpoint.get('vocab_size', model_params['vocab_size'])
    pad_id = checkpoint.get('pad_id', model_params['pad_id'])
    
    # クラス数をチェックポイントから取得
    num_main_classes = len(checkpoint['id_to_main'])
    num_arg1_classes = len(checkpoint['id_to_arg1'])
    num_arg2_classes = len(checkpoint['id_to_arg2'])
    
    model = TransformerClassifier(
        vocab_size=vocab_size,
        pad_id=pad_id,
        max_seq_len=model_params['max_seq_len'],
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
    model.to(device)
    model.eval()
    
    # ラベルマッピングを取得
    label_mappings = {
        'main_to_id': checkpoint['main_to_id'],
        'arg1_to_id': checkpoint['arg1_to_id'],
        'arg2_to_id': checkpoint['arg2_to_id'],
        'id_to_main': checkpoint['id_to_main'],
        'id_to_arg1': checkpoint['id_to_arg1'],
        'id_to_arg2': checkpoint['id_to_arg2'],
    }
    
    return model, label_mappings


def predict_tactic(
    model: TransformerClassifier,
    tokenizer: CharTokenizer,
    premises: List[str],
    goal: str,
    label_mappings: Dict[str, Any],
    device: torch.device,
    banned_tactics: set = None
) -> Tuple[str, float, float, float]:
    """
    タクティクを予測
    
    Returns:
        (tactic_string, main_confidence, arg1_confidence, arg2_confidence)
    """
    if banned_tactics is None:
        banned_tactics = set()
    
    # 入力をエンコード
    input_ids, attention_mask, segment_ids = tokenizer.encode(goal, premises)
    input_ids = input_ids.unsqueeze(0).to(device)
    attention_mask = attention_mask.unsqueeze(0).to(device)
    segment_ids = segment_ids.unsqueeze(0).to(device)
    
    with torch.no_grad():
        main_logits, arg1_logits, arg2_logits = model(input_ids, attention_mask, segment_ids)
        
        # 主タクティクを予測
        main_probs = torch.softmax(main_logits, dim=-1)
        main_pred_id = torch.argmax(main_probs, dim=-1).item()
        main_confidence = main_probs[0, main_pred_id].item()
        
        # 禁止されたタクティクをマスク
        if banned_tactics:
            for tactic in banned_tactics:
                if tactic in label_mappings['main_to_id']:
                    tactic_id = label_mappings['main_to_id'][tactic]
                    main_probs[0, tactic_id] = 0.0
            
            # 再正規化
            main_probs = main_probs / main_probs.sum(dim=-1, keepdim=True)
            main_pred_id = torch.argmax(main_probs, dim=-1).item()
            main_confidence = main_probs[0, main_pred_id].item()
        
        # 引数を予測
        arg1_probs = torch.softmax(arg1_logits, dim=-1)
        arg1_pred_id = torch.argmax(arg1_probs, dim=-1).item()
        arg1_confidence = arg1_probs[0, arg1_pred_id].item()
        
        arg2_probs = torch.softmax(arg2_logits, dim=-1)
        arg2_pred_id = torch.argmax(arg2_probs, dim=-1).item()
        arg2_confidence = arg2_probs[0, arg2_pred_id].item()
        
        # タクティク文字列を構築
        main_tactic = label_mappings['id_to_main'][main_pred_id]
        arg1_value = label_mappings['id_to_arg1'][arg1_pred_id]
        arg2_value = label_mappings['id_to_arg2'][arg2_pred_id]
        
        # 引数が不要なタクティクの場合は引数を無視
        if main_tactic in ['assumption', 'intro', 'split', 'left', 'right', 'add_dn']:
            tactic_string = main_tactic
        elif main_tactic in ['apply', 'destruct']:
            tactic_string = f"{main_tactic} {arg1_value}"
        elif main_tactic == 'specialize':
            tactic_string = f"{main_tactic} {arg1_value} {arg2_value}"
        else:
            tactic_string = main_tactic
        
        return tactic_string, main_confidence, arg1_confidence, arg2_confidence


def apply_tactic_from_label(prover, label) -> bool:
    """タクティクを適用"""
    if isinstance(label, dict):
        tactic_str = format_tactic_string(label)
    else:
        tactic_str = label
    
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


def main():
    parser = argparse.ArgumentParser(description="Run hierarchical tactic inference")
    parser.add_argument("--model_path", type=str, default="hierarchical_model.pth", help="model path")
    parser.add_argument("--count", type=int, default=10, help="number of examples to run")
    parser.add_argument("--max_steps", type=int, default=5, help="max steps per example")
    parser.add_argument("--device", type=str, default="auto", help="device")
    parser.add_argument("--verbose", action="store_true", help="verbose output")
    
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
        print("Please train a model first using train_hierarchical.py")
        return
    
    model, label_mappings = load_hierarchical_model(args.model_path, device)
    print(f"Loaded model from {args.model_path}")
    
    # トークナイザーを作成
    root_dir = os.path.dirname(__file__)
    token_py_path = os.path.join(root_dir, "fof_tokens.py")
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
    
    # 簡単なテスト例を実行
    print(f"\nRunning {args.count} examples...")
    
    solved_count = 0
    
    for i in range(args.count):
        # 簡単な例を作成
        if i % 3 == 0:
            # (a → a) の例
            goal_str = "(a → a)"
            premises = []
        elif i % 3 == 1:
            # (a → b), a から b を導く例
            goal_str = "b"
            premises = ["(a → b)", "a"]
        else:
            # (a ∨ b) から a を導く例
            goal_str = "a"
            premises = ["(a ∨ b)"]
        
        # パースしてproverを作成
        parse_tree = PropParseTree()
        goal_node = parse_tree.transform(prop_parser.parse(goal_str))
        prover = Prover(goal_node)
        
        # 前提を追加
        for prem_str in premises:
            prem_node = parse_tree.transform(prop_parser.parse(prem_str))
            prover.variables.append(prem_node)
        
        if args.verbose:
            print(f"\nExample {i+1}:")
            print(f"  Goal: {goal_str}")
            print(f"  Premises: {premises}")
        
        # 推論ループ
        step = 0
        solved = prover.goal is None
        banned_tactics = set()
        
        while not solved and step < args.max_steps:
            # 現在の状態を取得
            current_state = encode_prover_state(prover)
            current_premises = current_state["premises"]
            current_goal = current_state["goal"]
            
            # タクティクを予測
            tactic_str, main_conf, arg1_conf, arg2_conf = predict_tactic(
                model, tokenizer, current_premises, current_goal, 
                label_mappings, device, banned_tactics
            )
            
            if args.verbose:
                print(f"  Step {step+1}: {tactic_str} (conf: {main_conf:.3f}, {arg1_conf:.3f}, {arg2_conf:.3f})")
            
            # タクティクを適用
            success = apply_tactic_from_label(prover, tactic_str)
            
            if success:
                if args.verbose:
                    print(f"    Applied successfully")
                banned_tactics = set()  # 成功したら禁止リストをリセット
            else:
                if args.verbose:
                    print(f"    Failed")
                banned_tactics.add(tactic_str)
            
            step += 1
            solved = prover.goal is None
        
        if solved:
            solved_count += 1
            if args.verbose:
                print(f"  Result: SOLVED in {step} steps")
        else:
            if args.verbose:
                print(f"  Result: FAILED after {step} steps")
    
    print(f"\nResults: {solved_count}/{args.count} examples solved ({solved_count/args.count*100:.1f}%)")


if __name__ == "__main__":
    main()
