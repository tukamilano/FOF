#!/usr/bin/env python3
"""
真の証明検証システム - 実際に戦術を適用して証明を検証
"""
import os
import sys
import torch
import json
import argparse
import time
from typing import List, Tuple, Dict, Any, Optional

# プロジェクトルートをパスに追加
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_root)

from src.core.transformer_classifier import (
    load_tokens_and_labels_from_token_py,
    CharTokenizer,
    TransformerClassifier,
)

class ProofState:
    """証明の状態を管理"""
    
    def __init__(self, goal: str):
        self.goal = goal
        self.premises = []
        self.current_goal = goal
        self.step_count = 0
        self.is_proven = False
        
    def apply_tactic(self, tactic: str, arg1: str, arg2: str) -> bool:
        """戦術を適用して証明状態を更新"""
        self.step_count += 1
        
        if tactic == "assumption":
            # 前提から直接証明
            if self.current_goal in self.premises:
                self.is_proven = True
                return True
            return False
            
        elif tactic == "intro":
            # 含意の導入: A → B を証明するために A を前提に追加して B を証明
            if "→" in self.current_goal:
                # 含意の左側を前提に追加
                left_side = self.current_goal.split("→")[0].strip()
                self.premises.append(left_side)
                # 右側を新しい目標に
                right_side = self.current_goal.split("→")[1].strip()
                self.current_goal = right_side
                return True
            return False
            
        elif tactic == "destruct":
            # 論理積の分解: A ∧ B から A と B を分離
            if "∧" in self.current_goal:
                # 論理積を分解
                parts = self.current_goal.split("∧")
                for part in parts:
                    part = part.strip()
                    if part not in self.premises:
                        self.premises.append(part)
                return True
            return False
            
        elif tactic == "left":
            # 論理和の左側: A ∨ B から A を選択
            if "∨" in self.current_goal:
                left_side = self.current_goal.split("∨")[0].strip()
                self.current_goal = left_side
                return True
            return False
            
        elif tactic == "right":
            # 論理和の右側: A ∨ B から B を選択
            if "∨" in self.current_goal:
                right_side = self.current_goal.split("∨")[1].strip()
                self.current_goal = right_side
                return True
            return False
            
        elif tactic == "split":
            # 論理和の分解: A ∨ B を証明するために A または B を証明
            if "∨" in self.current_goal:
                # 論理和の左側を試す
                left_side = self.current_goal.split("∨")[0].strip()
                if self._try_prove(left_side):
                    return True
                # 右側を試す
                right_side = self.current_goal.split("∨")[1].strip()
                if self._try_prove(right_side):
                    return True
            return False
            
        return False
    
    def _try_prove(self, goal: str) -> bool:
        """目標を証明しようと試みる"""
        # 簡単な証明チェック
        if goal in self.premises:
            return True
        if goal == "True":
            return True
        if goal in ["a", "b", "c"] and goal in self.premises:
            return True
        return False
    
    def check_proof_complete(self) -> bool:
        """証明が完了したかチェック"""
        if self.is_proven:
            return True
        if self.current_goal in self.premises:
            return True
        if self.current_goal == "True":
            return True
        return False

def load_hierarchical_model(model_path: str, device: torch.device) -> Tuple[TransformerClassifier, Dict[str, Any]]:
    """階層分類モデルを読み込み"""
    checkpoint = torch.load(model_path, map_location=device)
    
    # モデルパラメータを取得
    if 'model_params' in checkpoint:
        model_params = checkpoint['model_params']
    else:
        model_params = {}
    
    vocab_size = checkpoint.get('vocab_size', 65)
    pad_id = checkpoint.get('pad_id', 0)
    max_seq_len = checkpoint.get('max_seq_len', 256)
    
    # デフォルトのクラス数
    num_main_classes = 59
    num_arg1_classes = 10
    num_arg2_classes = 10
    
    # モデルを作成
    model = TransformerClassifier(
        vocab_size=vocab_size,
        pad_id=pad_id,
        max_seq_len=max_seq_len,
        d_model=model_params.get('d_model', 128),
        nhead=model_params.get('nhead', 8),
        num_layers=model_params.get('num_layers', 2),
        dim_feedforward=model_params.get('dim_feedforward', 256),
        dropout=model_params.get('dropout', 0.1),
        num_main_classes=num_main_classes,
        num_arg1_classes=num_arg1_classes,
        num_arg2_classes=num_arg2_classes,
    ).to(device)
    
    # モデルの重みを読み込み
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    return model, {}

def get_tactic_name(tactic_id: int) -> str:
    """戦術IDから戦術名を取得"""
    tactic_names = [
        "assumption", "intro", "apply", "destruct", "left", "right",
        "split", "exfalso", "contradiction", "exact", "reflexivity",
        "symmetry", "transitivity", "congruence", "rewrite", "unfold",
        "fold", "simpl", "auto", "tauto", "intuition", "omega",
        "ring", "field", "fourier", "psatz", "lia", "nia", "lra",
        "nra", "sos", "nsatz", "field_simplify", "ring_simplify",
        "algebra", "romega", "rtauto", "firstorder", "eauto",
        "autorewrite", "autounfold", "autosimpl", "autoreflect",
        "autorewrite_with", "autounfold_with", "autosimpl_with",
        "autoreflect_with", "autorewrite_with_clear", "autounfold_with_clear",
        "autosimpl_with_clear", "autoreflect_with_clear", "autorewrite_with_clear_clear",
        "autounfold_with_clear_clear", "autosimpl_with_clear_clear", "autoreflect_with_clear_clear"
    ]
    
    if 0 <= tactic_id < len(tactic_names):
        return tactic_names[tactic_id]
    else:
        return f"unknown_tactic_{tactic_id}"

def get_argument_name(arg_id: int, arg_type: str) -> str:
    """引数IDから引数名を取得"""
    arg_names = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    
    if 0 <= arg_id < len(arg_names):
        return arg_names[arg_id]
    else:
        return f"unknown_{arg_type}_{arg_id}"

def real_proof_verification(model_path: str, test_count: int = 10, max_steps: int = 20, device: str = "auto"):
    """真の証明検証を実行"""
    print(f"🧪 Real proof verification (actual tactic application)")
    print(f"  Model: {model_path}")
    print(f"  Test count: {test_count}")
    print(f"  Max steps: {max_steps}")
    
    # デバイス設定
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    print(f"Using device: {device}")
    
    # モデルを読み込み
    model, _ = load_hierarchical_model(model_path, device)
    
    # トークナイザーを作成
    token_py_path = os.path.join(project_root, "src", "core", "fof_tokens.py")
    base_tokens, _ = load_tokens_and_labels_from_token_py(token_py_path)
    tokenizer = CharTokenizer(base_tokens)
    
    # バリデーションデータを読み込み
    validation_file = os.path.join(project_root, "validation", "validation_tautology.json")
    with open(validation_file, 'r', encoding='utf-8') as f:
        validation_data = json.load(f)
    
    # ランダムにサンプルを選択
    import random
    test_samples = random.sample(validation_data, min(test_count, len(validation_data)))
    
    print(f"\n📊 Testing {len(test_samples)} samples with real proof verification...")
    
    solved_count = 0
    step_counts = []
    tactic_usage = {}
    
    for i, formula in enumerate(test_samples):
        if i % 5 == 0:  # 5例ごとに進捗表示
            print(f"  Progress: {i}/{len(test_samples)} ({i/len(test_samples)*100:.1f}%)")
        
        print(f"\n--- Sample {i+1}/{len(test_samples)} ---")
        print(f"Formula: {formula}")
        
        try:
            # 証明状態を初期化
            proof_state = ProofState(formula)
            
            # 実際の証明を実行
            success, steps, tactics_used = execute_real_proof(
                model, tokenizer, proof_state, max_steps, device
            )
            
            step_counts.append(steps)
            
            # 戦術使用回数を記録
            for tactic in tactics_used:
                tactic_usage[tactic] = tactic_usage.get(tactic, 0) + 1
            
            if success:
                solved_count += 1
                print(f"✅ Success (solved in {steps} steps)")
                print(f"   Final goal: {proof_state.current_goal}")
                print(f"   Premises: {proof_state.premises}")
            else:
                print(f"❌ Failed (could not solve in {steps} steps)")
                print(f"   Final goal: {proof_state.current_goal}")
                print(f"   Premises: {proof_state.premises}")
                
        except Exception as e:
            print(f"❌ Error during proof: {e}")
            step_counts.append(max_steps)
    
    # 結果を計算
    total_examples = len(step_counts)
    success_rate = solved_count / total_examples if total_examples > 0 else 0.0
    avg_steps = sum(step_counts) / len(step_counts) if step_counts else 0
    
    print(f"\n📈 Results:")
    print(f"  Total examples: {total_examples}")
    print(f"  Solved: {solved_count}")
    print(f"  Success rate: {success_rate:.2%}")
    print(f"  Average steps: {avg_steps:.2f}")
    
    if tactic_usage:
        print(f"\n📊 Tactic usage:")
        for tactic, count in sorted(tactic_usage.items(), key=lambda x: x[1], reverse=True):
            print(f"  {tactic}: {count}")
    
    return {
        'success_rate': success_rate,
        'avg_steps': avg_steps,
        'solved_count': solved_count,
        'total_examples': total_examples,
        'tactic_usage': tactic_usage
    }

def execute_real_proof(model, tokenizer, proof_state, max_steps, device):
    """実際の証明を実行"""
    tactics_used = []
    
    for step in range(max_steps):
        try:
            # 現在の目標をエンコード
            input_ids, attention_mask, segment_ids = tokenizer.encode(
                goal=proof_state.current_goal,
                premises=proof_state.premises,
                max_seq_len=256
            )
            
            # バッチ次元を追加
            input_ids = input_ids.unsqueeze(0).to(device)
            attention_mask = attention_mask.unsqueeze(0).to(device)
            segment_ids = segment_ids.unsqueeze(0).to(device)
            
            # モデル推論
            with torch.no_grad():
                main_logits, arg1_logits, arg2_logits = model(
                    input_ids, attention_mask, segment_ids
                )
                
                # 予測を取得
                main_pred = torch.argmax(main_logits, dim=-1).item()
                arg1_pred = torch.argmax(arg1_logits, dim=-1).item()
                arg2_pred = torch.argmax(arg2_logits, dim=-1).item()
                
                # 実際の戦術名を取得
                tactic_name = get_tactic_name(main_pred)
                arg1_name = get_argument_name(arg1_pred, "arg1")
                arg2_name = get_argument_name(arg2_pred, "arg2")
                
                print(f"     Step {step + 1}: {tactic_name}({arg1_name}, {arg2_name})")
                
                # 戦術を実際に適用
                tactic_success = proof_state.apply_tactic(tactic_name, arg1_name, arg2_name)
                tactics_used.append(tactic_name)
                
                if not tactic_success:
                    print(f"       ❌ Tactic {tactic_name} failed to apply")
                    continue
                
                # 証明が完了したかチェック
                if proof_state.check_proof_complete():
                    print(f"       ✅ Proof completed!")
                    return True, step + 1, tactics_used
                
        except Exception as e:
            print(f"Error in step {step}: {e}")
            break
    
    return False, max_steps, tactics_used

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real proof verification with actual tactic application")
    parser.add_argument("--model_path", type=str, default="../models/pretrained_model.pth", help="Path to model")
    parser.add_argument("--count", type=int, default=10, help="Number of test samples")
    parser.add_argument("--max_steps", type=int, default=20, help="Maximum steps")
    parser.add_argument("--device", type=str, default="auto", help="Device to use")
    
    args = parser.parse_args()
    
    results = real_proof_verification(args.model_path, args.count, args.max_steps, args.device)
    
    print(f"\n🎯 Final Results:")
    print(f"  Success Rate: {results['success_rate']:.2%}")
    print(f"  Average Steps: {results['avg_steps']:.2f}")
    print(f"  Solved: {results['solved_count']}/{results['total_examples']}")

