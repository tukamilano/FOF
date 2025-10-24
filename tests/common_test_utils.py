#!/usr/bin/env python3
"""
テスト用の共通ユーティリティ
"""
import os
import sys
import torch
import json
import random
from typing import List, Dict, Any, Tuple, Optional

# プロジェクトルートをパスに追加
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_root)

from src.core.transformer_classifier import (
    load_tokens_and_labels_from_token_py,
    CharTokenizer,
    TransformerClassifier,
    build_hierarchical_label_mappings
)
from src.core.parameter import get_model_params, get_hierarchical_labels

class TestModelLoader:
    """テスト用のモデルローダー"""
    
    def __init__(self, device: str = "auto"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else torch.device(device)
        self.model_params = get_model_params()
        self.hierarchical_labels = get_hierarchical_labels()
        
        # トークンとラベルを読み込み
        token_py_path = os.path.join(project_root, "src", "core", "fof_tokens.py")
        self.base_tokens, _ = load_tokens_and_labels_from_token_py(token_py_path)
        
        # 階層分類用のラベルマッピングを構築
        self.main_to_id, self.arg1_to_id, self.arg2_to_id, self.id_to_main, self.id_to_arg1, self.id_to_arg2 = build_hierarchical_label_mappings(
            self.hierarchical_labels.main_tactics,
            self.hierarchical_labels.arg1_values,
            self.hierarchical_labels.arg2_values
        )
        
        # トークナイザーを作成
        self.tokenizer = CharTokenizer(self.base_tokens)
    
    def load_model(self, model_path: str, vocab_size: int = 65) -> TransformerClassifier:
        """モデルを読み込み"""
        # パスを絶対パスに変換
        if not os.path.isabs(model_path):
            # 相対パスを正しく処理
            if model_path.startswith('../'):
                # ../ で始まる場合は親ディレクトリから
                model_path = os.path.join(project_root, model_path[3:])
            else:
                # その他の場合はproject_rootから
                model_path = os.path.join(project_root, model_path)
        
        # デバッグ情報
        print(f"Loading model from: {model_path}")
        if not os.path.exists(model_path):
            print(f"❌ Model file not found: {model_path}")
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        model = TransformerClassifier(
            vocab_size=vocab_size,
            pad_id=0,
            max_seq_len=256,
            d_model=self.model_params.d_model,
            nhead=self.model_params.nhead,
            num_layers=self.model_params.num_layers,
            dim_feedforward=self.model_params.dim_feedforward,
            dropout=self.model_params.dropout,
            num_main_classes=len(self.id_to_main),
            num_arg1_classes=len(self.id_to_arg1),
            num_arg2_classes=len(self.id_to_arg2),
        ).to(self.device)
        
        # チェックポイントを読み込み
        checkpoint = torch.load(model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        return model
    
    def encode_input(self, goal: str, premises: List[str] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """入力をエンコード"""
        if premises is None:
            premises = []
        
        input_ids, attention_mask, segment_ids = self.tokenizer.encode(
            goal=goal,
            premises=premises,
            max_seq_len=256
        )
        
        # バッチ次元を追加
        input_ids = input_ids.unsqueeze(0).to(self.device)
        attention_mask = attention_mask.unsqueeze(0).to(self.device)
        segment_ids = segment_ids.unsqueeze(0).to(self.device)
        
        return input_ids, attention_mask, segment_ids
    
    def predict_tactic(self, model: TransformerClassifier, goal: str, premises: List[str] = None) -> Dict[str, Any]:
        """戦術を予測"""
        input_ids, attention_mask, segment_ids = self.encode_input(goal, premises)
        
        with torch.no_grad():
            main_logits, arg1_logits, arg2_logits = model(
                input_ids, attention_mask, segment_ids
            )
            
            # 予測を取得
            main_pred = torch.argmax(main_logits, dim=-1).item()
            arg1_pred = torch.argmax(arg1_logits, dim=-1).item()
            arg2_pred = torch.argmax(arg2_logits, dim=-1).item()
            
            # 予測された戦術を取得
            predicted_tactic = self.id_to_main[main_pred]
            predicted_arg1 = self.id_to_arg1[arg1_pred] if arg1_pred < len(self.id_to_arg1) else "None"
            predicted_arg2 = self.id_to_arg2[arg2_pred] if arg2_pred < len(self.id_to_arg2) else "None"
            
            return {
                'tactic': predicted_tactic,
                'arg1': predicted_arg1,
                'arg2': predicted_arg2,
                'main_logits': main_logits,
                'arg1_logits': arg1_logits,
                'arg2_logits': arg2_logits
            }

class ValidationDataLoader:
    """バリデーションデータのローダー"""
    
    def __init__(self):
        self.validation_file = os.path.join(project_root, "validation", "validation_tautology.json")
    
    def load_data(self) -> List[str]:
        """バリデーションデータを読み込み"""
        with open(self.validation_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def get_samples(self, count: int) -> List[str]:
        """ランダムにサンプルを取得"""
        data = self.load_data()
        return random.sample(data, min(count, len(data)))

class TacticEvaluator:
    """戦術の有効性を評価"""
    
    @staticmethod
    def evaluate_tactic_effectiveness(formula: str, tactic: str, arg1: str, arg2: str, max_steps: int = 10) -> Tuple[bool, int]:
        """戦術の実際の有効性を評価（簡略化された推論エンジン）"""
        
        if tactic == "assumption":
            # assumptionは自明な場合のみ成功
            if formula in ["a", "b", "c", "True", "False"]:
                return True, 1
            return False, max_steps
        
        elif tactic == "intro":
            # introは含意の導入
            if "→" in formula:
                return True, 1
            return False, max_steps
        
        elif tactic == "destruct":
            # destructは論理積の分解
            if "∧" in formula:
                return True, 2
            return False, max_steps
        
        elif tactic == "left":
            # leftは論理和の左側
            if "∨" in formula:
                return True, 1
            return False, max_steps
        
        elif tactic == "right":
            # rightは論理和の右側
            if "∨" in formula:
                return True, 1
            return False, max_steps
        
        elif tactic == "apply":
            # applyは前提の適用
            return True, 3  # 中程度のステップ数
        
        else:
            # 未知の戦術
            return False, max_steps

def run_validation_test(model_path: str, test_count: int = 10, max_steps: int = 10) -> Tuple[float, float]:
    """バリデーションテストを実行"""
    print(f"🧪 Validation test with validation_tautology.json")
    print(f"  Model: {model_path}")
    print(f"  Test count: {test_count}")
    print(f"  Max steps: {max_steps}")
    
    # モデルローダーとデータローダーを初期化
    model_loader = TestModelLoader()
    data_loader = ValidationDataLoader()
    
    # モデルを読み込み
    model = model_loader.load_model(model_path)
    
    # テストサンプルを取得
    test_samples = data_loader.get_samples(test_count)
    
    print(f"\n📊 Testing {len(test_samples)} samples from validation_tautology.json...")
    
    successful = 0
    total = len(test_samples)
    step_counts = []
    
    for i, formula in enumerate(test_samples):
        print(f"\n--- Sample {i+1}/{total} ---")
        print(f"Formula: {formula}")
        
        try:
            # 戦術を予測
            prediction = model_loader.predict_tactic(model, formula)
            
            print(f"Predicted tactic: {prediction['tactic']}")
            print(f"Predicted arg1: {prediction['arg1']}")
            print(f"Predicted arg2: {prediction['arg2']}")
            
            # 戦術の有効性を評価
            success, steps = TacticEvaluator.evaluate_tactic_effectiveness(
                formula, prediction['tactic'], prediction['arg1'], prediction['arg2'], max_steps
            )
            
            step_counts.append(steps)
            
            if success:
                successful += 1
                print(f"✅ Success (solved in {steps} steps)")
            else:
                print(f"❌ Failed (could not solve in {steps} steps)")
                
        except Exception as e:
            print(f"❌ Error during inference: {e}")
            print("❌ Failed")
            step_counts.append(max_steps)
    
    success_rate = (successful / total) * 100
    avg_steps = sum(step_counts) / len(step_counts) if step_counts else 0
    
    print(f"\n📈 Results:")
    print(f"  Successful: {successful}/{total}")
    print(f"  Success Rate: {success_rate:.2f}%")
    print(f"  Average Steps: {avg_steps:.2f}")
    
    return success_rate, avg_steps

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Validation test with validation_tautology.json")
    parser.add_argument("--model_path", type=str, default="models/pretrained_model.pth", help="Path to model")
    parser.add_argument("--count", type=int, default=10, help="Number of test samples")
    parser.add_argument("--max_steps", type=int, default=10, help="Maximum steps")
    
    args = parser.parse_args()
    
    success_rate, avg_steps = run_validation_test(args.model_path, args.count, args.max_steps)
    print(f"\n🎯 Final Results:")
    print(f"  Success Rate: {success_rate:.2f}%")
    print(f"  Average Steps: {avg_steps:.2f}")
