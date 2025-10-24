#!/usr/bin/env python3
"""
Actor-Criticモデル用のinference_hierarchical.pyライクな検証システム
Actor-CriticモデルからTransformerClassifierを抽出して検証
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
from src.core.actor_critic_model import ActorCriticModel

def load_actor_critic_model(model_path: str, device: torch.device) -> Tuple[ActorCriticModel, Dict[str, Any]]:
    """Actor-Criticモデルを読み込み"""
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
    
    # ベースTransformerを作成
    base_transformer = TransformerClassifier(
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
    
    # Actor-Criticモデルを作成
    actor_critic_model = ActorCriticModel(
        base_transformer=base_transformer,
        pretrained_model=base_transformer,  # 同じモデルをpretrained_modelとして使用
        critic_hidden_dim=512
    ).to(device)
    
    # モデルの重みを読み込み
    if 'model_state_dict' in checkpoint:
        actor_critic_model.load_state_dict(checkpoint['model_state_dict'])
    else:
        actor_critic_model.load_state_dict(checkpoint)
    
    actor_critic_model.eval()
    return actor_critic_model, {}

def extract_transformer_from_actor_critic(actor_critic_model: ActorCriticModel) -> TransformerClassifier:
    """Actor-CriticモデルからTransformerClassifierを抽出"""
    # shared_encoderを取得
    transformer = actor_critic_model.shared_encoder
    
    # 新しいTransformerClassifierを作成
    # d_modelはembeddingの次元から取得
    d_model = transformer.embedding.embedding_dim
    
    extracted_transformer = TransformerClassifier(
        vocab_size=transformer.vocab_size,
        pad_id=transformer.pad_id,
        max_seq_len=transformer.max_seq_len,
        d_model=d_model,
        nhead=transformer.encoder.layers[0].self_attn.num_heads,
        num_layers=len(transformer.encoder.layers),
        dim_feedforward=transformer.encoder.layers[0].linear1.out_features,
        dropout=transformer.dropout.p,
        num_main_classes=transformer.num_main_classes,
        num_arg1_classes=transformer.num_arg1_classes,
        num_arg2_classes=transformer.num_arg2_classes,
    )
    
    # 重みをコピー
    extracted_transformer.load_state_dict(transformer.state_dict())
    extracted_transformer.eval()
    
    return extracted_transformer

def get_tactic_name(tactic_id: int) -> str:
    """戦術IDから戦術名を取得（実際のfof_tokens.pyから）"""
    # 実際の戦術名をfof_tokens.pyから取得
    actual_tactics = [
        "assumption", "intro", "split", "left", "right", "add_dn",
        "apply 0", "destruct 0", "apply 1", "destruct 1", "apply 2", "destruct 2",
        "specialize 0 1", "specialize 0 2", "specialize 1 0", "specialize 1 2", 
        "specialize 2 0", "specialize 2 1"
    ]
    
    if 0 <= tactic_id < len(actual_tactics):
        return actual_tactics[tactic_id]
    else:
        return f"unknown_tactic_{tactic_id}"

def get_argument_name(arg_id: int, arg_type: str) -> str:
    """引数IDから引数名を取得"""
    arg_names = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    
    if 0 <= arg_id < len(arg_names):
        return arg_names[arg_id]
    else:
        return f"unknown_{arg_type}_{arg_id}"

def apply_tactic_from_label(prover, label) -> bool:
    """タクティクを適用（inference_hierarchical.pyからコピー）"""
    if isinstance(label, dict):
        from src.core.state_encoder import format_tactic_string
        tactic_str = format_tactic_string(label)
    else:
        tactic_str = label
    
    try:
        # 戦術を適用
        prover.apply_tactic(tactic_str)
        return True
    except Exception:
        return False

def actor_critic_hierarchical_verification(model_path: str, test_count: int = 10, max_steps: int = 20, device: str = "auto"):
    """Actor-Criticモデル用の階層検証を実行"""
    print(f"🧪 Actor-Critic hierarchical verification (inference_hierarchical.py style)")
    print(f"  Model: {model_path}")
    print(f"  Test count: {test_count}")
    print(f"  Max steps: {max_steps}")
    
    # デバイス設定
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    print(f"Using device: {device}")
    
    # Actor-Criticモデルを読み込み
    actor_critic_model, _ = load_actor_critic_model(model_path, device)
    
    # TransformerClassifierを抽出
    print("📤 Extracting TransformerClassifier from Actor-Critic model...")
    transformer_model = extract_transformer_from_actor_critic(actor_critic_model)
    transformer_model = transformer_model.to(device)
    
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
    
    print(f"\n📊 Testing {len(test_samples)} samples with Actor-Critic hierarchical verification...")
    
    solved_count = 0
    step_counts = []
    tactic_usage = {}
    
    for i, formula in enumerate(test_samples):
        if i % 5 == 0:  # 5例ごとに進捗表示
            print(f"  Progress: {i}/{len(test_samples)} ({i/len(test_samples)*100:.1f}%)")
        
        print(f"\n--- Sample {i+1}/{len(test_samples)} ---")
        print(f"Formula: {formula}")
        
        try:
            # 階層推論を実行
            success, steps, tactics_used = execute_hierarchical_inference(
                transformer_model, tokenizer, formula, max_steps, device, {}
            )
            
            step_counts.append(steps)
            
            # 戦術使用回数を記録
            for tactic in tactics_used:
                tactic_usage[tactic] = tactic_usage.get(tactic, 0) + 1
            
            if success:
                solved_count += 1
                print(f"✅ Success (solved in {steps} steps)")
            else:
                print(f"❌ Failed (could not solve in {steps} steps)")
                
        except Exception as e:
            print(f"❌ Error during inference: {e}")
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

def execute_hierarchical_inference(model, tokenizer, formula, max_steps, device, label_mappings):
    """真の階層推論を実行（inference_hierarchical.pyと同じ方針）"""
    tactics_used = []
    
    print("🔍 Executing true hierarchical inference (inference_hierarchical.py style)...")
    
    try:
        # pyproverをインポート
        pyprover_dir = os.path.join(project_root, "pyprover")
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
        
        # パースしてproverを作成
        parse_tree = PropParseTree()
        goal_node = parse_tree.transform(prop_parser.parse(formula))
        prover = Prover(goal_node)
        
        # 前提は空（トートロジーなので前提なしで証明可能）
        premises = []
        
        # 推論ループ
        step = 0
        solved = prover.goal is None
        
        while not solved and step < max_steps:
            # 現在の状態を取得
            from src.core.state_encoder import encode_prover_state
            current_state = encode_prover_state(prover)
            current_premises = current_state["premises"]
            current_goal = current_state["goal"]
            
            # 入力をエンコード
            input_ids, attention_mask, segment_ids = tokenizer.encode(
                goal=current_goal,
                premises=current_premises,
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
                
                # 階層分類の戦術予測を処理
                if main_pred < 6:  # 基本戦術（assumption, intro, split, left, right, add_dn）
                    tactic_str = get_tactic_name(main_pred)
                elif main_pred < 12:  # apply戦術
                    arg_id = arg1_pred % 10  # 0-9の範囲に制限
                    tactic_str = f"apply {arg_id}"
                elif main_pred < 18:  # destruct戦術
                    arg_id = arg1_pred % 10  # 0-9の範囲に制限
                    tactic_str = f"destruct {arg_id}"
                else:  # specialize戦術
                    arg1_id = arg1_pred % 10
                    arg2_id = arg2_pred % 10
                    if arg1_id != arg2_id:  # 異なる引数である必要がある
                        tactic_str = f"specialize {arg1_id} {arg2_id}"
                    else:
                        tactic_str = "assumption"  # フォールバック
                
                print(f"     Step {step + 1}: {tactic_str}")
                
                # 戦術を適用
                success = apply_tactic_from_label(prover, tactic_str)
                
                tactics_used.append(tactic_str)
                
                if success:
                    print(f"       ✅ Tactic applied successfully")
                else:
                    print(f"       ❌ Tactic failed")
                
                step += 1
                solved = prover.goal is None
                
                if solved:
                    print(f"       🎉 Proof completed!")
                    return True, step, tactics_used
        
        return False, step, tactics_used
        
    except Exception as e:
        print(f"❌ Error during true inference: {e}")
        return False, max_steps, tactics_used

def save_extracted_transformer(actor_critic_model: ActorCriticModel, output_path: str):
    """抽出されたTransformerClassifierを保存"""
    transformer = extract_transformer_from_actor_critic(actor_critic_model)
    
    # チェックポイントを作成
    checkpoint = {
        'model_state_dict': transformer.state_dict(),
        'vocab_size': transformer.vocab_size,
        'pad_id': transformer.pad_id,
        'max_seq_len': transformer.max_seq_len,
        'model_params': {
            'd_model': transformer.d_model,
            'nhead': transformer.nhead,
            'num_layers': transformer.num_layers,
            'dim_feedforward': transformer.dim_feedforward,
            'dropout': transformer.dropout,
        }
    }
    
    # 保存
    torch.save(checkpoint, output_path)
    print(f"💾 Extracted TransformerClassifier saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Actor-Critic hierarchical verification")
    parser.add_argument("--model_path", type=str, required=True, help="Path to Actor-Critic model")
    parser.add_argument("--count", type=int, default=10, help="Number of test samples")
    parser.add_argument("--max_steps", type=int, default=20, help="Maximum steps")
    parser.add_argument("--device", type=str, default="auto", help="Device to use")
    parser.add_argument("--save_extracted", type=str, default=None, help="Save extracted transformer to this path")
    
    args = parser.parse_args()
    
    results = actor_critic_hierarchical_verification(
        args.model_path, args.count, args.max_steps, args.device
    )
    
    print(f"\n🎯 Final Results:")
    print(f"  Success Rate: {results['success_rate']:.2%}")
    print(f"  Average Steps: {results['avg_steps']:.2f}")
    print(f"  Solved: {results['solved_count']}/{results['total_examples']}")
    
    # 抽出されたTransformerを保存する場合
    if args.save_extracted:
        print(f"\n💾 Saving extracted transformer...")
        actor_critic_model, _ = load_actor_critic_model(args.model_path, torch.device("cpu"))
        save_extracted_transformer(actor_critic_model, args.save_extracted)
