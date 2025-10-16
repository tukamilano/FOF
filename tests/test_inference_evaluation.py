#!/usr/bin/env python3
"""
推論評価のみをテストするスクリプト
訓練をスキップして、異なるdifficulty値での推論性能を比較
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, Any

# プロジェクトルートをパスに追加
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_root)

import torch
import torch.nn as nn

from src.core.transformer_classifier import (
    load_tokens_and_labels_from_token_py,
    CharTokenizer,
    TransformerClassifier,
    build_hierarchical_label_mappings,
)
from src.core.generate_prop import FormulaGenerator, filter_formulas
from src.core.parameter import (
    get_model_params, get_training_params, 
    get_system_params, get_hierarchical_labels, DeviceType
)
from src.training.inference_hierarchical import evaluate_inference_performance


def test_difficulty_impact():
    """異なるdifficulty値での推論性能をテスト"""
    
    # パラメータを初期化
    model_params = get_model_params()
    training_params = get_training_params()
    system_params = get_system_params()
    hierarchical_labels = get_hierarchical_labels()
    
    # デバイス設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # トークンとラベルを読み込み
    token_py_path = os.path.join(project_root, "src", "core", "fof_tokens.py")
    print(f"Loading tokens from: {token_py_path}")
    base_tokens, _ = load_tokens_and_labels_from_token_py(token_py_path)
    
    # 階層分類用のラベルマッピングを構築
    main_to_id, arg1_to_id, arg2_to_id, id_to_main, id_to_arg1, id_to_arg2 = build_hierarchical_label_mappings(
        hierarchical_labels.main_tactics,
        hierarchical_labels.arg1_values,
        hierarchical_labels.arg2_values
    )
    
    print(f"Main tactics: {len(id_to_main)} classes")
    print(f"Arg1 values: {len(id_to_arg1)} classes")
    print(f"Arg2 values: {len(id_to_arg2)} classes")
    
    # トークナイザーを作成
    tokenizer = CharTokenizer(
        base_tokens=base_tokens,
        add_tactic_tokens=model_params.add_tactic_tokens,
        num_tactic_tokens=model_params.num_tactic_tokens
    )
    
    # モデルを作成（ランダム初期化）
    model = TransformerClassifier(
        vocab_size=tokenizer.vocab_size,
        pad_id=tokenizer.pad_id,
        max_seq_len=512,
        d_model=model_params.d_model,
        nhead=model_params.nhead,
        num_layers=model_params.num_layers,
        dim_feedforward=model_params.dim_feedforward,
        dropout=model_params.dropout,
        num_main_classes=len(id_to_main),
        num_arg1_classes=len(id_to_arg1),
        num_arg2_classes=len(id_to_arg2),
    )
    
    model = model.to(device)
    model.eval()  # 評価モード
    
    # ラベルマッピングを作成
    label_mappings = {
        'main_to_id': main_to_id,
        'arg1_to_id': arg1_to_id,
        'arg2_to_id': arg2_to_id,
        'id_to_main': id_to_main,
        'id_to_arg1': id_to_arg1,
        'id_to_arg2': id_to_arg2
    }
    
    # 異なるdifficulty値でテスト
    difficulty_values = [0.3, 0.5, 0.7, 0.9]
    num_examples = 50
    max_steps = 20
    
    print(f"\n🧪 Testing inference performance with different difficulty values")
    print(f"📊 Number of examples per test: {num_examples}")
    print(f"📊 Max steps per example: {max_steps}")
    print("=" * 80)
    
    results = {}
    
    for difficulty in difficulty_values:
        print(f"\n🔍 Testing difficulty = {difficulty}")
        print("-" * 40)
        
        # 指定されたdifficultyでトートロジーを生成
        gen = FormulaGenerator(
            variables=["a", "b", "c"],
            allow_const=False,
            difficulty=difficulty,
            max_depth=4,
            seed=42  # 再現性のため固定シード
        )
        
        # トートロジーを生成して表示
        tautologies = filter_formulas(
            gen=gen,
            max_len=100,
            require_tautology=True,
            limit=num_examples
        )
        
        if not tautologies:
            print(f"❌ Failed to generate tautologies for difficulty {difficulty}")
            continue
        
        print(f"Generated {len(tautologies)} tautologies for difficulty {difficulty}")
        print("First 5 tautologies:")
        for i in range(min(5, len(tautologies))):
            print(f"  {i+1}: {tautologies[i]}")
        
        # 推論性能を評価
        start_time = time.time()
        success_rate, avg_steps = evaluate_inference_performance(
            model, tokenizer, label_mappings, device, 512,
            num_examples=num_examples, 
            max_steps=max_steps, 
            temperature=1.0,
            difficulty=difficulty,  # 同じdifficulty値を使用
            seed=42,  # 同じシードを使用
            max_depth=4  # 同じmax_depth値を使用
        )
        eval_time = time.time() - start_time
        
        print(f"  Success rate: {success_rate:.3f}")
        print(f"  Avg steps (when solved): {avg_steps:.2f}")
        print(f"  Evaluation time: {eval_time:.2f}s")
        
        results[difficulty] = {
            'success_rate': success_rate,
            'avg_steps': avg_steps,
            'eval_time': eval_time,
            'tautologies': tautologies[:5]  # 最初の5個を保存
        }
    
    # 結果をまとめて表示
    print(f"\n📊 Summary of Results")
    print("=" * 80)
    print(f"{'Difficulty':<12} {'Success Rate':<15} {'Avg Steps':<12} {'Eval Time':<12}")
    print("-" * 80)
    
    for difficulty in difficulty_values:
        if difficulty in results:
            r = results[difficulty]
            print(f"{difficulty:<12.1f} {r['success_rate']:<15.3f} {r['avg_steps']:<12.2f} {r['eval_time']:<12.2f}")
    
    # トートロジーの複雑さを分析
    print(f"\n🔍 Tautology Complexity Analysis")
    print("=" * 80)
    
    for difficulty in difficulty_values:
        if difficulty in results:
            print(f"\nDifficulty {difficulty} - First 5 tautologies:")
            for i, tautology in enumerate(results[difficulty]['tautologies']):
                print(f"  {i+1}: {tautology} (length: {len(tautology)})")
    
    return results


def test_tautology_generation_detailed():
    """トートロジー生成の詳細テスト"""
    
    print(f"\n🔍 Detailed Tautology Generation Test")
    print("=" * 80)
    
    difficulty_values = [0.3, 0.5, 0.7, 0.9]
    num_examples = 20
    
    for difficulty in difficulty_values:
        print(f"\n📊 Difficulty = {difficulty}")
        print("-" * 40)
        
        gen = FormulaGenerator(
            variables=["a", "b", "c"],
            allow_const=False,
            difficulty=difficulty,
            max_depth=4,
            seed=42
        )
        
        tautologies = filter_formulas(
            gen=gen,
            max_len=100,
            require_tautology=True,
            limit=num_examples
        )
        
        if tautologies:
            # 統計を計算
            lengths = [len(t) for t in tautologies]
            avg_length = sum(lengths) / len(lengths)
            max_length = max(lengths)
            min_length = min(lengths)
            
            print(f"Generated {len(tautologies)} tautologies")
            print(f"Average length: {avg_length:.1f}")
            print(f"Min length: {min_length}")
            print(f"Max length: {max_length}")
            print("All generated tautologies:")
            for i, tautology in enumerate(tautologies):
                print(f"  {i+1:2d}: {tautology}")
        else:
            print("❌ Failed to generate tautologies")


def main():
    parser = argparse.ArgumentParser(description="Test inference evaluation with different difficulty values")
    parser.add_argument("--num_examples", type=int, default=50, help="number of examples for evaluation")
    parser.add_argument("--max_steps", type=int, default=30, help="max steps for inference")
    parser.add_argument("--test_generation", action="store_true", help="test tautology generation in detail")
    args = parser.parse_args()
    
    print("🚀 Inference Evaluation Test")
    print(f"   Script: {sys.argv[0]}")
    print(f"   Arguments: {' '.join(sys.argv[1:])}")
    print("=" * 80)
    
    # 再現性のためのシード設定
    import random
    import numpy as np
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
    
    # 推論性能テスト
    results = test_difficulty_impact()
    
    # 詳細な生成テスト（オプション）
    if args.test_generation:
        test_tautology_generation_detailed()
    
    print(f"\n🎉 Test completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
