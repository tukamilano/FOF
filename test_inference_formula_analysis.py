#!/usr/bin/env python3
"""
inferenceで生成される論理式の分析テストファイル
学習データと比較してdifficultyや複雑さをチェックする
"""

import os
import sys
import json
import random
import time
from typing import List, Dict, Any, Tuple
import argparse

# プロジェクトルートをパスに追加
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.core.generate_prop import FormulaGenerator, filter_formulas
from src.core.parameter import get_generation_params
from src.core.transformer_classifier import load_tokens_and_labels_from_token_py


def analyze_formula_complexity(formula: str) -> Dict[str, Any]:
    """論理式の複雑さを分析"""
    # 基本的な統計
    length = len(formula)
    
    # 演算子の数をカウント
    and_count = formula.count('∧')
    or_count = formula.count('∨')
    impl_count = formula.count('→')
    neg_count = formula.count('¬')
    
    # 括弧の深さを計算
    max_depth = 0
    current_depth = 0
    for char in formula:
        if char == '(':
            current_depth += 1
            max_depth = max(max_depth, current_depth)
        elif char == ')':
            current_depth -= 1
    
    # 変数の種類をカウント
    variables = set()
    for char in formula:
        if char.isalpha() and char.islower():
            variables.add(char)
    
    return {
        'length': length,
        'and_count': and_count,
        'or_count': or_count,
        'impl_count': impl_count,
        'neg_count': neg_count,
        'max_depth': max_depth,
        'variable_count': len(variables),
        'variables': sorted(list(variables)),
        'total_operators': and_count + or_count + impl_count + neg_count
    }


def load_training_data_samples(data_dir: str, num_samples: int = 1000) -> List[Dict[str, Any]]:
    """学習データからサンプルを読み込み"""
    import glob
    
    samples = []
    json_files = glob.glob(os.path.join(data_dir, "deduplicated_batch_*.json"))
    json_files.sort()
    
    print(f"Loading samples from {len(json_files)} batch files...")
    
    for json_file in json_files[:5]:  # 最初の5ファイルからサンプル
        with open(json_file, 'r') as f:
            batch_data = json.load(f)
        
        # 各バッチからランダムにサンプル
        batch_samples = random.sample(batch_data, min(200, len(batch_data)))
        samples.extend(batch_samples)
        
        if len(samples) >= num_samples:
            break
    
    return samples[:num_samples]


def generate_inference_formulas(num_formulas: int, difficulty: float = None) -> List[str]:
    """inference用の論理式を生成"""
    # パラメータを取得
    gen_params = get_generation_params()
    
    # 変数を取得
    token_py_path = os.path.join(project_root, "src", "core", "fof_tokens.py")
    base_tokens, _ = load_tokens_and_labels_from_token_py(token_py_path)
    variables = [t for t in ["a", "b", "c"] if t in base_tokens]
    if not variables:
        variables = ["a", "b", "c"]
    
    # 指定されたdifficultyを使用、またはデフォルト値
    if difficulty is None:
        difficulty = gen_params.difficulty
    
    # フォーミュラジェネレーターを作成
    gen = FormulaGenerator(
        variables=variables,
        allow_const=gen_params.allow_const,
        difficulty=difficulty,
        max_depth=gen_params.max_depth,
        seed=int(time.time() * 1000) % 2**32
    )
    
    # トートロジーを生成
    formulas = filter_formulas(
        gen=gen,
        max_len=100,
        require_tautology=True,
        limit=num_formulas
    )
    
    return formulas


def compare_formula_complexity(training_samples: List[Dict], inference_formulas: List[str]) -> None:
    """学習データとinferenceデータの複雑さを比較"""
    print("\n" + "="*80)
    print("FORMULA COMPLEXITY COMPARISON")
    print("="*80)
    
    # 学習データの分析
    training_goals = [sample['goal'] for sample in training_samples]
    training_complexities = [analyze_formula_complexity(goal) for goal in training_goals]
    
    # inferenceデータの分析
    inference_complexities = [analyze_formula_complexity(formula) for formula in inference_formulas]
    
    # 統計を計算
    def calculate_stats(complexities: List[Dict], name: str):
        print(f"\n{name} Statistics:")
        print("-" * 40)
        
        lengths = [c['length'] for c in complexities]
        depths = [c['max_depth'] for c in complexities]
        operators = [c['total_operators'] for c in complexities]
        negations = [c['neg_count'] for c in complexities]
        variables = [c['variable_count'] for c in complexities]
        
        print(f"  Length:     avg={sum(lengths)/len(lengths):.1f}, min={min(lengths)}, max={max(lengths)}")
        print(f"  Max Depth:  avg={sum(depths)/len(depths):.1f}, min={min(depths)}, max={max(depths)}")
        print(f"  Operators:  avg={sum(operators)/len(operators):.1f}, min={min(operators)}, max={max(operators)}")
        print(f"  Negations:  avg={sum(negations)/len(negations):.1f}, min={min(negations)}, max={max(negations)}")
        print(f"  Variables:  avg={sum(variables)/len(variables):.1f}, min={min(variables)}, max={max(variables)}")
        
        return {
            'length': lengths,
            'depth': depths,
            'operators': operators,
            'negations': negations,
            'variables': variables
        }
    
    training_stats = calculate_stats(training_complexities, "TRAINING DATA")
    inference_stats = calculate_stats(inference_complexities, "INFERENCE DATA")
    
    # 比較
    print(f"\nCOMPARISON:")
    print("-" * 40)
    print(f"Length ratio (inf/train): {sum(inference_stats['length'])/len(inference_stats['length']) / (sum(training_stats['length'])/len(training_stats['length'])):.2f}")
    print(f"Depth ratio (inf/train):  {sum(inference_stats['depth'])/len(inference_stats['depth']) / (sum(training_stats['depth'])/len(training_stats['depth'])):.2f}")
    print(f"Operators ratio (inf/train): {sum(inference_stats['operators'])/len(inference_stats['operators']) / (sum(training_stats['operators'])/len(training_stats['operators'])):.2f}")
    
    # 否定記号の比較（ゼロ除算を避ける）
    train_neg_avg = sum(training_stats['negations'])/len(training_stats['negations'])
    inf_neg_avg = sum(inference_stats['negations'])/len(inference_stats['negations'])
    if train_neg_avg > 0:
        print(f"Negations ratio (inf/train): {inf_neg_avg / train_neg_avg:.2f}")
    else:
        print(f"Negations ratio (inf/train): N/A (training data has no negations)")
        print(f"  Training negations: {train_neg_avg:.1f}, Inference negations: {inf_neg_avg:.1f}")


def analyze_difficulty_impact() -> None:
    """difficultyパラメータの影響を分析"""
    print("\n" + "="*80)
    print("DIFFICULTY IMPACT ANALYSIS")
    print("="*80)
    
    difficulties = [0.1, 0.3, 0.5, 0.7, 0.9]
    num_formulas = 50
    
    for difficulty in difficulties:
        print(f"\nDifficulty: {difficulty}")
        print("-" * 30)
        
        formulas = generate_inference_formulas(num_formulas, difficulty)
        if not formulas:
            print("  Failed to generate formulas")
            continue
        
        complexities = [analyze_formula_complexity(f) for f in formulas]
        
        avg_length = sum(c['length'] for c in complexities) / len(complexities)
        avg_depth = sum(c['max_depth'] for c in complexities) / len(complexities)
        avg_operators = sum(c['total_operators'] for c in complexities) / len(complexities)
        avg_negations = sum(c['neg_count'] for c in complexities) / len(complexities)
        
        print(f"  Generated: {len(formulas)} formulas")
        print(f"  Avg length: {avg_length:.1f}")
        print(f"  Avg depth: {avg_depth:.1f}")
        print(f"  Avg operators: {avg_operators:.1f}")
        print(f"  Avg negations: {avg_negations:.1f}")
        print(f"  Sample: {formulas[0] if formulas else 'None'}")


def show_sample_formulas(training_samples: List[Dict], inference_formulas: List[str], num_samples: int = 10) -> None:
    """サンプル論理式を表示"""
    print("\n" + "="*80)
    print("SAMPLE FORMULAS")
    print("="*80)
    
    print(f"\nTRAINING DATA SAMPLES (first {num_samples}):")
    print("-" * 50)
    for i, sample in enumerate(training_samples[:num_samples]):
        goal = sample['goal']
        complexity = analyze_formula_complexity(goal)
        print(f"{i+1:2d}. {goal}")
        print(f"    Length: {complexity['length']}, Depth: {complexity['max_depth']}, "
              f"Operators: {complexity['total_operators']}, Negations: {complexity['neg_count']}")
    
    print(f"\nINFERENCE DATA SAMPLES (first {num_samples}):")
    print("-" * 50)
    for i, formula in enumerate(inference_formulas[:num_samples]):
        complexity = analyze_formula_complexity(formula)
        print(f"{i+1:2d}. {formula}")
        print(f"    Length: {complexity['length']}, Depth: {complexity['max_depth']}, "
              f"Operators: {complexity['total_operators']}, Negations: {complexity['neg_count']}")


def main():
    parser = argparse.ArgumentParser(description="Analyze inference formula complexity")
    parser.add_argument("--data_dir", type=str, default="deduplicated_data", help="training data directory")
    parser.add_argument("--num_training_samples", type=int, default=1000, help="number of training samples to analyze")
    parser.add_argument("--num_inference_formulas", type=int, default=100, help="number of inference formulas to generate")
    parser.add_argument("--difficulty", type=float, default=None, help="difficulty for inference generation (default: use current params)")
    parser.add_argument("--analyze_difficulty", action="store_true", help="analyze difficulty impact")
    parser.add_argument("--show_samples", type=int, default=10, help="number of sample formulas to show")
    
    args = parser.parse_args()
    
    print("🔍 INFERENCE FORMULA COMPLEXITY ANALYSIS")
    print("="*80)
    
    # 学習データを読み込み
    data_dir = os.path.join(project_root, args.data_dir)
    if not os.path.exists(data_dir):
        print(f"❌ Data directory not found: {data_dir}")
        return
    
    print(f"Loading training data from: {data_dir}")
    training_samples = load_training_data_samples(data_dir, args.num_training_samples)
    print(f"Loaded {len(training_samples)} training samples")
    
    # inference用の論理式を生成
    print(f"Generating {args.num_inference_formulas} inference formulas...")
    inference_formulas = generate_inference_formulas(args.num_inference_formulas, args.difficulty)
    print(f"Generated {len(inference_formulas)} inference formulas")
    
    if not inference_formulas:
        print("❌ Failed to generate inference formulas")
        return
    
    # 複雑さを比較
    compare_formula_complexity(training_samples, inference_formulas)
    
    # サンプル論理式を表示
    show_sample_formulas(training_samples, inference_formulas, args.show_samples)
    
    # difficultyの影響を分析
    if args.analyze_difficulty:
        analyze_difficulty_impact()
    
    print("\n✅ Analysis completed!")


if __name__ == "__main__":
    main()
