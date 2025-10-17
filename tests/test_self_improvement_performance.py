"""
Self Improvement Data Collector Performance Test
処理速度を測定するためのテストコード
"""
import os
import sys
import time
import json
import torch
import numpy as np
from typing import List, Dict, Any, Tuple
import argparse

# psutilのインポート（オプション）
try:
    import psutil
    PSUTIL_AVAILABLE = True
    print(f"psutil imported successfully, version: {psutil.__version__}")
except ImportError as e:
    PSUTIL_AVAILABLE = False
    print(f"Warning: psutil not available. Memory monitoring will be disabled. Error: {e}")
    import traceback
    traceback.print_exc()

# プロジェクトルートをパスに追加
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_root)

from src.interaction.self_improvement_data_collector import (
    collect_self_improvement_data,
    generate_tactic_combinations,
    apply_tactic_from_label,
    load_hierarchical_model,
    initialize_global_constants,
    load_tautologies_from_generated_data,
    PYPROVER_MODULES
)
from src.core.transformer_classifier import (
    load_tokens_and_labels_from_token_py,
    CharTokenizer,
)
from src.core.state_encoder import encode_prover_state


class PerformanceProfiler:
    """パフォーマンス測定用のプロファイラー"""
    
    def __init__(self):
        self.metrics = {}
        self.start_times = {}
        if PSUTIL_AVAILABLE:
            self.process = psutil.Process()
        else:
            self.process = None
    
    def start_timer(self, name: str):
        """タイマーを開始"""
        self.start_times[name] = time.time()
    
    def end_timer(self, name: str):
        """タイマーを終了して時間を記録"""
        if name in self.start_times:
            elapsed = time.time() - self.start_times[name]
            self.metrics[name] = elapsed
            del self.start_times[name]
            return elapsed
        return 0.0
    
    def get_memory_usage(self) -> float:
        """現在のメモリ使用量をMBで取得"""
        if PSUTIL_AVAILABLE and self.process:
            try:
                return self.process.memory_info().rss / 1024 / 1024
            except Exception as e:
                print(f"Warning: Failed to get memory usage: {e}")
                return 0.0
        else:
            print(f"Warning: psutil not available (PSUTIL_AVAILABLE={PSUTIL_AVAILABLE}, process={self.process})")
            return 0.0
    
    def get_cpu_usage(self) -> float:
        """現在のCPU使用率を取得"""
        if PSUTIL_AVAILABLE and self.process:
            return self.process.cpu_percent()
        else:
            return 0.0


def measure_tactic_generation_performance(
    model,
    tokenizer,
    label_mappings,
    device,
    max_seq_len: int = 256,
    num_tests: int = 100
) -> Dict[str, Any]:
    """タクティク生成のパフォーマンスを測定"""
    print(f"Measuring tactic generation performance ({num_tests} tests)...")
    
    profiler = PerformanceProfiler()
    
    # テスト用の論理式を生成
    test_goals = [
        "P -> P",
        "P -> (Q -> P)",
        "(P -> Q) -> ((Q -> R) -> (P -> R))",
        "P -> (Q -> (P & Q))",
        "P & Q -> P",
        "P & Q -> Q",
        "P -> (P | Q)",
        "Q -> (P | Q)",
        "((P | Q) -> R) -> ((P -> R) & (Q -> R))",
        "P -> ~~P"
    ]
    
    # テスト用の前提を生成
    test_premises = [
        [],
        ["P"],
        ["P", "Q"],
        ["P -> Q", "Q -> R"],
        ["P & Q"],
        ["P | Q"],
        ["P -> Q", "P"],
        ["P -> (Q -> R)", "P", "Q"],
        ["P & Q", "P -> R", "Q -> R"],
        ["P | Q", "P -> R", "Q -> R"]
    ]
    
    total_tactics_generated = 0
    total_generation_time = 0.0
    generation_times = []
    
    for i in range(num_tests):
        # ランダムにテストケースを選択
        goal_idx = i % len(test_goals)
        premises_idx = i % len(test_premises)
        
        goal = test_goals[goal_idx]
        premises = test_premises[premises_idx]
        
        profiler.start_timer("tactic_generation")
        
        try:
            tactic_combinations = generate_tactic_combinations(
                model=model,
                tokenizer=tokenizer,
                premises=premises,
                goal=goal,
                label_mappings=label_mappings,
                device=device,
                max_seq_len=max_seq_len,
                probability_threshold=0.001
            )
            
            generation_time = profiler.end_timer("tactic_generation")
            
            tactics_count = len(tactic_combinations)
            total_tactics_generated += tactics_count
            total_generation_time += generation_time
            generation_times.append(generation_time)
            
            if i % 10 == 0:
                print(f"  Test {i+1}: Generated {tactics_count} tactics in {generation_time:.4f}s")
                
        except Exception as e:
            print(f"  Test {i+1} failed: {e}")
            profiler.end_timer("tactic_generation")
            continue
    
    # 統計を計算
    avg_generation_time = np.mean(generation_times) if generation_times else 0.0
    std_generation_time = np.std(generation_times) if generation_times else 0.0
    tactics_per_second = total_tactics_generated / total_generation_time if total_generation_time > 0 else 0.0
    
    results = {
        "total_tests": num_tests,
        "total_tactics_generated": total_tactics_generated,
        "total_generation_time": total_generation_time,
        "avg_generation_time": avg_generation_time,
        "std_generation_time": std_generation_time,
        "tactics_per_second": tactics_per_second,
        "generation_times": generation_times
    }
    
    print(f"Tactic Generation Results:")
    print(f"  Total tactics generated: {total_tactics_generated}")
    print(f"  Total time: {total_generation_time:.4f}s")
    print(f"  Average time per generation: {avg_generation_time:.4f}s ± {std_generation_time:.4f}s")
    print(f"  Tactics per second: {tactics_per_second:.2f}")
    
    return results


def measure_tactic_application_performance(
    num_tests: int = 100
) -> Dict[str, Any]:
    """タクティク適用のパフォーマンスを測定（簡略版）"""
    print(f"Measuring tactic application performance ({num_tests} tests)...")
    
    # グローバル定数を初期化
    initialize_global_constants()
    
    # PYPROVER_MODULESを再取得
    from src.interaction.self_improvement_data_collector import PYPROVER_MODULES
    
    # PYPROVER_MODULESの状態を確認
    if PYPROVER_MODULES is None:
        print("  Warning: PYPROVER_MODULES is None after initialization")
        return {
            "total_applications": 0,
            "success_count": 0,
            "success_rate": 0.0,
            "total_time": 0.0,
            "average_time_per_application": 0.0,
            "applications_per_second": 0.0,
            "error": "PYPROVER_MODULES not initialized"
        }
    
    profiler = PerformanceProfiler()
    
    # 簡単なテストケースのみ
    test_cases = [
        ("P -> P", "assumption"),
        ("P -> P", "intro"),
        ("P ∧ Q", "left"),
        ("P ∧ Q", "right"),
        ("P ∨ Q", "split"),
    ]
    
    total_applications = 0
    total_application_time = 0.0
    application_times = []
    success_count = 0
    
    for i in range(num_tests):
        # テストケースを循環的に使用
        case_idx = i % len(test_cases)
        goal, tactic = test_cases[case_idx]
        
        try:
            # プロバーを作成
            parse_tree = PYPROVER_MODULES['PropParseTree']()
            goal_node = parse_tree.transform(PYPROVER_MODULES['prop_parser'].parse(goal))
            prover = PYPROVER_MODULES['Prover'](goal_node)
            
            profiler.start_timer("tactic_application")
            
            success = apply_tactic_from_label(prover, tactic)
            
            application_time = profiler.end_timer("tactic_application")
            
            total_applications += 1
            total_application_time += application_time
            application_times.append(application_time)
            
            if success:
                success_count += 1
            
            if i % 20 == 0:
                print(f"  Test {i+1}: Applied '{tactic}' to '{goal}' in {application_time:.6f}s - {'Success' if success else 'Failed'}")
                
        except Exception as e:
            if i < 10:  # 最初の10回だけエラーを表示
                print(f"  Test {i+1} failed: {e}")
            profiler.end_timer("tactic_application")
            continue
    
    # 統計を計算
    avg_application_time = np.mean(application_times) if application_times else 0.0
    std_application_time = np.std(application_times) if application_times else 0.0
    applications_per_second = total_applications / total_application_time if total_application_time > 0 else 0.0
    success_rate = success_count / total_applications if total_applications > 0 else 0.0
    
    results = {
        "total_tests": num_tests,
        "total_applications": total_applications,
        "success_count": success_count,
        "success_rate": success_rate,
        "total_application_time": total_application_time,
        "avg_application_time": avg_application_time,
        "std_application_time": std_application_time,
        "applications_per_second": applications_per_second,
        "application_times": application_times
    }
    
    print(f"Tactic Application Results:")
    print(f"  Total applications: {total_applications}")
    print(f"  Success count: {success_count}")
    print(f"  Success rate: {success_rate:.2%}")
    print(f"  Total time: {total_application_time:.4f}s")
    print(f"  Average time per application: {avg_application_time:.6f}s ± {std_application_time:.6f}s")
    print(f"  Applications per second: {applications_per_second:.2f}")
    
    return results


def measure_overall_performance(
    model,
    tokenizer,
    label_mappings,
    device,
    max_seq_len: int = 256,
    num_examples: int = 10,
    max_steps: int = 10,
    temperature: float = 1.0
) -> Dict[str, Any]:
    """全体のパフォーマンスを測定"""
    print(f"Measuring overall performance ({num_examples} examples, max {max_steps} steps each)...")
    
    profiler = PerformanceProfiler()
    
    # メモリ使用量を記録
    initial_memory = profiler.get_memory_usage()
    print(f"  Initial memory: {initial_memory:.2f} MB")
    
    profiler.start_timer("overall_collection")
    
    try:
        successful_tactics = collect_self_improvement_data(
            model=model,
            tokenizer=tokenizer,
            label_mappings=label_mappings,
            device=device,
            max_seq_len=max_seq_len,
            num_examples=num_examples,
            max_steps=max_steps,
            probability_threshold=0.001,
            difficulty=0.5,
            seed=42,
            verbose=False,
            generated_data_dir="generated_data",
            temperature=temperature
        )
        
        total_time = profiler.end_timer("overall_collection")
        
        # 最終メモリ使用量を記録
        final_memory = profiler.get_memory_usage()
        print(f"  Final memory: {final_memory:.2f} MB")
        memory_used = final_memory - initial_memory
        print(f"  Memory used: {memory_used:.2f} MB")
        
        # 統計を計算
        total_tactics = len(successful_tactics)
        tactics_per_second = total_tactics / total_time if total_time > 0 else 0.0
        
        results = {
            "num_examples": num_examples,
            "max_steps": max_steps,
            "total_tactics_collected": total_tactics,
            "total_time": total_time,
            "tactics_per_second": tactics_per_second,
            "initial_memory_mb": initial_memory,
            "final_memory_mb": final_memory,
            "memory_used_mb": memory_used,
            "avg_tactics_per_example": total_tactics / num_examples if num_examples > 0 else 0.0
        }
        
        print(f"Overall Performance Results:")
        print(f"  Examples processed: {num_examples}")
        print(f"  Total tactics collected: {total_tactics}")
        print(f"  Total time: {total_time:.4f}s")
        print(f"  Tactics per second: {tactics_per_second:.2f}")
        print(f"  Average tactics per example: {total_tactics / num_examples if num_examples > 0 else 0:.2f}")
        print(f"  Memory used: {memory_used:.2f} MB")
        
        return results
        
    except Exception as e:
        profiler.end_timer("overall_collection")
        print(f"Overall performance test failed: {e}")
        return {"error": str(e)}


def run_comprehensive_performance_test(
    model_path: str,
    device: str = "auto",
    output_file: str = "performance_test_results.json",
    num_examples: int = 5,
    tactic_generation_tests: int = 50,
    tactic_application_tests: int = 100,
    temperature: float = 1.0,
    max_steps: int = 5
):
    """包括的なパフォーマンステストを実行"""
    print("Starting comprehensive performance test...")
    
    # デバイス設定
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    
    print(f"Using device: {device}")
    
    # グローバル定数を初期化
    initialize_global_constants()
    
    # モデルを読み込み
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return
    
    model, label_mappings = load_hierarchical_model(model_path, device)
    print(f"Loaded model from {model_path}")
    
    # トークナイザーを作成
    token_py_path = os.path.join(project_root, "src", "core", "fof_tokens.py")
    base_tokens, _ = load_tokens_and_labels_from_token_py(token_py_path)
    tokenizer = CharTokenizer(base_tokens=base_tokens)
    
    # モデルのmax_seq_lenを取得
    checkpoint = torch.load(model_path, map_location=device)
    max_seq_len = checkpoint.get('max_seq_len', 256)
    
    # パフォーマンステストを実行
    all_results = {}
    
    # 1. タクティク生成パフォーマンス
    print("\n" + "="*60)
    print("1. TACTIC GENERATION PERFORMANCE")
    print("="*60)
    generation_results = measure_tactic_generation_performance(
        model, tokenizer, label_mappings, device, max_seq_len, num_tests=tactic_generation_tests
    )
    all_results["tactic_generation"] = generation_results
    
    # 2. タクティク適用パフォーマンス
    print("\n" + "="*60)
    print("2. TACTIC APPLICATION PERFORMANCE")
    print("="*60)
    application_results = measure_tactic_application_performance(num_tests=tactic_application_tests)
    all_results["tactic_application"] = application_results
    
    # 3. 全体パフォーマンス
    print("\n" + "="*60)
    print("3. OVERALL PERFORMANCE")
    print("="*60)
    overall_results = measure_overall_performance(
        model, tokenizer, label_mappings, device, max_seq_len, num_examples=num_examples, max_steps=max_steps, temperature=temperature
    )
    all_results["overall"] = overall_results
    
    # 結果を保存
    output_path = os.path.join(project_root, "tests", output_file)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n" + "="*60)
    print("PERFORMANCE TEST SUMMARY")
    print("="*60)
    print(f"Results saved to: {output_path}")
    
    # サマリーを表示
    if "tactic_generation" in all_results:
        gen = all_results["tactic_generation"]
        print(f"Tactic Generation: {gen['tactics_per_second']:.2f} tactics/sec")
    
    if "tactic_application" in all_results:
        app = all_results["tactic_application"]
        print(f"Tactic Application: {app['applications_per_second']:.2f} applications/sec")
    
    if "overall" in all_results and "error" not in all_results["overall"]:
        overall = all_results["overall"]
        print(f"Overall Collection: {overall['tactics_per_second']:.2f} tactics/sec")
        print(f"Memory Usage: {overall['memory_used_mb']:.2f} MB")


def main():
    parser = argparse.ArgumentParser(description="Performance test for self improvement data collector")
    parser.add_argument("--model_path", type=str, default="models/pretrained_model.pth", help="model path")
    parser.add_argument("--device", type=str, default="auto", help="device (auto, cpu, cuda)")
    parser.add_argument("--output_file", type=str, default="performance_test_results.json", help="output file name")
    parser.add_argument("--num_examples", type=int, default=5, help="number of examples to process for overall performance test")
    parser.add_argument("--tactic_generation_tests", type=int, default=50, help="number of tests for tactic generation performance")
    parser.add_argument("--tactic_application_tests", type=int, default=100, help="number of tests for tactic application performance")
    parser.add_argument("--temperature", type=float, default=1.0, help="temperature for probabilistic tactic selection")
    parser.add_argument("--max_steps", type=int, default=5, help="maximum steps per example")
    
    args = parser.parse_args()
    
    print(f"Running performance test with {args.num_examples} examples...")
    run_comprehensive_performance_test(
        model_path=args.model_path,
        device=args.device,
        output_file=args.output_file,
        num_examples=args.num_examples,
        tactic_generation_tests=args.tactic_generation_tests,
        tactic_application_tests=args.tactic_application_tests,
        temperature=args.temperature,
        max_steps=args.max_steps
    )


if __name__ == "__main__":
    main()
