#!/usr/bin/env python3
"""
Script to create new training_data.json with tactics compressed using true BPE (v2)
"""

import json
import re
from typing import List, Dict, Tuple, Set

def extract_tactic_string(tactic: Dict) -> str:
    """
    Convert tactic object to string
    """
    main = tactic.get("main", "")
    arg1 = tactic.get("arg1")
    arg2 = tactic.get("arg2")
    
    # Build parts (exclude null)
    parts = [main]
    if arg1 is not None:
        parts.append(str(arg1))
    if arg2 is not None:
        parts.append(str(arg2))
    
    return "_".join(parts)

def parse_compressed_tactic(compressed_tactic: str) -> Dict:
    """
    圧縮was doneタクティク文字列 tacticオブジェクト 変換
    """
    # BPE with/at 圧縮was doneタクティクは複数のアンダースコア 含む
    if "_" in compressed_tactic and compressed_tactic.count("_") > 2:
        # 圧縮was doneタクティクの場合（複数のタクティク 結合されてexists）
        return {
            "main": f"compressed_{compressed_tactic}",
            "arg1": None,
            "arg2": None
        }
    else:
        # 通常のタクティクの場合（nullは既 除外されてexists）
        parts = compressed_tactic.split("_")
        if len(parts) == 1:
            main = parts[0]
            arg1 = None
            arg2 = None
        elif len(parts) == 2:
            main = parts[0]
            arg1 = parts[1]
            arg2 = None
        elif len(parts) == 3:
            main = parts[0]
            arg1 = parts[1]
            arg2 = parts[2]
        else:
            # 3以上の場合は最初 main、残り arg1 
            main = parts[0]
            arg1 = "_".join(parts[1:])
            arg2 = None
        
        return {
            "main": main,
            "arg1": arg1,
            "arg2": arg2
        }

def create_compressed_steps(original_steps: List[Dict], compressed_sequence: List[str], original_sequence: List[str]) -> List[Dict]:
    """
    元のstepsと圧縮was doneシーケンス from 新しいsteps 作成
    """
    if not original_steps:
        return original_steps
    
    # タクティクステップ 抽出
    tactic_steps = []
    for i, step in enumerate(original_steps):
        if "tactic" in step and step.get("tactic_apply", False):
            tactic_steps.append((i, step))
    
    if len(tactic_steps) == 0:
        return original_steps
    
    # 圧縮was doneシーケンス from 新しいsteps 作成
    new_steps = []
    tactic_idx = 0
    step_idx = 0
    
    # 元のシーケンスと圧縮was doneシーケンスの長さの差 計算
    original_length = len(original_sequence)
    compressed_length = len(compressed_sequence)
    total_skip = original_length - compressed_length
    
    while step_idx < len(original_steps):
        step = original_steps[step_idx]
        
        if "tactic" in step and step.get("tactic_apply", False):
            # タクティクステップの場合
            if tactic_idx < len(compressed_sequence):
                # 圧縮was doneタクティク 使用
                compressed_tactic = compressed_sequence[tactic_idx]
                new_step = step.copy()
                new_step["tactic"] = parse_compressed_tactic(compressed_tactic)
                new_step["step_index"] = len(new_steps)
                new_steps.append(new_step)
                
                # 圧縮was doneタクティクの場合、スキップdo/performステップ数 計算
                if "_" in compressed_tactic and compressed_tactic.count("_") > 2:
                    # 残りの圧縮タクティク数 基づいてスキップ数 計算
                    remaining_compressed = len(compressed_sequence) - tactic_idx - 1
                    if remaining_compressed > 0:
                        skip_count = total_skip // remaining_compressed
                        total_skip -= skip_count
                    else:
                        skip_count = total_skip
                        total_skip = 0
                    
                    # スキップdo/performステップ スキップ
                    for _ in range(skip_count):
                        step_idx += 1
                        if step_idx < len(original_steps):
                            next_step = original_steps[step_idx]
                            if "tactic" in next_step and next_step.get("tactic_apply", False):
                                continue  # タクティクステップ スキップ
                            else:
                                break  # タクティク with/at no/notステップはスキップしno/not
                
                tactic_idx += 1
            else:
                # 圧縮シーケンス 終わった場合、元のタクティク 使用
                new_step = step.copy()
                new_step["step_index"] = len(new_steps)
                new_steps.append(new_step)
        else:
            # タクティク with/at no/notステップはそのまま追加
            new_step = step.copy()
            new_step["step_index"] = len(new_steps)
            new_steps.append(new_step)
        
        step_idx += 1
    
    return new_steps

def main():
    # Move to project root
    import os
    script_dir = os.path.dirname(__file__)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
    os.chdir(project_root)
    
    # 元のtraining_data.json 読み込み
    print("Loading original training_data.json...")
    with open("data/training_data.json", "r") as f:
        original_data = json.load(f)
    
    # BPE分析結果 読み込み
    print("Loading BPE analysis...")
    with open("data/tactic_compression_bpe_analysis.json", "r") as f:
        bpe_data = json.load(f)
    
    compressed_sequences = bpe_data["compressed_sequences"]
    original_sequences = bpe_data["original_sequences"]
    print(f"Loaded {len(compressed_sequences)} compressed sequences")
    
    # 圧縮was doneデータ 作成
    print("Creating compressed training data...")
    compressed_data = []
    sequence_idx = 0
    
    for i, example in enumerate(original_data):
        if i % 100 == 0:
            print(f"Processing example {i+1}/{len(original_data)}")
        
        compressed_example = example.copy()
        
        if "steps" in example and sequence_idx < len(compressed_sequences):
            # 対応do/perform圧縮was doneシーケンスと元のシーケンス 使用
            compressed_sequence = compressed_sequences[sequence_idx]
            original_sequence = original_sequences[sequence_idx]
            compressed_example["steps"] = create_compressed_steps(example["steps"], compressed_sequence, original_sequence)
            sequence_idx += 1
        else:
            # 圧縮シーケンス no/not場合は元のまま
            compressed_example["steps"] = example.get("steps", [])
        
        compressed_data.append(compressed_example)
    
    # 圧縮was doneデータ 保存
    output_file = "data/training_data_compressed_bpe.json"
    print(f"Saving compressed data to {output_file}...")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(compressed_data, f, ensure_ascii=False, indent=2)
    
    # 統計情報 表示
    original_steps = sum(len(example.get("steps", [])) for example in original_data)
    compressed_steps = sum(len(example.get("steps", [])) for example in compressed_data)
    
    print(f"\nCompression results:")
    print(f"Original examples: {len(original_data)}")
    print(f"Compressed examples: {len(compressed_data)}")
    print(f"Original total steps: {original_steps}")
    print(f"Compressed total steps: {compressed_steps}")
    print(f"Step reduction: {original_steps - compressed_steps} steps")
    print(f"Step compression ratio: {(original_steps - compressed_steps) / original_steps * 100:.2f}%")
    
    # シーケンスレベルの圧縮率も表示
    original_sequence_length = sum(len(seq) for seq in original_sequences)
    compressed_sequence_length = sum(len(seq) for seq in compressed_sequences)
    sequence_compression_ratio = (original_sequence_length - compressed_sequence_length) / original_sequence_length * 100
    
    print(f"\nSequence-level compression:")
    print(f"Original sequence length: {original_sequence_length}")
    print(f"Compressed sequence length: {compressed_sequence_length}")
    print(f"Sequence compression ratio: {sequence_compression_ratio:.2f}%")
    
    print("Done!")

if __name__ == "__main__":
    main()
