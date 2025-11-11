#!/usr/bin/env python3
"""
Script to compress tactic sequences using true BPE algorithm
"""

import json
import re
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Set

def extract_tactic_string(tactic: Dict) -> str:
    """
    Convert tactic object to string
    Distinguish as different tactics if any of main, arg1, arg2 differs
    Don't include null
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

def extract_tactic_sequences(data: List[Dict]) -> List[List[str]]:
    """
    training_data from tactic_sequence 抽出
    """
    sequences = []
    
    for example in data:
        if "steps" in example:
            sequence = []
            for step in example["steps"]:
                if "tactic" in step and step.get("tactic_apply", False):
                    tactic_str = extract_tactic_string(step["tactic"])
                    sequence.append(tactic_str)
            if sequence:  # 空 with/at no/notシーケンスonly追加
                sequences.append(sequence)
    
    return sequences

def get_pairs(sequence: List[str]) -> List[Tuple[str, str]]:
    """
    シーケンス from 隣接do/performペア get
    """
    pairs = []
    for i in range(len(sequence) - 1):
        pairs.append((sequence[i], sequence[i + 1]))
    return pairs

def get_word_freqs(sequences: List[List[str]]) -> Counter:
    """
    全シーケンス from 単語（タクティク）の頻度 計算
    """
    word_freqs = Counter()
    for sequence in sequences:
        for word in sequence:
            word_freqs[word] += 1
    return word_freqs

def get_pair_freqs(sequences: List[List[str]]) -> Counter:
    """
    全シーケンス from ペアの頻度 計算
    """
    pair_freqs = Counter()
    for sequence in sequences:
        pairs = get_pairs(sequence)
        for pair in pairs:
            pair_freqs[pair] += 1
    return pair_freqs

def merge_vocab(pair: Tuple[str, str], sequences: List[List[str]]) -> List[List[str]]:
    """
    指定was doneペア マージand新しいシーケンス 作成
    """
    new_sequences = []
    for sequence in sequences:
        new_sequence = []
        i = 0
        while i < len(sequence):
            if i < len(sequence) - 1 and (sequence[i], sequence[i + 1]) == pair:
                # ペア マージ
                merged = f"{pair[0]}_{pair[1]}"
                new_sequence.append(merged)
                i += 2
            else:
                new_sequence.append(sequence[i])
                i += 1
        new_sequences.append(new_sequence)
    return new_sequences

def apply_bpe(sequences: List[List[str]], num_merges: int = 100) -> Tuple[List[List[str]], List[Tuple[str, str]], Dict[str, str]]:
    """
    真のBPEアルゴリズム 適用
    """
    print(f"Starting BPE with {num_merges} merges...")
    
    # 初期語彙 get
    vocab = set()
    for sequence in sequences:
        for word in sequence:
            vocab.add(word)
    
    print(f"Initial vocabulary size: {len(vocab)}")
    
    merges = []
    tactic_mapping = {}
    
    for merge_num in range(num_merges):
        # 現在のペアの頻度 計算
        pair_freqs = get_pair_freqs(sequences)
        
        if not pair_freqs:
            print(f"No more pairs to merge at iteration {merge_num}")
            break
        
        # 最も頻度の高いペア 選択
        best_pair = pair_freqs.most_common(1)[0][0]
        best_freq = pair_freqs[best_pair]
        
        if best_freq < 2:  # 最小頻度チェック
            print(f"Best pair frequency {best_freq} below threshold at iteration {merge_num}")
            break
        
        print(f"Iteration {merge_num + 1}: Merging {best_pair[0]} + {best_pair[1]} (freq: {best_freq})")
        
        # ペア マージ
        sequences = merge_vocab(best_pair, sequences)
        merges.append(best_pair)
        
        # マッピング 作成
        merged_name = f"compressed_{best_pair[0]}_{best_pair[1]}"
        tactic_mapping[f"{best_pair[0]}_{best_pair[1]}"] = merged_name
        
        # 語彙 更新
        vocab.add(merged_name)
        vocab.discard(best_pair[0])
        vocab.discard(best_pair[1])
        
        if (merge_num + 1) % 10 == 0:
            print(f"  Vocabulary size: {len(vocab)}")
    
    print(f"BPE completed. Final vocabulary size: {len(vocab)}")
    print(f"Total merges performed: {len(merges)}")
    
    return sequences, merges, tactic_mapping

def create_compressed_sequences(original_sequences: List[List[str]], merges: List[Tuple[str, str]]) -> List[List[str]]:
    """
    元のシーケンス マージ 適用and圧縮was doneシーケンス 作成
    """
    sequences = [seq.copy() for seq in original_sequences]
    
    for pair in merges:
        sequences = merge_vocab(pair, sequences)
    
    return sequences

def main():
    import sys
    import os
    
    # Move to project root
    script_dir = os.path.dirname(__file__)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
    os.chdir(project_root)
    
    # コマンドライン引数 from マージ数 get
    num_merges = 200  # デフォルト値
    if len(sys.argv) > 1:
        try:
            num_merges = int(sys.argv[1])
        except ValueError:
            print(f"Invalid number of merges: {sys.argv[1]}. Using default: {num_merges}")
    
    print(f"Using {num_merges} merges for BPE compression")
    
    # training_data.json 読み込み
    print("Loading training_data.json...")
    with open("data/training_data.json", "r") as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} examples")
    
    # tactic_sequence 抽出
    print("Extracting tactic sequences...")
    sequences = extract_tactic_sequences(data)
    print(f"Extracted {len(sequences)} tactic sequences")
    
    # 統計情報 表示
    all_tactics = []
    for seq in sequences:
        all_tactics.extend(seq)
    
    tactic_counter = Counter(all_tactics)
    print(f"Total unique tactics: {len(tactic_counter)}")
    print(f"Most common tactics:")
    for tactic, count in tactic_counter.most_common(10):
        print(f"  {tactic}: {count}")
    
    # 初期統計
    original_length = sum(len(seq) for seq in sequences)
    print(f"Original total tactics: {original_length}")
    
    # BPE 適用
    print("\nApplying BPE compression...")
    compressed_sequences, merges, tactic_mapping = apply_bpe(sequences, num_merges=num_merges)
    
    # 圧縮率 計算
    compressed_length = sum(len(seq) for seq in compressed_sequences)
    compression_ratio = (original_length - compressed_length) / original_length * 100
    
    print(f"\nCompression results:")
    print(f"Original total tactics: {original_length}")
    print(f"Compressed total tactics: {compressed_length}")
    print(f"Compression ratio: {compression_ratio:.2f}%")
    print(f"Number of merges: {len(merges)}")
    
    # 結果 保存
    result = {
        "original_sequences": sequences,
        "compressed_sequences": compressed_sequences,
        "merges": merges,
        "tactic_mapping": tactic_mapping,
        "statistics": {
            "original_sequences_count": len(sequences),
            "compressed_sequences_count": len(compressed_sequences),
            "original_total_tactics": original_length,
            "compressed_total_tactics": compressed_length,
            "compression_ratio": compression_ratio,
            "number_of_merges": len(merges)
        }
    }
    
    with open("data/tactic_compression_bpe_analysis.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"\nResults saved to data/tactic_compression_bpe_analysis.json")

if __name__ == "__main__":
    main()
