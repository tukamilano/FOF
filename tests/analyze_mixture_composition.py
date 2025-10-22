#!/usr/bin/env python3
"""
Analyze why deduplicated data overlap is lower than expected in mixture data.

This script investigates the discrepancy between expected 30% deduplicated data
and actual overlap percentages in mixture datasets.
"""

import json
import os
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Set
from collections import defaultdict, Counter
import argparse


def load_json_files(directory: str, max_files: int = None) -> List[Dict[str, Any]]:
    """Load all JSON files from a directory and return combined data."""
    if not os.path.exists(directory):
        print(f"Directory not found: {directory}")
        return []
    
    data = []
    json_files = sorted([f for f in os.listdir(directory) if f.endswith('.json')])
    
    if max_files:
        json_files = json_files[:max_files]
    
    print(f"Loading {len(json_files)} files from {directory}")
    
    for filename in json_files:
        filepath = os.path.join(directory, filename)
        try:
            with open(filepath, 'r') as f:
                file_data = json.load(f)
                if isinstance(file_data, list):
                    data.extend(file_data)
                else:
                    data.append(file_data)
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            continue
    
    print(f"Loaded {len(data)} total samples from {directory}")
    return data


def create_data_hash(data_item: Dict[str, Any]) -> str:
    """Create a hash for a data item to enable comparison."""
    data_str = json.dumps(data_item, sort_keys=True)
    return hashlib.md5(data_str.encode()).hexdigest()


def analyze_mixture_composition(base_dir: str = "/Users/milano/FOF", target_cycle: str = "RL2"):
    """Analyze the actual composition of mixture data."""
    
    print(f"Analyzing {target_cycle} mixture composition")
    print("=" * 60)
    
    # Load mixture data
    mixture_path = os.path.join(base_dir, f"generated_data_{target_cycle}", "temperature_1_mixture")
    mixture_data = load_json_files(mixture_path)
    
    if not mixture_data:
        print(f"No {target_cycle} mixture data found!")
        return
    
    print(f"Mixture data: {len(mixture_data)} samples")
    
    # Determine source cycle
    cycle_num = int(target_cycle[2:])
    source_cycle = f"RL{cycle_num - 1}" if cycle_num > 1 else "RL1"
    
    # Load source temperature_1 data
    source_temp1_path = os.path.join(base_dir, f"generated_data_{source_cycle}", "temperature_1")
    source_temp1_data = load_json_files(source_temp1_path)
    
    # Load deduplicated data (larger sample)
    dedup_path = os.path.join(base_dir, "deduplicated_data")
    dedup_data = load_json_files(dedup_path, max_files=100)  # Load more files
    
    print(f"\nSource data sizes:")
    print(f"  {source_cycle} temperature_1: {len(source_temp1_data)} samples")
    print(f"  Deduplicated data (sample): {len(dedup_data)} samples")
    
    # Create hashes
    mixture_hashes = set(create_data_hash(item) for item in mixture_data)
    source_temp1_hashes = set(create_data_hash(item) for item in source_temp1_data)
    dedup_hashes = set(create_data_hash(item) for item in dedup_data)
    
    print(f"\nUnique hashes:")
    print(f"  Mixture: {len(mixture_hashes)} unique")
    print(f"  {source_cycle} temperature_1: {len(source_temp1_hashes)} unique")
    print(f"  Deduplicated (sample): {len(dedup_hashes)} unique")
    
    # Calculate overlaps
    overlap_source = mixture_hashes.intersection(source_temp1_hashes)
    overlap_dedup = mixture_hashes.intersection(dedup_hashes)
    
    overlap_source_percent = len(overlap_source) / len(mixture_hashes) * 100
    overlap_dedup_percent = len(overlap_dedup) / len(mixture_hashes) * 100
    
    print(f"\nOverlap analysis:")
    print(f"  Mixture ∩ {source_cycle} temperature_1: {len(overlap_source)} samples ({overlap_source_percent:.2f}%)")
    print(f"  Mixture ∩ Deduplicated (sample): {len(overlap_dedup)} samples ({overlap_dedup_percent:.2f}%)")
    
    # Check for duplicates within source data
    source_dedup_overlap = source_temp1_hashes.intersection(dedup_hashes)
    source_dedup_percent = len(source_dedup_overlap) / len(source_temp1_hashes) * 100
    
    print(f"\nSource data analysis:")
    print(f"  {source_cycle} temperature_1 ∩ Deduplicated: {len(source_dedup_overlap)} samples ({source_dedup_percent:.2f}%)")
    
    # Analyze data structure differences
    print(f"\nData structure analysis:")
    
    # Sample some items to understand structure
    if mixture_data:
        sample_mixture = mixture_data[0]
        print(f"  Mixture data structure: {list(sample_mixture.keys())}")
    
    if source_temp1_data:
        sample_source = source_temp1_data[0]
        print(f"  {source_cycle} temperature_1 structure: {list(sample_source.keys())}")
    
    if dedup_data:
        sample_dedup = dedup_data[0]
        print(f"  Deduplicated data structure: {list(sample_dedup.keys())}")
    
    # Check if data structures are compatible
    print(f"\nStructure compatibility:")
    if mixture_data and source_temp1_data and dedup_data:
        mixture_keys = set(mixture_data[0].keys())
        source_keys = set(source_temp1_data[0].keys())
        dedup_keys = set(dedup_data[0].keys())
        
        print(f"  Mixture vs {source_cycle} temperature_1: {mixture_keys == source_keys}")
        print(f"  Mixture vs Deduplicated: {mixture_keys == dedup_keys}")
        print(f"  {source_cycle} temperature_1 vs Deduplicated: {source_keys == dedup_keys}")
        
        if mixture_keys != dedup_keys:
            print(f"  Key differences:")
            print(f"    Only in mixture: {mixture_keys - dedup_keys}")
            print(f"    Only in deduplicated: {dedup_keys - mixture_keys}")
    
    # Theoretical vs actual analysis
    print(f"\nTheoretical vs Actual Analysis:")
    print(f"  Expected deduplicated ratio: 30%")
    print(f"  Actual deduplicated overlap: {overlap_dedup_percent:.2f}%")
    
    if overlap_dedup_percent < 20:
        print(f"  ❌ Deduplicated overlap is much lower than expected")
        print(f"  Possible reasons:")
        print(f"    1. Data structures are different (different keys/format)")
        print(f"    2. Deduplicated data sample is not representative")
        print(f"    3. Mixture creation process has issues")
        print(f"    4. Deduplicated data was not properly included in mixture")
    elif overlap_dedup_percent > 25:
        print(f"  ✅ Deduplicated overlap is close to expected range")
    else:
        print(f"  ⚠️  Deduplicated overlap is lower than expected but not critically low")
    
    return {
        'mixture_count': len(mixture_data),
        'source_count': len(source_temp1_data),
        'dedup_count': len(dedup_data),
        'overlap_source_percent': overlap_source_percent,
        'overlap_dedup_percent': overlap_dedup_percent,
        'source_dedup_percent': source_dedup_percent
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze mixture data composition")
    parser.add_argument("--base-dir", default="/Users/milano/FOF", 
                       help="Base directory containing the data")
    parser.add_argument("--target-cycle", default="RL2",
                       help="Target cycle to analyze (e.g., RL2, RL3, RL4)")
    
    args = parser.parse_args()
    
    results = analyze_mixture_composition(args.base_dir, args.target_cycle)
    
    if results:
        print(f"\n{'='*60}")
        print(f"CONCLUSION")
        print(f"{'='*60}")
        
        if results['overlap_dedup_percent'] < 15:
            print("The low deduplicated overlap suggests:")
            print("1. Data structure incompatibility between mixture and deduplicated data")
            print("2. Deduplicated data was not properly included in the mixture")
            print("3. The mixture creation process may have issues")
            print("\nRecommendation: Check the mixture creation process and data formats")
        else:
            print("The deduplicated overlap is within acceptable range")
            print("The 7:3 ratio may be achieved through different data formats")


if __name__ == "__main__":
    main()
