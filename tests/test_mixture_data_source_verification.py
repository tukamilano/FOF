#!/usr/bin/env python3
"""
Test script to verify the source of mixture data.

This script checks whether RL2 mixture data comes from:
- RL1 temperature_1 data
- RL1 temperature_2 data
- Or some other source

Usage:
    python tests/test_mixture_data_source_verification.py
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
    # Convert to string and hash
    data_str = json.dumps(data_item, sort_keys=True)
    return hashlib.md5(data_str.encode()).hexdigest()


def analyze_data_sources(base_dir: str = "/Users/milano/FOF", target_cycle: str = "RL2"):
    """Analyze the source of mixture data."""
    
    # Load target cycle mixture data
    target_mixture_path = os.path.join(base_dir, f"generated_data_{target_cycle}", "temperature_1_mixture")
    print(f"\n{'='*60}")
    print(f"Analyzing {target_cycle} mixture data from: {target_mixture_path}")
    print(f"{'='*60}")
    
    target_mixture_data = load_json_files(target_mixture_path)
    
    if not target_mixture_data:
        print(f"No {target_cycle} mixture data found!")
        return
    
    # Create hashes for target mixture data
    target_hashes = set()
    for item in target_mixture_data:
        target_hashes.add(create_data_hash(item))
    
    print(f"{target_cycle} mixture data: {len(target_mixture_data)} samples, {len(target_hashes)} unique hashes")
    
    # Determine source cycle (previous cycle)
    cycle_num = int(target_cycle[2:])  # Extract number from "RL2", "RL3", etc.
    source_cycle = f"RL{cycle_num - 1}" if cycle_num > 1 else "RL1"
    
    # Load source cycle temperature data
    source_temp1_path = os.path.join(base_dir, f"generated_data_{source_cycle}", "temperature_1")
    source_temp2_path = os.path.join(base_dir, f"generated_data_{source_cycle}", "temperature_2")
    
    print(f"\n{'='*60}")
    print(f"Analyzing {source_cycle} temperature data sources")
    print(f"{'='*60}")
    
    # Load source temperature_1 data
    source_temp1_data = load_json_files(source_temp1_path)
    source_temp1_hashes = set()
    for item in source_temp1_data:
        source_temp1_hashes.add(create_data_hash(item))
    
    print(f"{source_cycle} temperature_1 data: {len(source_temp1_data)} samples, {len(source_temp1_hashes)} unique hashes")
    
    # Load source temperature_2 data
    source_temp2_data = load_json_files(source_temp2_path)
    source_temp2_hashes = set()
    for item in source_temp2_data:
        source_temp2_hashes.add(create_data_hash(item))
    
    print(f"{source_cycle} temperature_2 data: {len(source_temp2_data)} samples, {len(source_temp2_hashes)} unique hashes")
    
    # Load deduplicated data for comparison
    dedup_path = os.path.join(base_dir, "deduplicated_data")
    dedup_data = load_json_files(dedup_path, max_files=10)  # Sample first 10 files
    dedup_hashes = set()
    for item in dedup_data:
        dedup_hashes.add(create_data_hash(item))
    
    print(f"Deduplicated data (sample): {len(dedup_data)} samples, {len(dedup_hashes)} unique hashes")
    
    # Analyze overlaps
    print(f"\n{'='*60}")
    print(f"OVERLAP ANALYSIS")
    print(f"{'='*60}")
    
    # Target mixture vs source temperature_1
    overlap_temp1 = target_hashes.intersection(source_temp1_hashes)
    overlap_temp1_percent = len(overlap_temp1) / len(target_hashes) * 100 if target_hashes else 0
    
    print(f"{target_cycle} mixture ∩ {source_cycle} temperature_1: {len(overlap_temp1)} samples ({overlap_temp1_percent:.2f}%)")
    
    # Target mixture vs source temperature_2
    overlap_temp2 = target_hashes.intersection(source_temp2_hashes)
    overlap_temp2_percent = len(overlap_temp2) / len(target_hashes) * 100 if target_hashes else 0
    
    print(f"{target_cycle} mixture ∩ {source_cycle} temperature_2: {len(overlap_temp2)} samples ({overlap_temp2_percent:.2f}%)")
    
    # Target mixture vs deduplicated data
    overlap_dedup = target_hashes.intersection(dedup_hashes)
    overlap_dedup_percent = len(overlap_dedup) / len(target_hashes) * 100 if target_hashes else 0
    
    print(f"{target_cycle} mixture ∩ Deduplicated data: {len(overlap_dedup)} samples ({overlap_dedup_percent:.2f}%)")
    
    # Source temperature_1 vs source temperature_2 (check if they're different)
    overlap_temp1_temp2 = source_temp1_hashes.intersection(source_temp2_hashes)
    overlap_temp1_temp2_percent = len(overlap_temp1_temp2) / len(source_temp1_hashes) * 100 if source_temp1_hashes else 0
    
    print(f"{source_cycle} temperature_1 ∩ {source_cycle} temperature_2: {len(overlap_temp1_temp2)} samples ({overlap_temp1_temp2_percent:.2f}%)")
    
    # Analysis summary
    print(f"\n{'='*60}")
    print(f"ANALYSIS SUMMARY")
    print(f"{'='*60}")
    
    if overlap_temp1_percent > 50:
        print(f"✅ {target_cycle} mixture data appears to come primarily from {source_cycle} temperature_1 ({overlap_temp1_percent:.2f}% overlap)")
    elif overlap_temp2_percent > 50:
        print(f"✅ {target_cycle} mixture data appears to come primarily from {source_cycle} temperature_2 ({overlap_temp2_percent:.2f}% overlap)")
    elif overlap_dedup_percent > 30:
        print(f"✅ {target_cycle} mixture data appears to be a proper mixture with deduplicated data ({overlap_dedup_percent:.2f}% overlap)")
    else:
        print(f"❓ {target_cycle} mixture data source is unclear. Consider regenerating.")
    
    # Check if source temperature directories are different
    if overlap_temp1_temp2_percent < 10:
        print(f"✅ {source_cycle} temperature_1 and temperature_2 contain different data ({overlap_temp1_temp2_percent:.2f}% overlap)")
    else:
        print(f"⚠️  {source_cycle} temperature_1 and temperature_2 have significant overlap ({overlap_temp1_temp2_percent:.2f}% overlap)")
    
    # Sample analysis
    print(f"\n{'='*60}")
    print(f"SAMPLE DATA ANALYSIS")
    print(f"{'='*60}")
    
    if target_mixture_data:
        print(f"Sample {target_cycle} mixture data:")
        for i, item in enumerate(target_mixture_data[:3]):
            print(f"  {i+1}: {str(item)[:100]}...")
    
    if source_temp1_data:
        print(f"\nSample {source_cycle} temperature_1 data:")
        for i, item in enumerate(source_temp1_data[:3]):
            print(f"  {i+1}: {str(item)[:100]}...")
    
    if source_temp2_data:
        print(f"\nSample {source_cycle} temperature_2 data:")
        for i, item in enumerate(source_temp2_data[:3]):
            print(f"  {i+1}: {str(item)[:100]}...")
    
    return {
        'target_mixture_count': len(target_mixture_data),
        'source_temp1_count': len(source_temp1_data),
        'source_temp2_count': len(source_temp2_data),
        'overlap_temp1_percent': overlap_temp1_percent,
        'overlap_temp2_percent': overlap_temp2_percent,
        'overlap_dedup_percent': overlap_dedup_percent,
        'overlap_temp1_temp2_percent': overlap_temp1_temp2_percent
    }


def main():
    parser = argparse.ArgumentParser(description="Verify mixture data sources")
    parser.add_argument("--base-dir", default="/Users/milano/FOF", 
                       help="Base directory containing the data")
    parser.add_argument("--target-cycle", default="RL2",
                       help="Target cycle to analyze (e.g., RL2, RL3, RL4)")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose output")
    
    args = parser.parse_args()
    
    print(f"Mixture Data Source Verification for {args.target_cycle}")
    print("=" * 50)
    
    results = analyze_data_sources(args.base_dir, args.target_cycle)
    
    if results:
        print(f"\n{'='*60}")
        print(f"RECOMMENDATION")
        print(f"{'='*60}")
        
        cycle_num = int(args.target_cycle[2:])
        source_cycle = f"RL{cycle_num - 1}" if cycle_num > 1 else "RL1"
        
        if results['overlap_temp1_percent'] > 50:
            print(f"✅ {args.target_cycle} mixture data is correctly sourced from {source_cycle} temperature_1")
            print("   No regeneration needed.")
        elif results['overlap_temp2_percent'] > 50:
            print(f"❌ {args.target_cycle} mixture data incorrectly sourced from {source_cycle} temperature_2")
            print("   Consider regenerating with correct temperature_1 source.")
        elif results['overlap_dedup_percent'] > 30:
            print(f"✅ {args.target_cycle} mixture data appears to be a proper mixture")
            print("   No regeneration needed.")
        else:
            print(f"❓ {args.target_cycle} mixture data source is unclear")
            print("   Consider regenerating to ensure correct source.")


if __name__ == "__main__":
    main()
