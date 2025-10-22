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


def analyze_data_sources(base_dir: str = "/Users/milano/FOF"):
    """Analyze the source of mixture data."""
    
    # Load RL2 mixture data
    rl2_mixture_path = os.path.join(base_dir, "generated_data_RL2", "temperature_1_mixture")
    print(f"\n{'='*60}")
    print(f"Analyzing RL2 mixture data from: {rl2_mixture_path}")
    print(f"{'='*60}")
    
    rl2_mixture_data = load_json_files(rl2_mixture_path)
    
    if not rl2_mixture_data:
        print("No RL2 mixture data found!")
        return
    
    # Create hashes for RL2 mixture data
    rl2_hashes = set()
    for item in rl2_mixture_data:
        rl2_hashes.add(create_data_hash(item))
    
    print(f"RL2 mixture data: {len(rl2_mixture_data)} samples, {len(rl2_hashes)} unique hashes")
    
    # Load RL1 temperature data sources
    rl1_temp1_path = os.path.join(base_dir, "generated_data_RL1", "temperature_1")
    rl1_temp2_path = os.path.join(base_dir, "generated_data_RL1", "temperature_2")
    
    print(f"\n{'='*60}")
    print(f"Analyzing RL1 temperature data sources")
    print(f"{'='*60}")
    
    # Load RL1 temperature_1 data
    rl1_temp1_data = load_json_files(rl1_temp1_path)
    rl1_temp1_hashes = set()
    for item in rl1_temp1_data:
        rl1_temp1_hashes.add(create_data_hash(item))
    
    print(f"RL1 temperature_1 data: {len(rl1_temp1_data)} samples, {len(rl1_temp1_hashes)} unique hashes")
    
    # Load RL1 temperature_2 data
    rl1_temp2_data = load_json_files(rl1_temp2_path)
    rl1_temp2_hashes = set()
    for item in rl1_temp2_data:
        rl1_temp2_hashes.add(create_data_hash(item))
    
    print(f"RL1 temperature_2 data: {len(rl1_temp2_data)} samples, {len(rl1_temp2_hashes)} unique hashes")
    
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
    
    # RL2 mixture vs RL1 temperature_1
    overlap_temp1 = rl2_hashes.intersection(rl1_temp1_hashes)
    overlap_temp1_percent = len(overlap_temp1) / len(rl2_hashes) * 100 if rl2_hashes else 0
    
    print(f"RL2 mixture ∩ RL1 temperature_1: {len(overlap_temp1)} samples ({overlap_temp1_percent:.2f}%)")
    
    # RL2 mixture vs RL1 temperature_2
    overlap_temp2 = rl2_hashes.intersection(rl1_temp2_hashes)
    overlap_temp2_percent = len(overlap_temp2) / len(rl2_hashes) * 100 if rl2_hashes else 0
    
    print(f"RL2 mixture ∩ RL1 temperature_2: {len(overlap_temp2)} samples ({overlap_temp2_percent:.2f}%)")
    
    # RL2 mixture vs deduplicated data
    overlap_dedup = rl2_hashes.intersection(dedup_hashes)
    overlap_dedup_percent = len(overlap_dedup) / len(rl2_hashes) * 100 if rl2_hashes else 0
    
    print(f"RL2 mixture ∩ Deduplicated data: {len(overlap_dedup)} samples ({overlap_dedup_percent:.2f}%)")
    
    # RL1 temperature_1 vs RL1 temperature_2 (check if they're different)
    overlap_temp1_temp2 = rl1_temp1_hashes.intersection(rl1_temp2_hashes)
    overlap_temp1_temp2_percent = len(overlap_temp1_temp2) / len(rl1_temp1_hashes) * 100 if rl1_temp1_hashes else 0
    
    print(f"RL1 temperature_1 ∩ RL1 temperature_2: {len(overlap_temp1_temp2)} samples ({overlap_temp1_temp2_percent:.2f}%)")
    
    # Analysis summary
    print(f"\n{'='*60}")
    print(f"ANALYSIS SUMMARY")
    print(f"{'='*60}")
    
    if overlap_temp1_percent > 50:
        print(f"✅ RL2 mixture data appears to come primarily from RL1 temperature_1 ({overlap_temp1_percent:.2f}% overlap)")
    elif overlap_temp2_percent > 50:
        print(f"✅ RL2 mixture data appears to come primarily from RL1 temperature_2 ({overlap_temp2_percent:.2f}% overlap)")
    elif overlap_dedup_percent > 30:
        print(f"✅ RL2 mixture data appears to be a proper mixture with deduplicated data ({overlap_dedup_percent:.2f}% overlap)")
    else:
        print(f"❓ RL2 mixture data source is unclear. Consider regenerating.")
    
    # Check if RL1 temperature directories are different
    if overlap_temp1_temp2_percent < 10:
        print(f"✅ RL1 temperature_1 and temperature_2 contain different data ({overlap_temp1_temp2_percent:.2f}% overlap)")
    else:
        print(f"⚠️  RL1 temperature_1 and temperature_2 have significant overlap ({overlap_temp1_temp2_percent:.2f}% overlap)")
    
    # Sample analysis
    print(f"\n{'='*60}")
    print(f"SAMPLE DATA ANALYSIS")
    print(f"{'='*60}")
    
    if rl2_mixture_data:
        print("Sample RL2 mixture data:")
        for i, item in enumerate(rl2_mixture_data[:3]):
            print(f"  {i+1}: {str(item)[:100]}...")
    
    if rl1_temp1_data:
        print("\nSample RL1 temperature_1 data:")
        for i, item in enumerate(rl1_temp1_data[:3]):
            print(f"  {i+1}: {str(item)[:100]}...")
    
    if rl1_temp2_data:
        print("\nSample RL1 temperature_2 data:")
        for i, item in enumerate(rl1_temp2_data[:3]):
            print(f"  {i+1}: {str(item)[:100]}...")
    
    return {
        'rl2_mixture_count': len(rl2_mixture_data),
        'rl1_temp1_count': len(rl1_temp1_data),
        'rl1_temp2_count': len(rl1_temp2_data),
        'overlap_temp1_percent': overlap_temp1_percent,
        'overlap_temp2_percent': overlap_temp2_percent,
        'overlap_dedup_percent': overlap_dedup_percent,
        'overlap_temp1_temp2_percent': overlap_temp1_temp2_percent
    }


def main():
    parser = argparse.ArgumentParser(description="Verify mixture data sources")
    parser.add_argument("--base-dir", default="/Users/milano/FOF", 
                       help="Base directory containing the data")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose output")
    
    args = parser.parse_args()
    
    print("Mixture Data Source Verification")
    print("=" * 50)
    
    results = analyze_data_sources(args.base_dir)
    
    if results:
        print(f"\n{'='*60}")
        print(f"RECOMMENDATION")
        print(f"{'='*60}")
        
        if results['overlap_temp1_percent'] > 50:
            print("✅ RL2 mixture data is correctly sourced from RL1 temperature_1")
            print("   No regeneration needed.")
        elif results['overlap_temp2_percent'] > 50:
            print("❌ RL2 mixture data incorrectly sourced from RL1 temperature_2")
            print("   Consider regenerating with correct temperature_1 source.")
        elif results['overlap_dedup_percent'] > 30:
            print("✅ RL2 mixture data appears to be a proper mixture")
            print("   No regeneration needed.")
        else:
            print("❓ RL2 mixture data source is unclear")
            print("   Consider regenerating to ensure correct source.")


if __name__ == "__main__":
    main()
