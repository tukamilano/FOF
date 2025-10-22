#!/usr/bin/env python3
"""
Create temperature mixture datasets for RL stability.

This script creates mixture datasets by combining:
- All data from each temperature directory (e.g., temperature_1)
- Randomly selected 30% of deduplicated_data

The ratio is 7:3 (temperature_data:deduplicated_data).
"""

import json
import os
import random
import argparse
from pathlib import Path
from typing import List, Dict, Any
import math


def load_json_files(directory: str) -> List[Dict[str, Any]]:
    """Load all JSON files from a directory and return combined data."""
    data = []
    json_files = sorted([f for f in os.listdir(directory) if f.endswith('.json')])
    
    print(f"Loading {len(json_files)} files from {directory}")
    
    for filename in json_files:
        filepath = os.path.join(directory, filename)
        with open(filepath, 'r') as f:
            file_data = json.load(f)
            data.extend(file_data)
    
    print(f"Loaded {len(data)} total samples from {directory}")
    return data


def sample_deduplicated_data(deduplicated_data: List[Dict[str, Any]], 
                           target_size: int, 
                           seed: int = 42) -> List[Dict[str, Any]]:
    """Randomly sample target_size samples from deduplicated_data."""
    random.seed(seed)
    if len(deduplicated_data) <= target_size:
        print(f"Warning: deduplicated_data size ({len(deduplicated_data)}) <= target_size ({target_size})")
        return deduplicated_data
    
    sampled = random.sample(deduplicated_data, target_size)
    print(f"Sampled {len(sampled)} samples from deduplicated_data")
    return sampled


def create_mixture_dataset(temperature_data: List[Dict[str, Any]], 
                          deduplicated_data: List[Dict[str, Any]], 
                          ratio: float = 0.7) -> List[Dict[str, Any]]:
    """Create mixture dataset with specified ratio."""
    total_temp_size = len(temperature_data)
    
    # Calculate how many deduplicated samples we need
    # ratio = temp_size / (temp_size + dedup_size)
    # dedup_size = temp_size * (1 - ratio) / ratio
    target_dedup_size = int(total_temp_size * (1 - ratio) / ratio)
    
    print(f"Temperature data size: {total_temp_size}")
    print(f"Target deduplicated data size: {target_dedup_size}")
    print(f"Target ratio: {ratio:.1f}:{1-ratio:.1f}")
    
    # Sample deduplicated data
    sampled_dedup = sample_deduplicated_data(deduplicated_data, target_dedup_size)
    
    # Combine datasets
    mixture = temperature_data + sampled_dedup
    
    # Shuffle the mixture
    random.shuffle(mixture)
    
    print(f"Final mixture size: {len(mixture)}")
    print(f"Actual ratio: {len(temperature_data)}:{len(sampled_dedup)} = {len(temperature_data)/len(mixture):.3f}:{len(sampled_dedup)/len(mixture):.3f}")
    
    return mixture


def save_mixture_dataset(mixture_data: List[Dict[str, Any]], 
                        output_dir: str, 
                        samples_per_file: int = 10000):
    """Save mixture dataset to multiple JSON files."""
    os.makedirs(output_dir, exist_ok=True)
    
    num_files = math.ceil(len(mixture_data) / samples_per_file)
    
    for i in range(num_files):
        start_idx = i * samples_per_file
        end_idx = min((i + 1) * samples_per_file, len(mixture_data))
        
        file_data = mixture_data[start_idx:end_idx]
        filename = f"mixture_data_{i:05d}.json"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(file_data, f, indent=2)
        
        print(f"Saved {len(file_data)} samples to {filepath}")
    
    print(f"Saved mixture dataset to {output_dir} ({num_files} files)")


def main():
    parser = argparse.ArgumentParser(description="Create temperature mixture datasets")
    parser.add_argument("--base-dir", default="/Users/milano/FOF", 
                       help="Base directory containing generated_data_RL1 and deduplicated_data")
    parser.add_argument("--output-dir", required=True,
                       help="Output directory for mixture datasets")
    parser.add_argument("--ratio", type=float, default=0.7, 
                       help="Ratio of temperature data to deduplicated data (default: 0.7)")
    parser.add_argument("--samples-per-file", type=int, default=10000,
                       help="Number of samples per output file (default: 10000)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--temperatures", nargs="+", 
                       default=["temperature_1", "temperature_1.25", "temperature_1.5", "temperature_2"],
                       help="Temperature directories to process")
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    # Load deduplicated data once (shared across all temperatures)
    deduplicated_path = os.path.join(args.base_dir, "deduplicated_data")
    print("Loading deduplicated data...")
    deduplicated_data = load_json_files(deduplicated_path)
    
    # Process each temperature directory
    for temp_dir in args.temperatures:
        print(f"\n{'='*60}")
        print(f"Processing {temp_dir}")
        print(f"{'='*60}")
        
        # Load temperature data
        temp_path = os.path.join(args.base_dir, "generated_data_RL1", temp_dir)
        if not os.path.exists(temp_path):
            print(f"Warning: {temp_path} does not exist, skipping...")
            continue
            
        temperature_data = load_json_files(temp_path)
        
        # Create mixture
        mixture_data = create_mixture_dataset(temperature_data, deduplicated_data, args.ratio)
        
        # Save mixture
        output_dir = os.path.join(args.base_dir, args.output_dir, f"{temp_dir}_mixture")
        save_mixture_dataset(mixture_data, output_dir, args.samples_per_file)
        
        print(f"Completed {temp_dir} -> {temp_dir}_mixture")


if __name__ == "__main__":
    main()

