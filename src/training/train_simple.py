#!/usr/bin/env python3
"""
Simple non-batch training script (for deduplicated data only)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import glob
import time
from typing import List, Tuple, Dict, Any

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Install with: pip install wandb")

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, project_root)

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from tqdm import tqdm

from src.core.transformer_classifier import (
    load_tokens_and_labels_from_token_py,
    CharTokenizer,
    TransformerClassifier,
    build_hierarchical_label_mappings,
)
from src.core.state_encoder import parse_tactic_string
from src.core.parameter import (
    get_model_params, get_training_params, 
    get_system_params, get_hierarchical_labels
)


class SimpleDataset(Dataset):
    """Simple dataset (without batching)"""
    
    def __init__(
        self, 
        data_dir: str,
        tokenizer: CharTokenizer,
        main_to_id: Dict[str, int],
        arg1_to_id: Dict[str, int], 
        arg2_to_id: Dict[str, int],
        max_seq_len: int = 512
    ):
        self.tokenizer = tokenizer
        self.main_to_id = main_to_id
        self.arg1_to_id = arg1_to_id
        self.arg2_to_id = arg2_to_id
        self.max_seq_len = max_seq_len
        
        # Load data
        self.data = self._load_batch_data(data_dir)
    
    def _load_batch_data(self, data_dir: str) -> List[Dict[str, Any]]:
        """Load data（ディレクトリ配下のallのJSONファイル）"""
        data = []
        
        # Find all JSON files in directory
        json_files = glob.glob(os.path.join(data_dir, "*.json"))
        json_files.sort()  # Load files in order
        
        print(f"Found {len(json_files)} JSON files in {data_dir}")
        
        for json_file in json_files:
            print(f"Loading {os.path.basename(json_file)}...")
            with open(json_file, 'r') as f:
                file_data = json.load(f)
            
            # Process according to data format
            if isinstance(file_data, list):
                # List format case (deduplicated batch data or steps format)
                if file_data and isinstance(file_data[0], dict):
                    # If each element is a dictionary
                    if 'premises' in file_data[0] and 'goal' in file_data[0] and 'tactic' in file_data[0]:
                        # If already in training format
                        data.extend(file_data)
                    else:
                        # If in steps format, extract each step
                        for example in file_data:
                            steps = example.get('steps', [])
                            for step in steps:
                                data.append(step)
                else:
                    # If empty list
                    continue
            else:
                # If single dictionary format
                if isinstance(file_data, dict) and 'steps' in file_data:
                    steps = file_data.get('steps', [])
                    for step in steps:
                        data.append(step)
        
        print(f"Loaded {len(data)} training examples")
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Encode input
        premises = item['premises']
        goal = item['goal']
        input_ids, attention_mask, segment_ids = self.tokenizer.encode(
            goal, premises, self.max_seq_len
        )
        
        # Parse tactic
        tactic = item['tactic']
        if isinstance(tactic, str):
            tactic_dict = parse_tactic_string(tactic)
        else:
            tactic_dict = tactic
        
        # Get label
        main_tactic = tactic_dict['main']
        arg1 = tactic_dict['arg1']
        arg2 = tactic_dict['arg2']
        
        # Convert to ID
        main_label = self.main_to_id.get(main_tactic, 0)
        arg1_label = self.arg1_to_id.get(arg1, 0) if arg1 is not None else -1  # -1はInvalid value
        arg2_label = self.arg2_to_id.get(arg2, 0) if arg2 is not None else -1  # -1はInvalid value
        
        return input_ids, attention_mask, main_label, arg1_label, arg2_label


def train_single_example(
    model: TransformerClassifier,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    main_label: int,
    arg1_label: int,
    arg2_label: int,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    arg1_loss_weight: float = 0.8,
    arg2_loss_weight: float = 0.8,
) -> float:
    """Execute training with single example"""
    model.train()
    
    # Reset optimizer gradients
    optimizer.zero_grad()
    
    # Model inference
    main_logits, arg1_logits, arg2_logits = model(input_ids, attention_mask)
    
    # 損失計算（Invalid value-1 除外）
    main_loss = criterion(main_logits, torch.tensor([main_label], device=main_logits.device))
    
    # arg1の損失計算（Invalid value-1 除外）
    arg1_loss = 0.0
    if arg1_label != -1:
        arg1_loss = criterion(arg1_logits, torch.tensor([arg1_label], device=arg1_logits.device))
    
    # arg2の損失計算（Invalid value-1 除外）
    arg2_loss = 0.0
    if arg2_label != -1:
        arg2_loss = criterion(arg2_logits, torch.tensor([arg2_label], device=arg2_logits.device))
    
    # Total loss (weighted)
    total_loss = main_loss + arg1_loss_weight * arg1_loss + arg2_loss_weight * arg2_loss
    
    # Backpropagation
    total_loss.backward()
    optimizer.step()
    
    return total_loss.item()


def main():
    parser = argparse.ArgumentParser(description="Simple training script for hierarchical tactic classifier")
    parser.add_argument("--data_dir", type=str, default="deduplicated_data", help="deduplicated data directory")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="learning rate")
    parser.add_argument("--num_epochs", type=int, default=1, help="number of epochs")
    parser.add_argument("--device", type=str, default="auto", help="device")
    parser.add_argument("--save_path", type=str, default=None, help="model save path (auto-generated from data_dir if not specified)")
    parser.add_argument("--max_seq_len", type=int, default=256, help="maximum sequence length")
    parser.add_argument("--use_wandb", action="store_true", help="use wandb for logging")
    parser.add_argument("--wandb_project", type=str, default="fof-simple-training", help="wandb project name")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="wandb run name")
    parser.add_argument("--arg1_loss_weight", type=float, default=0.8, help="weight for arg1 loss")
    parser.add_argument("--arg2_loss_weight", type=float, default=0.8, help="weight for arg2 loss")
    parser.add_argument("--random_seed", type=int, default=42, help="random seed for reproducibility")
    parser.add_argument("--log_frequency", type=int, default=1000, help="log training loss every n examples")
    parser.add_argument("--save_checkpoints", action="store_true", help="save model checkpoint after each epoch")
    parser.add_argument("--load_model_path", type=str, default=None, help="path to pretrained model to load")
    
    args = parser.parse_args()
    
    # Log executed command line arguments
    print("🚀 Command line arguments:")
    print(f"   Script: {sys.argv[0]}")
    print(f"   Arguments: {' '.join(sys.argv[1:])}")
    print("=" * 60)
    
    # Set seed for reproducibility
    import random
    import numpy as np
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.random_seed)
        torch.cuda.manual_seed_all(args.random_seed)
    
    # Initialize parameters
    model_params = get_model_params()
    training_params = get_training_params()
    system_params = get_system_params()
    hierarchical_labels = get_hierarchical_labels()
    
    # Device setup
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Log execution configuration details
    print("\n📋 Training Configuration:")
    print(f"   Data directory: {args.data_dir}")
    print(f"   Learning rate: {args.learning_rate}")
    print(f"   Number of epochs: {args.num_epochs}")
    print(f"   Max sequence length: {args.max_seq_len}")
    print(f"   Random seed: {args.random_seed}")
    print(f"   Use wandb: {args.use_wandb}")
    if args.use_wandb:
        print(f"   Wandb project: {args.wandb_project}")
        print(f"   Wandb run name: {args.wandb_run_name}")
    print("=" * 60)
    
    # Data directory configuration
    data_dir = os.path.join(project_root, args.data_dir)
    
    # Auto-generate save path (if not specified)
    if args.save_path is None:
        # Generate save name from data directory name
        data_dir_name = os.path.basename(args.data_dir.rstrip('/'))
        args.save_path = f"models/{data_dir_name}.pth"
        print(f"Auto-generated save path: {args.save_path}")
    
    # Check data directory exists
    if not os.path.exists(data_dir):
        print(f"❌ Data directory not found: {data_dir}")
        print("Please run: python src/training/deduplicate_generated_data.py")
        return
    
    # Check data files exist
    json_files = glob.glob(os.path.join(data_dir, "*.json"))
    if not json_files:
        print(f"❌ No JSON files found in {data_dir}")
        print("Please ensure the directory contains JSON files with training data")
        return
    
    print(f"✅ Using data from: {data_dir}")
    print(f"   Found {len(json_files)} JSON files")
    
    # Load tokens and labels
    token_py_path = os.path.join(project_root, "src", "core", "fof_tokens.py")
    base_tokens, _ = load_tokens_and_labels_from_token_py(token_py_path)
    
    # Build label mapping for hierarchical classification
    main_to_id, arg1_to_id, arg2_to_id, id_to_main, id_to_arg1, id_to_arg2 = build_hierarchical_label_mappings(
        hierarchical_labels.main_tactics,
        hierarchical_labels.arg1_values,
        hierarchical_labels.arg2_values
    )
    
    print(f"Main tactics: {len(id_to_main)} classes")
    print(f"Arg1 values: {len(id_to_arg1)} classes")
    print(f"Arg2 values: {len(id_to_arg2)} classes")
    
    # Create tokenizer
    tokenizer = CharTokenizer(
        base_tokens=base_tokens,
        add_tactic_tokens=model_params.add_tactic_tokens,
        num_tactic_tokens=model_params.num_tactic_tokens
    )
    
    # Create dataset
    print("📊 Creating SimpleDataset")
    dataset = SimpleDataset(
        data_dir=data_dir,
        tokenizer=tokenizer,
        main_to_id=main_to_id,
        arg1_to_id=arg1_to_id,
        arg2_to_id=arg2_to_id,
        max_seq_len=args.max_seq_len
    )
    
    if len(dataset) == 0:
        print("No training data found. Please check the data directory.")
        return
    
    # Create model
    model = TransformerClassifier(
        vocab_size=tokenizer.vocab_size,
        pad_id=tokenizer.pad_id,
        max_seq_len=args.max_seq_len,
        d_model=model_params.d_model,
        nhead=model_params.nhead,
        num_layers=model_params.num_layers,
        dim_feedforward=model_params.dim_feedforward,
        dropout=model_params.dropout,
        num_main_classes=len(id_to_main),
        num_arg1_classes=len(id_to_arg1),
        num_arg2_classes=len(id_to_arg2),
    )
    
    print(f"Model vocab_size: {tokenizer.vocab_size}")
    print(f"Model pad_id: {tokenizer.pad_id}")
    
    # Load pretrained model (if specified)
    if args.load_model_path:
        load_model_path = os.path.join(project_root, args.load_model_path)
        if os.path.exists(load_model_path):
            print(f"🔄 Loading pretrained model from: {load_model_path}")
            try:
                # Load state_dict
                state_dict = torch.load(load_model_path, map_location=device)
                
                # ModelのLoad state_dict
                model.load_state_dict(state_dict)
                print("✅ Pretrained model loaded successfully!")
                
                # Check model structure compatibility
                print(f"   Loaded model vocab_size: {model.vocab_size}")
                print(f"   Loaded model pad_id: {model.pad_id}")
                print(f"   Loaded model num_main_classes: {model.num_main_classes}")
                print(f"   Loaded model num_arg1_classes: {model.num_arg1_classes}")
                print(f"   Loaded model num_arg2_classes: {model.num_arg2_classes}")
                
            except Exception as e:
                print(f"❌ Error loading pretrained model: {e}")
                print("   Continuing with randomly initialized model...")
        else:
            print(f"❌ Pretrained model not found: {load_model_path}")
            print("   Continuing with randomly initialized model...")
    else:
        print("🆕 Using randomly initialized model")
    
    # Move model to device
    model = model.to(device)
    
    # Create optimizer and loss function
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    # Initialize wandb
    if args.use_wandb and WANDB_AVAILABLE:
        run_name = args.wandb_run_name or f"simple_training_{int(time.time())}"
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config={
                "data_dir": args.data_dir,
                "learning_rate": args.learning_rate,
                "num_epochs": args.num_epochs,
                "max_seq_len": args.max_seq_len,
                "device": str(device),
            }
        )
        print(f"Wandb initialized: {args.wandb_project}/{run_name}")
    elif args.use_wandb and not WANDB_AVAILABLE:
        print("Warning: wandb requested but not available. Continuing without logging.")
    
    # Create label mapping (for inference evaluation)
    label_mappings = {
        'main_to_id': main_to_id,
        'arg1_to_id': arg1_to_id,
        'arg2_to_id': arg2_to_id,
        'id_to_main': id_to_main,
        'id_to_arg1': id_to_arg1,
        'id_to_arg2': id_to_arg2
    }
    
    # Training loop
    print(f"\n🚀 Starting simple training for {args.num_epochs} epochs...")
    print(f"📊 Training data: {len(dataset)} examples")
    print(f"📊 Learning rate: {args.learning_rate}")
    print(f"📊 Log frequency: every {args.log_frequency} examples")
    print("=" * 60)
    
    # Training loop
    total_examples = 0
    epoch_losses = []
    
    # Queue to record recent log_frequency losses
    recent_losses = []
    
    for epoch in range(args.num_epochs):
        print(f"\n🚀 Starting epoch {epoch+1}/{args.num_epochs}")
        
        # Record loss within epoch
        epoch_loss = 0.0
        num_examples = 0
        
        # Shuffle data
        indices = list(range(len(dataset)))
        random.shuffle(indices)
        
        # Create progress bar
        pbar = tqdm(indices, desc=f"Epoch {epoch+1}", unit="example")
        
        for example_idx, data_idx in enumerate(pbar):
            # Get data
            input_ids, attention_mask, main_label, arg1_label, arg2_label = dataset[data_idx]
            
            # Convert to tensor and move to device
            input_ids = input_ids.unsqueeze(0).to(device)  # Add batch dimension
            attention_mask = attention_mask.unsqueeze(0).to(device)  # Add batch dimension
            
            # 単一の例 with/at Training
            loss = train_single_example(
                model=model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                main_label=main_label,
                arg1_label=arg1_label,
                arg2_label=arg2_label,
                optimizer=optimizer,
                criterion=criterion,
                arg1_loss_weight=args.arg1_loss_weight,
                arg2_loss_weight=args.arg2_loss_weight,
            )
            
            epoch_loss += loss
            num_examples += 1
            total_examples += 1
            
            # Record recent log_frequency losses
            recent_losses.append(loss)
            if len(recent_losses) > args.log_frequency:
                recent_losses.pop(0)  # Remove old loss
            
            # Update progress bar
            pbar.set_postfix({'Loss': f'{loss:.4f}', 'Avg Loss': f'{epoch_loss / num_examples:.4f}'})
            
            # Log to wandb at specified frequency
            if args.use_wandb and WANDB_AVAILABLE and total_examples % args.log_frequency == 0:
                recent_avg_loss = sum(recent_losses) / len(recent_losses) if recent_losses else 0.0
                wandb.log({
                    "loss": recent_avg_loss  # Average of recent log_frequency
                })
            
        
        # Calculate epoch average loss
        avg_epoch_loss = epoch_loss / num_examples if num_examples > 0 else 0.0
        epoch_losses.append(avg_epoch_loss)
        
        print(f"Epoch {epoch+1} completed. Average loss: {avg_epoch_loss:.4f}")
        
        # Save model after each epoch
        if args.save_checkpoints:
            checkpoint_path = f"models/simple_model_epoch_{epoch+1}.pth"
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_epoch_loss,
            }, checkpoint_path)
            print(f"💾 Epoch checkpoint saved: {checkpoint_path}")
        
        # Log to wandb
        if args.use_wandb and WANDB_AVAILABLE:
            wandb.log({
                "epoch": epoch + 1,
                "epoch_loss": avg_epoch_loss
            })
    
    # Save final model
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    torch.save(model.state_dict(), args.save_path)
    
    print(f"\n🎉 Training completed!")
    print(f"📁 Model saved to: {args.save_path}")
    print(f"📊 Total examples processed: {total_examples}")
    print(f"📊 Average loss per epoch: {[f'{loss:.4f}' for loss in epoch_losses]}")
    
    # Terminate wandb
    if args.use_wandb and WANDB_AVAILABLE:
        wandb.finish()
        print("📈 Wandb logging completed!")


if __name__ == "__main__":
    main()
