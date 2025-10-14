#!/usr/bin/env python3
"""
シンプルな学習スクリプト（重複排除済みバッチデータ専用）
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

# プロジェクトルートをパスに追加
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, project_root)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DataParallel
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from src.core.transformer_classifier import (
    load_tokens_and_labels_from_token_py,
    CharTokenizer,
    TransformerClassifier,
    build_hierarchical_label_mappings,
    hierarchical_collate,
)
from src.core.state_encoder import parse_tactic_string, encode_prover_state, format_tactic_string
from src.core.parameter import (
    default_params, get_model_params, get_training_params, 
    get_system_params, get_hierarchical_labels, DeviceType
)
from src.training.inference_hierarchical import evaluate_inference_performance


class DeduplicatedDataDataset(Dataset):
    """重複排除済みバッチデータセット（シンプル版）"""
    
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
        
        # 重複排除済みバッチデータを読み込み
        self.data = self._load_batch_data(data_dir)
    
    def _load_batch_data(self, data_dir: str) -> List[Dict[str, Any]]:
        """重複排除済みバッチデータを読み込み"""
        data = []
        json_files = glob.glob(os.path.join(data_dir, "deduplicated_batch_*.json"))
        json_files.sort()  # バッチファイルを順序よく読み込み
        
        print(f"Found {len(json_files)} batch files in {data_dir}")
        
        for json_file in json_files:
            print(f"Loading {os.path.basename(json_file)}...")
            with open(json_file, 'r') as f:
                batch_data = json.load(f)
            
            # バッチデータを直接追加（既に重複排除済み）
            data.extend(batch_data)
        
        print(f"Loaded {len(data)} training examples")
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 入力をエンコード
        premises = item['premises']
        goal = item['goal']
        input_ids, attention_mask, segment_ids = self.tokenizer.encode(
            goal, premises, self.max_seq_len
        )
        
        # タクティクを解析
        tactic = item['tactic']
        if isinstance(tactic, str):
            tactic_dict = parse_tactic_string(tactic)
        else:
            tactic_dict = tactic
        
        # ラベルを取得
        main_tactic = tactic_dict['main']
        arg1 = tactic_dict['arg1']
        arg2 = tactic_dict['arg2']
        
        # IDに変換
        main_label = self.main_to_id.get(main_tactic, 0)
        arg1_label = self.arg1_to_id.get(arg1, 0) if arg1 is not None else -1  # -1は無効値
        arg2_label = self.arg2_to_id.get(arg2, 0) if arg2 is not None else -1  # -1は無効値
        
        return input_ids, attention_mask, main_label, arg1_label, arg2_label


def train_epoch(
    model: TransformerClassifier,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    arg1_loss_weight: float = 0.8,
    arg2_loss_weight: float = 0.8,
    use_amp: bool = False,
    scaler: GradScaler = None,
    gradient_accumulation_steps: int = 1,
    use_wandb: bool = False,
    epoch: int = 0,
    log_frequency: int = 1000
) -> float:
    """1エポックの学習を実行"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc="Training", unit="batch")
    
    for batch_idx, batch in enumerate(pbar):
        input_ids, attention_mask, main_labels, arg1_labels, arg2_labels = batch
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        main_labels = main_labels.to(device)
        arg1_labels = arg1_labels.to(device)
        arg2_labels = arg2_labels.to(device)
        
        # オプティマイザーの勾配をリセット
        optimizer.zero_grad()
        
        # 混合精度での推論
        if use_amp and scaler is not None:
            with autocast():
                # モデル推論
                main_logits, arg1_logits, arg2_logits = model(input_ids, attention_mask)
                
                # シンプルな損失計算（無効値-1を除外）
                main_loss = criterion(main_logits, main_labels)
                
                # arg1の損失計算（無効値-1を除外）
                arg1_valid_mask = arg1_labels != -1
                arg1_loss = 0.0
                if arg1_valid_mask.any():
                    arg1_loss = criterion(arg1_logits[arg1_valid_mask], arg1_labels[arg1_valid_mask])
                
                # arg2の損失計算（無効値-1を除外）
                arg2_valid_mask = arg2_labels != -1
                arg2_loss = 0.0
                if arg2_valid_mask.any():
                    arg2_loss = criterion(arg2_logits[arg2_valid_mask], arg2_labels[arg2_valid_mask])
                
                # 総損失（重み付き）
                total_loss_batch = main_loss + arg1_loss_weight * arg1_loss + arg2_loss_weight * arg2_loss
                total_loss_batch = total_loss_batch / gradient_accumulation_steps
            
            # 混合精度での逆伝播
            scaler.scale(total_loss_batch).backward()
        else:
            # 通常の推論
            main_logits, arg1_logits, arg2_logits = model(input_ids, attention_mask)
            
            # シンプルな損失計算（無効値-1を除外）
            main_loss = criterion(main_logits, main_labels)
            
            # arg1の損失計算（無効値-1を除外）
            arg1_valid_mask = arg1_labels != -1
            arg1_loss = 0.0
            if arg1_valid_mask.any():
                arg1_loss = criterion(arg1_logits[arg1_valid_mask], arg1_labels[arg1_valid_mask])
            
            # arg2の損失計算（無効値-1を除外）
            arg2_valid_mask = arg2_labels != -1
            arg2_loss = 0.0
            if arg2_valid_mask.any():
                arg2_loss = criterion(arg2_logits[arg2_valid_mask], arg2_labels[arg2_valid_mask])
            
            # 総損失（重み付き）
            total_loss_batch = main_loss + arg1_loss_weight * arg1_loss + arg2_loss_weight * arg2_loss
            total_loss_batch = total_loss_batch / gradient_accumulation_steps
            
            # 通常の逆伝播
            total_loss_batch.backward()
        
        # 勾配累積のステップが完了したらオプティマイザーを更新
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            if use_amp and scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
        
        total_loss += total_loss_batch.item() * gradient_accumulation_steps
        num_batches += 1
        
        # プログレスバーを更新
        pbar.set_postfix({'Loss': f'{total_loss / num_batches:.4f}'})
        
        # 指定された頻度でwandbにログ
        if use_wandb and WANDB_AVAILABLE and batch_idx % log_frequency == 0:
            wandb.log({
                "batch_loss": total_loss_batch.item() * gradient_accumulation_steps,
                "running_avg_loss": total_loss / num_batches,
                "batch": epoch * len(dataloader) + batch_idx,
                "epoch": epoch + 1
            })
    
    return total_loss / num_batches if num_batches > 0 else 0.0


def main():
    parser = argparse.ArgumentParser(description="Train hierarchical tactic classifier with deduplicated data")
    parser.add_argument("--data_dir", type=str, default="deduplicated_data", help="deduplicated data directory")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="learning rate")
    parser.add_argument("--num_epochs", type=int, default=1, help="number of epochs")
    parser.add_argument("--device", type=str, default="auto", help="device")
    parser.add_argument("--save_path", type=str, default="models/hierarchical_model_generated.pth", help="model save path")
    parser.add_argument("--max_seq_len", type=int, default=512, help="maximum sequence length")
    parser.add_argument("--use_wandb", action="store_true", help="use wandb for logging")
    parser.add_argument("--wandb_project", type=str, default="fof-training", help="wandb project name")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="wandb run name")
    parser.add_argument("--arg1_loss_weight", type=float, default=0.8, help="weight for arg1 loss")
    parser.add_argument("--arg2_loss_weight", type=float, default=0.8, help="weight for arg2 loss")
    parser.add_argument("--random_seed", type=int, default=42, help="random seed for reproducibility")
    parser.add_argument("--save_checkpoints", action="store_true", help="save model checkpoint after each epoch")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="directory to save checkpoints")
    
    # 並列化関連の引数
    parser.add_argument("--num_workers", type=int, default=1, help="number of data loading workers")
    parser.add_argument("--use_data_parallel", action="store_true", help="use DataParallel for multi-GPU training")
    parser.add_argument("--gpu_ids", type=str, default=None, help="comma-separated GPU IDs to use")
    parser.add_argument("--use_amp", action="store_true", help="use Automatic Mixed Precision")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="number of gradient accumulation steps")
    
    # 推論評価関連の引数
    parser.add_argument("--inference_eval_examples", type=int, default=50, help="number of examples for inference evaluation")
    parser.add_argument("--inference_max_steps", type=int, default=30, help="max steps for inference evaluation")
    parser.add_argument("--inference_temperature", type=float, default=1.0, help="temperature for inference evaluation")
    
    # ログ関連の引数
    parser.add_argument("--log_frequency", type=int, default=1000, help="log training loss every n batches (default: 1000)")
    
    args = parser.parse_args()
    
    # 実行されたコマンドライン引数をログ出力
    print("🚀 Command line arguments:")
    print(f"   Script: {sys.argv[0]}")
    print(f"   Arguments: {' '.join(sys.argv[1:])}")
    print("=" * 60)
    
    # 再現性のためのシード設定
    import random
    import numpy as np
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.random_seed)
        torch.cuda.manual_seed_all(args.random_seed)
    
    # パラメータを初期化
    model_params = get_model_params()
    training_params = get_training_params()
    system_params = get_system_params()
    hierarchical_labels = get_hierarchical_labels()
    
    # デバイス設定
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # 実行設定の詳細をログ出力
    print("\n📋 Training Configuration:")
    print(f"   Data directory: {args.data_dir}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Learning rate: {args.learning_rate}")
    print(f"   Number of epochs: {args.num_epochs}")
    print(f"   Max sequence length: {args.max_seq_len}")
    print(f"   Number of workers: {args.num_workers}")
    print(f"   Use AMP: {args.use_amp}")
    print(f"   Use DataParallel: {args.use_data_parallel}")
    print(f"   GPU IDs: {args.gpu_ids}")
    print(f"   Gradient accumulation steps: {args.gradient_accumulation_steps}")
    print(f"   Random seed: {args.random_seed}")
    print(f"   Save checkpoints: {args.save_checkpoints}")
    print(f"   Use wandb: {args.use_wandb}")
    if args.use_wandb:
        print(f"   Wandb project: {args.wandb_project}")
        print(f"   Wandb run name: {args.wandb_run_name}")
    print("=" * 60)
    
    # 混合精度のスケーラー初期化
    scaler = None
    use_amp = args.use_amp and device.type == 'cuda'
    if use_amp:
        scaler = GradScaler()
        print("Using Automatic Mixed Precision (AMP)")
    
    # データディレクトリの設定
    data_dir = os.path.join(project_root, args.data_dir)
    
    # データディレクトリの存在確認
    if not os.path.exists(data_dir):
        print(f"❌ Data directory not found: {data_dir}")
        print("Please run: python src/training/deduplicate_generated_data.py")
        return
    
    # バッチファイルの存在確認
    batch_files = glob.glob(os.path.join(data_dir, "deduplicated_batch_*.json"))
    if not batch_files:
        print(f"❌ No deduplicated batch files found in {data_dir}")
        print("Please run: python src/training/deduplicate_generated_data.py")
        return
    
    print(f"✅ Using deduplicated data from: {data_dir}")
    print(f"   Found {len(batch_files)} batch files")
    
    # トークンとラベルを読み込み
    token_py_path = os.path.join(project_root, "src", "core", "fof_tokens.py")
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
    
    # データセットを作成
    print("📊 Creating DeduplicatedDataDataset")
    dataset = DeduplicatedDataDataset(
        data_dir=data_dir,
        tokenizer=tokenizer,
        main_to_id=main_to_id,
        arg1_to_id=arg1_to_id,
        arg2_to_id=arg2_to_id,
        max_seq_len=args.max_seq_len
    )
    
    if len(dataset) == 0:
        print("No training data found. Please check the deduplicated data directory.")
        return
    
    # データローダーを作成
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False,
        collate_fn=hierarchical_collate
    )
    
    # モデルを作成
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
    
    # モデルをデバイスに移動
    model = model.to(device)
    
    # 並列化設定
    if args.use_data_parallel and device.type == 'cuda':
        if args.gpu_ids:
            if args.gpu_ids == "all":
                gpu_ids = list(range(torch.cuda.device_count()))
            else:
                gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(",")]
            print(f"Using GPU IDs: {gpu_ids}")
        else:
            gpu_ids = None
        
        if gpu_ids and len(gpu_ids) > 1:
            model = DataParallel(model, device_ids=gpu_ids)
            print(f"Using DataParallel with GPU IDs: {gpu_ids}")
        else:
            print(f"Only one GPU specified, using single GPU")
    else:
        gpu_ids = None
    
    # オプティマイザーと損失関数を作成
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    # チェックポイントディレクトリの作成
    if args.save_checkpoints:
        checkpoint_dir = os.path.join(project_root, args.checkpoint_dir)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
            print(f"Created checkpoint directory: {checkpoint_dir}")
        else:
            print(f"Using existing checkpoint directory: {checkpoint_dir}")
    
    # wandb初期化
    if args.use_wandb and WANDB_AVAILABLE:
        run_name = args.wandb_run_name or f"training_{int(time.time())}"
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config={
                "data_dir": args.data_dir,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "num_epochs": args.num_epochs,
                "max_seq_len": args.max_seq_len,
                "device": str(device),
                "save_checkpoints": args.save_checkpoints,
                "checkpoint_dir": args.checkpoint_dir
            }
        )
        print(f"Wandb initialized: {args.wandb_project}/{run_name}")
    elif args.use_wandb and not WANDB_AVAILABLE:
        print("Warning: wandb requested but not available. Continuing without logging.")
    
    # ラベルマッピングを作成（推論評価用）
    label_mappings = {
        'main_to_id': main_to_id,
        'arg1_to_id': arg1_to_id,
        'arg2_to_id': arg2_to_id,
        'id_to_main': id_to_main,
        'id_to_arg1': id_to_arg1,
        'id_to_arg2': id_to_arg2
    }
    
    # 学習ループ
    print(f"\n🚀 Starting training for {args.num_epochs} epochs...")
    print(f"📊 Training data: {len(dataset)} examples")
    print(f"📊 Batch size: {args.batch_size}")
    print(f"📊 Learning rate: {args.learning_rate}")
    print(f"📊 Log frequency: every {args.log_frequency} batches")
    print("=" * 60)
    
    # 学習開始前のベースライン推論評価
    print(f"\n🔍 Evaluating baseline inference performance (before training)...")
    baseline_success_rate, baseline_avg_steps = evaluate_inference_performance(
        model, tokenizer, label_mappings, device, args.max_seq_len,
        num_examples=args.inference_eval_examples, 
        max_steps=args.inference_max_steps, 
        temperature=args.inference_temperature
    )
    print(f"  Baseline inference success rate: {baseline_success_rate:.3f}")
    print(f"  Baseline inference avg steps (when solved): {baseline_avg_steps:.2f}")
    
    # ベースライン結果をwandbに記録
    if args.use_wandb and WANDB_AVAILABLE:
        wandb.log({
            "inference/success_rate": baseline_success_rate,
            "inference/avg_steps": baseline_avg_steps,
            "epoch": 0  # 学習前なのでepoch 0
        })
    
    print("=" * 60)
    
    for epoch in range(args.num_epochs):
        print(f"\n🚀 Starting epoch {epoch+1}/{args.num_epochs}")
        
        # 1エポックの学習を実行
        avg_loss = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            arg1_loss_weight=args.arg1_loss_weight,
            arg2_loss_weight=args.arg2_loss_weight,
            use_amp=use_amp,
            scaler=scaler,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            use_wandb=args.use_wandb and WANDB_AVAILABLE,
            epoch=epoch,
            log_frequency=args.log_frequency
        )
        
        print(f"Epoch {epoch+1} completed. Average loss: {avg_loss:.4f}")
        
        # 推論性能を評価（毎エポック）
        if True:  # 毎エポック実行
            print(f"\n🔍 Evaluating inference performance after epoch {epoch+1}...")
            inference_success_rate, inference_avg_steps = evaluate_inference_performance(
                model, tokenizer, label_mappings, device, args.max_seq_len,
                num_examples=args.inference_eval_examples, 
                max_steps=args.inference_max_steps, 
                temperature=args.inference_temperature
            )
            print(f"  Inference success rate: {inference_success_rate:.3f}")
            print(f"  Inference avg steps (when solved): {inference_avg_steps:.2f}")
        else:
            inference_success_rate = None
            inference_avg_steps = None
        
        # wandbにログ
        if args.use_wandb and WANDB_AVAILABLE:
            log_data = {
                "loss": avg_loss
            }
            if inference_success_rate is not None:
                log_data.update({
                    "inference/success_rate": inference_success_rate,
                    "inference/avg_steps": inference_avg_steps
                })
            wandb.log(log_data)
        
        # チェックポイント保存
        if args.save_checkpoints:
            checkpoint_path = os.path.join(args.checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth")
            os.makedirs(args.checkpoint_dir, exist_ok=True)
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")
    
    # モデルを保存
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    torch.save(model.state_dict(), args.save_path)
    
    print(f"\n🎉 Training completed!")
    print(f"📁 Model saved to: {args.save_path}")
    
    # wandb終了
    if args.use_wandb and WANDB_AVAILABLE:
        wandb.finish()
        print("📈 Wandb logging completed!")


if __name__ == "__main__":
    main()
