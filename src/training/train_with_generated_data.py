#!/usr/bin/env python3
"""
並列化バッチ学習スクリプト（重複排除済みデータ専用）
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


class BatchDataset(Dataset):
    """バッチデータセット（並列化対応版）"""
    
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
        """データを読み込み（バッチファイルまたは通常のJSONファイル）"""
        data = []
        
        # まずバッチファイルを探す
        batch_files = glob.glob(os.path.join(data_dir, "deduplicated_batch_*.json"))
        if batch_files:
            # バッチファイルが存在する場合
            batch_files.sort()
            print(f"Found {len(batch_files)} batch files in {data_dir}")
            
            for json_file in batch_files:
                print(f"Loading {os.path.basename(json_file)}...")
                with open(json_file, 'r') as f:
                    batch_data = json.load(f)
                data.extend(batch_data)
        else:
            # バッチファイルがない場合は通常のJSONファイルを探す
            json_files = glob.glob(os.path.join(data_dir, "*.json"))
            json_files.sort()
            print(f"Found {len(json_files)} JSON files in {data_dir}")
            
            for json_file in json_files:
                print(f"Loading {os.path.basename(json_file)}...")
                with open(json_file, 'r') as f:
                    file_data = json.load(f)
                
                # データの形式に応じて処理
                if isinstance(file_data, list):
                    if file_data and isinstance(file_data[0], dict):
                        if 'premises' in file_data[0] and 'goal' in file_data[0] and 'tactic' in file_data[0]:
                            data.extend(file_data)
                        else:
                            for example in file_data:
                                steps = example.get('steps', [])
                                for step in steps:
                                    data.append(step)
                elif isinstance(file_data, dict) and 'steps' in file_data:
                    steps = file_data.get('steps', [])
                    for step in steps:
                        data.append(step)
        
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
    entropy_reg_weight: float = 0.0,
    kl_penalty_weight: float = 0.0,
    reference_model: TransformerClassifier | DataParallel | None = None,
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
    
    # 直近1000バッチの損失を記録するためのキュー
    recent_losses = []
    recent_entropy_reg_losses = []
    recent_kl_penalties = []

    pbar = tqdm(dataloader, desc="Training", unit="batch")

    def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
        """Compute mean entropy for the provided logits."""
        if logits.numel() == 0:
            return torch.tensor(0.0, device=device)
        log_probs = torch.log_softmax(logits, dim=-1)
        probs = torch.softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1)
        return entropy.mean()

    def compute_kl_to_reference(
        student_logits: torch.Tensor,
        reference_logits: torch.Tensor | None,
    ) -> torch.Tensor:
        """Compute mean KL divergence from student predictions to reference model."""
        if reference_logits is None or reference_logits.numel() == 0:
            return torch.tensor(0.0, device=device)
        student_log_probs = torch.log_softmax(student_logits, dim=-1)
        student_probs = torch.softmax(student_logits, dim=-1)
        reference_log_probs = torch.log_softmax(reference_logits, dim=-1)
        kl = (student_probs * (student_log_probs - reference_log_probs)).sum(dim=-1)
        return kl.mean()

    def compute_batch_loss(
        main_logits,
        arg1_logits,
        arg2_logits,
        reference_main_logits=None,
        reference_arg1_logits=None,
        reference_arg2_logits=None,
    ):
        main_loss = criterion(main_logits, main_labels)

        arg1_valid_mask = arg1_labels != -1
        if arg1_valid_mask.any():
            arg1_loss = criterion(arg1_logits[arg1_valid_mask], arg1_labels[arg1_valid_mask])
        else:
            arg1_loss = torch.tensor(0.0, device=device)

        arg2_valid_mask = arg2_labels != -1
        if arg2_valid_mask.any():
            arg2_loss = criterion(arg2_logits[arg2_valid_mask], arg2_labels[arg2_valid_mask])
        else:
            arg2_loss = torch.tensor(0.0, device=device)

        total_loss_local = main_loss + arg1_loss_weight * arg1_loss + arg2_loss_weight * arg2_loss

        entropy_loss = torch.tensor(0.0, device=device)
        if entropy_reg_weight != 0.0:
            entropy_terms = [compute_entropy(main_logits)]
            if arg1_valid_mask.any():
                entropy_terms.append(compute_entropy(arg1_logits[arg1_valid_mask]))
            if arg2_valid_mask.any():
                entropy_terms.append(compute_entropy(arg2_logits[arg2_valid_mask]))
            entropy_loss = -torch.stack(entropy_terms).mean()
            total_loss_local = total_loss_local + entropy_reg_weight * entropy_loss

        kl_penalty_loss = torch.tensor(0.0, device=device)
        if kl_penalty_weight != 0.0:
            kl_terms = [
                compute_kl_to_reference(main_logits, reference_main_logits)
            ]
            if arg1_valid_mask.any():
                kl_terms.append(
                    compute_kl_to_reference(
                        arg1_logits[arg1_valid_mask],
                        None
                        if reference_arg1_logits is None
                        else reference_arg1_logits[arg1_valid_mask],
                    )
                )
            if arg2_valid_mask.any():
                kl_terms.append(
                    compute_kl_to_reference(
                        arg2_logits[arg2_valid_mask],
                        None
                        if reference_arg2_logits is None
                        else reference_arg2_logits[arg2_valid_mask],
                    )
                )
            kl_penalty_loss = torch.stack(kl_terms).mean()
            total_loss_local = total_loss_local + kl_penalty_weight * kl_penalty_loss

        loss_components = {
            "main_loss": main_loss.detach(),
            "arg1_loss": arg1_loss.detach(),
            "arg2_loss": arg2_loss.detach(),
            "entropy_reg": (entropy_reg_weight * entropy_loss).detach() if entropy_reg_weight != 0.0 else torch.tensor(0.0, device=device),
            "kl_penalty": (kl_penalty_weight * kl_penalty_loss).detach() if kl_penalty_weight != 0.0 else torch.tensor(0.0, device=device),
        }

        return total_loss_local, loss_components

    for batch_idx, batch in enumerate(pbar):
        input_ids, attention_mask, main_labels, arg1_labels, arg2_labels = batch
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        main_labels = main_labels.to(device)
        arg1_labels = arg1_labels.to(device)
        arg2_labels = arg2_labels.to(device)

        # オプティマイザーの勾配をリセット
        optimizer.zero_grad()

        reference_main_logits = None
        reference_arg1_logits = None
        reference_arg2_logits = None
        if reference_model is not None and kl_penalty_weight != 0.0:
            with torch.no_grad():
                reference_outputs = reference_model(input_ids, attention_mask)
            if isinstance(reference_outputs, tuple) and len(reference_outputs) == 3:
                reference_main_logits, reference_arg1_logits, reference_arg2_logits = reference_outputs
            else:
                raise ValueError("Reference model must return a tuple of three logits")

        # 混合精度での推論
        if use_amp and scaler is not None:
            with autocast():
                # モデル推論
                main_logits, arg1_logits, arg2_logits = model(input_ids, attention_mask)

                # 正則化込みの損失計算
                total_loss_batch, loss_components = compute_batch_loss(
                    main_logits,
                    arg1_logits,
                    arg2_logits,
                    reference_main_logits,
                    reference_arg1_logits,
                    reference_arg2_logits,
                )
                total_loss_batch = total_loss_batch / gradient_accumulation_steps

            # 混合精度での逆伝播
            scaler.scale(total_loss_batch).backward()
        else:
            # 通常の推論
            main_logits, arg1_logits, arg2_logits = model(input_ids, attention_mask)

            # 正則化込みの損失計算
            total_loss_batch, loss_components = compute_batch_loss(
                main_logits,
                arg1_logits,
                arg2_logits,
                reference_main_logits,
                reference_arg1_logits,
                reference_arg2_logits,
            )
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

        # 直近1000バッチの損失を記録
        current_loss = total_loss_batch.item() * gradient_accumulation_steps
        recent_losses.append(current_loss)
        if len(recent_losses) > 1000:
            recent_losses.pop(0)  # 古い損失を削除

        if entropy_reg_weight != 0.0:
            entropy_contribution = loss_components["entropy_reg"].item()
            recent_entropy_reg_losses.append(entropy_contribution)
            if len(recent_entropy_reg_losses) > 1000:
                recent_entropy_reg_losses.pop(0)

        if kl_penalty_weight != 0.0:
            kl_contribution = loss_components["kl_penalty"].item()
            recent_kl_penalties.append(kl_contribution)
            if len(recent_kl_penalties) > 1000:
                recent_kl_penalties.pop(0)

        # プログレスバーを更新
        pbar.set_postfix({'Loss': f'{total_loss / num_batches:.4f}'})

        # 指定された頻度でwandbにログ
        if use_wandb and WANDB_AVAILABLE and batch_idx % log_frequency == 0:
            recent_avg_loss = sum(recent_losses) / len(recent_losses) if recent_losses else 0.0
            log_payload = {
                "recent_avg_loss": recent_avg_loss
            }
            if entropy_reg_weight != 0.0 and recent_entropy_reg_losses:
                log_payload["recent_entropy_reg_loss"] = sum(recent_entropy_reg_losses) / len(recent_entropy_reg_losses)
            if kl_penalty_weight != 0.0 and recent_kl_penalties:
                log_payload["recent_kl_penalty"] = sum(recent_kl_penalties) / len(recent_kl_penalties)
            wandb.log(log_payload)

    return total_loss / num_batches if num_batches > 0 else 0.0


def main():
    parser = argparse.ArgumentParser(description="Parallel batch training script for hierarchical tactic classifier")
    parser.add_argument("--data_dir", type=str, default="deduplicated_data", help="data directory")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="learning rate")
    parser.add_argument("--num_epochs", type=int, default=1, help="number of epochs")
    parser.add_argument("--device", type=str, default="auto", help="device")
    parser.add_argument("--save_path", type=str, default=None, help="model save path (auto-generated from data_dir if not specified)")
    parser.add_argument("--max_seq_len", type=int, default=256, help="maximum sequence length")
    parser.add_argument("--use_wandb", action="store_true", help="use wandb for logging")
    parser.add_argument("--wandb_project", type=str, default="fof-parallel-training", help="wandb project name")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="wandb run name")
    parser.add_argument("--arg1_loss_weight", type=float, default=0.8, help="weight for arg1 loss")
    parser.add_argument("--arg2_loss_weight", type=float, default=0.8, help="weight for arg2 loss")
    parser.add_argument("--entropy_reg_weight", type=float, default=0.0, help="weight for entropy regularization (positive values penalize high entropy)")
    parser.add_argument("--kl_penalty_weight", type=float, default=0.0, help="weight for KL penalty against reference model")
    parser.add_argument(
        "--kl_reference_model_path",
        type=str,
        default=None,
        help="path to frozen reference model used for KL regularization",
    )
    parser.add_argument("--random_seed", type=int, default=42, help="random seed for reproducibility")
    parser.add_argument("--log_frequency", type=int, default=1000, help="log training loss every n batches")
    parser.add_argument("--save_checkpoints", action="store_true", help="save model checkpoint after each epoch")
    parser.add_argument("--load_model_path", type=str, default=None, help="path to pretrained model to load")
    
    # 並列化関連の引数（オプション）
    parser.add_argument("--num_workers", type=int, default=4, help="number of data loading workers")
    parser.add_argument("--use_data_parallel", action="store_true", help="use DataParallel for multi-GPU training")
    parser.add_argument("--gpu_ids", type=str, default=None, help="comma-separated GPU IDs to use")
    parser.add_argument("--use_amp", action="store_true", help="use Automatic Mixed Precision")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="number of gradient accumulation steps")
    
    # 推論評価関連の引数（オプション）
    parser.add_argument("--inference_eval_examples", type=int, default=100, help="number of examples for inference evaluation")
    parser.add_argument("--inference_max_steps", type=int, default=30, help="max steps for inference evaluation")
    parser.add_argument("--inference_temperature", type=float, default=1.0, help="temperature for inference evaluation")
    
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
    print(f"   Entropy regularization weight: {args.entropy_reg_weight}")
    print(f"   KL penalty weight: {args.kl_penalty_weight}")
    print(f"   KL reference model path: {args.kl_reference_model_path}")
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
    
    # 保存パスの自動生成（指定されていない場合）
    if args.save_path is None:
        # データディレクトリ名から保存名を生成
        data_dir_name = os.path.basename(args.data_dir.rstrip('/'))
        args.save_path = f"models/{data_dir_name}_parallel.pth"
        print(f"Auto-generated save path: {args.save_path}")
    
    # データディレクトリの存在確認
    if not os.path.exists(data_dir):
        print(f"❌ Data directory not found: {data_dir}")
        print("Please ensure the directory contains training data")
        return
    
    # データファイルの存在確認
    json_files = glob.glob(os.path.join(data_dir, "*.json"))
    if not json_files:
        print(f"❌ No JSON files found in {data_dir}")
        print("Please ensure the directory contains JSON files with training data")
        return
    
    print(f"✅ Using data from: {data_dir}")
    print(f"   Found {len(json_files)} JSON files")
    
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
    print("📊 Creating BatchDataset")
    dataset = BatchDataset(
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
    
    # 事前学習済みモデルを読み込み（指定されている場合）
    if args.load_model_path:
        load_model_path = os.path.join(project_root, args.load_model_path)
        if os.path.exists(load_model_path):
            print(f"🔄 Loading pretrained model from: {load_model_path}")
            try:
                # state_dictを読み込み
                state_dict = torch.load(load_model_path, map_location=device)
                
                # モデルのstate_dictを読み込み
                model.load_state_dict(state_dict)
                print("✅ Pretrained model loaded successfully!")
                
                # モデル構造の互換性をチェック
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

    reference_model = None
    if args.kl_penalty_weight != 0.0:
        if not args.kl_reference_model_path:
            print("❌ KL penalty weight specified but no kl_reference_model_path provided.")
            return
        reference_model_path = os.path.join(project_root, args.kl_reference_model_path)
        if not os.path.exists(reference_model_path):
            print(f"❌ Reference model not found: {reference_model_path}")
            return
        print(f"🔒 Loading KL reference model from: {reference_model_path}")
        reference_model = TransformerClassifier(
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
        try:
            reference_state = torch.load(reference_model_path, map_location=device)
            reference_model.load_state_dict(reference_state)
            print("✅ Reference model loaded successfully")
        except Exception as exc:
            print(f"❌ Failed to load reference model: {exc}")
            return
        reference_model = reference_model.to(device)
        if args.use_data_parallel and device.type == 'cuda' and gpu_ids and len(gpu_ids) > 1:
            reference_model = DataParallel(reference_model, device_ids=gpu_ids)
        reference_model.eval()
        for param in reference_model.parameters():
            param.requires_grad = False

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
                "arg1_loss_weight": args.arg1_loss_weight,
                "arg2_loss_weight": args.arg2_loss_weight,
                "entropy_reg_weight": args.entropy_reg_weight,
                "kl_penalty_weight": args.kl_penalty_weight,
                "kl_reference_model_path": args.kl_reference_model_path,
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
    print(f"\n🚀 Starting parallel training for {args.num_epochs} epochs...")
    print(f"📊 Training data: {len(dataset)} examples")
    print(f"📊 Batch size: {args.batch_size}")
    print(f"📊 Learning rate: {args.learning_rate}")
    print(f"📊 Number of workers: {args.num_workers}")
    print(f"📊 Log frequency: every {args.log_frequency} batches")
    print("=" * 60)
    
    # 学習開始前のベースライン推論評価
    print(f"\n🔍 Evaluating baseline inference performance (before training)...")
    baseline_success_rate, baseline_avg_steps = evaluate_inference_performance(
        model, tokenizer, label_mappings, device, args.max_seq_len,
        num_examples=args.inference_eval_examples, 
        max_steps=args.inference_max_steps, 
        temperature=args.inference_temperature,
        difficulty=0.7,  # デフォルトのdifficulty値を使用
        max_depth=4,  # データ生成時と同じmax_depth値を使用
        seed=42  # 再現性のため固定シードを使用
    )
    print(f"  Baseline inference success rate: {baseline_success_rate:.3f}")
    print(f"  Baseline inference avg steps (when solved): {baseline_avg_steps:.2f}")
    
    # ベースライン結果をwandbに記録
    if args.use_wandb and WANDB_AVAILABLE:
        wandb.log({
            "inference/success_rate": baseline_success_rate,
            "inference/avg_steps": baseline_avg_steps
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
            entropy_reg_weight=args.entropy_reg_weight,
            kl_penalty_weight=args.kl_penalty_weight,
            reference_model=reference_model,
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
                temperature=args.inference_temperature,
                difficulty=0.7,  # デフォルトのdifficulty値を使用
                max_depth=4  # データ生成時と同じmax_depth値を使用
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
        
        # エポックごとにモデルを保存
        if args.save_checkpoints:
            checkpoint_path = f"models/parallel_model_epoch_{epoch+1}.pth"
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f"💾 Epoch checkpoint saved: {checkpoint_path}")
    
    # モデルを保存
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    torch.save(model.state_dict(), args.save_path)
    
    print(f"\n🎉 Parallel training completed!")
    print(f"📁 Model saved to: {args.save_path}")
    
    # wandb終了
    if args.use_wandb and WANDB_AVAILABLE:
        wandb.finish()
        print("📈 Wandb logging completed!")


if __name__ == "__main__":
    main()
