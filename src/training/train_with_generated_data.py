"""
generated_dataディレクトリのデータを使用した学習スクリプト
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


class GeneratedDataDataset(Dataset):
    """generated_dataディレクトリのデータセット"""
    
    def __init__(
        self, 
        data_dir: str,
        tokenizer: CharTokenizer,
        main_to_id: Dict[str, int],
        arg1_to_id: Dict[str, int], 
        arg2_to_id: Dict[str, int],
        max_seq_len: int = 512,
        remove_duplicates: bool = True
    ):
        self.tokenizer = tokenizer
        self.main_to_id = main_to_id
        self.arg1_to_id = arg1_to_id
        self.arg2_to_id = arg2_to_id
        self.max_seq_len = max_seq_len
        self.remove_duplicates = remove_duplicates
        
        # generated_dataディレクトリから全JSONファイルを読み込み
        self.data = []
        json_files = glob.glob(os.path.join(data_dir, "*.json"))
        
        print(f"Found {len(json_files)} JSON files in {data_dir}")
        
        # 重複チェック用のセット
        seen_hashes = set()
        duplicate_count = 0
        total_before_dedup = 0
        
        for json_file in json_files:
            print(f"Loading {json_file}...")
            with open(json_file, 'r') as f:
                file_data = json.load(f)
            
            # 各例の各ステップを個別の訓練例として追加
            for example in file_data:
                for step in example.get('steps', []):
                    # tactic_applyがTrueのもののみを使用
                    if step.get('tactic_apply', False):
                        total_before_dedup += 1
                        
                        # 重複チェック
                        if self.remove_duplicates:
                            state_hash = step.get('state_hash', '')
                            if state_hash in seen_hashes:
                                duplicate_count += 1
                                continue
                            seen_hashes.add(state_hash)
                        
                        self.data.append(step)
        
        # ログ出力
        print(f"\n=== Data Loading Summary ===")
        print(f"Total examples before deduplication: {total_before_dedup}")
        print(f"Total examples after deduplication: {len(self.data)}")
        
        if self.remove_duplicates:
            print(f"Removed duplicates: {duplicate_count}")
            print(f"Duplicate rate: {duplicate_count / total_before_dedup * 100:.2f}%")
        else:
            print("Duplicates kept (--keep_duplicates enabled)")
    
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


def compute_complete_tactic_accuracy(main_pred, arg1_pred, arg2_pred, 
                                   main_labels, arg1_labels, arg2_labels,
                                   tactic_arg_mask, main_to_id):
    """完全なタクティクが正しく予測された割合を計算"""
    id_to_main = {v: k for k, v in main_to_id.items()}
    correct_count = 0
    total_count = len(main_labels)
    
    for i in range(total_count):
        main_tactic = id_to_main[main_labels[i].item()]
        arg1_required, arg2_required = tactic_arg_mask.get(main_tactic, (False, False))
        
        # メインタクティクが正しいかチェック
        main_correct = main_pred[i] == main_labels[i]
        
        # 必要な引数が正しいかチェック（無効値-1を考慮）
        arg1_correct = not arg1_required or (arg1_labels[i] != -1 and arg1_pred[i] == arg1_labels[i])
        arg2_correct = not arg2_required or (arg2_labels[i] != -1 and arg2_pred[i] == arg2_labels[i])
        
        # すべてが正しい場合のみ完全正解
        if main_correct and arg1_correct and arg2_correct:
            correct_count += 1
    
    return correct_count / total_count if total_count > 0 else 0.0




# inference_hierarchical.pyから関数をインポート
from src.training.inference_hierarchical import evaluate_inference_performance


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
    gradient_accumulation_steps: int = 1
) -> float:
    """1エポックの学習（シンプル化：マスクなし損失計算）"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    total_batches = len(dataloader)
    
    print(f"  Training on {total_batches} batches...")
    
    # tqdmプログレスバーを使用
    pbar = tqdm(dataloader, desc="Training", leave=False)
    
    for batch_idx, batch in enumerate(pbar):
        input_ids, attention_mask, main_labels, arg1_labels, arg2_labels = batch
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        main_labels = main_labels.to(device)
        arg1_labels = arg1_labels.to(device)
        arg2_labels = arg2_labels.to(device)
        
        # 勾配累積のため、最初のステップでのみzero_grad
        if batch_idx % gradient_accumulation_steps == 0:
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
        
        total_loss += total_loss_batch.item() * gradient_accumulation_steps
        num_batches += 1
        
        # プログレスバーに現在の平均損失を表示
        avg_loss = total_loss / num_batches
        pbar.set_postfix({'Loss': f'{avg_loss:.4f}'})
    
    return total_loss / num_batches if num_batches > 0 else 0.0


# evaluate function removed - using inference performance evaluation instead


def main():
    parser = argparse.ArgumentParser(description="Train hierarchical tactic classifier with generated data")
    parser.add_argument("--data_dir", type=str, default="generated_data", help="generated data directory")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="learning rate")
    parser.add_argument("--num_epochs", type=int, default=None, help="number of epochs (default: data-point based training)")
    parser.add_argument("--device", type=str, default="auto", help="device")
    parser.add_argument("--save_path", type=str, default="models/hierarchical_model_generated.pth", help="model save path")
    # Removed eval_split argument - using all data for training
    parser.add_argument("--max_seq_len", type=int, default=512, help="maximum sequence length")
    parser.add_argument("--keep_duplicates", action="store_true", help="keep duplicate examples (default: remove duplicates)")
    parser.add_argument("--use_wandb", action="store_true", help="use wandb for logging")
    parser.add_argument("--wandb_project", type=str, default="fof-training", help="wandb project name")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="wandb run name")
    parser.add_argument("--arg1_loss_weight", type=float, default=0.8, help="weight for arg1 loss")
    parser.add_argument("--arg2_loss_weight", type=float, default=0.8, help="weight for arg2 loss")
    parser.add_argument("--inference_eval_examples", type=int, default=50, help="number of examples for inference evaluation")
    parser.add_argument("--inference_max_steps", type=int, default=30, help="max steps for inference evaluation")
    parser.add_argument("--inference_temperature", type=float, default=1.0, help="temperature for inference evaluation")
    parser.add_argument("--validation_frequency", type=int, default=10000, help="run validation every n data points (default: 10000)")
    parser.add_argument("--max_data_points", type=int, default=None, help="maximum number of data points to train on (default: all)")
    parser.add_argument("--skip_inference_eval", action="store_true", help="skip inference performance evaluation (faster training)")
    parser.add_argument("--max_eval_examples", type=int, default=50, help="maximum number of examples for evaluation (default: 50)")
    parser.add_argument("--random_seed", type=int, default=42, help="random seed for reproducibility (default: 42)")
    
    # 並列化関連の引数
    parser.add_argument("--num_workers", type=int, default=4, help="number of data loading workers (default: 4)")
    parser.add_argument("--use_data_parallel", action="store_true", help="use DataParallel for multi-GPU training")
    parser.add_argument("--gpu_ids", type=str, default=None, help="comma-separated GPU IDs to use (e.g., '0,1,2') or 'all' for all available GPUs")
    parser.add_argument("--use_amp", action="store_true", help="use Automatic Mixed Precision for faster training")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="number of gradient accumulation steps (default: 1)")
    
    args = parser.parse_args()
    
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
    
    # コマンドライン引数から重みを設定
    training_params.arg1_loss_weight = args.arg1_loss_weight
    training_params.arg2_loss_weight = args.arg2_loss_weight
    
    # GPU IDの処理
    if args.gpu_ids is not None:
        if args.gpu_ids == "all":
            gpu_ids = list(range(torch.cuda.device_count()))
        else:
            gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(",")]
        print(f"Using GPU IDs: {gpu_ids}")
    else:
        gpu_ids = None
    
    # デバイス設定
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # 混合精度のスケーラー初期化
    scaler = None
    if args.use_amp and device.type == 'cuda':
        scaler = GradScaler()
        print("Using Automatic Mixed Precision (AMP)")
    
    # 勾配累積の確認
    if args.gradient_accumulation_steps > 1:
        print(f"Using gradient accumulation with {args.gradient_accumulation_steps} steps")
    
    # 重複削除設定（デフォルトで削除、--keep_duplicatesで保持）
    remove_duplicates = not args.keep_duplicates
    
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
                "remove_duplicates": remove_duplicates,
                "eval_split": "all_data_for_training",
                "device": str(device)
            }
        )
        print(f"Wandb initialized: {args.wandb_project}/{run_name}")
    elif args.use_wandb and not WANDB_AVAILABLE:
        print("Warning: wandb requested but not available. Continuing without logging.")
    
    # generated_dataディレクトリの存在確認
    data_dir = os.path.join(project_root, args.data_dir)
    if not os.path.exists(data_dir):
        print(f"Generated data directory not found: {data_dir}")
        print("Please ensure the generated_data directory exists and contains JSON files")
        return
    
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
    
    # トークナイザーを作成（tactic用トークンを追加）
    tokenizer = CharTokenizer(
        base_tokens=base_tokens,
        add_tactic_tokens=model_params.add_tactic_tokens,
        num_tactic_tokens=model_params.num_tactic_tokens
    )
    
    # データセットを作成
    dataset = GeneratedDataDataset(
        data_dir=data_dir,
        tokenizer=tokenizer,
        main_to_id=main_to_id,
        arg1_to_id=arg1_to_id,
        arg2_to_id=arg2_to_id,
        max_seq_len=args.max_seq_len,
        remove_duplicates=remove_duplicates
    )
    
    if len(dataset) == 0:
        print("No training data found. Please check the generated_data directory.")
        return
    
    # 全データを訓練に使用（validationは推論性能評価で行う）
    total_size = len(dataset)
    train_dataset = dataset
    
    print(f"Train examples: {len(train_dataset)} (using all data)")
    print(f"Validation: Using inference performance evaluation with random generation")
    print(f"Random seed used: {args.random_seed}")
    
    # データローダーを作成（訓練用のみ）
    train_loader = DataLoader(
        train_dataset,
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
    
    model.to(device)
    
    # モデルの並列化
    if args.use_data_parallel and torch.cuda.device_count() > 1:
        if gpu_ids is not None:
            # 指定されたGPU IDを使用
            if len(gpu_ids) > 1:
                model = DataParallel(model, device_ids=gpu_ids)
                print(f"Using DataParallel with GPU IDs: {gpu_ids}")
            else:
                print(f"Only one GPU specified ({gpu_ids[0]}), using single GPU")
        else:
            # すべてのGPUを使用（警告付き）
            print(f"⚠️  WARNING: Using all available GPUs ({torch.cuda.device_count()})")
            print(f"⚠️  Consider using --gpu_ids to specify specific GPUs")
            print(f"⚠️  This may conflict with other processes")
            model = DataParallel(model)
            print(f"Using DataParallel with all available GPUs: {torch.cuda.device_count()}")
    
    # オプティマイザーと損失関数
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    # モデル保存ディレクトリを作成
    save_dir = os.path.dirname(args.save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 学習ループ
    best_eval_loss = float('inf')
    
    # 学習モードの設定
    if args.num_epochs is not None:
        print(f"\n🚀 Starting epoch-based training for {args.num_epochs} epochs...")
        print(f"📊 Training data: {len(train_loader.dataset)} examples")
        print(f"📊 Validation: Using inference performance evaluation")
        print(f"📊 Batch size: {args.batch_size}")
        print(f"📊 Learning rate: {args.learning_rate}")
        print("=" * 60)
    else:
        # データポイントベースの学習設定
        total_data_points = len(train_loader.dataset)
        if args.max_data_points is not None:
            total_data_points = min(total_data_points, args.max_data_points)
        
        print(f"\n🚀 Starting data-point based training for {total_data_points} data points...")
        print(f"📊 Training data: {len(train_loader.dataset)} examples")
        print(f"📊 Validation: Using inference performance evaluation")
        print(f"📊 Batch size: {args.batch_size}")
        print(f"📊 Learning rate: {args.learning_rate}")
        print(f"📊 Validation frequency: every {args.validation_frequency} data points")
        print("=" * 60)
    
    # タクティク引数マスクを取得
    tactic_arg_mask = hierarchical_labels.TACTIC_ARG_MASK
    
    # ラベルマッピングを作成（推論性能評価用）
    label_mappings = {
        'main_to_id': main_to_id,
        'arg1_to_id': arg1_to_id,
        'arg2_to_id': arg2_to_id,
        'id_to_main': id_to_main,
        'id_to_arg1': id_to_arg1,
        'id_to_arg2': id_to_arg2,
    }
    
    # 学習開始前の推論性能を評価（スキップ可能）
    if not args.skip_inference_eval:
        print("\n🔍 Evaluating initial inference performance...")
        # 並列化されたモデルの場合は元のモデルを取得
        eval_model = model.module if hasattr(model, 'module') else model
        initial_success_rate, initial_avg_steps = evaluate_inference_performance(
            eval_model, tokenizer, label_mappings, device, args.max_seq_len,
            num_examples=args.inference_eval_examples,
            max_steps=args.inference_max_steps,
            temperature=args.inference_temperature
        )
        print(f"  Initial success rate: {initial_success_rate:.3f}")
        print(f"  Initial avg steps (when solved): {initial_avg_steps:.2f}")
        
        # 初期性能をwandbにログ
        if args.use_wandb and WANDB_AVAILABLE:
            wandb.log({
                "inference/success_rate": initial_success_rate,
                "inference/avg_steps": initial_avg_steps
            })
    else:
        print("\n⏭️  Skipping initial inference performance evaluation")
        initial_success_rate, initial_avg_steps = 0.0, 0.0
    
    if args.num_epochs is not None:
        # エポックベースの学習ループ（データポイントベースのvalidation付き）
        model.train()
        total_loss = 0.0
        num_batches = 0
        data_points_processed = 0
        validation_count = 0
        
        for epoch in range(args.num_epochs):
            print(f"\n🚀 Starting epoch {epoch+1}/{args.num_epochs}")
            
            # エポック内でデータポイントベースのvalidationを実行
            pbar = tqdm(total=len(train_loader.dataset), desc=f"Epoch {epoch+1}", unit="samples")
            train_loader_iter = iter(train_loader)
            
            while data_points_processed < len(train_loader.dataset):
                try:
                    batch = next(train_loader_iter)
                except StopIteration:
                    break
                
                input_ids, attention_mask, main_labels, arg1_labels, arg2_labels = batch
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                main_labels = main_labels.to(device)
                arg1_labels = arg1_labels.to(device)
                arg2_labels = arg2_labels.to(device)
                
                optimizer.zero_grad()
                
                # 混合精度での推論
                if args.use_amp and scaler is not None:
                    with autocast():
                        # モデル推論
                        main_logits, arg1_logits, arg2_logits = model(input_ids, attention_mask)
                        
                        # 損失計算
                        main_loss = criterion(main_logits, main_labels)
                        
                        arg1_valid_mask = arg1_labels != -1
                        arg1_loss = 0.0
                        if arg1_valid_mask.any():
                            arg1_loss = criterion(arg1_logits[arg1_valid_mask], arg1_labels[arg1_valid_mask])
                        
                        arg2_valid_mask = arg2_labels != -1
                        arg2_loss = 0.0
                        if arg2_valid_mask.any():
                            arg2_loss = criterion(arg2_logits[arg2_valid_mask], arg2_labels[arg2_valid_mask])
                        
                        total_loss_batch = main_loss + training_params.arg1_loss_weight * arg1_loss + training_params.arg2_loss_weight * arg2_loss
                        total_loss_batch = total_loss_batch / args.gradient_accumulation_steps
                    
                    # 混合精度での逆伝播
                    scaler.scale(total_loss_batch).backward()
                else:
                    # 通常の推論
                    main_logits, arg1_logits, arg2_logits = model(input_ids, attention_mask)
                    
                    # 損失計算
                    main_loss = criterion(main_logits, main_labels)
                    
                    arg1_valid_mask = arg1_labels != -1
                    arg1_loss = 0.0
                    if arg1_valid_mask.any():
                        arg1_loss = criterion(arg1_logits[arg1_valid_mask], arg1_labels[arg1_valid_mask])
                    
                    arg2_valid_mask = arg2_labels != -1
                    arg2_loss = 0.0
                    if arg2_valid_mask.any():
                        arg2_loss = criterion(arg2_logits[arg2_valid_mask], arg2_labels[arg2_valid_mask])
                    
                    total_loss_batch = main_loss + training_params.arg1_loss_weight * arg1_loss + training_params.arg2_loss_weight * arg2_loss
                    total_loss_batch = total_loss_batch / args.gradient_accumulation_steps
                    
                    # 通常の逆伝播
                    total_loss_batch.backward()
                
                total_loss += total_loss_batch.item()
                num_batches += 1
                batch_size = input_ids.size(0)
                data_points_processed += batch_size
                
                # 勾配累積のステップが完了したらオプティマイザーを更新
                if data_points_processed % (args.gradient_accumulation_steps * args.batch_size) == 0:
                    if args.use_amp and scaler is not None:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad()
                
                # プログレスバーを更新
                avg_loss = total_loss / num_batches
                pbar.update(batch_size)
                pbar.set_postfix({'Loss': f'{avg_loss:.4f}'})
                
                # 指定されたデータポイント数ごとにvalidationを実行
                next_validation_threshold = (validation_count + 1) * args.validation_frequency
                if data_points_processed >= next_validation_threshold:
                    validation_count += 1
                    
                    # プログレスバーを一時停止
                    pbar.set_description(f"Epoch {epoch+1} - Validating")
                    pbar.refresh()
                    
                    # 現在の平均損失を計算
                    current_avg_loss = total_loss / num_batches
                    
                    print(f"\n📈 Validation {validation_count} (after {data_points_processed} data points)")
                    print(f"  🔥 Current Train Loss: {current_avg_loss:.4f}")
                    
                    # 推論性能を評価（従来のvalidationの代わり）
                    print(f"  📊 Skipping traditional validation - using inference performance evaluation")
                    print("-" * 60)
                    
                # 推論性能を評価
                print(f"\n🔍 Evaluating inference performance...")
                # 並列化されたモデルの場合は元のモデルを取得
                eval_model = model.module if hasattr(model, 'module') else model
                inference_success_rate, inference_avg_steps = evaluate_inference_performance(
                    eval_model, tokenizer, label_mappings, device, args.max_seq_len,
                    num_examples=args.inference_eval_examples, 
                    max_steps=args.inference_max_steps, 
                    temperature=args.inference_temperature
                )
                print(f"  Inference success rate: {inference_success_rate:.3f}")
                print(f"  Inference avg steps (when solved): {inference_avg_steps:.2f}")
                
                # wandbにログ
                if args.use_wandb and WANDB_AVAILABLE:
                    wandb.log({
                        "train_loss": current_avg_loss,
                        "inference/success_rate": inference_success_rate,
                        "inference/avg_steps": inference_avg_steps,
                        "learning_rate": optimizer.param_groups[0]['lr']
                    })
                
                # ベストモデルを保存（推論性能ベース）
                if inference_success_rate > best_eval_loss:  # 推論成功率をベスト指標として使用
                    best_eval_loss = inference_success_rate
                    # 並列化されたモデルの場合は元のモデルを取得
                    model_to_save = model.module if hasattr(model, 'module') else model
                    torch.save({
                        'model_state_dict': model_to_save.state_dict(),
                        'main_to_id': main_to_id,
                        'arg1_to_id': arg1_to_id,
                        'arg2_to_id': arg2_to_id,
                        'id_to_main': id_to_main,
                        'id_to_arg1': id_to_arg1,
                        'id_to_arg2': id_to_arg2,
                        'model_params': model_params.__dict__,
                        'vocab_size': tokenizer.vocab_size,
                        'pad_id': tokenizer.pad_id,
                        'max_seq_len': args.max_seq_len,
                    }, args.save_path)
                    print(f"Best model saved to {args.save_path} (inference success rate: {inference_success_rate:.3f})")
                    
                    # ベストモデル保存をwandbにログ
                    if args.use_wandb and WANDB_AVAILABLE:
                        wandb.log({"best_inference_success_rate": inference_success_rate})
                    
                    # モデルを訓練モードに戻す
                    model.train()
                    
                    # プログレスバーを再開
                    pbar.set_description(f"Epoch {epoch+1}")
                    pbar.refresh()
            
            # プログレスバーを閉じる
            pbar.close()
            
            # エポック終了時の最終評価
            train_loss = total_loss / num_batches if num_batches > 0 else 0.0
            
            print(f"\n📈 Epoch {epoch+1}/{args.num_epochs} completed")
            print(f"  🔥 Train Loss: {train_loss:.4f}")
            print(f"  📊 Data points processed: {data_points_processed}")
            print(f"  📊 Validations performed: {validation_count}")
            print("-" * 60)
            
            # データポイントカウンターをリセット（次のエポック用）
            data_points_processed = 0
            validation_count = 0
            total_loss = 0.0
            num_batches = 0
    else:
        # データポイントベースの学習ループ
        model.train()
        total_loss = 0.0
        num_batches = 0
        data_points_processed = 0
        validation_count = 0
        
        # tqdmプログレスバーを作成
        pbar = tqdm(total=total_data_points, desc="Training", unit="samples")
        
        # 無限ループでデータローダーを回す
        train_loader_iter = iter(train_loader)
        
        while data_points_processed < total_data_points:
            try:
                # 次のバッチを取得
                batch = next(train_loader_iter)
            except StopIteration:
                # データローダーが終了したら再開
                train_loader_iter = iter(train_loader)
                batch = next(train_loader_iter)
            
            input_ids, attention_mask, main_labels, arg1_labels, arg2_labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            main_labels = main_labels.to(device)
            arg1_labels = arg1_labels.to(device)
            arg2_labels = arg2_labels.to(device)
            
            optimizer.zero_grad()
            
            # 混合精度での推論
            if args.use_amp and scaler is not None:
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
                    total_loss_batch = main_loss + training_params.arg1_loss_weight * arg1_loss + training_params.arg2_loss_weight * arg2_loss
                    total_loss_batch = total_loss_batch / args.gradient_accumulation_steps
                
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
                total_loss_batch = main_loss + training_params.arg1_loss_weight * arg1_loss + training_params.arg2_loss_weight * arg2_loss
                total_loss_batch = total_loss_batch / args.gradient_accumulation_steps
                
                # 通常の逆伝播
                total_loss_batch.backward()
            
            total_loss += total_loss_batch.item()
            num_batches += 1
            batch_size = input_ids.size(0)
            data_points_processed += batch_size
            
            # 勾配累積のステップが完了したらオプティマイザーを更新
            if data_points_processed % (args.gradient_accumulation_steps * args.batch_size) == 0:
                if args.use_amp and scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
            
            # プログレスバーを更新
            avg_loss = total_loss / num_batches
            pbar.update(batch_size)
            pbar.set_postfix({
                'Loss': f'{avg_loss:.4f}',
                'Batches': num_batches
            })
            
            # 指定されたデータポイント数ごとにvalidationを実行
            next_validation_threshold = (validation_count + 1) * args.validation_frequency
            if data_points_processed >= next_validation_threshold:
                validation_count += 1
                
                # プログレスバーを一時停止
                pbar.set_description("Validating")
                pbar.refresh()
                
                # 現在の平均損失を計算
                current_avg_loss = total_loss / num_batches
                
                print(f"\n📈 Validation {validation_count} (after {data_points_processed} data points)")
                print(f"  🔥 Current Train Loss: {current_avg_loss:.4f}")
                
                # 推論性能を評価（従来のvalidationの代わり）
                print(f"  📊 Skipping traditional validation - using inference performance evaluation")
                print("-" * 60)
                
                # 推論性能を評価
                print(f"\n🔍 Evaluating inference performance...")
                # 並列化されたモデルの場合は元のモデルを取得
                eval_model = model.module if hasattr(model, 'module') else model
                inference_success_rate, inference_avg_steps = evaluate_inference_performance(
                    eval_model, tokenizer, label_mappings, device, args.max_seq_len,
                    num_examples=args.inference_eval_examples, 
                    max_steps=args.inference_max_steps, 
                    temperature=args.inference_temperature
                )
                print(f"  Inference success rate: {inference_success_rate:.3f}")
                print(f"  Inference avg steps (when solved): {inference_avg_steps:.2f}")
                
                # wandbにログ
                if args.use_wandb and WANDB_AVAILABLE:
                    wandb.log({
                        "train_loss": current_avg_loss,
                        "inference/success_rate": inference_success_rate,
                        "inference/avg_steps": inference_avg_steps,
                        "learning_rate": optimizer.param_groups[0]['lr']
                    })
                
                # ベストモデルを保存（推論性能ベース）
                if inference_success_rate > best_eval_loss:  # 推論成功率をベスト指標として使用
                    best_eval_loss = inference_success_rate
                    # 並列化されたモデルの場合は元のモデルを取得
                    model_to_save = model.module if hasattr(model, 'module') else model
                    torch.save({
                        'model_state_dict': model_to_save.state_dict(),
                        'main_to_id': main_to_id,
                        'arg1_to_id': arg1_to_id,
                        'arg2_to_id': arg2_to_id,
                        'id_to_main': id_to_main,
                        'id_to_arg1': id_to_arg1,
                        'id_to_arg2': id_to_arg2,
                        'model_params': model_params.__dict__,
                        'vocab_size': tokenizer.vocab_size,
                        'pad_id': tokenizer.pad_id,
                        'max_seq_len': args.max_seq_len,
                    }, args.save_path)
                    print(f"Best model saved to {args.save_path} (inference success rate: {inference_success_rate:.3f})")
                    
                    # ベストモデル保存をwandbにログ
                    if args.use_wandb and WANDB_AVAILABLE:
                        wandb.log({"best_inference_success_rate": inference_success_rate})
                
                # モデルを訓練モードに戻す
                model.train()
                
                # プログレスバーを再開
                pbar.set_description("Training")
                pbar.refresh()
        
        # プログレスバーを閉じる
        pbar.close()
    
    print("\n🎉 Training completed!")
    print(f"📁 Best model saved to: {args.save_path}")
    print(f"📊 Best inference success rate: {best_eval_loss:.4f}")
    
    # wandb終了
    if args.use_wandb and WANDB_AVAILABLE:
        wandb.finish()
        print("📈 Wandb logging completed!")


if __name__ == "__main__":
    main()
