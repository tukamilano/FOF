#!/usr/bin/env python3
"""
シンプルなバッチなし学習スクリプト（重複排除済みデータ専用）
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
from validation.inference_hierarchical import evaluate_inference_performance


class SimpleDataset(Dataset):
    """シンプルなデータセット（バッチなし）"""
    
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
        
        # データを読み込み
        self.data = self._load_batch_data(data_dir)
    
    def _load_batch_data(self, data_dir: str) -> List[Dict[str, Any]]:
        """データを読み込み（ディレクトリ配下のすべてのJSONファイル）"""
        data = []
        
        # ディレクトリ配下のすべてのJSONファイルを探す
        json_files = glob.glob(os.path.join(data_dir, "*.json"))
        json_files.sort()  # ファイルを順序よく読み込み
        
        print(f"Found {len(json_files)} JSON files in {data_dir}")
        
        for json_file in json_files:
            print(f"Loading {os.path.basename(json_file)}...")
            with open(json_file, 'r') as f:
                file_data = json.load(f)
            
            # データの形式に応じて処理
            if isinstance(file_data, list):
                # リスト形式の場合（重複排除済みバッチデータまたはsteps形式）
                if file_data and isinstance(file_data[0], dict):
                    # 各要素が辞書の場合
                    if 'premises' in file_data[0] and 'goal' in file_data[0] and 'tactic' in file_data[0]:
                        # 既に学習用形式の場合
                        data.extend(file_data)
                    else:
                        # steps形式の場合、各stepを抽出
                        for example in file_data:
                            steps = example.get('steps', [])
                            for step in steps:
                                data.append(step)
                else:
                    # 空のリストの場合
                    continue
            else:
                # 単一の辞書形式の場合
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
    """単一の例で学習を実行"""
    model.train()
    
    # オプティマイザーの勾配をリセット
    optimizer.zero_grad()
    
    # モデル推論
    main_logits, arg1_logits, arg2_logits = model(input_ids, attention_mask)
    
    # 損失計算（無効値-1を除外）
    main_loss = criterion(main_logits, torch.tensor([main_label], device=main_logits.device))
    
    # arg1の損失計算（無効値-1を除外）
    arg1_loss = 0.0
    if arg1_label != -1:
        arg1_loss = criterion(arg1_logits, torch.tensor([arg1_label], device=arg1_logits.device))
    
    # arg2の損失計算（無効値-1を除外）
    arg2_loss = 0.0
    if arg2_label != -1:
        arg2_loss = criterion(arg2_logits, torch.tensor([arg2_label], device=arg2_logits.device))
    
    # 総損失（重み付き）
    total_loss = main_loss + arg1_loss_weight * arg1_loss + arg2_loss_weight * arg2_loss
    
    # 逆伝播
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
    
    # 推論評価関連の引数
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
    print(f"   Learning rate: {args.learning_rate}")
    print(f"   Number of epochs: {args.num_epochs}")
    print(f"   Max sequence length: {args.max_seq_len}")
    print(f"   Random seed: {args.random_seed}")
    print(f"   Use wandb: {args.use_wandb}")
    if args.use_wandb:
        print(f"   Wandb project: {args.wandb_project}")
        print(f"   Wandb run name: {args.wandb_run_name}")
    print("=" * 60)
    
    # データディレクトリの設定
    data_dir = os.path.join(project_root, args.data_dir)
    
    # 保存パスの自動生成（指定されていない場合）
    if args.save_path is None:
        # データディレクトリ名から保存名を生成
        data_dir_name = os.path.basename(args.data_dir.rstrip('/'))
        args.save_path = f"models/{data_dir_name}.pth"
        print(f"Auto-generated save path: {args.save_path}")
    
    # データディレクトリの存在確認
    if not os.path.exists(data_dir):
        print(f"❌ Data directory not found: {data_dir}")
        print("Please run: python src/training/deduplicate_generated_data.py")
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
    
    # オプティマイザーと損失関数を作成
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    # wandb初期化
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
    print(f"\n🚀 Starting simple training for {args.num_epochs} epochs...")
    print(f"📊 Training data: {len(dataset)} examples")
    print(f"📊 Learning rate: {args.learning_rate}")
    print(f"📊 Log frequency: every {args.log_frequency} examples")
    print("=" * 60)
    
    # 学習開始前のベースライン推論評価
    print(f"\n🔍 Evaluating baseline inference performance (before training)...")
    baseline_success_rate, baseline_avg_steps = evaluate_inference_performance(
        model, tokenizer, label_mappings, device, args.max_seq_len,
        num_examples=args.inference_eval_examples, 
        max_steps=args.inference_max_steps, 
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
    
    # 学習ループ
    total_examples = 0
    epoch_losses = []
    
    # 直近log_frequency分の損失を記録するためのキュー
    recent_losses = []
    
    for epoch in range(args.num_epochs):
        print(f"\n🚀 Starting epoch {epoch+1}/{args.num_epochs}")
        
        # エポック内の損失を記録
        epoch_loss = 0.0
        num_examples = 0
        
        # データをシャッフル
        indices = list(range(len(dataset)))
        random.shuffle(indices)
        
        # プログレスバーを作成
        pbar = tqdm(indices, desc=f"Epoch {epoch+1}", unit="example")
        
        for example_idx, data_idx in enumerate(pbar):
            # データを取得
            input_ids, attention_mask, main_label, arg1_label, arg2_label = dataset[data_idx]
            
            # テンソルに変換してデバイスに移動
            input_ids = input_ids.unsqueeze(0).to(device)  # バッチ次元を追加
            attention_mask = attention_mask.unsqueeze(0).to(device)  # バッチ次元を追加
            
            # 単一の例で学習
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
            
            # 直近log_frequency分の損失を記録
            recent_losses.append(loss)
            if len(recent_losses) > args.log_frequency:
                recent_losses.pop(0)  # 古い損失を削除
            
            # プログレスバーを更新
            pbar.set_postfix({'Loss': f'{loss:.4f}', 'Avg Loss': f'{epoch_loss / num_examples:.4f}'})
            
            # 指定された頻度でwandbにログ
            if args.use_wandb and WANDB_AVAILABLE and total_examples % args.log_frequency == 0:
                recent_avg_loss = sum(recent_losses) / len(recent_losses) if recent_losses else 0.0
                wandb.log({
                    "loss": recent_avg_loss  # 直近log_frequency分の平均
                })
            
        
        # エポックの平均損失を計算
        avg_epoch_loss = epoch_loss / num_examples if num_examples > 0 else 0.0
        epoch_losses.append(avg_epoch_loss)
        
        print(f"Epoch {epoch+1} completed. Average loss: {avg_epoch_loss:.4f}")
        
        # 推論性能を評価（毎エポック）
        print(f"\n🔍 Evaluating inference performance after epoch {epoch+1}...")
        inference_success_rate, inference_avg_steps = evaluate_inference_performance(
            model, tokenizer, label_mappings, device, args.max_seq_len,
            num_examples=args.inference_eval_examples, 
            max_steps=args.inference_max_steps, 
            seed=42  # 再現性のため固定シードを使用
        )
        print(f"  Inference success rate: {inference_success_rate:.3f}")
        print(f"  Inference avg steps (when solved): {inference_avg_steps:.2f}")
        
        # エポックごとにモデルを保存
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
        
        # wandbにログ
        if args.use_wandb and WANDB_AVAILABLE:
            wandb.log({
                "epoch": epoch + 1,
                "epoch_loss": avg_epoch_loss,
                "inference/success_rate": inference_success_rate,
                "inference/avg_steps": inference_avg_steps
            })
    
    # 最終モデルを保存
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    torch.save(model.state_dict(), args.save_path)
    
    print(f"\n🎉 Training completed!")
    print(f"📁 Model saved to: {args.save_path}")
    print(f"📊 Total examples processed: {total_examples}")
    print(f"📊 Average loss per epoch: {[f'{loss:.4f}' for loss in epoch_losses]}")
    
    # wandb終了
    if args.use_wandb and WANDB_AVAILABLE:
        wandb.finish()
        print("📈 Wandb logging completed!")


if __name__ == "__main__":
    main()
