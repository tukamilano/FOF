"""
階層分類対応の学習スクリプト
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Tuple, Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformer_classifier import (
    load_tokens_and_labels_from_token_py,
    CharTokenizer,
    TransformerClassifier,
    build_hierarchical_label_mappings,
    hierarchical_collate,
)
from state_encoder import parse_tactic_string
from parameter import (
    default_params, get_model_params, get_training_params, 
    get_system_params, get_hierarchical_labels, DeviceType
)


class HierarchicalTacticDataset(Dataset):
    """階層分類用のデータセット"""
    
    def __init__(
        self, 
        data_file: str,
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
        with open(data_file, 'r') as f:
            self.data = json.load(f)
        
        # フィルタリング：tactic_applyがTrueのもののみ
        self.data = [item for item in self.data if item.get('tactic_apply', False)]
        
        print(f"Loaded {len(self.data)} training examples")
    
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
        arg1_label = self.arg1_to_id.get(arg1, 0) if arg1 is not None else 0
        arg2_label = self.arg2_to_id.get(arg2, 0) if arg2 is not None else 0
        
        return input_ids, attention_mask, main_label, arg1_label, arg2_label


def train_epoch(
    model: TransformerClassifier,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device
) -> float:
    """1エポックの学習"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch in dataloader:
        input_ids, attention_mask, main_labels, arg1_labels, arg2_labels = batch
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        main_labels = main_labels.to(device)
        arg1_labels = arg1_labels.to(device)
        arg2_labels = arg2_labels.to(device)
        
        optimizer.zero_grad()
        
        # モデル推論
        main_logits, arg1_logits, arg2_logits = model(input_ids, attention_mask)
        
        # 損失計算
        main_loss = criterion(main_logits, main_labels)
        arg1_loss = criterion(arg1_logits, arg1_labels)
        arg2_loss = criterion(arg2_logits, arg2_labels)
        
        # 総損失（重み付き）
        total_loss_batch = main_loss + 0.5 * arg1_loss + 0.5 * arg2_loss
        
        total_loss_batch.backward()
        optimizer.step()
        
        total_loss += total_loss_batch.item()
        num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else 0.0


def evaluate(
    model: TransformerClassifier,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float, float, float]:
    """評価"""
    model.eval()
    total_loss = 0.0
    main_correct = 0
    arg1_correct = 0
    arg2_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask, main_labels, arg1_labels, arg2_labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            main_labels = main_labels.to(device)
            arg1_labels = arg1_labels.to(device)
            arg2_labels = arg2_labels.to(device)
            
            # モデル推論
            main_logits, arg1_logits, arg2_logits = model(input_ids, attention_mask)
            
            # 損失計算
            main_loss = criterion(main_logits, main_labels)
            arg1_loss = criterion(arg1_logits, arg1_labels)
            arg2_loss = criterion(arg2_logits, arg2_labels)
            total_loss_batch = main_loss + 0.5 * arg1_loss + 0.5 * arg2_loss
            
            total_loss += total_loss_batch.item()
            
            # 精度計算
            main_pred = torch.argmax(main_logits, dim=-1)
            arg1_pred = torch.argmax(arg1_logits, dim=-1)
            arg2_pred = torch.argmax(arg2_logits, dim=-1)
            
            main_correct += (main_pred == main_labels).sum().item()
            arg1_correct += (arg1_pred == arg1_labels).sum().item()
            arg2_correct += (arg2_pred == arg2_labels).sum().item()
            total_samples += main_labels.size(0)
    
    avg_loss = total_loss / len(dataloader)
    main_acc = main_correct / total_samples
    arg1_acc = arg1_correct / total_samples
    arg2_acc = arg2_correct / total_samples
    
    return avg_loss, main_acc, arg1_acc, arg2_acc


def main():
    parser = argparse.ArgumentParser(description="Train hierarchical tactic classifier")
    parser.add_argument("--data_file", type=str, default="training_data.json", help="training data file")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="learning rate")
    parser.add_argument("--num_epochs", type=int, default=10, help="number of epochs")
    parser.add_argument("--device", type=str, default="auto", help="device")
    parser.add_argument("--save_path", type=str, default="hierarchical_model.pth", help="model save path")
    parser.add_argument("--eval_split", type=float, default=0.2, help="evaluation split ratio")
    
    args = parser.parse_args()
    
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
    
    # トークンとラベルを読み込み
    root_dir = os.path.dirname(__file__)
    token_py_path = os.path.join(root_dir, "fof_tokens.py")
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
    tokenizer = CharTokenizer(base_tokens=base_tokens)
    
    # データセットを作成
    dataset = HierarchicalTacticDataset(
        data_file=args.data_file,
        tokenizer=tokenizer,
        main_to_id=main_to_id,
        arg1_to_id=arg1_to_id,
        arg2_to_id=arg2_to_id,
        max_seq_len=model_params.max_seq_len
    )
    
    # 訓練・評価分割
    total_size = len(dataset)
    eval_size = int(total_size * args.eval_split)
    train_size = total_size - eval_size
    
    train_dataset, eval_dataset = torch.utils.data.random_split(
        dataset, [train_size, eval_size]
    )
    
    print(f"Train examples: {len(train_dataset)}")
    print(f"Eval examples: {len(eval_dataset)}")
    
    # データローダーを作成
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=hierarchical_collate
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=hierarchical_collate
    )
    
    # モデルを作成
    model = TransformerClassifier(
        vocab_size=tokenizer.vocab_size,
        pad_id=tokenizer.pad_id,
        max_seq_len=model_params.max_seq_len,
        d_model=model_params.d_model,
        nhead=model_params.nhead,
        num_layers=model_params.num_layers,
        dim_feedforward=model_params.dim_feedforward,
        dropout=model_params.dropout,
        use_hierarchical_classification=True,
        num_main_classes=len(id_to_main),
        num_arg1_classes=len(id_to_arg1),
        num_arg2_classes=len(id_to_arg2),
    )
    
    print(f"Model vocab_size: {tokenizer.vocab_size}")
    print(f"Model pad_id: {tokenizer.pad_id}")
    
    model.to(device)
    
    # オプティマイザーと損失関数
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    # 学習ループ
    best_eval_loss = float('inf')
    
    for epoch in range(args.num_epochs):
        # 訓練
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # 評価
        eval_loss, main_acc, arg1_acc, arg2_acc = evaluate(model, eval_loader, criterion, device)
        
        print(f"Epoch {epoch+1}/{args.num_epochs}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Eval Loss: {eval_loss:.4f}")
        print(f"  Main Acc: {main_acc:.4f}")
        print(f"  Arg1 Acc: {arg1_acc:.4f}")
        print(f"  Arg2 Acc: {arg2_acc:.4f}")
        print()
        
        # ベストモデルを保存
        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'main_to_id': main_to_id,
                'arg1_to_id': arg1_to_id,
                'arg2_to_id': arg2_to_id,
                'id_to_main': id_to_main,
                'id_to_arg1': id_to_arg1,
                'id_to_arg2': id_to_arg2,
                'model_params': model_params.__dict__,
                'vocab_size': tokenizer.vocab_size,
                'pad_id': tokenizer.pad_id,
            }, args.save_path)
            print(f"Best model saved to {args.save_path}")
    
    print("Training completed!")


if __name__ == "__main__":
    main()
