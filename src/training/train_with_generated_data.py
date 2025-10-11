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




def evaluate_inference_performance(
    model: TransformerClassifier,
    tokenizer: CharTokenizer,
    label_mappings: Dict[str, Any],
    device: torch.device,
    max_seq_len: int,
    num_examples: int = 50,
    max_steps: int = 5,
    temperature: float = 1.0
) -> Tuple[float, float]:
    """
    推論性能を評価（inference_hierarchical.pyのロジックを使用）
    
    Returns:
        (success_rate, avg_steps_when_solved)
    """
    import sys
    import glob
    import json
    
    # pyproverをインポート（inference_hierarchical.pyと同じ方法）
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    pyprover_dir = os.path.join(project_root, "pyprover")
    sys.path.insert(0, pyprover_dir)
    
    original_cwd = os.getcwd()
    os.chdir(pyprover_dir)
    try:
        import proposition as proposition_mod
        import prover as prover_mod
    finally:
        os.chdir(original_cwd)
    
    PropParseTree = proposition_mod.PropParseTree
    prop_parser = proposition_mod.parser
    Prover = prover_mod.Prover
    
    # generated_dataから例を取得（inference_hierarchical.pyと同じ方法）
    generated_data_dir = os.path.join(project_root, "generated_data")
    json_files = glob.glob(os.path.join(generated_data_dir, "*.json"))
    
    all_examples = []
    for json_file in json_files:
        with open(json_file, 'r') as f:
            file_data = json.load(f)
            all_examples.extend(file_data)
    
    # 証明済みの例のみをフィルタリング
    proved_examples = [ex for ex in all_examples if ex.get('meta', {}).get('is_proved', False)]
    
    if not proved_examples:
        print("No proved examples found for inference evaluation!")
        return 0.0, 0.0
    
    # 推論性能評価用の関数（inference_hierarchical.pyから移植）
    def sample_from_logits(logits: torch.Tensor, temperature: float = 1.0) -> Tuple[int, float]:
        if temperature != 1.0:
            logits = logits / temperature
        probs = torch.softmax(logits, dim=-1)
        sampled_id = torch.multinomial(probs, 1).item()
        confidence = probs[0, sampled_id].item()
        return sampled_id, confidence
    
    def predict_tactic_inference(
        model: TransformerClassifier,
        tokenizer: CharTokenizer,
        premises: List[str],
        goal: str,
        label_mappings: Dict[str, Any],
        device: torch.device,
        max_seq_len: int = 512,
        temperature: float = 1.0
    ) -> Tuple[str, float, float, float]:
        # 入力をエンコード
        input_ids, attention_mask, segment_ids = tokenizer.encode(goal, premises, max_seq_len)
        input_ids = input_ids.unsqueeze(0).to(device)
        attention_mask = attention_mask.unsqueeze(0).to(device)
        segment_ids = segment_ids.unsqueeze(0).to(device)
        
        with torch.no_grad():
            main_logits, arg1_logits, arg2_logits = model(input_ids, attention_mask, segment_ids)
            
            # 確率的サンプリングで予測（禁止タクティクマスクなし）
            main_pred_id, main_confidence = sample_from_logits(main_logits, temperature)
            arg1_pred_id, arg1_confidence = sample_from_logits(arg1_logits, temperature)
            arg2_pred_id, arg2_confidence = sample_from_logits(arg2_logits, temperature)
            
            # タクティク文字列を構築
            main_tactic = label_mappings['id_to_main'][main_pred_id]
            arg1_value = label_mappings['id_to_arg1'][arg1_pred_id]
            arg2_value = label_mappings['id_to_arg2'][arg2_pred_id]
            
            # 引数が不要なタクティクの場合は引数を無視
            if main_tactic in ['assumption', 'intro', 'split', 'left', 'right', 'add_dn']:
                tactic_string = main_tactic
            elif main_tactic in ['apply', 'destruct']:
                tactic_string = f"{main_tactic} {arg1_value}"
            elif main_tactic == 'specialize':
                tactic_string = f"{main_tactic} {arg1_value} {arg2_value}"
            else:
                tactic_string = main_tactic
            
            return tactic_string, main_confidence, arg1_confidence, arg2_confidence
    
    def apply_tactic_from_label(prover, label) -> bool:
        """タクティクを適用（inference_hierarchical.pyから移植）"""
        if isinstance(label, dict):
            tactic_str = format_tactic_string(label)
        else:
            tactic_str = label
        
        try:
            if tactic_str == "assumption":
                return not prover.assumption()
            if tactic_str == "intro":
                return not prover.intro()
            if tactic_str == "split":
                return not prover.split()
            if tactic_str == "left":
                return not prover.left()
            if tactic_str == "right":
                return not prover.right()
            if tactic_str == "add_dn":
                return not prover.add_dn()
            
            parts = tactic_str.split()
            if parts[0] == "apply" and len(parts) == 2 and parts[1].isdigit():
                idx = int(parts[1])
                if idx >= len(prover.variables):
                    return False
                return not prover.apply(idx)
            if parts[0] == "destruct" and len(parts) == 2 and parts[1].isdigit():
                idx = int(parts[1])
                if idx >= len(prover.variables):
                    return False
                return not prover.destruct(idx)
            if parts[0] == "specialize" and len(parts) == 3 and parts[1].isdigit() and parts[2].isdigit():
                func_idx = int(parts[1])
                domain_idx = int(parts[2])
                if func_idx >= len(prover.variables) or domain_idx >= len(prover.variables):
                    return False
                return not prover.specialize(func_idx, domain_idx)
            return False
        except Exception as e:
            # pyproverのエラーをキャッチしてFalseを返す
            return False
    
    # 推論性能評価を実行
    solved_count = 0
    solved_steps = []
    
    for i in range(min(num_examples, len(proved_examples))):
        # 例を循環して選択
        example = proved_examples[i % len(proved_examples)]
        
        # 最初のステップから初期状態を取得
        first_step = example['steps'][0]
        goal_str = first_step['goal']
        premises = first_step['premises']
        
        # パースしてproverを作成
        parse_tree = PropParseTree()
        goal_node = parse_tree.transform(prop_parser.parse(goal_str))
        prover = Prover(goal_node)
        
        # 前提を追加
        for prem_str in premises:
            prem_node = parse_tree.transform(prop_parser.parse(prem_str))
            prover.variables.append(prem_node)
        
        # 推論ループ（シンプル化：禁止タクティクシステムなし）
        step = 0
        solved = prover.goal is None
        
        while not solved and step < max_steps:
            # 現在の状態を取得
            current_state = encode_prover_state(prover)
            current_premises = current_state["premises"]
            current_goal = current_state["goal"]
            
            # タクティクを予測（純粋な言語モデル性能）
            tactic_str, main_conf, arg1_conf, arg2_conf = predict_tactic_inference(
                model, tokenizer, current_premises, current_goal, 
                label_mappings, device, max_seq_len, temperature
            )
            
            # タクティクを適用
            success = apply_tactic_from_label(prover, tactic_str)
            
            step += 1
            solved = prover.goal is None
        
        if solved:
            solved_count += 1
            solved_steps.append(step)
    
    # 統計を計算
    success_rate = solved_count / num_examples
    avg_steps_when_solved = sum(solved_steps) / len(solved_steps) if solved_steps else 0.0
    
    return success_rate, avg_steps_when_solved


def train_epoch(
    model: TransformerClassifier,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    arg1_loss_weight: float = 0.8,
    arg2_loss_weight: float = 0.8
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
        
        optimizer.zero_grad()
        
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
        
        total_loss_batch.backward()
        optimizer.step()
        
        total_loss += total_loss_batch.item()
        num_batches += 1
        
        # プログレスバーに現在の平均損失を表示
        avg_loss = total_loss / num_batches
        pbar.set_postfix({'Loss': f'{avg_loss:.4f}'})
    
    return total_loss / num_batches if num_batches > 0 else 0.0


def evaluate(
    model: TransformerClassifier,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    tactic_arg_mask: Dict[str, tuple],
    main_to_id: Dict[str, int],
    arg1_loss_weight: float = 0.8,
    arg2_loss_weight: float = 0.8
) -> Tuple[float, float, float, float, float, int, int]:
    """評価（シンプル化：マスクなし損失計算、完全タクティク精度付き）"""
    model.eval()
    total_loss = 0.0
    main_correct = 0
    arg1_correct = 0
    arg2_correct = 0
    total_samples = 0
    total_batches = len(dataloader)
    
    # 完全タクティク精度用の累積データ
    all_main_preds = []
    all_arg1_preds = []
    all_arg2_preds = []
    all_main_labels = []
    all_arg1_labels = []
    all_arg2_labels = []
    
    print(f"  Evaluating on {total_batches} batches...")
    
    # tqdmプログレスバーを使用
    pbar = tqdm(dataloader, desc="Evaluating", leave=False)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(pbar):
            input_ids, attention_mask, main_labels, arg1_labels, arg2_labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            main_labels = main_labels.to(device)
            arg1_labels = arg1_labels.to(device)
            arg2_labels = arg2_labels.to(device)
            
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
            
            total_loss += total_loss_batch.item()
            
            # 精度計算
            main_pred = torch.argmax(main_logits, dim=-1)
            arg1_pred = torch.argmax(arg1_logits, dim=-1)
            arg2_pred = torch.argmax(arg2_logits, dim=-1)
            
            main_correct += (main_pred == main_labels).sum().item()
            
            # arg1の精度計算（無効値-1を除外）
            arg1_valid_mask = arg1_labels != -1
            if arg1_valid_mask.any():
                arg1_correct += (arg1_pred[arg1_valid_mask] == arg1_labels[arg1_valid_mask]).sum().item()
            
            # arg2の精度計算（無効値-1を除外）
            arg2_valid_mask = arg2_labels != -1
            if arg2_valid_mask.any():
                arg2_correct += (arg2_pred[arg2_valid_mask] == arg2_labels[arg2_valid_mask]).sum().item()
            
            total_samples += main_labels.size(0)
            
            # 完全タクティク精度計算用にデータを蓄積
            all_main_preds.append(main_pred.cpu())
            all_arg1_preds.append(arg1_pred.cpu())
            all_arg2_preds.append(arg2_pred.cpu())
            all_main_labels.append(main_labels.cpu())
            all_arg1_labels.append(arg1_labels.cpu())
            all_arg2_labels.append(arg2_labels.cpu())
            
            # プログレスバーに現在の精度を表示
            current_main_acc = main_correct / total_samples if total_samples > 0 else 0.0
            
            # 現在のバッチでの有効サンプル数を計算
            current_arg1_valid = (arg1_labels != -1).sum().item()
            current_arg2_valid = (arg2_labels != -1).sum().item()
            
            current_arg1_acc = arg1_correct / current_arg1_valid if current_arg1_valid > 0 else 0.0
            current_arg2_acc = arg2_correct / current_arg2_valid if current_arg2_valid > 0 else 0.0
            
            pbar.set_postfix({
                'Main': f'{current_main_acc:.3f}',
                'Arg1': f'{current_arg1_acc:.3f}',
                'Arg2': f'{current_arg2_acc:.3f}'
            })
    
    avg_loss = total_loss / len(dataloader)
    main_acc = main_correct / total_samples
    
    # arg1とarg2の有効サンプル数を計算
    all_arg1_labels_tensor = torch.cat(all_arg1_labels, dim=0) if all_arg1_labels else torch.tensor([])
    all_arg2_labels_tensor = torch.cat(all_arg2_labels, dim=0) if all_arg2_labels else torch.tensor([])
    
    arg1_valid_count = (all_arg1_labels_tensor != -1).sum().item() if len(all_arg1_labels_tensor) > 0 else 0
    arg2_valid_count = (all_arg2_labels_tensor != -1).sum().item() if len(all_arg2_labels_tensor) > 0 else 0
    
    arg1_acc = arg1_correct / arg1_valid_count if arg1_valid_count > 0 else 0.0
    arg2_acc = arg2_correct / arg2_valid_count if arg2_valid_count > 0 else 0.0
    
    # 完全タクティク精度を計算
    if all_main_preds:
        all_main_preds = torch.cat(all_main_preds, dim=0)
        all_arg1_preds = torch.cat(all_arg1_preds, dim=0)
        all_arg2_preds = torch.cat(all_arg2_preds, dim=0)
        all_main_labels = torch.cat(all_main_labels, dim=0)
        all_arg1_labels = torch.cat(all_arg1_labels, dim=0)
        all_arg2_labels = torch.cat(all_arg2_labels, dim=0)
        
        complete_tactic_acc = compute_complete_tactic_accuracy(
            all_main_preds, all_arg1_preds, all_arg2_preds,
            all_main_labels, all_arg1_labels, all_arg2_labels,
            tactic_arg_mask, main_to_id
        )
    else:
        complete_tactic_acc = 0.0
    
    return avg_loss, main_acc, arg1_acc, arg2_acc, complete_tactic_acc, arg1_valid_count, arg2_valid_count


def main():
    parser = argparse.ArgumentParser(description="Train hierarchical tactic classifier with generated data")
    parser.add_argument("--data_dir", type=str, default="generated_data", help="generated data directory")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="learning rate")
    parser.add_argument("--num_epochs", type=int, default=10, help="number of epochs")
    parser.add_argument("--device", type=str, default="auto", help="device")
    parser.add_argument("--save_path", type=str, default="models/hierarchical_model_generated.pth", help="model save path")
    parser.add_argument("--eval_split", type=float, default=0.2, help="evaluation split ratio")
    parser.add_argument("--max_seq_len", type=int, default=512, help="maximum sequence length")
    parser.add_argument("--remove_duplicates", action="store_true", default=True, help="remove duplicate examples based on state_hash (default: True)")
    parser.add_argument("--keep_duplicates", action="store_true", help="keep duplicate examples (overrides --remove_duplicates)")
    parser.add_argument("--use_wandb", action="store_true", help="use wandb for logging")
    parser.add_argument("--wandb_project", type=str, default="fof-training", help="wandb project name")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="wandb run name")
    parser.add_argument("--arg1_loss_weight", type=float, default=0.8, help="weight for arg1 loss")
    parser.add_argument("--arg2_loss_weight", type=float, default=0.8, help="weight for arg2 loss")
    parser.add_argument("--inference_eval_examples", type=int, default=50, help="number of examples for inference evaluation")
    parser.add_argument("--inference_max_steps", type=int, default=30, help="max steps for inference evaluation")
    parser.add_argument("--inference_temperature", type=float, default=1.0, help="temperature for inference evaluation")
    parser.add_argument("--validation_frequency", type=int, default=1000, help="run validation every n data points (default: 1000)")
    parser.add_argument("--max_data_points", type=int, default=None, help="maximum number of data points to train on (default: all)")
    
    args = parser.parse_args()
    
    # パラメータを初期化
    model_params = get_model_params()
    training_params = get_training_params()
    system_params = get_system_params()
    hierarchical_labels = get_hierarchical_labels()
    
    # コマンドライン引数から重みを設定
    training_params.arg1_loss_weight = args.arg1_loss_weight
    training_params.arg2_loss_weight = args.arg2_loss_weight
    
    # デバイス設定
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # 重複削除設定
    remove_duplicates = args.remove_duplicates and not args.keep_duplicates
    
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
                "eval_split": args.eval_split,
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
    
    # オプティマイザーと損失関数
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    # モデル保存ディレクトリを作成
    save_dir = os.path.dirname(args.save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 学習ループ
    best_eval_loss = float('inf')
    
    # データポイントベースの学習設定
    total_data_points = len(train_loader.dataset)
    if args.max_data_points is not None:
        total_data_points = min(total_data_points, args.max_data_points)
    
    print(f"\n🚀 Starting training for {total_data_points} data points...")
    print(f"📊 Training data: {len(train_loader.dataset)} examples")
    print(f"📊 Evaluation data: {len(eval_loader.dataset)} examples")
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
    
    # 学習開始前の推論性能を評価
    print("\n🔍 Evaluating initial inference performance...")
    initial_success_rate, initial_avg_steps = evaluate_inference_performance(
        model, tokenizer, label_mappings, device, args.max_seq_len,
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
            "inference/avg_steps": initial_avg_steps,
            "epoch": 0  # 初期状態をepoch 0として記録
        })
    
    # データポイントベースの学習ループ
    model.train()
    total_loss = 0.0
    num_batches = 0
    data_points_processed = 0
    validation_count = 0
    
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
        
        total_loss_batch.backward()
        optimizer.step()
        
        total_loss += total_loss_batch.item()
        num_batches += 1
        data_points_processed += input_ids.size(0)
        
        # バッチごとの進捗表示
        if num_batches % 10 == 0:
            avg_loss = total_loss / num_batches
            print(f"  Processed {data_points_processed}/{total_data_points} data points, avg loss: {avg_loss:.4f}")
        
        # 指定されたデータポイント数ごとにvalidationを実行
        if data_points_processed >= (validation_count + 1) * args.validation_frequency:
            validation_count += 1
            
            # 現在の平均損失を計算
            current_avg_loss = total_loss / num_batches
            
            print(f"\n📈 Validation {validation_count} (after {data_points_processed} data points)")
            print(f"  🔥 Current Train Loss: {current_avg_loss:.4f}")
            
            # 評価を実行
            eval_loss, main_acc, arg1_acc, arg2_acc, complete_tactic_acc, arg1_valid_count, arg2_valid_count = evaluate(
                model, eval_loader, criterion, device,
                tactic_arg_mask, main_to_id,
                training_params.arg1_loss_weight, training_params.arg2_loss_weight
            )
            
            print(f"  📊 Eval Loss: {eval_loss:.4f}")
            print(f"  🎯 Main Acc: {main_acc:.4f}")
            print(f"  🎯 Arg1 Acc: {arg1_acc:.4f} (valid samples: {arg1_valid_count})")
            print(f"  🎯 Arg2 Acc: {arg2_acc:.4f} (valid samples: {arg2_valid_count})")
            print(f"  ✅ Complete Tactic Acc: {complete_tactic_acc:.4f}")
            print("-" * 60)
            
            # 推論性能を評価
            print(f"\n🔍 Evaluating inference performance...")
            inference_success_rate, inference_avg_steps = evaluate_inference_performance(
                model, tokenizer, label_mappings, device, args.max_seq_len,
                num_examples=args.inference_eval_examples, 
                max_steps=args.inference_max_steps, 
                temperature=args.inference_temperature
            )
            print(f"  Inference success rate: {inference_success_rate:.3f}")
            print(f"  Inference avg steps (when solved): {inference_avg_steps:.2f}")
            
            # wandbにログ
            if args.use_wandb and WANDB_AVAILABLE:
                wandb.log({
                    "data_points_processed": data_points_processed,
                    "validation_count": validation_count,
                    "train_loss": current_avg_loss,
                    "eval_loss": eval_loss,
                    "main_accuracy": main_acc,
                    "arg1_accuracy": arg1_acc,
                    "arg2_accuracy": arg2_acc,
                    "complete_tactic_accuracy": complete_tactic_acc,
                    "inference/success_rate": inference_success_rate,
                    "inference/avg_steps": inference_avg_steps,
                    "learning_rate": optimizer.param_groups[0]['lr']
                })
            
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
                    'max_seq_len': args.max_seq_len,
                }, args.save_path)
                print(f"Best model saved to {args.save_path}")
                
                # ベストモデル保存をwandbにログ
                if args.use_wandb and WANDB_AVAILABLE:
                    wandb.log({"best_eval_loss": eval_loss})
            
            # モデルを訓練モードに戻す
            model.train()
    
    print("\n🎉 Training completed!")
    print(f"📁 Best model saved to: {args.save_path}")
    print(f"📊 Best evaluation loss: {best_eval_loss:.4f}")
    
    # wandb終了
    if args.use_wandb and WANDB_AVAILABLE:
        wandb.finish()
        print("📈 Wandb logging completed!")


if __name__ == "__main__":
    main()
