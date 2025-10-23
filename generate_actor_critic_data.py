#!/usr/bin/env python3
"""
Actor-Critic学習用データ生成スクリプト
trial1データからpretrained_modelを使って包括的データを収集
"""
import os
import sys
import json
import torch
import argparse
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = os.path.dirname(__file__)
sys.path.insert(0, project_root)

from src.core.transformer_classifier import (
    load_tokens_and_labels_from_token_py,
    CharTokenizer,
    TransformerClassifier,
    build_hierarchical_label_mappings,
)
from src.core.parameter import get_model_params, get_hierarchical_labels
from src.interaction.self_improvement_data_collector import collect_comprehensive_rl_data


def load_trial1_data(trial1_dir: str = "tautology/trial1", max_files: int = None) -> list:
    """
    trial1ディレクトリから論理式データを読み込み
    
    Args:
        trial1_dir: trial1ディレクトリのパス
        max_files: 読み込む最大ファイル数（Noneの場合は全て）
    
    Returns:
        論理式のリスト
    """
    trial1_path = Path(trial1_dir)
    if not trial1_path.exists():
        print(f"❌ Trial1 directory not found: {trial1_dir}")
        return []
    
    json_files = sorted(trial1_path.glob("tautology_data_*.json"))
    if max_files:
        json_files = json_files[:max_files]
    
    print(f"📁 Found {len(json_files)} JSON files in {trial1_dir}")
    
    all_formulas = []
    for json_file in json_files:
        print(f"📖 Loading {json_file.name}...")
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                formulas = json.load(f)
            
            if isinstance(formulas, list):
                all_formulas.extend(formulas)
                print(f"   Loaded {len(formulas)} formulas")
            else:
                print(f"   Warning: {json_file.name} is not a list format")
                
        except Exception as e:
            print(f"   Error loading {json_file.name}: {e}")
            continue
    
    print(f"📊 Total formulas loaded: {len(all_formulas)}")
    return all_formulas


def save_trial1_data_as_generated_data(formulas: list, output_dir: str = "generated_data_trial1"):
    """
    trial1データをgenerated_data形式で保存
    
    Args:
        formulas: 論理式のリスト
        output_dir: 出力ディレクトリ
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # バッチサイズで分割して保存
    batch_size = 1000
    for i in range(0, len(formulas), batch_size):
        batch_formulas = formulas[i:i + batch_size]
        batch_num = i // batch_size + 1
        
        filename = f"tautology_data_{batch_num:05d}.json"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(batch_formulas, f, ensure_ascii=False, indent=2)
        
        print(f"💾 Saved {len(batch_formulas)} formulas to {filename}")
    
    print(f"✅ All formulas saved to {output_dir}")


def generate_actor_critic_data(
    pretrained_model_path: str = "models/pretrained_model.pth",
    trial1_dir: str = "tautology/trial1",
    output_dir: str = "actor_critic_data",
    max_trial1_files: int = 10,  # テスト用に少なめ
    num_examples: int = 100,
    max_steps: int = 20,
    temperature: float = 1.0,
    success_reward: float = 1.0,
    step_penalty: float = 0.01,
    failure_penalty: float = -0.1,
    device: str = "auto"
):
    """
    Actor-Critic学習用データを生成
    
    Args:
        pretrained_model_path: 事前学習済みモデルのパス
        trial1_dir: trial1ディレクトリのパス
        output_dir: 出力ディレクトリ
        max_trial1_files: 使用するtrial1ファイル数
        num_examples: 処理する例数
        max_steps: 最大ステップ数
        temperature: 温度パラメータ
        success_reward: 成功時の報酬
        step_penalty: ステップペナルティ
        failure_penalty: 失敗時のペナルティ
        device: デバイス
    """
    print("🚀 Starting Actor-Critic data generation...")
    
    # デバイス設定
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    print(f"Using device: {device}")
    
    # パラメータを初期化
    model_params = get_model_params()
    hierarchical_labels = get_hierarchical_labels()
    
    # トークンとラベルを読み込み
    token_py_path = os.path.join(project_root, "src", "core", "fof_tokens.py")
    base_tokens, _ = load_tokens_and_labels_from_token_py(token_py_path)
    
    # 階層分類用のラベルマッピングを構築
    main_to_id, arg1_to_id, arg2_to_id, id_to_main, id_to_arg1, id_to_arg2 = build_hierarchical_label_mappings(
        hierarchical_labels.main_tactics,
        hierarchical_labels.arg1_values,
        hierarchical_labels.arg2_values
    )
    
    label_mappings = {
        'main_to_id': main_to_id,
        'arg1_to_id': arg1_to_id,
        'arg2_to_id': arg2_to_id,
        'id_to_main': id_to_main,
        'id_to_arg1': id_to_arg1,
        'id_to_arg2': id_to_arg2
    }
    
    # トークナイザーを作成
    tokenizer = CharTokenizer(
        base_tokens=base_tokens,
        add_tactic_tokens=model_params.add_tactic_tokens,
        num_tactic_tokens=model_params.num_tactic_tokens
    )
    
    # モデルを作成
    print("🔄 Loading pretrained model...")
    model = TransformerClassifier(
        vocab_size=tokenizer.vocab_size,
        pad_id=tokenizer.pad_id,
        max_seq_len=256,
        d_model=model_params.d_model,
        nhead=model_params.nhead,
        num_layers=model_params.num_layers,
        dim_feedforward=model_params.dim_feedforward,
        dropout=model_params.dropout,
        num_main_classes=len(id_to_main),
        num_arg1_classes=len(id_to_arg1),
        num_arg2_classes=len(id_to_arg2),
    )
    
    # 事前学習済みモデルを読み込み
    if os.path.exists(pretrained_model_path):
        print(f"📥 Loading pretrained model from: {pretrained_model_path}")
        state_dict = torch.load(pretrained_model_path, map_location=device)
        model.load_state_dict(state_dict)
        model = model.to(device)
        print("✅ Pretrained model loaded successfully!")
    else:
        print(f"❌ Pretrained model not found: {pretrained_model_path}")
        return
    
    # trial1データを読み込み
    print(f"\n📚 Loading trial1 data...")
    trial1_formulas = load_trial1_data(trial1_dir, max_trial1_files)
    
    if not trial1_formulas:
        print("❌ No trial1 data loaded. Exiting.")
        return
    
    # trial1データをgenerated_data形式で保存
    temp_generated_data_dir = "temp_generated_data_trial1"
    print(f"\n💾 Saving trial1 data as generated_data format...")
    save_trial1_data_as_generated_data(trial1_formulas, temp_generated_data_dir)
    
    # 包括的データ収集を実行
    print(f"\n📊 Collecting comprehensive RL data...")
    successful_tactics, failed_tactics = collect_comprehensive_rl_data(
        model=model,
        tokenizer=tokenizer,
        label_mappings=label_mappings,
        device=device,
        max_seq_len=256,
        num_examples=num_examples,
        max_steps=max_steps,
        verbose=True,
        generated_data_dir=temp_generated_data_dir,
        temperature=temperature,
        include_failures=True,
        success_reward=success_reward,
        step_penalty=step_penalty,
        failure_penalty=failure_penalty
    )
    
    print(f"\n📈 Data collection completed:")
    print(f"  Successful tactics: {len(successful_tactics)}")
    print(f"  Failed tactics: {len(failed_tactics)}")
    
    # 結果を保存
    os.makedirs(output_dir, exist_ok=True)
    
    # 成功データを保存
    if successful_tactics:
        success_file = os.path.join(output_dir, "successful_tactics.json")
        with open(success_file, 'w', encoding='utf-8') as f:
            json.dump(successful_tactics, f, ensure_ascii=False, indent=2)
        print(f"💾 Successful tactics saved to: {success_file}")
    
    # 失敗データを保存
    if failed_tactics:
        failed_file = os.path.join(output_dir, "failed_tactics.json")
        with open(failed_file, 'w', encoding='utf-8') as f:
            json.dump(failed_tactics, f, ensure_ascii=False, indent=2)
        print(f"💾 Failed tactics saved to: {failed_file}")
    
    # 統計情報を保存
    stats = {
        "total_examples_processed": num_examples,
        "successful_tactics": len(successful_tactics),
        "failed_tactics": len(failed_tactics),
        "success_rate": len(successful_tactics) / (len(successful_tactics) + len(failed_tactics)) if (len(successful_tactics) + len(failed_tactics)) > 0 else 0,
        "trial1_files_used": max_trial1_files,
        "temperature": temperature,
        "max_steps": max_steps,
        "success_reward": success_reward,
        "step_penalty": step_penalty,
        "failure_penalty": failure_penalty
    }
    
    stats_file = os.path.join(output_dir, "data_generation_stats.json")
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"📊 Statistics saved to: {stats_file}")
    
    # 一時ファイルを削除
    import shutil
    if os.path.exists(temp_generated_data_dir):
        shutil.rmtree(temp_generated_data_dir)
        print(f"🗑️  Cleaned up temporary directory: {temp_generated_data_dir}")
    
    print(f"\n🎉 Actor-Critic data generation completed!")
    print(f"📁 Output directory: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate Actor-Critic training data from trial1")
    parser.add_argument("--pretrained_model", type=str, default="models/pretrained_model.pth", help="pretrained model path")
    parser.add_argument("--trial1_dir", type=str, default="tautology/trial1", help="trial1 directory")
    parser.add_argument("--output_dir", type=str, default="actor_critic_data", help="output directory")
    parser.add_argument("--max_trial1_files", type=int, default=10, help="max trial1 files to use")
    parser.add_argument("--num_examples", type=int, default=100, help="number of examples to process")
    parser.add_argument("--max_steps", type=int, default=20, help="max steps per example")
    parser.add_argument("--temperature", type=float, default=1.0, help="temperature for sampling")
    parser.add_argument("--success_reward", type=float, default=1.0, help="success reward")
    parser.add_argument("--step_penalty", type=float, default=0.01, help="step penalty")
    parser.add_argument("--failure_penalty", type=float, default=-0.1, help="failure penalty")
    parser.add_argument("--device", type=str, default="auto", help="device (auto/cuda/cpu)")
    
    args = parser.parse_args()
    
    generate_actor_critic_data(
        pretrained_model_path=args.pretrained_model,
        trial1_dir=args.trial1_dir,
        output_dir=args.output_dir,
        max_trial1_files=args.max_trial1_files,
        num_examples=args.num_examples,
        max_steps=args.max_steps,
        temperature=args.temperature,
        success_reward=args.success_reward,
        step_penalty=args.step_penalty,
        failure_penalty=args.failure_penalty,
        device=args.device
    )


if __name__ == "__main__":
    main()

