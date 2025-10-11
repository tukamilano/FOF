#!/usr/bin/env python3
"""
generated_dataディレクトリの全データを使って学習して動作確認するテストスクリプト
重複排除を行い、データn個ごとにvalidationを実行
"""
import os
import sys
import argparse

# プロジェクトルートをパスに追加
project_root = os.path.dirname(os.path.dirname(__file__))  # tests/の親ディレクトリ
sys.path.insert(0, project_root)

def main():
    parser = argparse.ArgumentParser(description="Test training with generated_data")
    parser.add_argument("--validation_frequency", type=int, default=10000, 
                       help="Run validation every n data points (default: 10000)")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--max_seq_len", type=int, default=128, help="max sequence length")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="learning rate")
    parser.add_argument("--keep_duplicates", action="store_true", help="keep duplicate examples (default: remove duplicates)")
    parser.add_argument("--use_wandb", action="store_true", help="use wandb for logging")
    parser.add_argument("--wandb_project", type=str, default="fof-test-generated-data", 
                       help="wandb project name")
    parser.add_argument("--save_path", type=str, default="models/test_generated_data_model.pth",
                       help="model save path")
    parser.add_argument("--num_epochs", type=int, default=None, help="number of epochs (default: data-point based training)")
    
    # 並列化関連の引数
    parser.add_argument("--num_workers", type=int, default=4, help="number of data loading workers (default: 4)")
    parser.add_argument("--use_data_parallel", action="store_true", help="use DataParallel for multi-GPU training")
    parser.add_argument("--gpu_ids", type=str, default=None, help="comma-separated GPU IDs to use (e.g., '0,1,2') or 'all' for all available GPUs")
    parser.add_argument("--use_amp", action="store_true", help="use Automatic Mixed Precision for faster training")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="number of gradient accumulation steps (default: 1)")
    
    args = parser.parse_args()
    
    # generated_dataディレクトリを直接使用
    data_dir = os.path.join(project_root, "generated_data")
    
    if not os.path.exists(data_dir):
        print(f"❌ Generated data directory not found: {data_dir}")
        return
    
    print(f"📁 Using data directory: {data_dir}")
    print(f"📊 Validation frequency: every {args.validation_frequency} data points")
    if args.num_epochs is not None:
        print(f"📊 Number of epochs: {args.num_epochs}")
    else:
        print(f"📊 Training mode: data-point based (no epoch limit)")
    
    # 学習スクリプトを実行
    cmd = [
        sys.executable, "src/training/train_with_generated_data.py",
        "--data_dir", data_dir,
        "--batch_size", str(args.batch_size),
        "--learning_rate", str(args.learning_rate),
        "--max_seq_len", str(args.max_seq_len),
        "--arg1_loss_weight", "0.8",
        "--arg2_loss_weight", "0.8",
        # 重複排除はデフォルトで有効（--keep_duplicatesで無効化可能）
        "--validation_frequency", str(args.validation_frequency),
        "--save_path", args.save_path,
        "--num_workers", str(args.num_workers),
        "--gradient_accumulation_steps", str(args.gradient_accumulation_steps)
    ]
    
    if args.use_wandb:
        cmd.extend(["--use_wandb", "--wandb_project", args.wandb_project])
    
    if args.keep_duplicates:
        cmd.append("--keep_duplicates")
    
    if args.use_data_parallel:
        cmd.append("--use_data_parallel")
    
    if args.gpu_ids is not None:
        cmd.extend(["--gpu_ids", args.gpu_ids])
    
    if args.use_amp:
        cmd.append("--use_amp")
    
    if args.num_epochs is not None:
        cmd.extend(["--num_epochs", str(args.num_epochs)])
    
    print(f"🚀 Running command: {' '.join(cmd)}")
    print("=" * 60)
    
    # コマンドを実行
    import subprocess
    result = subprocess.run(cmd, cwd=project_root)
    
    if result.returncode == 0:
        print("\n✅ Test completed successfully!")
    else:
        print(f"\n❌ Test failed with return code: {result.returncode}")

if __name__ == "__main__":
    main()
