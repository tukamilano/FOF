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
    parser.add_argument("--validation_frequency", type=int, default=1000, 
                       help="Run validation every n data points (default: 1000)")
    parser.add_argument("--batch_size", type=int, default=2, help="batch size")
    parser.add_argument("--max_seq_len", type=int, default=128, help="max sequence length")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="learning rate")
    parser.add_argument("--use_wandb", action="store_true", help="use wandb for logging")
    parser.add_argument("--wandb_project", type=str, default="fof-test-generated-data", 
                       help="wandb project name")
    parser.add_argument("--save_path", type=str, default="models/test_generated_data_model.pth",
                       help="model save path")
    
    args = parser.parse_args()
    
    # generated_dataディレクトリを直接使用
    data_dir = os.path.join(project_root, "generated_data")
    
    if not os.path.exists(data_dir):
        print(f"❌ Generated data directory not found: {data_dir}")
        return
    
    print(f"📁 Using data directory: {data_dir}")
    print(f"📊 Validation frequency: every {args.validation_frequency} data points")
    
    # 学習スクリプトを実行
    cmd = [
        sys.executable, "src/training/train_with_generated_data.py",
        "--data_dir", data_dir,
        "--batch_size", str(args.batch_size),
        "--learning_rate", str(args.learning_rate),
        "--max_seq_len", str(args.max_seq_len),
        "--arg1_loss_weight", "0.8",
        "--arg2_loss_weight", "0.8",
        "--remove_duplicates",  # 重複排除を有効化
        "--validation_frequency", str(args.validation_frequency),
        "--save_path", args.save_path
    ]
    
    if args.use_wandb:
        cmd.extend(["--use_wandb", "--wandb_project", args.wandb_project])
    
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
