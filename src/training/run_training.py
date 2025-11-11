#!/usr/bin/env python3
"""
generated_data 使用didTraining 実行do/perform簡単なスクリプト
"""
import os
import sys
import subprocess

def main():
    # プロジェクトルート 移動
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    os.chdir(project_root)
    
    # Trainingスクリプト 実行
    train_script = os.path.join(script_dir, "train_with_generated_data.py")
    
    cmd = [
        sys.executable, train_script,
        "--data_dir", "generated_data",
        "--batch_size", "16",  # メモリ使用量 考慮and小さめ 設定
        "--learning_rate", "3e-4",
        "--num_epochs", "5",   # 最初は少なめ 設定
        "--save_path", "models/hierarchical_model_generated.pth",
        "--eval_split", "0.2",
        "--max_seq_len", "256",  # メモリ使用量 考慮and小さめ 設定
        "--use_wandb",
        "--wandb_project", "fof-training",
        "--wandb_run_name", "default_training_run"
        # --remove_duplicates はデフォルト with/at 有効
    ]
    
    print("Starting training with generated data...")
    print(f"Command: {' '.join(cmd)}")
    print()
    
    try:
        subprocess.run(cmd, check=True)
        print("\nTraining completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"\nTraining failed with error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        sys.exit(1)

if __name__ == "__main__":
    main()
