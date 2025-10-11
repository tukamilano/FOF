#!/usr/bin/env python3
"""
generated_dataを使用した学習を実行する簡単なスクリプト
"""
import os
import sys
import subprocess

def main():
    # プロジェクトルートに移動
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    os.chdir(project_root)
    
    # 学習スクリプトを実行
    train_script = os.path.join(script_dir, "train_with_generated_data.py")
    
    cmd = [
        sys.executable, train_script,
        "--data_dir", "generated_data",
        "--batch_size", "16",  # メモリ使用量を考慮して小さめに設定
        "--learning_rate", "3e-4",
        "--num_epochs", "5",   # 最初は少なめに設定
        "--save_path", "models/hierarchical_model_generated.pth",
        "--eval_split", "0.2",
        "--max_seq_len", "256",  # メモリ使用量を考慮して小さめに設定
        "--use_wandb",
        "--wandb_project", "fof-training",
        "--wandb_run_name", "default_training_run"
        # --remove_duplicates はデフォルトで有効
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
