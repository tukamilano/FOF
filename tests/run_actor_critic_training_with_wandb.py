#!/usr/bin/env python3
"""
Actor-Critic学習をwandbで監視するスクリプト
重い実行を避けて、学習の状況をチェック
"""
import os
import sys
import subprocess
import argparse
import time
from pathlib import Path


def check_wandb_connection():
    """wandbの接続をテスト"""
    print("🔍 Checking wandb connection...")
    try:
        import wandb
        print("✅ Wandb is available")
        return True
    except ImportError:
        print("❌ Wandb is not installed")
        print("   Install with: pip install wandb")
        return False


def run_training_with_wandb(
    data_dir: str = "actor_critic_data_combined",
    output_dir: str = "actor_critic_models",
    num_epochs: int = 5,
    batch_size: int = 16,
    learning_rate: float = 1e-4,
    wandb_project: str = "fof-actor-critic",
    wandb_run_name: str = None,
    dry_run: bool = False
):
    """
    wandbで学習を監視しながらActor-Critic学習を実行
    
    Args:
        data_dir: データディレクトリ
        output_dir: 出力ディレクトリ
        num_epochs: エポック数
        batch_size: バッチサイズ
        learning_rate: 学習率
        wandb_project: wandbプロジェクト名
        wandb_run_name: wandbラン名
        dry_run: 実際には実行せずにコマンドを表示
    """
    print("🚀 Actor-Critic Training with Wandb Monitoring")
    print("=" * 60)
    
    # wandb接続チェック
    if not check_wandb_connection():
        print("❌ Cannot proceed without wandb")
        return False
    
    # データディレクトリの存在確認
    if not os.path.exists(data_dir):
        print(f"❌ Data directory not found: {data_dir}")
        print("   Run: python3 combine_actor_critic_data.py")
        return False
    
    # 事前学習済みモデルの存在確認
    pretrained_model_path = "models/pretrained_model.pth"
    if not os.path.exists(pretrained_model_path):
        print(f"❌ Pretrained model not found: {pretrained_model_path}")
        return False
    
    # 学習コマンドを構築
    cmd = [
        "python3", "train_actor_critic.py",
        "--data_dir", data_dir,
        "--output_dir", output_dir,
        "--num_epochs", str(num_epochs),
        "--batch_size", str(batch_size),
        "--learning_rate", str(learning_rate),
        "--use_wandb",
        "--wandb_project", wandb_project
    ]
    
    if wandb_run_name:
        cmd.extend(["--wandb_run_name", wandb_run_name])
    
    print(f"📋 Training command:")
    print(f"   {' '.join(cmd)}")
    print()
    
    if dry_run:
        print("🔍 Dry run mode - command not executed")
        print("   Remove --dry_run to actually run the training")
        return True
    
    # 学習を実行
    print("🎯 Starting training...")
    print("   Monitor progress at: https://wandb.ai")
    print("   Press Ctrl+C to stop training")
    print("=" * 60)
    
    try:
        # 学習プロセスを開始
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # リアルタイムで出力を表示
        for line in iter(process.stdout.readline, ''):
            print(line.rstrip())
        
        # プロセスが終了するまで待機
        process.wait()
        
        if process.returncode == 0:
            print("\n🎉 Training completed successfully!")
            print(f"📁 Output directory: {output_dir}")
            print("📊 Check wandb dashboard for detailed metrics")
            return True
        else:
            print(f"\n❌ Training failed with return code: {process.returncode}")
            return False
            
    except KeyboardInterrupt:
        print("\n⏹️  Training interrupted by user")
        process.terminate()
        return False
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        return False


def show_wandb_info():
    """wandbの情報を表示"""
    print("📊 Wandb Information:")
    print("=" * 40)
    print("1. Create account at: https://wandb.ai")
    print("2. Login with: wandb login")
    print("3. View dashboard at: https://wandb.ai")
    print("4. Project will be: fof-actor-critic")
    print()


def main():
    parser = argparse.ArgumentParser(description="Run Actor-Critic training with wandb monitoring")
    parser.add_argument("--data_dir", type=str, default="actor_critic_data_combined", help="data directory")
    parser.add_argument("--output_dir", type=str, default="actor_critic_models", help="output directory")
    parser.add_argument("--num_epochs", type=int, default=5, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--wandb_project", type=str, default="fof-actor-critic", help="wandb project name")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="wandb run name")
    parser.add_argument("--dry_run", action="store_true", help="show command without executing")
    parser.add_argument("--show_wandb_info", action="store_true", help="show wandb setup information")
    
    args = parser.parse_args()
    
    if args.show_wandb_info:
        show_wandb_info()
        return
    
    success = run_training_with_wandb(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        dry_run=args.dry_run
    )
    
    if success:
        print("\n✅ Script completed successfully!")
    else:
        print("\n❌ Script failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
"""
Actor-Critic学習をwandbで監視するスクリプト
重い実行を避けて、学習の状況をチェック
"""
import os
import sys
import subprocess
import argparse
import time
from pathlib import Path


def check_wandb_connection():
    """wandbの接続をテスト"""
    print("🔍 Checking wandb connection...")
    try:
        import wandb
        print("✅ Wandb is available")
        return True
    except ImportError:
        print("❌ Wandb is not installed")
        print("   Install with: pip install wandb")
        return False


def run_training_with_wandb(
    data_dir: str = "actor_critic_data_combined",
    output_dir: str = "actor_critic_models",
    num_epochs: int = 5,
    batch_size: int = 16,
    learning_rate: float = 1e-4,
    wandb_project: str = "fof-actor-critic",
    wandb_run_name: str = None,
    dry_run: bool = False
):
    """
    wandbで学習を監視しながらActor-Critic学習を実行
    
    Args:
        data_dir: データディレクトリ
        output_dir: 出力ディレクトリ
        num_epochs: エポック数
        batch_size: バッチサイズ
        learning_rate: 学習率
        wandb_project: wandbプロジェクト名
        wandb_run_name: wandbラン名
        dry_run: 実際には実行せずにコマンドを表示
    """
    print("🚀 Actor-Critic Training with Wandb Monitoring")
    print("=" * 60)
    
    # wandb接続チェック
    if not check_wandb_connection():
        print("❌ Cannot proceed without wandb")
        return False
    
    # データディレクトリの存在確認
    if not os.path.exists(data_dir):
        print(f"❌ Data directory not found: {data_dir}")
        print("   Run: python3 combine_actor_critic_data.py")
        return False
    
    # 事前学習済みモデルの存在確認
    pretrained_model_path = "models/pretrained_model.pth"
    if not os.path.exists(pretrained_model_path):
        print(f"❌ Pretrained model not found: {pretrained_model_path}")
        return False
    
    # 学習コマンドを構築
    cmd = [
        "python3", "train_actor_critic.py",
        "--data_dir", data_dir,
        "--output_dir", output_dir,
        "--num_epochs", str(num_epochs),
        "--batch_size", str(batch_size),
        "--learning_rate", str(learning_rate),
        "--use_wandb",
        "--wandb_project", wandb_project
    ]
    
    if wandb_run_name:
        cmd.extend(["--wandb_run_name", wandb_run_name])
    
    print(f"📋 Training command:")
    print(f"   {' '.join(cmd)}")
    print()
    
    if dry_run:
        print("🔍 Dry run mode - command not executed")
        print("   Remove --dry_run to actually run the training")
        return True
    
    # 学習を実行
    print("🎯 Starting training...")
    print("   Monitor progress at: https://wandb.ai")
    print("   Press Ctrl+C to stop training")
    print("=" * 60)
    
    try:
        # 学習プロセスを開始
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # リアルタイムで出力を表示
        for line in iter(process.stdout.readline, ''):
            print(line.rstrip())
        
        # プロセスが終了するまで待機
        process.wait()
        
        if process.returncode == 0:
            print("\n🎉 Training completed successfully!")
            print(f"📁 Output directory: {output_dir}")
            print("📊 Check wandb dashboard for detailed metrics")
            return True
        else:
            print(f"\n❌ Training failed with return code: {process.returncode}")
            return False
            
    except KeyboardInterrupt:
        print("\n⏹️  Training interrupted by user")
        process.terminate()
        return False
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        return False


def show_wandb_info():
    """wandbの情報を表示"""
    print("📊 Wandb Information:")
    print("=" * 40)
    print("1. Create account at: https://wandb.ai")
    print("2. Login with: wandb login")
    print("3. View dashboard at: https://wandb.ai")
    print("4. Project will be: fof-actor-critic")
    print()


def main():
    parser = argparse.ArgumentParser(description="Run Actor-Critic training with wandb monitoring")
    parser.add_argument("--data_dir", type=str, default="actor_critic_data_combined", help="data directory")
    parser.add_argument("--output_dir", type=str, default="actor_critic_models", help="output directory")
    parser.add_argument("--num_epochs", type=int, default=5, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--wandb_project", type=str, default="fof-actor-critic", help="wandb project name")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="wandb run name")
    parser.add_argument("--dry_run", action="store_true", help="show command without executing")
    parser.add_argument("--show_wandb_info", action="store_true", help="show wandb setup information")
    
    args = parser.parse_args()
    
    if args.show_wandb_info:
        show_wandb_info()
        return
    
    success = run_training_with_wandb(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        dry_run=args.dry_run
    )
    
    if success:
        print("\n✅ Script completed successfully!")
    else:
        print("\n❌ Script failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
