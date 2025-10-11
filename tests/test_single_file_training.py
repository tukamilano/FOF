#!/usr/bin/env python3
"""
1ファイル分だけ学習して動作確認するテストスクリプト
"""
import os
import sys
import shutil
import tempfile

# プロジェクトルートをパスに追加
project_root = os.path.dirname(os.path.dirname(__file__))  # tests/の親ディレクトリ
sys.path.insert(0, project_root)

def main():
    # テスト用の一時ディレクトリを作成
    temp_dir = tempfile.mkdtemp(prefix="fof_test_")
    test_data_dir = os.path.join(temp_dir, "test_data")
    os.makedirs(test_data_dir)
    
    try:
        # 1つのファイルだけをコピー
        source_file = os.path.join(project_root, "generated_data", "test_output_00001.json")
        dest_file = os.path.join(test_data_dir, "test_output_00001.json")
        shutil.copy2(source_file, dest_file)
        
        print(f"📁 Test data directory: {test_data_dir}")
        print(f"📄 Using file: test_output_00001.json")
        
        # 学習スクリプトを実行
        cmd = [
            sys.executable, "src/training/train_with_generated_data.py",
            "--data_dir", test_data_dir,
            "--use_wandb",
            "--wandb_project", "fof-test-single-file",
            "--num_epochs", "3",
            "--batch_size", "2",  # さらに小さく
            "--arg1_loss_weight", "0.8",
            "--arg2_loss_weight", "0.8",
            "--max_seq_len", "128",  # シーケンス長も短く
            "--save_path", "models/test_single_file_model.pth"
        ]
        
        print(f"🚀 Running command: {' '.join(cmd)}")
        print("=" * 60)
        
        # コマンドを実行
        import subprocess
        result = subprocess.run(cmd, cwd=project_root)
        
        if result.returncode == 0:
            print("\n✅ Test completed successfully!")
        else:
            print(f"\n❌ Test failed with return code: {result.returncode}")
            
    finally:
        # 一時ディレクトリを削除
        shutil.rmtree(temp_dir)
        print(f"🧹 Cleaned up temporary directory: {temp_dir}")

if __name__ == "__main__":
    main()
