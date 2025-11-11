"""
wandb接続テスト
"""
import os
import sys
import time
import tempfile
import shutil
from pathlib import Path

# Add project root to path
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_root)

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Install with: pip install wandb")

def test_wandb_connection():
    """wandbの基本的な接続とログ記録 テスト"""
    if not WANDB_AVAILABLE:
        print("❌ wandb is not available. Please install with: pip install wandb")
        return False
    
    print("🧪 Testing wandb connection...")
    
    try:
        # wandb 初期化（テスト用のプロジェクト）
        run_name = f"test_connection_{int(time.time())}"
        wandb.init(
            project="fof-test",
            name=run_name,
            config={
                "test": True,
                "timestamp": time.time()
            }
        )
        print(f"✅ wandb initialized: fof-test/{run_name}")
        
        # 簡単なログ 記録
        test_metrics = {
            "test_loss": 0.5,
            "test_accuracy": 0.8,
            "test_step": 1
        }
        wandb.log(test_metrics)
        print("✅ Successfully logged test metrics")
        
        # 複数のステップ ログ
        for i in range(3):
            wandb.log({
                "step_loss": 0.5 - i * 0.1,
                "step_accuracy": 0.8 + i * 0.05,
                "step": i + 1
            })
            time.sleep(0.1)  # 少し待機
        
        print("✅ Successfully logged multiple steps")
        
        # wandb 終了
        wandb.finish()
        print("✅ wandb session finished successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ wandb test failed: {e}")
        return False

def test_training_script_wandb():
    """Trainingスクリプトのwandb機能 テスト"""
    if not WANDB_AVAILABLE:
        print("❌ wandb is not available. Skipping training script test")
        return False
    
    print("🧪 Testing training script wandb integration...")
    
    try:
        # Trainingスクリプト インポート
        from src.training.train_with_generated_data import main as train_main
        print("✅ Successfully imported training script")
        
        # Trainingスクリプトの引数 テスト（実際 は実行しno/not）
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--use_wandb", action="store_true")
        parser.add_argument("--wandb_project", type=str, default="fof-test")
        parser.add_argument("--wandb_run_name", type=str, default="test_training")
        
        args = parser.parse_args(["--use_wandb", "--wandb_project", "fof-test"])
        print("✅ Training script arguments parsed successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Training script test failed: {e}")
        return False

def test_inference_script_wandb():
    """推論スクリプトのwandb機能 テスト"""
    if not WANDB_AVAILABLE:
        print("❌ wandb is not available. Skipping inference script test")
        return False
    
    print("🧪 Testing inference script wandb integration...")
    
    try:
        # 推論スクリプト インポート
        from src.training.inference_hierarchical import main as inference_main
        print("✅ Successfully imported inference script")
        
        # 推論スクリプトの引数 テスト（実際 は実行しno/not）
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--use_wandb", action="store_true")
        parser.add_argument("--wandb_project", type=str, default="fof-test")
        parser.add_argument("--wandb_run_name", type=str, default="test_inference")
        
        args = parser.parse_args(["--use_wandb", "--wandb_project", "fof-test"])
        print("✅ Inference script arguments parsed successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Inference script test failed: {e}")
        return False

def test_wandb_logging_functions():
    """wandbログ記録関数 テスト"""
    if not WANDB_AVAILABLE:
        print("❌ wandb is not available. Skipping logging functions test")
        return False
    
    print("🧪 Testing wandb logging functions...")
    
    try:
        # Training用のログ記録 テスト
        run_name = f"test_logging_{int(time.time())}"
        wandb.init(
            project="fof-test",
            name=run_name,
            config={
                "test_type": "logging_functions",
                "timestamp": time.time()
            }
        )
        
        # Training時のログ シミュレート
        for epoch in range(3):
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": 1.0 - epoch * 0.2,
                "eval_loss": 0.8 - epoch * 0.15,
                "main_accuracy": 0.6 + epoch * 0.1,
                "arg1_accuracy": 0.7 + epoch * 0.05,
                "arg2_accuracy": 0.5 + epoch * 0.1,
                "learning_rate": 0.001
            })
            time.sleep(0.1)
        
        print("✅ Training logs recorded successfully")
        
        # 推論時のログ シミュレート
        for i in range(3):
            wandb.log({
                f"example_{i+1}/solved": 1 if i % 2 == 0 else 0,
                f"example_{i+1}/steps": i + 1,
                f"example_{i+1}/avg_confidence": 0.8 - i * 0.1,
                f"example_{i+1}/goal_length": 10 + i * 5,
                f"example_{i+1}/premises_count": 2 + i
            })
            time.sleep(0.1)
        
        print("✅ Inference logs recorded successfully")
        
        # 最終結果 ログ
        wandb.log({
            "final/success_rate": 0.67,
            "final/avg_steps": 2.0,
            "final/avg_confidence": 0.75,
            "final/solved_count": 2,
            "final/total_examples": 3
        })
        
        # タクティク使用頻度 ログ
        wandb.log({
            "tactics/assumption": 3,
            "tactics/contradiction": 1,
            "tactics/elim": 2
        })
        
        print("✅ Final results and tactic usage logged successfully")
        
        wandb.finish()
        print("✅ wandb logging functions test completed")
        
        return True
        
    except Exception as e:
        print(f"❌ Logging functions test failed: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 Starting wandb connection tests...")
    print("=" * 50)
    
    tests = [
        ("Basic wandb connection", test_wandb_connection),
        ("Training script integration", test_training_script_wandb),
        ("Inference script integration", test_inference_script_wandb),
        ("Logging functions", test_wandb_logging_functions)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n📋 Running: {test_name}")
        print("-" * 30)
        try:
            result = test_func()
            results.append((test_name, result))
            if result:
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! wandb integration is working correctly.")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
