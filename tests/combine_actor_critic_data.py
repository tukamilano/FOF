#!/usr/bin/env python3
"""
Actor-Criticデータを統合するスクリプト
複数のバッチファイルを統合して学習用の単一ファイルを作成
"""
import os
import json
import glob
from pathlib import Path
from typing import List, Dict, Any


def combine_actor_critic_data(
    data_dir: str = "actor_critic_data",
    output_dir: str = "actor_critic_data_combined"
):
    """
    Actor-Criticデータを統合
    
    Args:
        data_dir: 入力データディレクトリ
        output_dir: 出力ディレクトリ
    """
    print("🔄 Combining Actor-Critic data...")
    
    # 出力ディレクトリを作成
    os.makedirs(output_dir, exist_ok=True)
    
    # 成功データを統合
    successful_files = glob.glob(os.path.join(data_dir, "successful_tactics_*.json"))
    successful_files.sort()
    
    print(f"📁 Found {len(successful_files)} successful tactics files")
    
    all_successful = []
    for file_path in successful_files:
        print(f"  Loading {os.path.basename(file_path)}...")
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            all_successful.extend(data)
    
    # 失敗データを統合
    failed_files = glob.glob(os.path.join(data_dir, "failed_tactics_*.json"))
    failed_files.sort()
    
    print(f"📁 Found {len(failed_files)} failed tactics files")
    
    all_failed = []
    for file_path in failed_files:
        print(f"  Loading {os.path.basename(file_path)}...")
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            all_failed.extend(data)
    
    print(f"\n📊 Data summary:")
    print(f"  Total successful tactics: {len(all_successful)}")
    print(f"  Total failed tactics: {len(all_failed)}")
    print(f"  Total examples: {len(all_successful) + len(all_failed)}")
    
    # 統合されたデータを保存
    success_output_path = os.path.join(output_dir, "successful_tactics.json")
    failed_output_path = os.path.join(output_dir, "failed_tactics.json")
    
    print(f"\n💾 Saving combined data...")
    print(f"  Successful tactics: {success_output_path}")
    with open(success_output_path, 'w', encoding='utf-8') as f:
        json.dump(all_successful, f, ensure_ascii=False, indent=2)
    
    print(f"  Failed tactics: {failed_output_path}")
    with open(failed_output_path, 'w', encoding='utf-8') as f:
        json.dump(all_failed, f, ensure_ascii=False, indent=2)
    
    # 統計情報を保存
    stats = {
        "total_successful_tactics": len(all_successful),
        "total_failed_tactics": len(all_failed),
        "total_examples": len(all_successful) + len(all_failed),
        "successful_files_processed": len(successful_files),
        "failed_files_processed": len(failed_files),
        "success_rate": len(all_successful) / (len(all_successful) + len(all_failed)) if (len(all_successful) + len(all_failed)) > 0 else 0
    }
    
    stats_path = os.path.join(output_dir, "data_stats.json")
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    print(f"📊 Statistics saved to: {stats_path}")
    print(f"\n🎉 Data combination completed!")
    print(f"📁 Output directory: {output_dir}")
    
    return stats


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Combine Actor-Critic data files")
    parser.add_argument("--data_dir", type=str, default="actor_critic_data", help="Input data directory")
    parser.add_argument("--output_dir", type=str, default="actor_critic_data_combined", help="Output directory")
    
    args = parser.parse_args()
    
    stats = combine_actor_critic_data(args.data_dir, args.output_dir)
    
    print(f"\n📈 Final statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
"""
Actor-Criticデータを統合するスクリプト
複数のバッチファイルを統合して学習用の単一ファイルを作成
"""
import os
import json
import glob
from pathlib import Path
from typing import List, Dict, Any


def combine_actor_critic_data(
    data_dir: str = "actor_critic_data",
    output_dir: str = "actor_critic_data_combined"
):
    """
    Actor-Criticデータを統合
    
    Args:
        data_dir: 入力データディレクトリ
        output_dir: 出力ディレクトリ
    """
    print("🔄 Combining Actor-Critic data...")
    
    # 出力ディレクトリを作成
    os.makedirs(output_dir, exist_ok=True)
    
    # 成功データを統合
    successful_files = glob.glob(os.path.join(data_dir, "successful_tactics_*.json"))
    successful_files.sort()
    
    print(f"📁 Found {len(successful_files)} successful tactics files")
    
    all_successful = []
    for file_path in successful_files:
        print(f"  Loading {os.path.basename(file_path)}...")
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            all_successful.extend(data)
    
    # 失敗データを統合
    failed_files = glob.glob(os.path.join(data_dir, "failed_tactics_*.json"))
    failed_files.sort()
    
    print(f"📁 Found {len(failed_files)} failed tactics files")
    
    all_failed = []
    for file_path in failed_files:
        print(f"  Loading {os.path.basename(file_path)}...")
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            all_failed.extend(data)
    
    print(f"\n📊 Data summary:")
    print(f"  Total successful tactics: {len(all_successful)}")
    print(f"  Total failed tactics: {len(all_failed)}")
    print(f"  Total examples: {len(all_successful) + len(all_failed)}")
    
    # 統合されたデータを保存
    success_output_path = os.path.join(output_dir, "successful_tactics.json")
    failed_output_path = os.path.join(output_dir, "failed_tactics.json")
    
    print(f"\n💾 Saving combined data...")
    print(f"  Successful tactics: {success_output_path}")
    with open(success_output_path, 'w', encoding='utf-8') as f:
        json.dump(all_successful, f, ensure_ascii=False, indent=2)
    
    print(f"  Failed tactics: {failed_output_path}")
    with open(failed_output_path, 'w', encoding='utf-8') as f:
        json.dump(all_failed, f, ensure_ascii=False, indent=2)
    
    # 統計情報を保存
    stats = {
        "total_successful_tactics": len(all_successful),
        "total_failed_tactics": len(all_failed),
        "total_examples": len(all_successful) + len(all_failed),
        "successful_files_processed": len(successful_files),
        "failed_files_processed": len(failed_files),
        "success_rate": len(all_successful) / (len(all_successful) + len(all_failed)) if (len(all_successful) + len(all_failed)) > 0 else 0
    }
    
    stats_path = os.path.join(output_dir, "data_stats.json")
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    print(f"📊 Statistics saved to: {stats_path}")
    print(f"\n🎉 Data combination completed!")
    print(f"📁 Output directory: {output_dir}")
    
    return stats


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Combine Actor-Critic data files")
    parser.add_argument("--data_dir", type=str, default="actor_critic_data", help="Input data directory")
    parser.add_argument("--output_dir", type=str, default="actor_critic_data_combined", help="Output directory")
    
    args = parser.parse_args()
    
    stats = combine_actor_critic_data(args.data_dir, args.output_dir)
    
    print(f"\n📈 Final statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
