#!/usr/bin/env python3
"""
parameter.pyの使用例

このスクリプトは、新しく作成したparameter.pyシステムの使用方法を示します。
"""

from parameter import (
    default_params, get_model_params, get_training_params, 
    get_generation_params, get_system_params, 
    update_parameters, apply_preset, PRESETS
)


def demonstrate_basic_usage():
    """基本的な使用方法のデモンストレーション"""
    print("=== 基本的な使用方法 ===")
    
    # デフォルトパラメータの取得
    model_params = get_model_params()
    training_params = get_training_params()
    generation_params = get_generation_params()
    
    print(f"デフォルトのモデル d_model: {model_params.d_model}")
    print(f"デフォルトの学習 batch_size: {training_params.batch_size}")
    print(f"デフォルトの生成 count: {generation_params.count}")
    print()


def demonstrate_parameter_updates():
    """パラメータ更新のデモンストレーション"""
    print("=== パラメータ更新 ===")
    
    # 個別パラメータの更新
    default_params.update_model_params(d_model=256, nhead=8)
    default_params.update_training_params(batch_size=64, learning_rate=1e-4)
    default_params.update_generation_params(count=100, difficulty=0.8)
    
    print("更新後のパラメータ:")
    print(f"モデル d_model: {get_model_params().d_model}")
    print(f"学習 batch_size: {get_training_params().batch_size}")
    print(f"生成 count: {get_generation_params().count}")
    print()


def demonstrate_preset_usage():
    """プリセット使用のデモンストレーション"""
    print("=== プリセット使用 ===")
    
    # 利用可能なプリセットを表示
    print("利用可能なプリセット:")
    for preset_name in PRESETS.keys():
        print(f"  - {preset_name}")
    print()
    
    # プリセットの適用
    print("fast_trainingプリセットを適用:")
    apply_preset("fast_training")
    print(f"適用後 d_model: {get_model_params().d_model}")
    print(f"適用後 batch_size: {get_training_params().batch_size}")
    print(f"適用後 count: {get_generation_params().count}")
    print()


def demonstrate_batch_updates():
    """一括更新のデモンストレーション"""
    print("=== 一括更新 ===")
    
    # 複数カテゴリのパラメータを一度に更新
    update_parameters(
        model={
            "d_model": 512,
            "nhead": 16,
            "num_layers": 6
        },
        training={
            "batch_size": 128,
            "learning_rate": 5e-5
        },
        generation={
            "count": 200,
            "max_steps": 10
        }
    )
    
    print("一括更新後のパラメータ:")
    print(f"モデル d_model: {get_model_params().d_model}, nhead: {get_model_params().nhead}")
    print(f"学習 batch_size: {get_training_params().batch_size}, learning_rate: {get_training_params().learning_rate}")
    print(f"生成 count: {get_generation_params().count}, max_steps: {get_generation_params().max_steps}")
    print()


def demonstrate_device_management():
    """デバイス管理のデモンストレーション"""
    print("=== デバイス管理 ===")
    
    # デバイス設定
    default_params.update_system_params(device="auto")
    print(f"デバイス設定: {default_params.get_device()}")
    
    # フィルタリングフラグの取得
    filter_flags = default_params.get_filter_flags()
    print(f"フィルタリングフラグ: {filter_flags}")
    print()


def demonstrate_parameter_export():
    """パラメータエクスポートのデモンストレーション"""
    print("=== パラメータエクスポート ===")
    
    # 全パラメータを辞書形式で取得
    all_params = default_params.to_dict()
    print("全パラメータ（辞書形式）:")
    for category, params in all_params.items():
        print(f"{category}: {params}")
    print()


def demonstrate_custom_preset():
    """カスタムプリセットの作成例"""
    print("=== カスタムプリセット作成例 ===")
    
    # カスタムプリセットの定義
    custom_preset = {
        "model": {
            "d_model": 64,
            "nhead": 2,
            "num_layers": 1,
            "dim_feedforward": 128
        },
        "training": {
            "batch_size": 8,
            "learning_rate": 1e-2
        },
        "generation": {
            "count": 5,
            "max_steps": 3,
            "difficulty": 0.2
        }
    }
    
    # カスタムプリセットの適用
    update_parameters(**custom_preset)
    print("カスタムプリセット適用後:")
    print(f"モデル d_model: {get_model_params().d_model}")
    print(f"学習 batch_size: {get_training_params().batch_size}")
    print(f"生成 count: {get_generation_params().count}")
    print()


if __name__ == "__main__":
    print("parameter.py システムの使用例")
    print("=" * 50)
    print()
    
    demonstrate_basic_usage()
    demonstrate_parameter_updates()
    demonstrate_preset_usage()
    demonstrate_batch_updates()
    demonstrate_device_management()
    demonstrate_parameter_export()
    demonstrate_custom_preset()
    
    print("=" * 50)
    print("デモンストレーション完了！")
    print()
    print("実際の使用では、以下のようにしてパラメータを取得できます：")
    print("  from parameter import get_model_params, get_training_params")
    print("  model_params = get_model_params()")
    print("  training_params = get_training_params()")
    print("  # パラメータを使用...")
