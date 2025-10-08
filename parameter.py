"""
ハイパーパラメータとデフォルト値の管理モジュール

このモジュールは、プロジェクト全体で使用されるハイパーパラメータとデフォルト値を
一元管理し、設定の一貫性と保守性を向上させます。
"""

from enum import Enum
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


class DeviceType(Enum):
    """デバイス選択の列挙型"""
    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"


class DataFilterType(Enum):
    """データフィルタリングの列挙型"""
    ALL = "all"
    SUCCESSFUL_ONLY = "successful_only"
    TACTIC_SUCCESS_ONLY = "tactic_success_only"


@dataclass
class ModelParameters:
    """Transformerモデルのハイパーパラメータ"""
    # 基本設定
    vocab_size: int = 0  # 動的に設定される
    pad_id: int = 0  # 動的に設定される
    
    # 階層分類設定
    use_hierarchical_classification: bool = True  # 階層分類を使用するか
    num_main_classes: int = 0  # 主タクティクのクラス数（動的に設定）
    num_arg1_classes: int = 0  # 第1引数のクラス数（動的に設定）
    num_arg2_classes: int = 0  # 第2引数のクラス数（動的に設定）
    
    # シーケンス長設定
    max_seq_len: int = 512
    
    # Transformerアーキテクチャ
    d_model: int = 128
    nhead: int = 4
    num_layers: int = 2
    dim_feedforward: int = 256
    dropout: float = 0.1
    
    # 位置エンコーディング
    max_position_len: int = 512
    positional_dropout: float = 0.1
    
    # セグメントエンコーディング
    max_segments: int = 100  # 前提の最大数に対応


@dataclass
class TrainingParameters:
    """学習関連のパラメータ"""
    # 基本学習設定
    learning_rate: float = 3e-4
    batch_size: int = 32
    num_epochs: int = 10
    
    # データ収集設定
    collect_data: bool = False
    work_file: str = "temp_work.json"
    dataset_file: str = "training_data.json"
    filter_type: DataFilterType = DataFilterType.ALL
    
    # 評価設定
    warmup_iterations: int = 2
    max_interactions: int = 5


@dataclass
class GenerationParameters:
    """公式生成と評価のパラメータ"""
    # 基本生成設定
    count: int = 10
    difficulty: float = 0.5
    seed: int = 7
    max_len: int = 50
    
    # 変数設定
    variables: List[str] = None
    allow_const: bool = False
    
    # 証明探索設定
    max_steps: int = 5
    max_depth: int = 8  # auto_classical用
    
    def __post_init__(self):
        if self.variables is None:
            self.variables = ["a", "b", "c"]


@dataclass
class SystemParameters:
    """システムとデバイス関連のパラメータ"""
    device: DeviceType = DeviceType.AUTO
    root_dir: str = ""  # 動的に設定される
    token_py_path: str = ""  # 動的に設定される
    pyprover_dir: str = ""  # 動的に設定される


@dataclass
class SpecialTokens:
    """特殊トークンの定義"""
    PAD = "[PAD]"
    CLS = "[CLS]"
    SEP = "[SEP]"
    UNK = "[UNK]"
    EOS = "[EOS]"


@dataclass
class HierarchicalLabels:
    """階層分類用のラベル管理"""
    main_tactics: List[str] = None  # 主タクティクのリスト
    arg1_values: List[str] = None   # 第1引数の値のリスト
    arg2_values: List[str] = None   # 第2引数の値のリスト
    
    def __post_init__(self):
        if self.main_tactics is None:
            # デフォルトの主タクティク
            self.main_tactics = [
                "assumption", "intro", "split", "left", "right", "add_dn", 
                "apply", "destruct", "specialize"
            ]
        if self.arg1_values is None:
            # デフォルトの第1引数（数値）
            self.arg1_values = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
        if self.arg2_values is None:
            # デフォルトの第2引数（数値）
            self.arg2_values = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    
    def get_main_to_id(self) -> Dict[str, int]:
        """主タクティクの文字列からIDへのマッピング"""
        return {tactic: i for i, tactic in enumerate(self.main_tactics)}
    
    def get_arg1_to_id(self) -> Dict[str, int]:
        """第1引数の文字列からIDへのマッピング"""
        return {arg: i for i, arg in enumerate(self.arg1_values)}
    
    def get_arg2_to_id(self) -> Dict[str, int]:
        """第2引数の文字列からIDへのマッピング"""
        return {arg: i for i, arg in enumerate(self.arg2_values)}
    
    def get_id_to_main(self) -> List[str]:
        """IDから主タクティクの文字列へのマッピング"""
        return self.main_tactics.copy()
    
    def get_id_to_arg1(self) -> List[str]:
        """IDから第1引数の文字列へのマッピング"""
        return self.arg1_values.copy()
    
    def get_id_to_arg2(self) -> List[str]:
        """IDから第2引数の文字列へのマッピング"""
        return self.arg2_values.copy()


class ParameterManager:
    """パラメータ管理クラス"""
    
    def __init__(self):
        self.model = ModelParameters()
        self.training = TrainingParameters()
        self.generation = GenerationParameters()
        self.system = SystemParameters()
        self.special_tokens = SpecialTokens()
        self.hierarchical_labels = HierarchicalLabels()
    
    def update_model_params(self, **kwargs) -> None:
        """モデルパラメータを更新"""
        for key, value in kwargs.items():
            if hasattr(self.model, key):
                setattr(self.model, key, value)
            else:
                raise ValueError(f"Unknown model parameter: {key}")
    
    def update_training_params(self, **kwargs) -> None:
        """学習パラメータを更新"""
        for key, value in kwargs.items():
            if hasattr(self.training, key):
                # filter_typeの場合は文字列をDataFilterTypeに変換
                if key == "filter_type" and isinstance(value, str):
                    try:
                        value = DataFilterType(value)
                    except ValueError:
                        # 無効な値の場合はそのまま文字列として保持
                        pass
                setattr(self.training, key, value)
            else:
                raise ValueError(f"Unknown training parameter: {key}")
    
    def update_generation_params(self, **kwargs) -> None:
        """生成パラメータを更新"""
        for key, value in kwargs.items():
            if hasattr(self.generation, key):
                setattr(self.generation, key, value)
            else:
                raise ValueError(f"Unknown generation parameter: {key}")
    
    def update_system_params(self, **kwargs) -> None:
        """システムパラメータを更新"""
        for key, value in kwargs.items():
            if hasattr(self.system, key):
                # deviceの場合は文字列をDeviceTypeに変換
                if key == "device" and isinstance(value, str):
                    try:
                        value = DeviceType(value)
                    except ValueError:
                        # 無効な値の場合はそのまま文字列として保持
                        pass
                setattr(self.system, key, value)
            else:
                raise ValueError(f"Unknown system parameter: {key}")
    
    def get_device(self) -> str:
        """デバイス文字列を取得"""
        if isinstance(self.system.device, DeviceType):
            if self.system.device == DeviceType.AUTO:
                try:
                    import torch
                    return "cuda" if torch.cuda.is_available() else "cpu"
                except ImportError:
                    return "cpu"
            return self.system.device.value
        else:
            # 文字列の場合はそのまま返す
            if self.system.device == "auto":
                try:
                    import torch
                    return "cuda" if torch.cuda.is_available() else "cpu"
                except ImportError:
                    return "cpu"
            return self.system.device
    
    def get_filter_flags(self) -> Dict[str, bool]:
        """データフィルタリングフラグを取得"""
        if isinstance(self.training.filter_type, DataFilterType):
            return {
                "filter_successful_only": self.training.filter_type == DataFilterType.SUCCESSFUL_ONLY,
                "filter_tactic_success_only": self.training.filter_type == DataFilterType.TACTIC_SUCCESS_ONLY
            }
        else:
            # 文字列の場合は比較
            return {
                "filter_successful_only": self.training.filter_type == "successful_only",
                "filter_tactic_success_only": self.training.filter_type == "tactic_success_only"
            }
    
    def to_dict(self) -> Dict[str, Any]:
        """全パラメータを辞書形式で取得"""
        return {
            "model": self.model.__dict__,
            "training": self.training.__dict__,
            "generation": self.generation.__dict__,
            "system": self.system.__dict__,
            "special_tokens": self.special_tokens.__dict__
        }


# デフォルトのパラメータマネージャーインスタンス
default_params = ParameterManager()


# 便利な関数群
def get_model_params() -> ModelParameters:
    """モデルパラメータを取得"""
    return default_params.model


def get_training_params() -> TrainingParameters:
    """学習パラメータを取得"""
    return default_params.training


def get_generation_params() -> GenerationParameters:
    """生成パラメータを取得"""
    return default_params.generation


def get_system_params() -> SystemParameters:
    """システムパラメータを取得"""
    return default_params.system


def get_special_tokens() -> SpecialTokens:
    """特殊トークンを取得"""
    return default_params.special_tokens


def get_hierarchical_labels() -> HierarchicalLabels:
    """階層分類用ラベルを取得"""
    return default_params.hierarchical_labels


def update_parameters(**kwargs) -> None:
    """パラメータを一括更新"""
    for category, params in kwargs.items():
        if category == "model":
            default_params.update_model_params(**params)
        elif category == "training":
            default_params.update_training_params(**params)
        elif category == "generation":
            default_params.update_generation_params(**params)
        elif category == "system":
            default_params.update_system_params(**params)
        else:
            raise ValueError(f"Unknown parameter category: {category}")


# よく使用される設定のプリセット
PRESETS = {
    "fast_training": {
        "model": {"d_model": 64, "nhead": 2, "num_layers": 1, "dim_feedforward": 128},
        "training": {"batch_size": 16, "learning_rate": 1e-3},
        "generation": {"count": 5, "max_steps": 3}
    },
    "high_quality": {
        "model": {"d_model": 256, "nhead": 8, "num_layers": 4, "dim_feedforward": 512},
        "training": {"batch_size": 64, "learning_rate": 1e-4},
        "generation": {"count": 50, "max_steps": 10}
    },
    "debug": {
        "model": {"d_model": 32, "nhead": 1, "num_layers": 1, "dim_feedforward": 64},
        "training": {"batch_size": 4, "learning_rate": 1e-2},
        "generation": {"count": 2, "max_steps": 2}
    }
}


def apply_preset(preset_name: str) -> None:
    """プリセット設定を適用"""
    if preset_name not in PRESETS:
        raise ValueError(f"Unknown preset: {preset_name}. Available: {list(PRESETS.keys())}")
    
    update_parameters(**PRESETS[preset_name])


# 使用例とドキュメント
if __name__ == "__main__":
    # 基本的な使用方法
    print("=== パラメータ管理システムの使用例 ===")
    
    # デフォルトパラメータの表示
    print("\n1. デフォルトパラメータ:")
    print(f"モデル d_model: {get_model_params().d_model}")
    print(f"学習 batch_size: {get_training_params().batch_size}")
    print(f"生成 count: {get_generation_params().count}")
    
    # パラメータの更新
    print("\n2. パラメータの更新:")
    update_parameters(
        model={"d_model": 256, "nhead": 8},
        training={"batch_size": 64},
        generation={"count": 100}
    )
    print(f"更新後 d_model: {get_model_params().d_model}")
    print(f"更新後 batch_size: {get_training_params().batch_size}")
    
    # プリセットの適用
    print("\n3. プリセットの適用:")
    apply_preset("fast_training")
    print(f"fast_training適用後 d_model: {get_model_params().d_model}")
    print(f"fast_training適用後 batch_size: {get_training_params().batch_size}")
    
    # デバイス取得
    print(f"\n4. デバイス: {default_params.get_device()}")
    
    # 全パラメータの表示
    print("\n5. 全パラメータ:")
    all_params = default_params.to_dict()
    for category, params in all_params.items():
        print(f"{category}: {params}")