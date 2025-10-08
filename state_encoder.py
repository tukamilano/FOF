"""
推論時と学習時で一貫した状態エンコードを行うモジュール
前提の数と長さに制限なし
"""
from typing import Dict, List, Tuple, Any


def encode_prover_state(prover) -> Dict[str, List[str]]:
    """
    推論時と学習時で同じ形式でproverの状態をエンコードする
    前提の数と長さに制限なし
    
    Args:
        prover: Proverインスタンス
        
    Returns:
        {"premises": [str, ...], "goal": str} の形式（premisesの数は実際の数に応じて変動）
    """
    # 前提を取得（すべての前提を使用、制限なし）
    vars_as_str = [str(v) for v in getattr(prover, "variables", [])]
    premises = vars_as_str  # すべての前提を使用
    
    # ゴールを取得
    goal = str(getattr(prover, "goal", "")) if getattr(prover, "goal", None) is not None else ""
    
    return {
        "premises": premises,
        "goal": goal
    }


