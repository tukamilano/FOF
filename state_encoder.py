"""
推論時と学習時で一貫した状態エンコードを行うモジュール
"""
from typing import Dict, List, Tuple, Any


def encode_prover_state(prover, max_len: int = None) -> Dict[str, List[str]]:
    """
    推論時と学習時で同じ形式でproverの状態をエンコードする
    
    Args:
        prover: Proverインスタンス
        max_len: 各文字列の最大長（Noneの場合は切り詰めない）
        
    Returns:
        {"premises": [str, str, str], "goal": str} の形式
    """
    # 前提を取得（最初の3つまで）
    vars_as_str = [str(v) for v in getattr(prover, "variables", [])]
    premises = vars_as_str[:3]  # 最初の3つ
    
    # 3つ未満の場合は空文字列で埋める
    while len(premises) < 3:
        premises.append("")
    
    # ゴールを取得
    goal = str(getattr(prover, "goal", "")) if getattr(prover, "goal", None) is not None else ""
    
    # 長さ制限を適用（max_lenが指定されている場合のみ）
    if max_len is not None:
        premises = [p[:max_len] for p in premises]
        goal = goal[:max_len]
    
    return {
        "premises": premises,
        "goal": goal
    }


def encode_prover_state_for_transformer(prover, max_len: int) -> Tuple[str, str, str, str]:
    """
    既存のTransformer用の形式でエンコード（後方互換性のため）
    
    Args:
        prover: Proverインスタンス
        max_len: 各文字列の最大長
        
    Returns:
        (premise1, premise2, premise3, goal) のタプル形式
    """
    state = encode_prover_state(prover, max_len)
    return state["premises"][0], state["premises"][1], state["premises"][2], state["goal"]


def truncate_state_for_transformer(state: Dict[str, List[str]], max_len: int) -> Tuple[str, str, str, str]:
    """
    完全な状態データをTransformer用に切り詰める
    
    Args:
        state: 完全な状態データ {"premises": [...], "goal": "..."}
        max_len: 最大長
        
    Returns:
        (premise1, premise2, premise3, goal) のタプル形式
    """
    premises = state["premises"]
    goal = state["goal"]
    
    # 長さ制限を適用
    p1 = premises[0][:max_len] if len(premises) > 0 else ""
    p2 = premises[1][:max_len] if len(premises) > 1 else ""
    p3 = premises[2][:max_len] if len(premises) > 2 else ""
    goal_truncated = goal[:max_len]
    
    return p1, p2, p3, goal_truncated


def test_encoding_consistency(prover, max_len: int = 50):
    """
    エンコードの一貫性をテストする
    
    Args:
        prover: テスト用のProverインスタンス
        max_len: 最大長
        
    Returns:
        bool: 一貫性があるかどうか
    """
    # 両方の形式でエンコード
    state_dict = encode_prover_state(prover, max_len)
    state_tuple = encode_prover_state_for_transformer(prover, max_len)
    
    # 一貫性をチェック
    expected_tuple = (
        state_dict["premises"][0],
        state_dict["premises"][1], 
        state_dict["premises"][2],
        state_dict["goal"]
    )
    
    return state_tuple == expected_tuple
