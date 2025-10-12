"""
推論時と学習時で一貫した状態エンコードを行うモジュール
前提の数と長さに制限なし
"""
import hashlib
from typing import Dict, List, Tuple, Any, Optional


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


def parse_tactic_string(tactic_str: str) -> Dict[str, Any]:
    """
    文字列形式のtacticを構造化されたJSON形式に変換する
    
    Args:
        tactic_str: 文字列形式のtactic（例: "apply 0", "specialize 1 2", "add_dn"）
        
    Returns:
        構造化されたtactic辞書
    """
    parts = tactic_str.split()
    
    if len(parts) == 1:
        # 引数なしのtactic
        return {
            "main": parts[0],
            "arg1": None,
            "arg2": None
        }
    elif len(parts) == 2:
        # 引数1つのtactic
        return {
            "main": parts[0],
            "arg1": parts[1],
            "arg2": None
        }
    elif len(parts) == 3:
        # 引数2つのtactic
        return {
            "main": parts[0],
            "arg1": parts[1],
            "arg2": parts[2]
        }
    else:
        # 予期しない形式
        return {
            "main": tactic_str,
            "arg1": None,
            "arg2": None
        }


def format_tactic_string(tactic_dict: Dict[str, Any]) -> str:
    """
    構造化されたtactic辞書を文字列形式に変換する
    
    Args:
        tactic_dict: 構造化されたtactic辞書
        
    Returns:
        文字列形式のtactic
    """
    main = tactic_dict["main"]
    arg1 = tactic_dict["arg1"]
    arg2 = tactic_dict["arg2"]
    
    if arg1 is None:
        return main
    elif arg2 is None:
        return f"{main} {arg1}"
    else:
        return f"{main} {arg1} {arg2}"


def state_hash(premises: List[str], goal: str) -> str:
    """
    状態のみのハッシュ（tactic を含まない）
    強化学習で同じ状態での複数のアクション試行を管理するために使用
    
    Args:
        premises: 前提のリスト
        goal: ゴール
        
    Returns:
        状態のハッシュ値
    """
    state_str = f"{'|'.join(premises)}|{goal}"
    return hashlib.md5(state_str.encode()).hexdigest()


def state_tactic_hash(premises: List[str], goal: str, tactic: str) -> str:
    """
    状態とtacticの組み合わせのハッシュ
    重複チェックやデータ管理に使用
    
    Args:
        premises: 前提のリスト
        goal: ゴール
        tactic: 戦略文字列
        
    Returns:
        状態とtacticの組み合わせのハッシュ値
    """
    record_str = f"{'|'.join(premises)}|{goal}|{tactic}"
    return hashlib.md5(record_str.encode()).hexdigest()


