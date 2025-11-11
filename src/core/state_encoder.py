"""
推論時とTraining時 with/at 一貫did状態エンコード 行うモジュール
No limit on number and length of premises
"""
import hashlib
from typing import Dict, List, Tuple, Any, Optional


def encode_prover_state(prover) -> Dict[str, List[str]]:
    """
    推論時とTraining時 with/at 同じ形式 with/at proverの状態 エンコードdo/perform
    No limit on number and length of premises
    
    Args:
        prover: Proverインスタンス
        
    Returns:
        {"premises": [str, ...], "goal": str} の形式（premisesの数は実際の数 応じて変動）
    """
    # 前提 get（allの前提 使用、制限なし）
    vars_as_str = [str(v) for v in getattr(prover, "variables", [])]
    premises = vars_as_str  # allの前提 使用
    
    # ゴール get
    goal = str(getattr(prover, "goal", "")) if getattr(prover, "goal", None) is not None else ""
    
    return {
        "premises": premises,
        "goal": goal
    }


def parse_tactic_string(tactic_str: str) -> Dict[str, Any]:
    """
    文字列形式のtactic 構造化was doneJSON形式 変換do/perform
    
    Args:
        tactic_str: 文字列形式のtactic（例: "apply 0", "specialize 1 2", "add_dn"）
        
    Returns:
        構造化was donetactic辞書
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
        # 引数1のtactic
        return {
            "main": parts[0],
            "arg1": parts[1],
            "arg2": None
        }
    elif len(parts) == 3:
        # 引数2のtactic
        return {
            "main": parts[0],
            "arg1": parts[1],
            "arg2": parts[2]
        }
    else:
        # 予期しno/not形式
        return {
            "main": tactic_str,
            "arg1": None,
            "arg2": None
        }


def format_tactic_string(tactic_dict: Dict[str, Any]) -> str:
    """
    構造化was donetactic辞書 文字列形式 変換do/perform
    
    Args:
        tactic_dict: 構造化was donetactic辞書
        
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
    状態onlyのハッシュ（tactic  含まno/not）
    強化Training with/at 同じ状態 with/at の複数のアクション試行 管理do/performため 使用
    
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
    重複チェックやデータ管理 使用
    
    Args:
        premises: 前提のリスト
        goal: ゴール
        tactic: 戦略文字列
        
    Returns:
        状態とtacticの組み合わせのハッシュ値
    """
    record_str = f"{'|'.join(premises)}|{goal}|{tactic}"
    return hashlib.md5(record_str.encode()).hexdigest()


