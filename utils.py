"""
共通ユーティリティ関数
"""
import contextlib
import os
import sys
from typing import Tuple, Any


@contextlib.contextmanager
def pushd(path: str):
    """指定されたディレクトリに一時的に移動するコンテキストマネージャー"""
    prev = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def import_pyprover(pyprover_dir: str) -> Tuple[Any, Any]:
    """pyproverモジュールをインポートする"""
    with pushd(pyprover_dir):
        if pyprover_dir not in sys.path:
            sys.path.insert(0, pyprover_dir)
        # Local imports after chdir so that proposition.py can open its grammar
        import proposition as proposition_mod  # type: ignore
        import prover as prover_mod  # type: ignore
    return proposition_mod, prover_mod
