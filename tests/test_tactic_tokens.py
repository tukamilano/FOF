#!/usr/bin/env python3
"""
tactic tokensのテスト
"""

import sys
import os

# プロジェクトルートをパスに追加
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_root)

from src.core.transformer_classifier import CharTokenizer, load_tokens_and_labels_from_token_py
from src.core.state_encoder import parse_tactic_string, format_tactic_string


def test_tokenizer():
    """トークナイザーのテスト"""
    print("Testing tokenizer...")
    
    # トークンを読み込み
    token_py_path = os.path.join(project_root, "src", "core", "fof_tokens.py")
    base_tokens, _ = load_tokens_and_labels_from_token_py(token_py_path)
    
    # トークナイザーを作成
    tokenizer = CharTokenizer(base_tokens=base_tokens)
    
    # 基本的なエンコードテスト
    goal = "a"
    premises = ["(a → b)"]
    input_ids, attention_mask, segment_ids = tokenizer.encode(goal, premises)
    
    assert len(input_ids) > 0, "Input IDs should not be empty"
    assert len(attention_mask) == len(input_ids), "Attention mask length mismatch"
    assert len(segment_ids) == len(input_ids), "Segment IDs length mismatch"
    
    print("✓ Tokenizer test passed")


def test_tactic_parsing():
    """タクティク解析のテスト"""
    print("Testing tactic parsing...")
    
    # 文字列から構造化形式への変換
    tactic_str = "apply 0"
    parsed = parse_tactic_string(tactic_str)
    expected = {"main": "apply", "arg1": "0", "arg2": None}
    assert parsed == expected, f"Parsing failed: {parsed} != {expected}"
    
    # 構造化形式から文字列への変換
    formatted = format_tactic_string(parsed)
    assert formatted == tactic_str, f"Formatting failed: {formatted} != {tactic_str}"
    
    # 引数なしのタクティク
    tactic_str2 = "assumption"
    parsed2 = parse_tactic_string(tactic_str2)
    expected2 = {"main": "assumption", "arg1": None, "arg2": None}
    assert parsed2 == expected2, f"Parsing failed: {parsed2} != {expected2}"
    
    print("✓ Tactic parsing test passed")


if __name__ == "__main__":
    test_tokenizer()
    test_tactic_parsing()
    print("\nAll tactic token tests passed!")
