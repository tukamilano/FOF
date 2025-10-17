"""
Self Improvement Data Collector
解けたexampleの成功したタクティクのみを収集して学習データとして保存するスクリプト
"""
from __future__ import annotations

import argparse
import os
import sys
import json
import time
import hashlib
import random
import numpy as np
import subprocess
import multiprocessing as mp
from typing import List, Tuple, Dict, Any
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# プロジェクトルートをパスに追加
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, project_root)

import torch

# グローバル定数（一度だけ初期化）
PYPROVER_MODULES = None
GENERATION_PARAMS = None
BASE_TOKENS = None
TACTIC_PARSER_CACHE = {}

from src.core.transformer_classifier import (
    load_tokens_and_labels_from_token_py,
    CharTokenizer,
    TransformerClassifier,
)
from src.core.state_encoder import encode_prover_state, format_tactic_string


def process_single_tautology_worker(args: Tuple) -> Dict[str, Any]:
    """ワーカー関数：単一の論理式を並列処理
    
    Args:
        args: (goal_str, max_steps, probability_threshold, temperature, pyprover_dir)
    
    Returns:
        処理結果の辞書
    """
    goal_str, max_steps, probability_threshold, temperature, pyprover_dir = args
    
    try:
        # pyproverをインポート
        sys.path.insert(0, pyprover_dir)
        original_cwd = os.getcwd()
        os.chdir(pyprover_dir)
        try:
            import proposition as proposition_mod
            import prover as prover_mod
        finally:
            os.chdir(original_cwd)
        
        PropParseTree = proposition_mod.PropParseTree
        prop_parser = proposition_mod.parser
        Prover = prover_mod.Prover
        
        # 論理式をパースしてproverを作成
        parse_tree = PropParseTree()
        goal_node = parse_tree.transform(prop_parser.parse(goal_str))
        prover = Prover(goal_node)
        
        # 成功したタクティクを収集
        successful_tactics = []
        solved = prover.goal is None
        step = 0
        consecutive_failures = 0
        max_consecutive_failures = 5
        
        while not solved and step < max_steps and consecutive_failures < max_consecutive_failures:
            # 現在の状態を取得
            current_state = encode_prover_state(prover)
            current_premises = current_state["premises"]
            current_goal = current_state["goal"]
            
            # 確率閾値を満たすタクティク組み合わせを生成（簡略化版）
            # 実際の実装では、モデルを使ってタクティクを生成する必要がある
            # ここでは簡略化してランダムにタクティクを選択
            available_tactics = [
                "assumption", "intro", "split", "left", "right", "add_dn"
            ]
            
            # 引数付きタクティクも追加
            for i in range(len(prover.variables)):
                available_tactics.extend([f"apply {i}", f"destruct {i}"])
                for j in range(len(prover.variables)):
                    available_tactics.append(f"specialize {i} {j}")
            
            # ランダムにタクティクを選択（実際の実装ではモデル予測を使用）
            if available_tactics:
                tactic = random.choice(available_tactics)
                
                # タクティクを適用
                success = apply_tactic_from_label(prover, tactic)
                
                if success:
                    # 成功したタクティクを記録
                    tactic_data = {
                        "premises": current_premises,
                        "goal": current_goal,
                        "tactic": tactic,
                        "step": step
                    }
                    successful_tactics.append(tactic_data)
                    consecutive_failures = 0
                else:
                    consecutive_failures += 1
                
                step += 1
                solved = prover.goal is None
            else:
                break
        
        return {
            'goal': goal_str,
            'solved': solved,
            'successful_tactics': successful_tactics,
            'total_steps': step,
            'worker_id': os.getpid()
        }
        
    except Exception as e:
        return {
            'goal': goal_str,
            'solved': False,
            'successful_tactics': [],
            'total_steps': 0,
            'error': str(e),
            'worker_id': os.getpid()
        }


def apply_tactic_from_label(prover, label: str) -> bool:
    """タクティクを適用"""
    if label == "assumption":
        return not prover.assumption()
    if label == "intro":
        return not prover.intro()
    if label == "split":
        return not prover.split()
    if label == "left":
        return not prover.left()
    if label == "right":
        return not prover.right()
    if label == "add_dn":
        return not prover.add_dn()
    
    parts = label.split()
    if parts[0] == "apply" and len(parts) == 2 and parts[1].isdigit():
        idx = int(parts[1])
        if idx >= len(prover.variables):
            return False
        return not prover.apply(idx)
    if parts[0] == "destruct" and len(parts) == 2 and parts[1].isdigit():
        idx = int(parts[1])
        if idx >= len(prover.variables):
            return False
        return not prover.destruct(idx)
    if parts[0] == "specialize" and len(parts) == 3 and parts[1].isdigit() and parts[2].isdigit():
        func_idx = int(parts[1])
        domain_idx = int(parts[2])
        if func_idx >= len(prover.variables) or domain_idx >= len(prover.variables):
            return False
        return not prover.specialize(func_idx, domain_idx)
    
    return False


def upload_file_to_gcs(local_file_path: str, gcs_path: str) -> bool:
    """ローカルファイルをGCSにアップロード"""
    try:
        # gcloudコマンドでアップロード
        result = subprocess.run([
            'gcloud', 'storage', 'cp', local_file_path, gcs_path
        ], capture_output=True, text=True, check=True)
        
        print(f"Uploaded: {os.path.basename(local_file_path)} -> {gcs_path}")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"Upload failed: {os.path.basename(local_file_path)}")
        print(f"Error: {e.stderr}")
        return False
    except Exception as e:
        print(f"Upload failed: {os.path.basename(local_file_path)}")
        print(f"Error: {e}")
        return False


def upload_directory_to_gcs(local_dir: str, gcs_bucket: str, gcs_prefix: str) -> bool:
    """ローカルディレクトリ全体をGCSにアップロード"""
    if not os.path.exists(local_dir):
        print(f"Local directory not found: {local_dir}")
        return False
    
    success_count = 0
    total_count = 0
    
    # ディレクトリ内のすべてのファイルをアップロード
    for root, dirs, files in os.walk(local_dir):
        for file in files:
            if file.endswith('.json'):
                local_file_path = os.path.join(root, file)
                # 相対パスを計算
                rel_path = os.path.relpath(local_file_path, local_dir)
                gcs_file_path = f"gs://{gcs_bucket}/{gcs_prefix}{rel_path}"
                
                total_count += 1
                if upload_file_to_gcs(local_file_path, gcs_file_path):
                    success_count += 1
    
    print(f"Upload completed: {success_count}/{total_count} files uploaded successfully")
    return success_count > 0


def load_tautologies_from_generated_data(
    generated_data_dir: str = "generated_data",
    max_examples: int = None,
    specific_file: str = None
) -> List[str]:
    """
    generated_data配下のファイルから論理式を読み出す
    
    Args:
        generated_data_dir: generated_dataディレクトリのパス
        max_examples: 読み込む最大例数（Noneの場合は全て）
        specific_file: 特定のファイル名を指定（Noneの場合はすべてのファイルを処理）
    
    Returns:
        論理式のリスト
    """
    tautologies = []
    
    # generated_dataディレクトリ内のJSONファイルを検索
    if not os.path.exists(generated_data_dir):
        print(f"Warning: Generated data directory not found: {generated_data_dir}")
        return []
    
    if specific_file:
        # 特定のファイルのみを処理
        if not specific_file.endswith('.json'):
            specific_file += '.json'
        json_files = [specific_file] if os.path.exists(os.path.join(generated_data_dir, specific_file)) else []
        if not json_files:
            print(f"Warning: Specified file not found: {specific_file}")
            return []
    else:
        # すべてのJSONファイルを処理
        json_files = [f for f in os.listdir(generated_data_dir) if f.endswith('.json')]
        json_files.sort()  # ファイル名でソート
    
    if not json_files:
        print(f"Warning: No JSON files found in {generated_data_dir}")
        return []
    
    print(f"Found {len(json_files)} JSON files in {generated_data_dir}")
    
    for json_file in json_files:
        if max_examples and len(tautologies) >= max_examples:
            break
            
        file_path = os.path.join(generated_data_dir, json_file)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                # リスト形式の場合
                for formula in data:
                    if max_examples and len(tautologies) >= max_examples:
                        break
                    if isinstance(formula, str) and formula.strip():
                        tautologies.append(formula.strip())
            elif isinstance(data, dict):
                # 辞書形式の場合（将来的な拡張に対応）
                if 'formulas' in data and isinstance(data['formulas'], list):
                    for formula in data['formulas']:
                        if max_examples and len(tautologies) >= max_examples:
                            break
                        if isinstance(formula, str) and formula.strip():
                            tautologies.append(formula.strip())
            
            print(f"Loaded {len(data) if isinstance(data, list) else 'unknown'} formulas from {json_file}")
            
        except Exception as e:
            print(f"Warning: Failed to load {json_file}: {e}")
            continue
    
    print(f"Total loaded tautologies: {len(tautologies)}")
    return tautologies


def initialize_global_constants():
    """グローバル定数を一度だけ初期化"""
    global PYPROVER_MODULES, GENERATION_PARAMS, BASE_TOKENS
    
    if PYPROVER_MODULES is None:
        # pyproverモジュールを初期化
        pyprover_dir = os.path.join(project_root, "pyprover")
        
        # パスを追加
        if pyprover_dir not in sys.path:
            sys.path.insert(0, pyprover_dir)
        
        # ディレクトリを変更してからモジュールをインポート
        original_cwd = os.getcwd()
        os.chdir(pyprover_dir)
        
        try:
            # モジュールをインポート
            import proposition as proposition_mod
            import prover as prover_mod
            
            PYPROVER_MODULES = {
                'PropParseTree': proposition_mod.PropParseTree,
                'prop_parser': proposition_mod.parser,
                'Prover': prover_mod.Prover
            }
            print(f"PYPROVER_MODULES initialized successfully")
        except Exception as e:
            print(f"Error initializing PYPROVER_MODULES: {e}")
            import traceback
            traceback.print_exc()
            PYPROVER_MODULES = None
        finally:
            os.chdir(original_cwd)
    
    if GENERATION_PARAMS is None:
        from src.core.parameter import get_generation_params
        GENERATION_PARAMS = get_generation_params()
    
    if BASE_TOKENS is None:
        token_py_path = os.path.join(project_root, "src", "core", "fof_tokens.py")
        BASE_TOKENS, _ = load_tokens_and_labels_from_token_py(token_py_path)


def parse_tactic_string_cached(tactic_str: str) -> Dict[str, Any]:
    """タクティク文字列をパース（キャッシュ付き）"""
    if tactic_str not in TACTIC_PARSER_CACHE:
        parts = tactic_str.split()
        
        if len(parts) == 1:
            TACTIC_PARSER_CACHE[tactic_str] = {
                "main": parts[0],
                "arg1": None,
                "arg2": None
            }
        elif len(parts) == 2:
            TACTIC_PARSER_CACHE[tactic_str] = {
                "main": parts[0],
                "arg1": parts[1],
                "arg2": None
            }
        elif len(parts) == 3:
            TACTIC_PARSER_CACHE[tactic_str] = {
                "main": parts[0],
                "arg1": parts[1],
                "arg2": parts[2]
            }
        else:
            TACTIC_PARSER_CACHE[tactic_str] = {
                "main": tactic_str,
                "arg1": None,
                "arg2": None
            }
    
    return TACTIC_PARSER_CACHE[tactic_str]


def create_state_hash(premises: List[str], goal: str, tactic_str: str) -> str:
    """状態ハッシュを効率的に作成"""
    # 文字列結合を最適化
    state_tactic_str = f"{'|'.join(premises)}|{goal}|{tactic_str}"
    return hashlib.md5(state_tactic_str.encode()).hexdigest()


def load_hierarchical_model(model_path: str, device: torch.device) -> Tuple[TransformerClassifier, Dict[str, Any]]:
    """階層分類モデルを読み込み"""
    checkpoint = torch.load(model_path, map_location=device)
    
    # 新しい形式のチェックポイントかどうかを判定
    if 'model_params' in checkpoint:
        # 新しい形式のチェックポイント
        model_params = checkpoint['model_params']
        vocab_size = checkpoint.get('vocab_size', model_params['vocab_size'])
        pad_id = checkpoint.get('pad_id', model_params['pad_id'])
        max_seq_len = checkpoint.get('max_seq_len', model_params['max_seq_len'])
        
        # クラス数をチェックポイントから取得
        num_main_classes = len(checkpoint['id_to_main'])
        num_arg1_classes = len(checkpoint['id_to_arg1'])
        num_arg2_classes = len(checkpoint['id_to_arg2'])
        
        # ラベルマッピングを取得
        label_mappings = {
            'main_to_id': checkpoint['main_to_id'],
            'arg1_to_id': checkpoint['arg1_to_id'],
            'arg2_to_id': checkpoint['arg2_to_id'],
            'id_to_main': checkpoint['id_to_main'],
            'id_to_arg1': checkpoint['id_to_arg1'],
            'id_to_arg2': checkpoint['id_to_arg2'],
        }
        
        model = TransformerClassifier(
            vocab_size=vocab_size,
            pad_id=pad_id,
            max_seq_len=max_seq_len,
            d_model=model_params['d_model'],
            nhead=model_params['nhead'],
            num_layers=model_params['num_layers'],
            dim_feedforward=model_params['dim_feedforward'],
            dropout=model_params['dropout'],
            num_main_classes=num_main_classes,
            num_arg1_classes=num_arg1_classes,
            num_arg2_classes=num_arg2_classes,
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # 古い形式のチェックポイント（重みのみ）
        print("Loading old format checkpoint (weights only)")
        
        # デフォルトのモデルパラメータを取得
        from src.core.parameter import get_model_params
        model_params = get_model_params()
        
        # チェックポイントからvocab_sizeを取得（embedding層のサイズから）
        vocab_size = checkpoint['embedding.weight'].shape[0]
        print(f"Detected vocab_size from checkpoint: {vocab_size}")
        
        # デフォルトのラベルマッピングを取得
        from src.core.parameter import get_hierarchical_labels
        from src.core.transformer_classifier import build_hierarchical_label_mappings
        hierarchical_labels = get_hierarchical_labels()
        main_to_id, arg1_to_id, arg2_to_id, id_to_main, id_to_arg1, id_to_arg2 = build_hierarchical_label_mappings(
            hierarchical_labels.main_tactics,
            hierarchical_labels.arg1_values,
            hierarchical_labels.arg2_values
        )
        
        label_mappings = {
            'main_to_id': main_to_id,
            'arg1_to_id': arg1_to_id,
            'arg2_to_id': arg2_to_id,
            'id_to_main': id_to_main,
            'id_to_arg1': id_to_arg1,
            'id_to_arg2': id_to_arg2,
        }
        
        # モデルを作成（チェックポイントから取得したvocab_sizeを使用）
        model = TransformerClassifier(
            vocab_size=vocab_size,
            pad_id=model_params.pad_id,
            max_seq_len=model_params.max_seq_len,
            d_model=model_params.d_model,
            nhead=model_params.nhead,
            num_layers=model_params.num_layers,
            dim_feedforward=model_params.dim_feedforward,
            dropout=model_params.dropout,
            num_main_classes=len(id_to_main),
            num_arg1_classes=len(id_to_arg1),
            num_arg2_classes=len(id_to_arg2),
        )
        
        # 重みを読み込み
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    return model, label_mappings


def calculate_tactic_probability(
    main_tactic: str,
    arg1_value: str,
    arg2_value: str,
    main_confidence: float,
    arg1_confidence: float,
    arg2_confidence: float
) -> float:
    """
    タクティクの種類に応じて適切な確率を計算
    
    Args:
        main_tactic: メインタクティク
        arg1_value: 第1引数
        arg2_value: 第2引数
        main_confidence: メインタクティクの確信度
        arg1_confidence: 第1引数の確信度
        arg2_confidence: 第2引数の確信度
    
    Returns:
        計算された確率
    """
    # 引数不要なタクティク
    if main_tactic in ['assumption', 'intro', 'split', 'left', 'right', 'add_dn']:
        return main_confidence
    
    # 引数1つのタクティク
    elif main_tactic in ['apply', 'destruct']:
        return main_confidence * arg1_confidence
    
    # 引数2つのタクティク
    elif main_tactic == 'specialize':
        return main_confidence * arg1_confidence * arg2_confidence
    
    # その他のタクティク（引数不要として扱う）
    else:
        return main_confidence


def select_tactic_probabilistically(
    tactic_combinations: List[Tuple[str, float]], 
    temperature: float = 1.0,
    failed_tactics: set = None
) -> Tuple[str, float]:
    """
    temperatureを使用してタクティクを確率的に選択
    
    Args:
        tactic_combinations: [(tactic_string, probability), ...] のリスト
        temperature: 温度パラメータ（高いほどランダム、低いほど確率に従う）
        failed_tactics: 失敗したタクティクのセット
    
    Returns:
        選択されたタクティクとその調整後の確率
    """
    if failed_tactics is None:
        failed_tactics = set()
    
    # 失敗していないタクティクのみをフィルタリング
    available_tactics = [(tactic, prob) for tactic, prob in tactic_combinations 
                        if tactic not in failed_tactics]
    
    if not available_tactics:
        # 利用可能なタクティクがない場合は空を返す
        return "", 0.0
    
    # 確率を温度で調整
    tactics, probabilities = zip(*available_tactics)
    probabilities = np.array(probabilities)
    
    # 温度を適用（log確率に変換してから温度で割る）
    log_probs = np.log(probabilities + 1e-8)  # 数値安定性のため小さな値を追加
    scaled_log_probs = log_probs / temperature
    
    # softmaxで正規化
    exp_probs = np.exp(scaled_log_probs - np.max(scaled_log_probs))  # 数値安定性のため
    softmax_probs = exp_probs / np.sum(exp_probs)
    
    # 確率的に選択
    selected_idx = np.random.choice(len(tactics), p=softmax_probs)
    selected_tactic = tactics[selected_idx]
    selected_prob = softmax_probs[selected_idx]  # 調整後の確率を返す
    
    return selected_tactic, selected_prob


def generate_tactic_combinations(
    model: TransformerClassifier,
    tokenizer: CharTokenizer,
    premises: List[str],
    goal: str,
    label_mappings: Dict[str, Any],
    device: torch.device,
    max_seq_len: int = 256,
    probability_threshold: float = 0.001
) -> List[Tuple[str, float]]:
    """
    確率閾値を満たすタクティク組み合わせを生成
    
    Args:
        probability_threshold: 確率の閾値（これ以下の組み合わせは生成しない）
    
    Returns:
        [(tactic_string, probability), ...] のリスト（確率の高い順）
    """
    
    # 入力をエンコード
    input_ids, attention_mask, segment_ids = tokenizer.encode(goal, premises, max_seq_len)
    input_ids = input_ids.unsqueeze(0).to(device)
    attention_mask = attention_mask.unsqueeze(0).to(device)
    segment_ids = segment_ids.unsqueeze(0).to(device)
    
    with torch.no_grad():
        main_logits, arg1_logits, arg2_logits = model(input_ids, attention_mask)
        
        # softmaxで確率に変換
        main_probs = torch.softmax(main_logits, dim=-1)
        arg1_probs = torch.softmax(arg1_logits, dim=-1)
        arg2_probs = torch.softmax(arg2_logits, dim=-1)
        
        # 確率閾値を満たすタクティクを収集
        tactic_combinations = []
        
        # 引数が不要なタクティクを処理
        for main_id, main_tactic in enumerate(label_mappings['id_to_main']):
            main_confidence = main_probs[0, main_id].item()
            
            # 確率閾値チェック
            if main_confidence < probability_threshold:
                continue
                
            if main_tactic in ['assumption', 'intro', 'split', 'left', 'right', 'add_dn']:
                tactic_string = main_tactic
                probability = calculate_tactic_probability(
                    main_tactic, "", "",
                    main_confidence, 0.0, 0.0
                )
                
                tactic_combinations.append((tactic_string, probability))
            
            # 引数1つのタクティクの場合
            elif main_tactic in ['apply', 'destruct']:
                for arg1_id, arg1_value in enumerate(label_mappings['id_to_arg1']):
                    arg1_confidence = arg1_probs[0, arg1_id].item()
                    
                    # 確率閾値チェック（main * arg1）
                    if main_confidence * arg1_confidence < probability_threshold:
                        continue
                        
                    tactic_string = f"{main_tactic} {arg1_value}"
                    probability = calculate_tactic_probability(
                        main_tactic, arg1_value, "",
                        main_confidence, arg1_confidence, 0.0
                    )
                    
                    tactic_combinations.append((tactic_string, probability))
            
            # 引数2つのタクティクの場合
            elif main_tactic == 'specialize':
                for arg1_id, arg1_value in enumerate(label_mappings['id_to_arg1']):
                    arg1_confidence = arg1_probs[0, arg1_id].item()
                    
                    # 確率閾値チェック（main * arg1）
                    if main_confidence * arg1_confidence < probability_threshold:
                        continue
                        
                    for arg2_id, arg2_value in enumerate(label_mappings['id_to_arg2']):
                        arg2_confidence = arg2_probs[0, arg2_id].item()
                        
                        # 確率閾値チェック（main * arg1 * arg2）
                        if main_confidence * arg1_confidence * arg2_confidence < probability_threshold:
                            continue
                            
                        tactic_string = f"{main_tactic} {arg1_value} {arg2_value}"
                        probability = calculate_tactic_probability(
                            main_tactic, arg1_value, arg2_value,
                            main_confidence, arg1_confidence, arg2_confidence
                        )
                        
                        tactic_combinations.append((tactic_string, probability))
            
            # その他のタクティク（引数不要として扱う）
            else:
                tactic_string = main_tactic
                probability = calculate_tactic_probability(
                    main_tactic, "", "",
                    main_confidence, 0.0, 0.0
                )
                
                tactic_combinations.append((tactic_string, probability))
        
        # 確率の高い順にソート
        tactic_combinations.sort(key=lambda x: x[1], reverse=True)
        
        return tactic_combinations


def apply_tactic_from_label(prover, label) -> bool:
    """タクティクを適用"""
    if isinstance(label, dict):
        tactic_str = format_tactic_string(label)
    else:
        tactic_str = label
    
    try:
        if tactic_str == "assumption":
            return not prover.assumption()
        if tactic_str == "intro":
            return not prover.intro()
        if tactic_str == "split":
            return not prover.split()
        if tactic_str == "left":
            return not prover.left()
        if tactic_str == "right":
            return not prover.right()
        if tactic_str == "add_dn":
            return not prover.add_dn()
        
        parts = tactic_str.split()
        if parts[0] == "apply" and len(parts) == 2 and parts[1].isdigit():
            idx = int(parts[1])
            if idx >= len(prover.variables):
                return False
            return not prover.apply(idx)
        if parts[0] == "destruct" and len(parts) == 2 and parts[1].isdigit():
            idx = int(parts[1])
            if idx >= len(prover.variables):
                return False
            return not prover.destruct(idx)
        if parts[0] == "specialize" and len(parts) == 3 and parts[1].isdigit() and parts[2].isdigit():
            func_idx = int(parts[1])
            domain_idx = int(parts[2])
            if func_idx >= len(prover.variables) or domain_idx >= len(prover.variables):
                return False
            return not prover.specialize(func_idx, domain_idx)
        return False
    except Exception as e:
        # pyproverのエラーをキャッチしてFalseを返す
        return False




def collect_self_improvement_data_parallel(
    num_examples: int = None,
    max_steps: int = 30,
    probability_threshold: float = 0.001,
    temperature: float = 1.0,
    generated_data_dir: str = "generated_data",
    specific_file: str = None,
    output_dir: str = "self_improvement_data",
    batch_size: int = 1000,
    save_interval: int = 10,
    num_workers: int = None,
    gcs_bucket: str = None,
    gcs_prefix: str = ""
) -> List[Dict[str, Any]]:
    """
    並列化されたSelf improvement用のデータ収集
    
    Args:
        num_examples: 処理する例数（Noneの場合はすべて）
        max_steps: 各例の最大ステップ数
        probability_threshold: タクティク生成の確率閾値
        temperature: 確率的タクティク選択の温度
        generated_data_dir: generated_dataディレクトリのパス
        specific_file: 特定のファイル名（Noneの場合はすべてのファイル）
        output_dir: 出力ディレクトリ
        batch_size: バッチサイズ
        save_interval: 保存間隔（例数）
        num_workers: 並列ワーカー数（Noneの場合はCPU数）
    
    Returns:
        成功したタクティクのリスト
    """
    # ワーカー数を設定
    if num_workers is None:
        num_workers = min(mp.cpu_count(), 8)  # 最大8ワーカーに制限
    
    # generated_data配下のファイルから論理式を読み込み
    tautologies = load_tautologies_from_generated_data(
        generated_data_dir=generated_data_dir,
        max_examples=num_examples,
        specific_file=specific_file
    )
    
    if not tautologies:
        print("Failed to load tautologies from generated_data directory!")
        return []
    
    print(f"Loaded {len(tautologies)} tautologies from generated_data directory")
    print(f"Using {num_workers} parallel workers")
    
    # pyproverディレクトリのパス
    pyprover_dir = os.path.join(project_root, "pyprover")
    
    # 成功したタクティクを収集
    all_successful_tactics = []
    solved_count = 0
    processed_count = 0
    file_batch_num = 1  # ファイル番号（1から開始）
    
    # バッチサイズを設定（ワーカー数の4倍）
    batch_size_parallel = num_workers * 4
    
    print(f"Starting parallel data collection with {num_workers} workers...")
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        with tqdm(total=len(tautologies), desc="Processing tautologies", unit="formula", 
                 bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:
            
            # バッチごとに処理
            for i in range(0, len(tautologies), batch_size_parallel):
                batch_tautologies = tautologies[i:i + batch_size_parallel]
                
                # ワーカー引数を準備
                worker_args = [
                    (goal_str, max_steps, probability_threshold, temperature, pyprover_dir)
                    for goal_str in batch_tautologies
                ]
                
                # バッチを並列処理
                future_to_index = {
                    executor.submit(process_single_tautology_worker, args): idx
                    for idx, args in enumerate(worker_args)
                }
                
                # 完了したタスクを処理
                for future in as_completed(future_to_index):
                    idx = future_to_index[future]
                    try:
                        result = future.result()
                        
                        # 成功したタクティクを収集
                        if result.get('solved', False) and result.get('successful_tactics'):
                            all_successful_tactics.extend(result['successful_tactics'])
                            solved_count += 1
                        
                        processed_count += 1
                        pbar.update(1)
                        
                        # 進捗情報を更新
                        if processed_count % 100 == 0:
                            pbar.set_postfix({
                                'solved': solved_count,
                                'tactics': len(all_successful_tactics)
                            })
                        
                        # 定期的に保存
                        if len(all_successful_tactics) >= batch_size:
                            save_tactics_batch(all_successful_tactics, output_dir, file_batch_num, gcs_bucket, gcs_prefix)
                            all_successful_tactics = []
                            file_batch_num += 1
                            
                    except Exception as e:
                        print(f"Error processing tautology: {e}")
                        pbar.update(1)
                        processed_count += 1
    
    # 残りのタクティクを保存
    if all_successful_tactics:
        save_tactics_batch(all_successful_tactics, output_dir, file_batch_num, gcs_bucket, gcs_prefix)
    
    print(f"Parallel data collection completed. {solved_count} problems solved, {len(all_successful_tactics)} tactics collected.")
    return all_successful_tactics


def save_tactics_batch(tactics: List[Dict], output_dir: str, batch_num: int, gcs_bucket: str = None, gcs_prefix: str = ""):
    """タクティクのバッチを保存し、GCSにアップロード"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 重複除去
    unique_tactics = []
    seen_hashes = set()
    
    for tactic in tactics:
        tactic_str = f"{tactic['premises']}|{tactic['goal']}|{tactic['tactic']}"
        tactic_hash = hashlib.md5(tactic_str.encode()).hexdigest()
        
        if tactic_hash not in seen_hashes:
            seen_hashes.add(tactic_hash)
            unique_tactics.append(tactic)
    
    # ファイルに保存
    filename = f"training_data_{batch_num:05d}.json"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(unique_tactics, f, ensure_ascii=False, indent=2)
    
    print(f"Saved {len(unique_tactics)} tactics to {filename}")
    
    # GCSにアップロード
    if gcs_bucket:
        gcs_path = f"gs://{gcs_bucket}/{gcs_prefix}{filename}"
        if upload_file_to_gcs(filepath, gcs_path):
            print(f"Uploaded {filename} to GCS")
        else:
            print(f"Failed to upload {filename} to GCS")


def collect_self_improvement_data(
    model: TransformerClassifier,
    tokenizer: CharTokenizer,
    label_mappings: Dict[str, Any],
    device: torch.device,
    max_seq_len: int,
    num_examples: int = None,
    max_steps: int = 30,
    probability_threshold: float = 0.001,
    verbose: bool = False,
    generated_data_dir: str = "generated_data",
    temperature: float = 1.0,
    specific_file: str = None,
    output_dir: str = "self_improvement_data",
    batch_size: int = 1000,
    save_interval: int = 10,
    gcs_bucket: str = None,
    gcs_prefix: str = ""
) -> List[Dict[str, Any]]:
    """
    Self improvement用のデータを収集
    
    Args:
        model: 学習済みモデル
        tokenizer: トークナイザー
        label_mappings: ラベルマッピング
        device: デバイス
        max_seq_len: 最大シーケンス長
        num_examples: 処理する例数（Noneの場合はすべて）
        max_steps: 各例の最大ステップ数
        probability_threshold: タクティク生成の確率閾値
        verbose: 詳細出力フラグ
        generated_data_dir: generated_dataディレクトリのパス
        temperature: 確率的タクティク選択の温度
        specific_file: 特定のファイル名（Noneの場合はすべてのファイル）
        output_dir: 出力ディレクトリ
        batch_size: バッチサイズ
        save_interval: 保存間隔（例数）
    
    Returns:
        成功したタクティクのリスト（deduplicated_batch形式）
    """
    # グローバル定数を初期化
    initialize_global_constants()
    
    # generated_data配下のファイルから論理式を読み込み
    tautologies = load_tautologies_from_generated_data(
        generated_data_dir=generated_data_dir,
        max_examples=num_examples,
        specific_file=specific_file
    )
    
    if not tautologies:
        print("Failed to load tautologies from generated_data directory!")
        return []
    
    print(f"Loaded {len(tautologies)} tautologies from generated_data directory")
    print("First 5 tautologies:")
    for i in range(min(5, len(tautologies))):
        print(f"  {i+1}: {tautologies[i]}")
    
    # 成功したタクティクを収集
    successful_tactics = []
    solved_count = 0
    pending_tactics = []  # 保存待ちのタクティク
    
    # 進捗表示付きでループ実行
    progress_bar = tqdm(tautologies, desc="Processing tautologies", unit="formula", 
                      bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
    
    for i, goal_str in enumerate(progress_bar):
        try:
            # パースしてproverを作成
            parse_tree = PYPROVER_MODULES['PropParseTree']()
            goal_node = parse_tree.transform(PYPROVER_MODULES['prop_parser'].parse(goal_str))
            prover = PYPROVER_MODULES['Prover'](goal_node)
            
            # 前提は空（トートロジーなので前提なしで証明可能）
            premises = []
            
            if verbose:
                print(f"\nExample {i+1}: {goal_str}")
                # 初期状態を表示
                initial_state = encode_prover_state(prover)
                initial_premises = initial_state["premises"]
                initial_goal = initial_state["goal"]
                print(f"  Initial state:")
                print(f"    Premises: {initial_premises if initial_premises else '[]'}")
                print(f"    Goal: {initial_goal}")
            
            # このexampleの成功したタクティクを一時的に保存
            example_successful_tactics = []
            
            # 失敗したタクティクを記録するセット（このexample内でのみ有効）
            failed_tactics = set()
            
            # 推論ループ
            step = 0
            solved = prover.goal is None
            example_terminated = False  # example全体の早期終了フラグ
            consecutive_failures = 0  # 連続失敗数をステップ間で保持
            
            while not solved and step < max_steps and not example_terminated:
                # 現在の状態を取得
                current_state = encode_prover_state(prover)
                current_premises = current_state["premises"]
                current_goal = current_state["goal"]
                
                if verbose:
                    print(f"\n  Step {step+1}:")
                    print(f"    Premises: {current_premises if current_premises else '[]'}")
                    print(f"    Goal: {current_goal}")
                
                # 確率閾値を満たすタクティク組み合わせを生成
                tactic_combinations = generate_tactic_combinations(
                    model, tokenizer, current_premises, current_goal,
                    label_mappings, device, max_seq_len, probability_threshold
                )
                
                if verbose:
                    print(f"    Generated {len(tactic_combinations)} tactic combinations")
                    if tactic_combinations:
                        print(f"    Threshold filtered tactics: {[(t, f'{p:.3f}') for t, p in tactic_combinations]}")
                    else:
                        print(f"    Threshold filtered tactics: []")
                    print(f"    Failed tactics so far: {failed_tactics}")
                
                # 確率的にタクティクを選択して適用
                success = False
                max_attempts = len(tactic_combinations)  # 利用可能なタクティク数
                attempts = 0
                
                # 利用可能なタクティクを事前に計算
                available_tactics = [tactic for tactic, _ in tactic_combinations 
                                   if tactic not in failed_tactics]
                
                while not success and attempts < max_attempts and not example_terminated and available_tactics:
                    # 確率的にタクティクを選択
                    selected_tactic, selected_prob = select_tactic_probabilistically(
                        tactic_combinations, temperature, failed_tactics
                    )
                    
                    if not selected_tactic:
                        # 利用可能なタクティクがない場合
                        if verbose:
                            print(f"    No available tactics (all failed)")
                        example_terminated = True
                        break
                    
                    # タクティクを適用
                    success = apply_tactic_from_label(prover, selected_tactic)
                    attempts += 1
                    
                    if verbose:
                        print(f"    Attempt {attempts}: {selected_tactic} (prob: {selected_prob:.3f}) - {'Success' if success else 'Failed'}")
                    
                    if success:
                        # 成功したタクティクを一時的に記録
                        tactic_dict = parse_tactic_string_cached(selected_tactic)
                        state_tactic_hash = create_state_hash(current_premises, current_goal, selected_tactic)
                        
                        example_successful_tactics.append({
                            "step_index": step,
                            "premises": current_premises.copy(),
                            "goal": current_goal,
                            "tactic": tactic_dict,
                            "tactic_apply": True,
                            "state_tactic_hash": state_tactic_hash
                        })
                        consecutive_failures = 0  # 成功したら連続失敗数をリセット
                        
                        if verbose:
                            # タクティク適用後の状態を表示
                            new_state = encode_prover_state(prover)
                            new_premises = new_state["premises"]
                            new_goal = new_state["goal"]
                            print(f"    → New state after {selected_tactic}:")
                            print(f"      Premises: {new_premises if new_premises else '[]'}")
                            print(f"      Goal: {new_goal if new_goal else 'SOLVED!'}")
                    else:
                        # 失敗したタクティクを記録
                        failed_tactics.add(selected_tactic)
                        consecutive_failures += 1
                        
                        # 利用可能なタクティクリストを更新
                        available_tactics = [tactic for tactic, _ in tactic_combinations 
                                           if tactic not in failed_tactics]
                        
                        if not available_tactics:
                            if verbose:
                                print(f"    All tactics failed in this step")
                            example_terminated = True
                            break
                
                step += 1
                solved = prover.goal is None
            
            # 解けた場合のみ、このexampleの成功したタクティクを追加
            if solved:
                solved_count += 1
                successful_tactics.extend(example_successful_tactics)
                pending_tactics.extend(example_successful_tactics)
                result_text = f"SOLVED in {step} steps"
                if verbose:
                    print(f"  Result: {result_text}")
            else:
                if example_terminated:
                    result_text = f"EARLY TERMINATED after {step} steps"
                    if verbose:
                        print(f"  Result: {result_text}")
                else:
                    result_text = f"FAILED after {step} steps"
                    if verbose:
                        print(f"  Result: {result_text}")
            
            # 定期的にデータを保存
            if len(pending_tactics) >= save_interval:
                append_self_improvement_data(pending_tactics, output_dir, batch_size, gcs_bucket, gcs_prefix)
                pending_tactics = []
            
            # 進捗バーに結果を表示
            progress_bar.set_postfix({
                'Solved': f"{solved_count}/{i+1}",
                'Tactics': len(successful_tactics),
                'Pending': len(pending_tactics),
                'Last': result_text
            })
                
        except Exception as e:
            # パースエラーなどで失敗した場合はスキップ
            if verbose:
                print(f"Warning: Failed to process tautology {i+1}: {e}")
            
            # 進捗バーにエラーを表示
            progress_bar.set_postfix({
                'Solved': f"{solved_count}/{i+1}",
                'Tactics': len(successful_tactics),
                'Pending': len(pending_tactics),
                'Last': f"ERROR: {str(e)[:20]}..."
            })
            continue
    
    # 進捗バーを閉じる
    progress_bar.close()
    
    # 残りのデータを保存
    if pending_tactics:
        append_self_improvement_data(pending_tactics, output_dir, batch_size, gcs_bucket, gcs_prefix)
    
    print(f"\nSelf improvement data collection completed:")
    print(f"  Solved examples: {solved_count}/{len(tautologies)}")
    print(f"  Successful tactics collected: {len(successful_tactics)}")
    
    return successful_tactics


def clear_self_improvement_data(output_dir: str = "self_improvement_data") -> None:
    """Self improvementデータディレクトリをクリア"""
    if os.path.exists(output_dir):
        import shutil
        shutil.rmtree(output_dir)
        print(f"Cleared existing data in {output_dir}")
    else:
        print(f"Directory {output_dir} does not exist, no need to clear")


def save_self_improvement_data(
    data: List[Dict[str, Any]],
    output_dir: str = "self_improvement_data",
    batch_size: int = 1000
) -> None:
    """Self improvementデータをファイルに保存"""
    # 出力ディレクトリを作成
    os.makedirs(output_dir, exist_ok=True)
    
    # バッチごとに分割して保存
    for i in range(0, len(data), batch_size):
        batch_data = data[i:i + batch_size]
        batch_num = i // batch_size
        
        filename = f"training_data_{batch_num:05d}.json"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(batch_data, f, ensure_ascii=False, indent=2)
        
        print(f"Saved {len(batch_data)} tactics to {filepath}")


def append_self_improvement_data(
    data: List[Dict[str, Any]],
    output_dir: str = "self_improvement_data",
    batch_size: int = 1000,
    gcs_bucket: str = None,
    gcs_prefix: str = ""
) -> None:
    """Self improvementデータをインクリメンタルに保存"""
    # 出力ディレクトリを作成
    os.makedirs(output_dir, exist_ok=True)
    
    if not data:
        return
    
    # 既存のファイルを確認して、最後のバッチファイルを見つける
    existing_files = [f for f in os.listdir(output_dir) if f.startswith('training_data_') and f.endswith('.json')]
    
    if existing_files:
        # 最後のファイル番号を取得
        last_file_num = max(int(f.split('_')[2].split('.')[0]) for f in existing_files)
        last_file_path = os.path.join(output_dir, f"training_data_{last_file_num:05d}.json")
        
        # 最後のファイルを読み込んで現在のデータ数を確認
        try:
            with open(last_file_path, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
            
            # 現在のファイルに追加できるかチェック
            if len(existing_data) < batch_size:
                # 現在のファイルに追加
                existing_data.extend(data)
                with open(last_file_path, 'w', encoding='utf-8') as f:
                    json.dump(existing_data, f, ensure_ascii=False, indent=2)
                
                # GCSにアップロード
                if gcs_bucket:
                    gcs_path = f"gs://{gcs_bucket}/{gcs_prefix}{os.path.basename(last_file_path)}"
                    if upload_file_to_gcs(last_file_path, gcs_path):
                        print(f"Updated and uploaded {os.path.basename(last_file_path)} to GCS")
                    else:
                        print(f"Failed to upload {os.path.basename(last_file_path)} to GCS")
                return
        except (json.JSONDecodeError, FileNotFoundError):
            # ファイルが壊れている場合は新しいファイルを作成
            pass
    
    # 新しいファイルを作成
    batch_num = len(existing_files) + 1 if existing_files else 1
    filename = f"training_data_{batch_num:05d}.json"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    # GCSにアップロード
    if gcs_bucket:
        gcs_path = f"gs://{gcs_bucket}/{gcs_prefix}{filename}"
        if upload_file_to_gcs(filepath, gcs_path):
            print(f"Created and uploaded {filename} to GCS")
        else:
            print(f"Failed to upload {filename} to GCS")


def main():
    parser = argparse.ArgumentParser(description="Collect self improvement data from solved examples")
    parser.add_argument("--model_path", type=str, default="models/pretrained_model.pth", help="model path")
    parser.add_argument("--count", type=int, default=None, help="number of examples to process (None for all examples)")
    parser.add_argument("--max_steps", type=int, default=30, help="max steps per example")
    parser.add_argument("--probability_threshold", type=float, default=0.001, help="probability threshold for tactic generation")
    parser.add_argument("--device", type=str, default="auto", help="device")
    parser.add_argument("--verbose", action="store_true", help="verbose output")
    parser.add_argument("--max_seq_len", type=int, default=256, help="maximum sequence length")
    parser.add_argument("--output_dir", type=str, default="self_improvement_data", help="output directory for collected data")
    parser.add_argument("--batch_size", type=int, default=10000, help="batch size for saving data")
    parser.add_argument("--generated_data_dir", type=str, default="generated_data", help="directory containing generated tautology data")
    parser.add_argument("--temperature", type=float, default=1.0, help="temperature for probabilistic tactic selection (higher = more random)")
    parser.add_argument("--specific_file", type=str, default=None, help="specific JSON file to process (e.g., 'tautology_data_00001.json')")
    parser.add_argument("--save_interval", type=int, default=10, help="save interval (number of examples)")
    parser.add_argument("--gcs_bucket", type=str, default=None, help="GCS bucket name for upload (e.g., fof-data-20251010-milano)")
    parser.add_argument("--gcs_prefix", type=str, default="generated_data_RL1/", help="GCS prefix for uploaded files (e.g., generated_data_RL1/)")
    parser.add_argument("--num_workers", type=int, default=None, help="number of parallel workers (if specified, enables parallel processing)")
    
    args = parser.parse_args()
    
    # デバイス設定
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # モデルを読み込み
    if not os.path.exists(args.model_path):
        print(f"Model file not found: {args.model_path}")
        print("Please provide a valid model path.")
        return
    
    model, label_mappings = load_hierarchical_model(args.model_path, device)
    print(f"Loaded model from {args.model_path}")
    
    # トークナイザーを作成
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    token_py_path = os.path.join(root_dir, "src", "core", "fof_tokens.py")
    base_tokens, _ = load_tokens_and_labels_from_token_py(token_py_path)
    tokenizer = CharTokenizer(base_tokens=base_tokens)
    
    # モデルのmax_seq_lenを取得
    checkpoint = torch.load(args.model_path, map_location=device)
    max_seq_len = checkpoint.get('max_seq_len', 256)
    
    print(f"Starting self improvement data collection...")
    print(f"  Examples to process: {args.count if args.count is not None else 'all'}")
    print(f"  Max steps per example: {args.max_steps}")
    print(f"  Probability threshold: {args.probability_threshold}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Generated data directory: {args.generated_data_dir}")
    print(f"  Output directory: {args.output_dir}")
    if args.gcs_bucket:
        print(f"  GCS upload: gs://{args.gcs_bucket}/{args.gcs_prefix}")
    if args.num_workers is not None:
        print(f"  Parallel processing: {args.num_workers} workers")
    
    # 既存のデータをクリア
    clear_self_improvement_data(args.output_dir)
    
    # Self improvementデータを収集
    if args.num_workers is not None:
        # 並列処理を使用
        successful_tactics = collect_self_improvement_data_parallel(
            num_examples=args.count,
            max_steps=args.max_steps,
            probability_threshold=args.probability_threshold,
            temperature=args.temperature,
            generated_data_dir=args.generated_data_dir,
            specific_file=args.specific_file,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            save_interval=args.save_interval,
            num_workers=args.num_workers,
            gcs_bucket=args.gcs_bucket,
            gcs_prefix=args.gcs_prefix
        )
    else:
        # 従来の順次処理を使用
        successful_tactics = collect_self_improvement_data(
            model=model,
            tokenizer=tokenizer,
            label_mappings=label_mappings,
            device=device,
            max_seq_len=max_seq_len,
            num_examples=args.count,
            max_steps=args.max_steps,
            probability_threshold=args.probability_threshold,
            verbose=args.verbose,
            generated_data_dir=args.generated_data_dir,
            temperature=args.temperature,
            specific_file=args.specific_file,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            save_interval=args.save_interval,
            gcs_bucket=args.gcs_bucket,
            gcs_prefix=args.gcs_prefix
        )
    
    if successful_tactics:
        print(f"Data collection completed. {len(successful_tactics)} tactics were collected and saved incrementally.")
        if args.gcs_bucket:
            print(f"All files have been uploaded to GCS: gs://{args.gcs_bucket}/{args.gcs_prefix}")
    else:
        print("No successful tactics collected. Please check your model and parameters.")


if __name__ == "__main__":
    main()
