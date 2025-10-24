"""Parallel Actor-Critic data collector.

This module provides a parallelised variant of
``self_improvement_data_collector.collect_comprehensive_rl_data``.
It follows the logic of the sequential implementation but distributes
the proof search for individual tautologies across multiple worker
processes. The workers lazily load the required model and pyprover
modules on first use which keeps the startup overhead small while
ensuring each process owns its resources.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple
import multiprocessing

import torch
from tqdm import tqdm
from google.cloud import storage

# プロジェクトルートをパスに追加
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.core.transformer_classifier import CharTokenizer, load_tokens_and_labels_from_token_py
from src.core.state_encoder import encode_prover_state
from src.interaction import self_improvement_data_collector as base_collector


# --- Worker global state -------------------------------------------------
_WORKER_MODEL: Optional[torch.nn.Module] = None
_WORKER_LABEL_MAPPINGS: Optional[Dict[str, Any]] = None
_WORKER_TOKENIZER: Optional[CharTokenizer] = None
_WORKER_DEVICE: Optional[torch.device] = None
_WORKER_MAX_SEQ_LEN: Optional[int] = None
_WORKER_MODEL_PATH: Optional[str] = None


def _ensure_worker_initialized(config: Dict[str, Any]) -> None:
    """Initialise heavy resources inside a worker process on demand."""

    global _WORKER_MODEL, _WORKER_LABEL_MAPPINGS, _WORKER_TOKENIZER
    global _WORKER_DEVICE, _WORKER_MAX_SEQ_LEN, _WORKER_MODEL_PATH

    model_path = config["model_path"]
    device_str = config["device"]
    max_seq_len = config["max_seq_len"]

    if _WORKER_MODEL is not None and _WORKER_MODEL_PATH == model_path:
        # Already initialised for this model path.
        return

    # Ensure the base collector initialises global constants (pyprover, tokens, ...)
    base_collector.initialize_global_constants()

    device = torch.device(device_str)
    model, label_mappings = base_collector.load_hierarchical_model(model_path, device)
    model.eval()

    # CharTokenizer requires base tokens.  ``initialize_global_constants`` already populates
    # ``BASE_TOKENS`` but we keep a fallback in case the sequential collector is updated.
    if base_collector.BASE_TOKENS is None:
        token_py_path = os.path.join(project_root, "src", "core", "fof_tokens.py")
        base_tokens, _ = load_tokens_and_labels_from_token_py(token_py_path)
    else:
        base_tokens = base_collector.BASE_TOKENS

    tokenizer = CharTokenizer(base_tokens=base_tokens)

    _WORKER_MODEL = model
    _WORKER_LABEL_MAPPINGS = label_mappings
    _WORKER_TOKENIZER = tokenizer
    _WORKER_DEVICE = device
    _WORKER_MAX_SEQ_LEN = max_seq_len
    _WORKER_MODEL_PATH = model_path


def _process_tautology_worker(args: Tuple[Any, ...]) -> Dict[str, Any]:
    """Process a single tautology inside a worker process."""

    (
        index,
        goal_str,
        max_steps,
        temperature,
        config,
    ) = args

    try:
        _ensure_worker_initialized(config)
        assert _WORKER_MODEL is not None
        assert _WORKER_TOKENIZER is not None
        assert _WORKER_LABEL_MAPPINGS is not None
        assert _WORKER_DEVICE is not None
        assert _WORKER_MAX_SEQ_LEN is not None

        # Process the tautology using the comprehensive RL data collection
        # 単一の論理式を処理するため、一時的なgenerated_dataディレクトリを作成
        import tempfile
        import shutil
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # 単一の論理式をgenerated_data形式で保存
            temp_file = os.path.join(temp_dir, "tautology_data_00001.json")
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump([goal_str], f, ensure_ascii=False, indent=2)
            
            successful_tactics, failed_tactics = base_collector.collect_comprehensive_rl_data(
                model=_WORKER_MODEL,
                tokenizer=_WORKER_TOKENIZER,
                label_mappings=_WORKER_LABEL_MAPPINGS,
                device=_WORKER_DEVICE,
                max_seq_len=_WORKER_MAX_SEQ_LEN,
                num_examples=1,  # Process single tautology
                max_steps=max_steps,
                verbose=False,
                generated_data_dir=temp_dir,
                temperature=temperature,
                include_failures=True,
                success_reward=1.0,
                step_penalty=0.0,
                failure_penalty=0.0
            )

        return {
            "index": index,
            "goal_str": goal_str,
            "successful_tactics": successful_tactics,
            "failed_tactics": failed_tactics,
            "success": True,
            "error": None,
        }

    except Exception as e:
        return {
            "index": index,
            "goal_str": goal_str,
            "successful_tactics": [],
            "failed_tactics": [],
            "success": False,
            "error": str(e),
        }


def collect_actor_critic_data_parallel(
    model_path: str,
    num_examples: int = None,
    max_steps: int = 20,
    temperature: float = 1.0,
    device: torch.device = None,
    generated_data_dir: str = "generated_data",
    max_seq_len: int = 256,
    num_workers: int = 1,
    output_dir: str = "actor_critic_data",
    batch_size: int = 1000,
    gcs_bucket: str = None,
    gcs_prefix: str = ""
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Collect Actor-Critic data using multiple worker processes."""

    if num_workers <= 1:
        # Fallback to the original sequential implementation.
        base_collector.initialize_global_constants()
        if base_collector.BASE_TOKENS is None:
            token_py_path = os.path.join(project_root, "src", "core", "fof_tokens.py")
            base_tokens, _ = load_tokens_and_labels_from_token_py(token_py_path)
        else:
            base_tokens = base_collector.BASE_TOKENS

        tokenizer = CharTokenizer(base_tokens=base_tokens)
        model, label_mappings = base_collector.load_hierarchical_model(model_path, device)
        return base_collector.collect_comprehensive_rl_data(
            model=model,
            tokenizer=tokenizer,
            label_mappings=label_mappings,
            device=device,
            max_seq_len=max_seq_len,
            num_examples=num_examples,
            max_steps=max_steps,
            verbose=False,
            generated_data_dir=generated_data_dir,
            temperature=temperature,
            include_failures=True,
            success_reward=1.0,
            step_penalty=0.0,
            failure_penalty=0.0,
        )

    base_collector.initialize_global_constants()
    tautologies = base_collector.load_tautologies_from_generated_data(
        generated_data_dir=generated_data_dir,
        max_examples=num_examples,  # Noneの場合は全てのデータを読み込み
    )

    if not tautologies:
        print("Failed to load tautologies from generated_data directory!")
        return [], []


    # Prepare worker configuration
    config = {
        "model_path": model_path,
        "device": str(device),
        "max_seq_len": max_seq_len,
    }

    # Prepare work items
    work_items = [
        (i, tautology, max_steps, temperature, config)
        for i, tautology in enumerate(tautologies)
    ]

    # Process tautologies in parallel
    all_successful_tactics = []
    all_failed_tactics = []
    successful_count = 0
    failed_count = 0
    
    # 逐次書き出し用のバッファ
    successful_buffer = []
    failed_buffer = []
    batch_counter = 0

    with tqdm(total=len(work_items), desc="Processing tautologies", unit="formula") as progress_bar:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Submit all work items
            future_to_index = {
                executor.submit(_process_tautology_worker, work_item): work_item[0]
                for work_item in work_items
            }

            # Process completed work items
            for future in as_completed(future_to_index):
                result = future.result()
                index = future_to_index[future]

                if result["success"]:
                    all_successful_tactics.extend(result["successful_tactics"])
                    all_failed_tactics.extend(result["failed_tactics"])
                    successful_buffer.extend(result["successful_tactics"])
                    failed_buffer.extend(result["failed_tactics"])
                    successful_count += 1
                    
                    # バッファが一定量に達したら書き出し
                    if len(successful_buffer) >= batch_size or len(failed_buffer) >= batch_size:
                        batch_counter = save_actor_critic_data_with_gcs(
                            successful_buffer, failed_buffer, output_dir, batch_size, gcs_bucket, gcs_prefix, batch_counter
                        )
                        successful_buffer = []
                        failed_buffer = []
                else:
                    failed_count += 1
                    # エラーログも簡潔化
                    pass
                
                progress_bar.update(1)

    # 残りのバッファを書き出し
    if successful_buffer or failed_buffer:
        batch_counter = save_actor_critic_data_with_gcs(
            successful_buffer, failed_buffer, output_dir, batch_size, gcs_bucket, gcs_prefix, batch_counter
        )

    return all_successful_tactics, all_failed_tactics


def upload_to_gcs(local_file_path: str, gcs_bucket: str, gcs_prefix: str) -> bool:
    """Upload a local file to Google Cloud Storage.
    
    Args:
        local_file_path: Path to the local file to upload
        gcs_bucket: GCS bucket name
        gcs_prefix: GCS prefix for the uploaded file
        
    Returns:
        True if upload successful, False otherwise
    """
    try:
        client = storage.Client()
        bucket = client.bucket(gcs_bucket)
        
        # Create the blob name by combining prefix and filename
        filename = os.path.basename(local_file_path)
        blob_name = f"{gcs_prefix.rstrip('/')}/{filename}" if gcs_prefix else filename
        
        blob = bucket.blob(blob_name)
        
        print(f"📤 Uploading {local_file_path} to gs://{gcs_bucket}/{blob_name}")
        blob.upload_from_filename(local_file_path)
        
        print(f"✅ Successfully uploaded to gs://{gcs_bucket}/{blob_name}")
        return True
        
    except Exception as e:
        print(f"❌ Error uploading {local_file_path} to GCS: {e}")
        return False


def save_actor_critic_data_with_gcs(
    successful_tactics: list,
    failed_tactics: list,
    output_dir: str = "actor_critic_data",
    batch_size: int = 10000,
    gcs_bucket: str = None,
    gcs_prefix: str = "",
    batch_counter: int = 0
) -> int:
    """Actor-Criticデータをファイルに保存し、GCSに逐次アップロード"""
    
    # 出力ディレクトリを作成
    os.makedirs(output_dir, exist_ok=True)
    
    # 成功データを保存・アップロード
    if successful_tactics:
        filename = f"successful_tactics_{batch_counter:05d}.json"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(successful_tactics, f, ensure_ascii=False, indent=2)
        
        print(f"💾 Saved {len(successful_tactics)} successful tactics to {filepath}")
        
        # GCSに逐次アップロード
        if gcs_bucket:
            if upload_to_gcs(filepath, gcs_bucket, gcs_prefix):
                print(f"✅ Uploaded {filename} to GCS")
            else:
                print(f"❌ Failed to upload {filename} to GCS")
    
    # 失敗データを保存・アップロード
    if failed_tactics:
        filename = f"failed_tactics_{batch_counter:05d}.json"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(failed_tactics, f, ensure_ascii=False, indent=2)
        
        print(f"💾 Saved {len(failed_tactics)} failed tactics to {filepath}")
        
        # GCSに逐次アップロード
        if gcs_bucket:
            if upload_to_gcs(filepath, gcs_bucket, gcs_prefix):
                print(f"✅ Uploaded {filename} to GCS")
            else:
                print(f"❌ Failed to upload {filename} to GCS")
    
    return batch_counter + 1


def main():
    # CUDAマルチプロセシング問題を解決するためspawnメソッドを使用
    multiprocessing.set_start_method('spawn', force=True)
    
    parser = argparse.ArgumentParser(description="Generate Actor-Critic training data in parallel")
    parser.add_argument("--pretrained_model", type=str, default="models/pretrained_model.pth", help="pretrained model path")
    parser.add_argument("--generated_data_dir", type=str, default="generated_data", help="generated data directory")
    parser.add_argument("--output_dir", type=str, default="actor_critic_data", help="output directory")
    parser.add_argument("--num_examples", type=int, default=None, help="number of examples to process (None for all)")
    parser.add_argument("--max_steps", type=int, default=20, help="max steps per example")
    parser.add_argument("--temperature", type=float, default=1.0, help="temperature for sampling")
    parser.add_argument("--device", type=str, default="auto", help="device (auto/cuda/cpu)")
    parser.add_argument("--max_seq_len", type=int, default=256, help="maximum sequence length")
    parser.add_argument("--num_workers", type=int, default=multiprocessing.cpu_count(), help="number of worker processes")
    parser.add_argument("--gcs_bucket", type=str, default=None, help="GCS bucket name for upload")
    parser.add_argument("--gcs_prefix", type=str, default="", help="GCS prefix for upload")
    parser.add_argument("--batch_size", type=int, default=10000, help="batch size for GCS upload")
    
    args = parser.parse_args()
    
    # デバイス設定
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"🚀 Starting parallel Actor-Critic data generation...")
    print(f"Using device: {device}")
    print(f"Number of workers: {args.num_workers}")
    
    # 並列データ収集を実行
    successful_tactics, failed_tactics = collect_actor_critic_data_parallel(
        model_path=args.pretrained_model,
        num_examples=args.num_examples,
        max_steps=args.max_steps,
        temperature=args.temperature,
        device=device,
        generated_data_dir=args.generated_data_dir,
        max_seq_len=args.max_seq_len,
        num_workers=args.num_workers,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        gcs_bucket=args.gcs_bucket,
        gcs_prefix=args.gcs_prefix
    )
    
    # 統計情報用にtautologies数を取得
    base_collector.initialize_global_constants()
    all_tautologies = base_collector.load_tautologies_from_generated_data(
        generated_data_dir=args.generated_data_dir,
        max_examples=None,  # 全てのデータを読み込み
    )
    
    print(f"\n📈 Data collection completed:")
    print(f"  Successful tactics: {len(successful_tactics)}")
    print(f"  Failed tactics: {len(failed_tactics)}")
    
    # 統計情報を保存
    stats = {
        "total_examples_available": len(all_tautologies),
        "total_examples_processed": len(all_tautologies) if args.num_examples is None else args.num_examples,
        "successful_tactics": len(successful_tactics),
        "failed_tactics": len(failed_tactics),
        "num_workers": args.num_workers,
        "temperature": args.temperature,
        "max_steps": args.max_steps,
        "processed_all_available": args.num_examples is None,
    }
    
    stats_file = os.path.join(args.output_dir, "data_generation_stats.json")
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"📊 Statistics saved to: {stats_file}")
    
    print(f"\n🎉 Parallel Actor-Critic data generation completed!")
    print(f"📁 Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
