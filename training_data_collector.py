"""
学習データ収集用のJSONファイル管理クラス
"""
import json
import os
from typing import Dict, List, Any, Optional
from datetime import datetime


class TrainingDataCollector:
    """学習データをJSON形式で収集・管理するクラス"""
    
    def __init__(self, 
                 work_file_path: str = "temp_work.json",
                 dataset_file_path: str = "training_data.json"):
        self.work_file_path = work_file_path
        self.dataset_file_path = dataset_file_path
        self.current_example = None
        
    def start_example(self, example_id: int, initial_premises: List[str], initial_goal: str):
        """新しい例を開始する"""
        self.current_example = {
            "example_id": example_id,
            "initial_state": {
                "premises": initial_premises,
                "goal": initial_goal
            },
            "tactic_applications": [],
            "is_proved": None
        }
        self._save_work_file()
        
    def add_tactic_application(self, 
                             step: int,
                             premises: List[str], 
                             goal: str, 
                             tactic: str, 
                             tactic_apply: bool):
        """戦略適用を記録する"""
        if self.current_example is None:
            raise ValueError("No current example. Call start_example first.")
            
        tactic_data = {
            "step": step,
            "premises": premises,
            "goal": goal,
            "tactic": tactic,
            "tactic_apply": tactic_apply
        }
        
        self.current_example["tactic_applications"].append(tactic_data)
        self._save_work_file()
        
    def update_last_tactic_apply(self, tactic_apply: bool):
        """最後に追加した戦略適用の結果を更新する"""
        if self.current_example is None or not self.current_example["tactic_applications"]:
            raise ValueError("No tactic applications to update.")
            
        self.current_example["tactic_applications"][-1]["tactic_apply"] = tactic_apply
        self._save_work_file()
        
    def finish_example(self, is_proved: bool):
        """例を完了し、データセットに追加する"""
        if self.current_example is None:
            raise ValueError("No current example to finish.")
            
        # is_provedを確定
        self.current_example["is_proved"] = is_proved
        
        # 各戦略適用を個別レコードとしてデータセットに追加
        self._add_to_dataset()
        
        # 作業ファイルをクリア
        self.current_example = None
        self._clear_work_file()
        
    def _add_to_dataset(self):
        """現在の例をデータセットに追加する"""
        if self.current_example is None:
            return
            
        # データセットファイルを読み込み
        dataset = self._load_dataset()
        
        # 各戦略適用を個別レコードとして追加
        for tactic_data in self.current_example["tactic_applications"]:
            record = {
                "premises": tactic_data["premises"],
                "goal": tactic_data["goal"],
                "tactic": tactic_data["tactic"],
                "tactic_apply": tactic_data["tactic_apply"],
                "is_proved": self.current_example["is_proved"]
            }
            dataset.append(record)
            
        # データセットを保存
        self._save_dataset(dataset)
        
    def _load_dataset(self) -> List[Dict[str, Any]]:
        """データセットファイルを読み込む"""
        if not os.path.exists(self.dataset_file_path):
            return []
            
        try:
            with open(self.dataset_file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return []
            
    def _save_dataset(self, dataset: List[Dict[str, Any]]):
        """データセットファイルを保存する"""
        with open(self.dataset_file_path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
            
    def _save_work_file(self):
        """作業ファイルを保存する"""
        if self.current_example is not None:
            with open(self.work_file_path, 'w', encoding='utf-8') as f:
                json.dump(self.current_example, f, ensure_ascii=False, indent=2)
                
    def _clear_work_file(self):
        """作業ファイルをクリアする"""
        if os.path.exists(self.work_file_path):
            os.remove(self.work_file_path)
            
    def get_dataset_stats(self) -> Dict[str, int]:
        """データセットの統計情報を取得する"""
        dataset = self._load_dataset()
        if not dataset:
            return {"total_records": 0, "proved_examples": 0, "failed_examples": 0}
            
        total_records = len(dataset)
        proved_examples = sum(1 for record in dataset if record["is_proved"])
        failed_examples = total_records - proved_examples
        
        return {
            "total_records": total_records,
            "proved_examples": proved_examples,
            "failed_examples": failed_examples
        }
        
    def cleanup(self):
        """作業ファイルをクリアする（終了時）"""
        self._clear_work_file()
