from __future__ import annotations

from typing import List, Dict, Tuple, Optional

import math
import importlib.util
import types

import torch
import torch.nn as nn


SPECIAL_PAD = "[PAD]"
SPECIAL_CLS = "[CLS]"
SPECIAL_SEP = "[SEP]"
SPECIAL_UNK = "[UNK]"
SPECIAL_EOS = "[EOS]"


def _load_module_from_path(module_name: str, file_path: str) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from path: {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


def load_tokens_and_labels_from_token_py(token_py_path: str) -> Tuple[List[str], List[str]]:
    module = _load_module_from_path("fof_token", token_py_path)
    if not hasattr(module, "input_token") or not hasattr(module, "output"):
        raise ValueError("token.py must define both `input_token` and `output` lists")
    input_tokens = list(getattr(module, "input_token"))
    labels = list(getattr(module, "output"))
    return input_tokens, labels


class CharTokenizer:
    def __init__(
        self,
        base_tokens: List[str],
        add_special_tokens: bool = True,
    ) -> None:

        vocab: List[str] = []
        if add_special_tokens:
            vocab.extend([SPECIAL_PAD, SPECIAL_CLS, SPECIAL_SEP, SPECIAL_UNK, SPECIAL_EOS])
        vocab.extend(base_tokens)

        self.token_to_id: Dict[str, int] = {tok: i for i, tok in enumerate(vocab)}
        self.id_to_token: List[str] = vocab

        self.pad_id = self.token_to_id[SPECIAL_PAD]
        self.cls_id = self.token_to_id[SPECIAL_CLS]
        self.sep_id = self.token_to_id[SPECIAL_SEP]
        self.unk_id = self.token_to_id[SPECIAL_UNK]
        self.eos_id = self.token_to_id[SPECIAL_EOS]

    @property
    def vocab_size(self) -> int:
        return len(self.id_to_token)

    def _encode_sentence(self, s: str, max_length: int = None) -> List[int]:
        # Treat each character as a token; ignore whitespace
        tokens: List[int] = []
        for ch in s:
            if ch.isspace():
                continue
            tokens.append(self.token_to_id.get(ch, self.unk_id))
            if max_length is not None and len(tokens) >= max_length:
                break
        return tokens





    def encode(
        self,
        goal: str,
        premises: List[str],
        max_seq_len: int = 512,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        エンコード: [CLS] Goal [SEP] Premise₁ [SEP] Premise₂ [SEP] ... [EOS]
        前提の数と長さに制限なし
        
        Args:
            goal: ゴール文字列
            premises: 前提のリスト（任意の数）
            max_seq_len: 最大シーケンス長
            
        Returns:
            (input_ids, attention_mask, segment_ids)
        """
        # シーケンスを構築: [CLS] Goal [SEP] Premise₁ [SEP] Premise₂ [SEP] ... [EOS]
        seq: List[int] = [self.cls_id]
        attn_mask: List[int] = [1]
        seg_ids: List[int] = [0]  # 0 for special tokens
        
        # Goalを追加
        goal_tokens = self._encode_sentence(goal)
        seq.extend(goal_tokens)
        attn_mask.extend([1] * len(goal_tokens))
        seg_ids.extend([1] * len(goal_tokens))  # 1 for goal
        seq.append(self.sep_id)
        attn_mask.append(1)
        seg_ids.append(0)  # 0 for special tokens
        
        # 各前提を追加
        for i, premise in enumerate(premises):
            premise_tokens = self._encode_sentence(premise)
            seq.extend(premise_tokens)
            attn_mask.extend([1] * len(premise_tokens))
            seg_ids.extend([2 + i] * len(premise_tokens))  # 2+ for premises
            seq.append(self.sep_id)
            attn_mask.append(1)
            seg_ids.append(0)  # 0 for special tokens
        
        # [EOS]を追加
        seq.append(self.eos_id)
        attn_mask.append(1)
        seg_ids.append(0)  # 0 for special tokens
        
        # パディング
        if len(seq) > max_seq_len:
            seq = seq[:max_seq_len]
            attn_mask = attn_mask[:max_seq_len]
            seg_ids = seg_ids[:max_seq_len]
        else:
            pad_len = max_seq_len - len(seq)
            seq.extend([self.pad_id] * pad_len)
            attn_mask.extend([0] * pad_len)
            seg_ids.extend([0] * pad_len)
        
        input_ids = torch.tensor(seq, dtype=torch.long)
        attention_mask = torch.tensor(attn_mask, dtype=torch.long)
        segment_ids = torch.tensor(seg_ids, dtype=torch.long)
        return input_ids, attention_mask, segment_ids


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape: (1, max_len, d_model)
        self.register_buffer("pe", pe)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]
        return self.dropout(x)


class TransformerClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        pad_id: int,
        max_seq_len: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        # Segment embeddings: 0=special([CLS]/[SEP]/[PAD]/[EOS]), 1=goal, 2+=premises
        # Support up to 100 segments to handle variable number of premises
        self.segment_embedding = nn.Embedding(100, d_model)
        self.positional_encoding = SinusoidalPositionalEncoding(d_model, max_len=max_seq_len, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        segment_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # input_ids: (batch, seq_len)
        # attention_mask: (batch, seq_len) where 1=keep, 0=pad
        # segment_ids: (batch, seq_len) in {0..4}
        x = self.embedding(input_ids)
        if segment_ids is not None:
            x = x + self.segment_embedding(segment_ids)
        x = self.positional_encoding(x)

        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = attention_mask == 0  # (batch, seq_len), True for pads

        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        x = self.norm(x)

        # Use first token ([CLS]) representation
        cls_repr = x[:, 0, :]
        logits = self.head(self.dropout(cls_repr))
        return logits


def build_label_mappings(labels: List[str]) -> Tuple[Dict[str, int], List[str]]:
    label_to_id = {label: i for i, label in enumerate(labels)}
    id_to_label = list(labels)
    return label_to_id, id_to_label


def simple_collate(batch: List[Tuple[torch.Tensor, torch.Tensor, int]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    input_ids = torch.stack([b[0] for b in batch], dim=0)
    attention_mask = torch.stack([b[1] for b in batch], dim=0)
    labels = torch.tensor([b[2] for b in batch], dtype=torch.long)
    return input_ids, attention_mask, labels


