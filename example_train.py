from __future__ import annotations

import os
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformer_classifier import (
    load_tokens_and_labels_from_token_py,
    CharTokenizer,
    TransformerClassifier,
    build_label_mappings,
    simple_collate,
)


class ToyTripleSentenceDataset(Dataset):
    def __init__(self, items: List[Tuple[str, str, str, int]]) -> None:
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Tuple[str, str, str, int]:
        return self.items[idx]


def main() -> None:
    token_py_path = os.path.join(os.path.dirname(__file__), "fof_tokens.py")
    base_tokens, labels = load_tokens_and_labels_from_token_py(token_py_path)
    label_to_id, id_to_label = build_label_mappings(labels)

    tokenizer = CharTokenizer(base_tokens=base_tokens, max_sentence_length=50)

    # Dummy few-shot examples: premise1, premise2, premise3, goal
    RawItem = Tuple[str, str, str, str, int]
    raw_items: List[RawItem] = [
        ("a", "a→b", "b→c", "c", label_to_id[labels[0]]),
        ("a∧b", "a→c", "b→c", "c", label_to_id[labels[1]]),
    ]

    class EncodedDataset(Dataset):
        def __init__(self, raw: List[RawItem]):
            self.encoded: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]] = []
            for p1, p2, p3, goal, y in raw:
                ids, mask, seg = tokenizer.encode_four_fixed_blocks(p1, p2, p3, goal)
                self.encoded.append((ids, mask, seg, y))

        def __len__(self) -> int:
            return len(self.encoded)

        def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
            return self.encoded[idx]

    dataset = EncodedDataset(raw_items)
    def collate(batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]]):
        input_ids = torch.stack([b[0] for b in batch], dim=0)
        attention_mask = torch.stack([b[1] for b in batch], dim=0)
        segment_ids = torch.stack([b[2] for b in batch], dim=0)
        labels_tensor = torch.tensor([b[3] for b in batch], dtype=torch.long)
        return input_ids, attention_mask, segment_ids, labels_tensor

    loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set max_seq_len to support 4 sentences: [CLS] + 4*(<=50 + [SEP])
    max_seq_len = 1 + 4 * (tokenizer.max_sentence_length + 1)

    model = TransformerClassifier(
        vocab_size=tokenizer.vocab_size,
        num_classes=len(labels),
        pad_id=tokenizer.pad_id,
        max_seq_len=max_seq_len,
        d_model=128,
        nhead=4,
        num_layers=2,
        dim_feedforward=256,
        dropout=0.1,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)

    model.train()
    for epoch in range(2):
        for input_ids, attention_mask, segment_ids, labels_tensor in loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            segment_ids = segment_ids.to(device)
            labels_tensor = labels_tensor.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(input_ids, attention_mask, segment_ids)
            loss = criterion(logits, labels_tensor)
            loss.backward()
            optimizer.step()

        print(f"epoch {epoch} loss {loss.item():.4f}")

    # Inference example
    model.eval()
    with torch.no_grad():
        ids, mask, seg = tokenizer.encode_four_fixed_blocks("a", "a→b", "b→c", "c")
        logits = model(
            ids.unsqueeze(0).to(device),
            mask.unsqueeze(0).to(device),
            seg.unsqueeze(0).to(device),
        )
        pred = torch.argmax(logits, dim=-1).item()
        print("prediction id:", pred, "label:", id_to_label[pred])


if __name__ == "__main__":
    main()


