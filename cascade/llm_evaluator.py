"""
CASCADEフレームワーク - LLM評価

LLMの評価関数。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import numpy as np
from typing import TYPE_CHECKING

from datasets import Dataset

from .cascade_dataset import iterate_batches

if TYPE_CHECKING:
    from .llm import LLM


def compute_ppl(
    llm: "LLM",
    dataset: Dataset,
    batch_size: int,
) -> float:
    """
    パープレキシティを計算。

    exit判定なしの単純なPPL計算。訓練中のearly stopping判定などに使用。

    Args:
        llm: 評価するLLM
        dataset: 検証用HF Dataset
        batch_size: バッチサイズ

    Returns:
        パープレキシティ
    """
    device = next(llm.parameters()).device
    llm.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for h, y in iterate_batches(dataset, batch_size, shuffle=False, device=device):
            # DatasetはCascade形式（hidden_states）
            h_out, _ = llm.forward_hidden_states(h)
            logits = llm.get_logits(h_out)
            batch_sz, seq_len, vocab_size = logits.shape
            loss = F.cross_entropy(
                logits.view(-1, vocab_size),
                y.view(-1),
                reduction='sum'
            )
            total_loss += loss.item()
            total_tokens += batch_sz * seq_len

    return float(np.exp(total_loss / total_tokens)) if total_tokens > 0 else float('inf')
