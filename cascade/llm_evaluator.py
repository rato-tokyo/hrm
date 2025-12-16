"""
CASCADEフレームワーク - LLM評価（評価専用）

LLMの評価関数。
訓練はllm_trainer.pyを使用。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Any, List, TYPE_CHECKING

if TYPE_CHECKING:
    from .llm import LLM
    from .sequence_data import SequenceData


def compute_ppl(
    llm: "LLM",
    data: "SequenceData",
    batch_size: int,
) -> float:
    """
    パープレキシティを計算。

    exit判定なしの単純なPPL計算。訓練中のearly stopping判定などに使用。

    Args:
        llm: 評価するLLM
        data: 検証用SequenceData
        batch_size: バッチサイズ

    Returns:
        パープレキシティ
    """
    device = next(llm.parameters()).device
    llm.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for h, y in data.to(str(device)).batches(batch_size, shuffle=False):
            h_out, _ = llm.forward(h)
            logits = llm.get_logits(h_out)
            batch_sz, seq_len, vocab_size = logits.shape
            loss = F.cross_entropy(
                logits.view(-1, vocab_size),
                y.view(-1),
                reduction='sum'
            )
            total_loss += loss.item()
            total_tokens += batch_sz * seq_len

    return float(np.exp(total_loss / total_tokens))


def evaluate_llm(
    llm: "LLM",
    data: "SequenceData",
    batch_size: int,
    is_last: bool = False,
) -> Tuple["SequenceData", Dict[str, Any]]:
    """
    LLMを評価し、exitトークンの統計と継続データを返す。

    Args:
        llm: 評価するLLM
        data: 入力SequenceData (hidden_states, targets)
        batch_size: バッチサイズ
        is_last: 最終LLMの場合True（全トークンをexit扱い）

    Returns:
        continue_data: 継続トークンのSequenceData（次LLM用）
        stats: loss, correct, input_tokens, exit_tokens, layers_computedを含むDict
    """
    from .sequence_data import SequenceData as SD

    device = next(llm.parameters()).device
    llm.eval()

    total_loss = 0.0
    total_correct = 0
    total_input = 0
    total_exit = 0

    continue_hidden: List[torch.Tensor] = []
    continue_targets: List[torch.Tensor] = []

    with torch.no_grad():
        for h, y in data.to(str(device)).batches(batch_size, shuffle=False):
            h_out, hidden_history = llm.forward(h)
            logits = llm.get_logits(h_out)
            should_exit = llm.should_exit(hidden_history)

            # フラット化
            batch_sz, seq_len, vocab_size = logits.shape
            num_tokens = batch_sz * seq_len
            total_input += num_tokens

            logits_flat = logits.view(-1, vocab_size)
            y_flat = y.view(-1)
            h_out_flat = h_out.view(-1, h_out.shape[-1])
            should_exit_flat = should_exit.view(-1)

            if is_last:
                # 最終LLM: 全トークンをexit
                exit_mask = torch.ones(num_tokens, dtype=torch.bool, device=device)
            else:
                exit_mask = should_exit_flat

            # exitトークンのloss/correct
            exit_logits = logits_flat[exit_mask]
            exit_targets = y_flat[exit_mask]
            if exit_logits.shape[0] > 0:
                total_loss += F.cross_entropy(exit_logits, exit_targets, reduction='sum').item()
                total_correct += int((exit_logits.argmax(dim=-1) == exit_targets).sum().item())
            total_exit += int(exit_mask.sum().item())

            # 継続トークン
            if not is_last:
                continue_mask = ~exit_mask
                if continue_mask.any():
                    continue_hidden.append(h_out_flat[continue_mask].cpu())
                    continue_targets.append(y_flat[continue_mask].cpu())

    # 継続データを構築
    if continue_hidden:
        all_h = torch.cat(continue_hidden)
        all_y = torch.cat(continue_targets)
        # シーケンスに再構成
        seq_len = data.seq_len
        num_complete = all_h.shape[0] // seq_len
        if num_complete > 0:
            usable = num_complete * seq_len
            continue_data = SD(
                all_h[:usable].view(num_complete, seq_len, -1),
                all_y[:usable].view(num_complete, seq_len),
            )
        else:
            continue_data = SD.empty(seq_len, llm.dim, str(device))
    else:
        continue_data = SD.empty(data.seq_len, llm.dim, str(device))

    stats = {
        'loss': total_loss,
        'correct': total_correct,
        'input_tokens': total_input,
        'exit_tokens': total_exit,
        'layers_computed': total_input * llm.num_layers,
    }

    return continue_data, stats
