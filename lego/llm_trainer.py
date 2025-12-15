"""
LEGOフレームワーク - LLM訓練（訓練専用）

EarlyExitLLMの訓練関数。
Hard token収集はEarlyExitLLM.collect_hard_tokens()で処理。

注意: このモジュールは訓練専用。評価はevaluator.pyを使用。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import Tuple, Dict, Any, List, TYPE_CHECKING

if TYPE_CHECKING:
    from .early_exit_llm import EarlyExitLLM
    from .sequence_data import SequenceData

from .config import TrainerConfig


def train_llm(
    llm: "EarlyExitLLM",
    train_data: "SequenceData",
    val_data: "SequenceData",
    optimizer: torch.optim.Optimizer,
    config: TrainerConfig,
) -> Tuple["SequenceData", Dict[str, Any]]:
    """
    EarlyExitLLMを訓練し、次のLLM用のhard tokensを返す。

    完全なLLM訓練ワークフローを調整:
    1. Early stopping付きLM訓練（val_dataで検証）
    2. llm.collect_hard_tokens()でhard tokens収集（閾値も設定）

    Args:
        llm: 訓練するEarlyExitLLM
        train_data: 訓練SequenceData (hidden_states, targets)
        val_data: Early stopping用の検証SequenceData
        optimizer: LLMパラメータ用オプティマイザ
        config: 訓練ハイパーパラメータを含むTrainerConfig

    Returns:
        タプル:
        - SequenceData: 次のLLM用のhard tokens（出力hidden states）
        - Dict: 訓練統計 (train_ppls, val_ppls, best_epoch等)
    """
    if llm.output_head is None:
        raise RuntimeError("output_headが未設定。先にset_output_head()を呼んでください。")

    if config.verbose:
        print(f"LLM訓練: {len(train_data)}訓練, {len(val_data)}検証シーケンス")
        print(f"  ({train_data.num_tokens}訓練, {val_data.num_tokens}検証トークン)")

    # 1. Early stopping付きLM訓練
    lm_stats = _train_lm(llm, train_data, val_data, optimizer, config)

    # 2. hard tokens収集（llm.thresholdも設定）
    hard_tokens = llm.collect_hard_tokens(train_data, config.hard_ratio, config.batch_size)

    # 3. 最終統計を構築
    actual_hard_ratio = hard_tokens.num_tokens / train_data.num_tokens if train_data.num_tokens > 0 else 0.0
    stats: Dict[str, Any] = {
        **lm_stats,
        'hard_ratio': actual_hard_ratio,
        'threshold': llm.threshold,
    }

    if config.verbose:
        print(f"  閾値 (cos_sim): {llm.threshold:.4f}")
        print(f"  Hard tokens: {len(hard_tokens)}シーケンス ({hard_tokens.num_tokens}トークン, {actual_hard_ratio*100:.1f}%)")

    return hard_tokens, stats


def _train_lm(
    llm: "EarlyExitLLM",
    train_data: "SequenceData",
    val_data: "SequenceData",
    optimizer: torch.optim.Optimizer,
    config: TrainerConfig,
) -> Dict[str, Any]:
    """
    Early stopping付き言語モデル訓練。

    Args:
        llm: 訓練するEarlyExitLLM
        train_data: 訓練SequenceData
        val_data: 検証SequenceData
        optimizer: LLMパラメータ用オプティマイザ
        config: 訓練ハイパーパラメータを含むTrainerConfig

    Returns:
        train_ppls, val_ppls, best_epoch, best_val_ppl, total_epochs, stopped_earlyを含むDict
    """
    device = next(llm.parameters()).device
    best_ppl = float('inf')
    best_state: Dict[str, torch.Tensor] | None = None
    patience_counter = 0
    best_epoch = 0
    epoch = 0

    train_ppls: List[float] = []
    val_ppls: List[float] = []

    for epoch in range(config.max_epochs):
        # 訓練エポック
        train_ppl = _run_train_epoch(llm, train_data, optimizer, config)
        train_ppls.append(train_ppl)

        # 検証PPL計算（early stopping用）
        val_ppl = _compute_val_ppl(llm, val_data, config.batch_size)
        val_ppls.append(val_ppl)

        # Early stoppingチェック
        is_best = val_ppl < best_ppl
        if is_best:
            best_ppl = val_ppl
            best_state = {k: v.cpu().clone() for k, v in llm.state_dict().items()}
            patience_counter = 0
            best_epoch = epoch
        else:
            patience_counter += 1

        if config.verbose:
            status = "best" if is_best else f"{patience_counter}/{config.patience}"
            print(f"  Epoch {epoch+1}/{config.max_epochs}: train_ppl={train_ppl:.2f}, val_ppl={val_ppl:.2f} [{status}]")

        if patience_counter >= config.patience:
            if config.verbose:
                print(f"  Early stopping at epoch {epoch+1}")
            break

    # ベストモデルを復元
    if best_state is not None:
        llm.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    return {
        'train_ppls': train_ppls,
        'val_ppls': val_ppls,
        'best_epoch': best_epoch,
        'best_val_ppl': best_ppl,
        'total_epochs': epoch + 1,
        'stopped_early': patience_counter >= config.patience,
    }


def _run_train_epoch(
    llm: "EarlyExitLLM",
    train_data: "SequenceData",
    optimizer: torch.optim.Optimizer,
    config: TrainerConfig,
) -> float:
    """
    【訓練用】1訓練エポックを実行。

    勾配計算とパラメータ更新を行う。

    Args:
        llm: 訓練するEarlyExitLLM
        train_data: 訓練SequenceData
        optimizer: LLMパラメータ用オプティマイザ
        config: 訓練ハイパーパラメータを含むTrainerConfig

    Returns:
        このエポックの訓練パープレキシティ
    """
    import numpy as np

    device = next(llm.parameters()).device
    llm.train()
    total_loss = 0.0
    total_tokens = 0

    for h, y in train_data.to(str(device)).batches(config.batch_size, shuffle=True):
        optimizer.zero_grad()
        _, logits, _ = llm.train_forward(h)

        batch_size, seq_len, vocab_size = logits.shape
        loss = F.cross_entropy(
            logits.view(-1, vocab_size),
            y.view(-1),
            reduction='sum'
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(llm.parameters(), config.grad_clip)
        optimizer.step()

        total_loss += loss.item()
        total_tokens += batch_size * seq_len

    return float(np.exp(total_loss / total_tokens))


def _compute_val_ppl(
    llm: "EarlyExitLLM",
    data: "SequenceData",
    batch_size: int,
) -> float:
    """
    【訓練用】検証データでパープレキシティを計算。

    訓練中のearly stopping判定用。推論時のexit判定は行わない。

    Args:
        llm: 検証するEarlyExitLLM
        data: 検証用SequenceData
        batch_size: バッチサイズ

    Returns:
        検証パープレキシティ
    """
    import numpy as np

    device = next(llm.parameters()).device
    llm.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for h, y in data.to(str(device)).batches(batch_size, shuffle=False):
            _, logits, _ = llm.train_forward(h)
            batch_size_actual, seq_len, vocab_size = logits.shape
            loss = F.cross_entropy(
                logits.view(-1, vocab_size),
                y.view(-1),
                reduction='sum'
            )
            total_loss += loss.item()
            total_tokens += batch_size_actual * seq_len

    return float(np.exp(total_loss / total_tokens))
