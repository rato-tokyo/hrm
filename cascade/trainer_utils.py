"""
DCA-LLM 訓練ユーティリティ

訓練・評価に必要な関数群:
- compute_ppl: Perplexity計算
- train_epoch: 1エポック訓練
- train_model: 完全な訓練ループ
- get_memory_usage: GPUメモリ使用量取得
- create_baseline_gpt2: ベースラインGPT-2モデル作成
"""

import time
import math
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from transformers import GPT2Config, GPT2LMHeadModel


def compute_ppl(
    model: nn.Module,
    batches: List[Tuple[torch.Tensor, torch.Tensor]],
    device: str
) -> float:
    """
    Perplexityを計算。

    Args:
        model: 評価対象モデル
        batches: (input_ids, labels) のリスト
        device: デバイス

    Returns:
        Perplexity値
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for x, y in batches:
            x, y = x.to(device), y.to(device)

            if hasattr(model, 'reset_memory'):
                model.reset_memory()

            outputs = model(input_ids=x, labels=y)
            loss = outputs.loss

            batch_tokens = y.numel()
            total_loss += loss.item() * batch_tokens
            total_tokens += batch_tokens

    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)
    return ppl


def train_epoch(
    model: nn.Module,
    batches: List[Tuple[torch.Tensor, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    device: str,
    max_grad_norm: float,
) -> float:
    """
    1エポック訓練。

    Args:
        model: 訓練対象モデル
        batches: (input_ids, labels) のリスト
        optimizer: オプティマイザ
        device: デバイス
        max_grad_norm: 勾配クリッピング閾値

    Returns:
        平均損失
    """
    model.train()
    total_loss = 0.0
    total_tokens = 0

    for x, y in batches:
        x, y = x.to(device), y.to(device)

        if hasattr(model, 'reset_memory'):
            model.reset_memory()

        optimizer.zero_grad()
        outputs = model(input_ids=x, labels=y)
        loss = outputs.loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        batch_tokens = y.numel()
        total_loss += loss.item() * batch_tokens
        total_tokens += batch_tokens

    avg_loss = total_loss / total_tokens
    return avg_loss


def train_model(
    model: nn.Module,
    train_batches: List[Tuple[torch.Tensor, torch.Tensor]],
    val_batches: List[Tuple[torch.Tensor, torch.Tensor]],
    device: str,
    num_epochs: int = 15,
    learning_rate: float = 2.5e-4,
    weight_decay: float = 0.01,
    max_grad_norm: float = 1.0,
    patience: int = 1,
    model_name: str = "model",
) -> Dict:
    """
    モデルを訓練。

    Args:
        model: 訓練対象モデル
        train_batches: 訓練データバッチ
        val_batches: 検証データバッチ
        device: デバイス
        num_epochs: エポック数
        learning_rate: 学習率
        weight_decay: 重み減衰
        max_grad_norm: 勾配クリッピング閾値
        patience: Early stoppingの忍耐度
        model_name: モデル名（ログ用）

    Returns:
        訓練結果の辞書
    """
    model = model.to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

    best_val_ppl = float('inf')
    best_epoch = 0
    patience_counter = 0
    history: dict[str, list[float]] = {
        'train_loss': [],
        'train_ppl': [],
        'val_ppl': [],
    }

    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"{'='*60}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"{'='*60}")

    start_time = time.time()
    best_state = None

    for epoch in range(num_epochs):
        epoch_start = time.time()

        train_loss = train_epoch(model, train_batches, optimizer, device, max_grad_norm)
        train_ppl = math.exp(train_loss)
        val_ppl = compute_ppl(model, val_batches, device)

        scheduler.step()
        epoch_time = time.time() - epoch_start

        history['train_loss'].append(train_loss)
        history['train_ppl'].append(train_ppl)
        history['val_ppl'].append(val_ppl)

        best_marker = ""
        if val_ppl < best_val_ppl:
            best_val_ppl = val_ppl
            best_epoch = epoch + 1
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_marker = " <- Best"
        else:
            patience_counter += 1

        print(f"Epoch {epoch+1:3d}/{num_epochs} | "
              f"Train PPL: {train_ppl:8.2f} | Val PPL: {val_ppl:8.2f} | "
              f"Time: {epoch_time:.1f}s{best_marker}")

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    total_time = time.time() - start_time

    if best_state:
        model.load_state_dict(best_state)
        model = model.to(device)

    final_train_ppl = compute_ppl(model, train_batches, device)
    final_val_ppl = compute_ppl(model, val_batches, device)

    print(f"\n{'='*60}")
    print(f"Training completed in {total_time:.1f}s")
    print(f"Best epoch: {best_epoch}")
    print(f"Final Train PPL: {final_train_ppl:.2f}")
    print(f"Final Val PPL: {final_val_ppl:.2f}")
    print(f"{'='*60}")

    return {
        'model_name': model_name,
        'train_ppl': final_train_ppl,
        'val_ppl': final_val_ppl,
        'best_epoch': best_epoch,
        'total_time': total_time,
        'history': history,
        'num_params': sum(p.numel() for p in model.parameters()),
    }


def get_memory_usage() -> Dict[str, float]:
    """
    GPUメモリ使用量を取得。

    Returns:
        メモリ使用量（MB）の辞書
    """
    if torch.cuda.is_available():
        return {
            'allocated_mb': torch.cuda.memory_allocated() / 1024 / 1024,
            'reserved_mb': torch.cuda.memory_reserved() / 1024 / 1024,
        }
    return {'allocated_mb': 0, 'reserved_mb': 0}


def create_baseline_gpt2(
    vocab_size: int,
    dim: int = 256,
    num_heads: int = 4,
    num_layers: int = 5,
    max_seq_len: int = 1024,
) -> nn.Module:
    """
    Baseline GPT-2モデルを作成。

    Args:
        vocab_size: 語彙サイズ
        dim: モデル次元
        num_heads: ヘッド数
        num_layers: レイヤー数
        max_seq_len: 最大シーケンス長

    Returns:
        GPT2LMHeadModel インスタンス
    """
    gpt2_config = GPT2Config(
        vocab_size=vocab_size,
        n_embd=dim,
        n_head=num_heads,
        n_layer=num_layers,
        n_positions=max_seq_len,
        loss_type="ForCausalLMLoss",
    )
    return GPT2LMHeadModel(gpt2_config)
