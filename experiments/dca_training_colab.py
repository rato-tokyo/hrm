#!/usr/bin/env python3
"""
DCA-LLM 訓練・評価スクリプト (Colab対応版)

Google Colab で実行可能なスタンドアロンスクリプト。
DCA（Dual-Context Attention）とベースラインの比較実験を行う。

使用方法 (Colab):
    !git clone https://github.com/rato-tokyo/hrm.git
    %cd hrm
    !pip install transformers datasets torch
    !python experiments/dca_training_colab.py
"""

import sys
import time
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# パスを追加
sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================
# Configuration
# ============================

@dataclass
class TrainingConfig:
    """訓練設定。"""
    # モデル設定
    dim: int = 256
    num_layers: int = 4
    num_heads: int = 4
    max_seq_len: int = 512

    # DCA設定
    window_size: int = 128
    max_representatives: int = 64
    encoder_mode: str = "pooling"

    # 訓練設定
    batch_size: int = 16
    seq_len: int = 128
    num_samples: int = 5000
    num_epochs: int = 15
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # Early stopping
    patience: int = 1

    # その他
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ============================
# Data Loading
# ============================

def create_wikitext_dataloaders(
    num_samples: int,
    batch_size: int,
    seq_len: int,
    seed: int,
    tokenizer_name: str = "gpt2",
) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor]],
           List[Tuple[torch.Tensor, torch.Tensor]],
           int]:
    """WikiText-2データローダーを作成。"""
    from datasets import load_dataset
    from transformers import AutoTokenizer

    torch.manual_seed(seed)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')

    # トークナイズ
    def tokenize_split(split_dataset):
        all_tokens = []
        for item in split_dataset:
            text = item['text']
            if text.strip():
                tokens = tokenizer.encode(text, add_special_tokens=False)
                all_tokens.extend(tokens)
        return torch.tensor(all_tokens, dtype=torch.long)

    train_tokens = tokenize_split(dataset['train'])
    val_tokens = tokenize_split(dataset['validation'])

    # サンプル数を制限
    max_tokens_train = num_samples * (seq_len + 1)
    max_tokens_val = int(num_samples * 0.2) * (seq_len + 1)
    train_tokens = train_tokens[:max_tokens_train]
    val_tokens = val_tokens[:max_tokens_val]

    # バッチ化
    def batchify(data, batch_size, seq_len):
        batches = []
        num_tokens = len(data)
        for i in range(0, num_tokens - seq_len - 1, batch_size * seq_len):
            batch_x, batch_y = [], []
            for j in range(batch_size):
                start_idx = i + j * seq_len
                if start_idx + seq_len + 1 <= num_tokens:
                    batch_x.append(data[start_idx:start_idx + seq_len])
                    batch_y.append(data[start_idx + 1:start_idx + seq_len + 1])
            if len(batch_x) == batch_size:
                batches.append((torch.stack(batch_x), torch.stack(batch_y)))
        return batches

    train_batches = batchify(train_tokens, batch_size, seq_len)
    val_batches = batchify(val_tokens, batch_size, seq_len)

    return train_batches, val_batches, tokenizer.vocab_size


# ============================
# DCA Components (Inline)
# ============================

class BidirectionalSpanEncoder(nn.Module):
    """双方向エンコーダでspanを圧縮。"""

    def __init__(self, dim: int, mode: str = "pooling", num_layers: int = 1,
                 num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.mode = mode

        if mode == "pooling":
            self.output_proj = nn.Sequential(
                nn.Linear(dim, dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dim, dim),
            )
        self.layer_norm = nn.LayerNorm(dim)

    def forward(self, hidden_states: Tensor) -> Tensor:
        squeeze_batch = False
        if hidden_states.dim() == 2:
            hidden_states = hidden_states.unsqueeze(0)
            squeeze_batch = True

        pooled = hidden_states.mean(dim=1)
        compressed = self.output_proj(pooled)
        compressed = self.layer_norm(compressed)

        if squeeze_batch:
            compressed = compressed.squeeze(0)
        return compressed


@dataclass
class DualContextState:
    """Dual-Contextの状態。"""
    l0_keys: Optional[Tensor] = None
    l0_values: Optional[Tensor] = None
    l1_representatives: List[Tensor] = field(default_factory=list)
    l1_keys: Optional[Tensor] = None
    total_processed: int = 0


class DualContextMemory(nn.Module):
    """L0/L1の2層コンテキストを管理するメモリ。"""

    def __init__(self, dim: int, window_size: int = 512, max_representatives: int = 256,
                 encoder_mode: str = "pooling", encoder_layers: int = 1,
                 encoder_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.max_representatives = max_representatives

        self.l0_key_proj = nn.Linear(dim, dim)
        self.l0_value_proj = nn.Linear(dim, dim)
        self.l1_key_proj = nn.Linear(dim, dim)

        self.span_encoder = BidirectionalSpanEncoder(
            dim=dim, mode=encoder_mode, num_layers=encoder_layers,
            num_heads=encoder_heads, dropout=dropout,
        )

        self._state = DualContextState()
        self._l0_buffer: List[Tensor] = []

    def reset(self):
        self._state = DualContextState()
        self._l0_buffer = []

    @property
    def state(self) -> DualContextState:
        return self._state

    def update(self, hidden_states: Tensor, force_compress: bool = False) -> DualContextState:
        if hidden_states.dim() == 2:
            hidden_states = hidden_states.unsqueeze(0)

        batch_size, seq_len, _ = hidden_states.shape

        for t in range(seq_len):
            self._l0_buffer.append(hidden_states[:, t:t+1, :])

        self._state.total_processed += seq_len

        while len(self._l0_buffer) > self.window_size or force_compress:
            if len(self._l0_buffer) <= self.window_size // 2:
                break

            compress_size = len(self._l0_buffer) - self.window_size // 2
            compress_size = min(compress_size, self.window_size // 2)

            if compress_size < 1:
                break

            span_hidden = torch.cat(self._l0_buffer[:compress_size], dim=1)
            self._l0_buffer = self._l0_buffer[compress_size:]

            representative = self.span_encoder(span_hidden.squeeze(0))
            self._state.l1_representatives.append(representative.detach())

            if len(self._state.l1_representatives) > self.max_representatives:
                self._state.l1_representatives.pop(0)

        if self._l0_buffer:
            l0_hidden = torch.cat(self._l0_buffer, dim=1)
            self._state.l0_keys = self.l0_key_proj(l0_hidden)
            self._state.l0_values = self.l0_value_proj(l0_hidden)
        else:
            self._state.l0_keys = None
            self._state.l0_values = None

        if self._state.l1_representatives:
            l1_reps = torch.stack(self._state.l1_representatives, dim=0)
            l1_reps = l1_reps.unsqueeze(0).expand(batch_size, -1, -1)
            self._state.l1_keys = self.l1_key_proj(l1_reps)
        else:
            self._state.l1_keys = None

        return self._state


class DualContextAttention(nn.Module):
    """L0とL1を統合するDual-Context Attention。"""

    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1, gate_init: float = 0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.gate = nn.Parameter(torch.tensor(gate_init))
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(self, query: Tensor, l0_keys: Optional[Tensor] = None,
                l0_values: Optional[Tensor] = None, l1_keys: Optional[Tensor] = None,
                l1_values: Optional[Tensor] = None) -> Tensor:
        batch_size, seq_len, _ = query.shape

        q = self.q_proj(query)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # L0 Attention
        if l0_keys is not None and l0_values is not None:
            l0_len = l0_keys.size(1)
            k0 = l0_keys.view(batch_size, l0_len, self.num_heads, self.head_dim).transpose(1, 2)
            v0 = l0_values.view(batch_size, l0_len, self.num_heads, self.head_dim).transpose(1, 2)
            attn_l0 = torch.matmul(q, k0.transpose(-2, -1)) * self.scale
            attn_l0_weights = F.softmax(attn_l0, dim=-1)
            attn_l0_weights = self.dropout(attn_l0_weights)
            out_l0 = torch.matmul(attn_l0_weights, v0)
        else:
            out_l0 = torch.zeros(batch_size, self.num_heads, seq_len, self.head_dim, device=query.device)

        # L1 Attention
        if l1_keys is not None and l1_values is not None:
            l1_len = l1_keys.size(1)
            k1 = l1_keys.view(batch_size, l1_len, self.num_heads, self.head_dim).transpose(1, 2)
            v1 = l1_values.view(batch_size, l1_len, self.num_heads, self.head_dim).transpose(1, 2)
            attn_l1 = torch.matmul(q, k1.transpose(-2, -1)) * self.scale
            attn_l1_weights = F.softmax(attn_l1, dim=-1)
            attn_l1_weights = self.dropout(attn_l1_weights)
            out_l1 = torch.matmul(attn_l1_weights, v1)
        else:
            out_l1 = torch.zeros(batch_size, self.num_heads, seq_len, self.head_dim, device=query.device)

        # Combine
        gate = torch.sigmoid(self.gate)
        if l1_keys is None:
            out = out_l0
        elif l0_keys is None:
            out = out_l1
        else:
            out = gate * out_l0 + (1 - gate) * out_l1

        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)
        out = self.out_proj(out)
        return out


# ============================
# DCA-LLM Model
# ============================

@dataclass
class DCALLMOutput:
    """DCA-LLMの出力。"""
    logits: Tensor
    loss: Optional[Tensor] = None


class DCALLM(nn.Module):
    """Dual-Context Attention統合言語モデル。"""

    def __init__(self, base_llm, window_size: int = 512, max_representatives: int = 256,
                 encoder_mode: str = "pooling", num_heads: Optional[int] = None,
                 dropout: float = 0.1, use_dca: bool = True):
        super().__init__()
        self.base_llm = base_llm
        self.use_dca = use_dca

        config = base_llm.config
        self.dim = config.n_embd if hasattr(config, 'n_embd') else config.hidden_size
        self.vocab_size = config.vocab_size
        self._num_heads = num_heads or (config.n_head if hasattr(config, 'n_head') else config.num_attention_heads)

        if use_dca:
            self.dca_memory = DualContextMemory(
                dim=self.dim, window_size=window_size,
                max_representatives=max_representatives,
                encoder_mode=encoder_mode, encoder_heads=self._num_heads,
                dropout=dropout,
            )
            self.dca_attention = DualContextAttention(
                dim=self.dim, num_heads=self._num_heads, dropout=dropout,
            )
            self.dca_ffn = nn.Sequential(
                nn.LayerNorm(self.dim),
                nn.Linear(self.dim, self.dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(self.dim * 4, self.dim),
                nn.Dropout(dropout),
            )
            self.dca_ln = nn.LayerNorm(self.dim)

    def reset_memory(self):
        if self.use_dca:
            self.dca_memory.reset()

    def forward(self, input_ids: Tensor, labels: Optional[Tensor] = None) -> DCALLMOutput:
        base_outputs = self.base_llm(
            input_ids=input_ids, output_hidden_states=True, return_dict=True,
        )
        hidden_states = base_outputs.hidden_states[-1]

        if self.use_dca:
            hidden_states = self._apply_dca(hidden_states)

        logits = self.base_llm.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return DCALLMOutput(logits=logits, loss=loss)

    def _apply_dca(self, hidden_states: Tensor) -> Tensor:
        """
        DCAを適用。

        訓練時: 各バッチは独立に処理（メモリはリセット済み）
        L0 = hidden_states自体をKey/Valueとして使用（self-attention的）
        L1 = 現在は空（ストリーミング推論時のみ使用）
        """
        batch_size, seq_len, dim = hidden_states.shape

        # 訓練時はhidden_states自体をL0として使用（メモリのストリーミングは不要）
        # これにより各バッチが独立に処理される
        l0_keys = self.dca_memory.l0_key_proj(hidden_states)
        l0_values = self.dca_memory.l0_value_proj(hidden_states)

        # L1は訓練時は使用しない（ストリーミング推論時のみ）
        l1_keys = None
        l1_values = None

        dca_output = self.dca_attention(
            query=hidden_states, l0_keys=l0_keys, l0_values=l0_values,
            l1_keys=l1_keys, l1_values=l1_values,
        )

        hidden_states = hidden_states + dca_output
        hidden_states = hidden_states + self.dca_ffn(self.dca_ln(hidden_states))
        return hidden_states


def create_dca_llm_from_scratch(vocab_size: int, dim: int = 256, num_layers: int = 4,
                                 num_heads: int = 4, max_seq_len: int = 512,
                                 window_size: int = 128, max_representatives: int = 64,
                                 encoder_mode: str = "pooling", use_dca: bool = True,
                                 device: Optional[str] = None) -> DCALLM:
    from transformers import GPT2Config, GPT2LMHeadModel

    config = GPT2Config(
        vocab_size=vocab_size, n_embd=dim, n_head=num_heads,
        n_layer=num_layers, n_positions=max_seq_len,
    )
    base_llm = GPT2LMHeadModel(config)

    dca_llm = DCALLM(
        base_llm=base_llm, window_size=window_size,
        max_representatives=max_representatives,
        encoder_mode=encoder_mode, use_dca=use_dca,
    )

    if device:
        dca_llm = dca_llm.to(device)
    return dca_llm


# ============================
# Training Functions
# ============================

def compute_ppl(model: nn.Module, batches: List[Tuple[torch.Tensor, torch.Tensor]], device: str) -> float:
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for x, y in batches:
            x, y = x.to(device), y.to(device)
            if hasattr(model, 'reset_memory'):
                model.reset_memory()
            outputs = model(input_ids=x, labels=y)
            batch_tokens = y.numel()
            total_loss += outputs.loss.item() * batch_tokens
            total_tokens += batch_tokens

    return math.exp(total_loss / total_tokens)


def train_epoch(model: nn.Module, batches: List[Tuple[torch.Tensor, torch.Tensor]],
                optimizer, device: str, max_grad_norm: float) -> float:
    model.train()
    total_loss = 0.0
    total_tokens = 0

    for x, y in batches:
        x, y = x.to(device), y.to(device)
        if hasattr(model, 'reset_memory'):
            model.reset_memory()

        optimizer.zero_grad()
        outputs = model(input_ids=x, labels=y)
        outputs.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        batch_tokens = y.numel()
        total_loss += outputs.loss.item() * batch_tokens
        total_tokens += batch_tokens

    return total_loss / total_tokens


def train_model(model: nn.Module, train_batches, val_batches, config: TrainingConfig,
                model_name: str = "model") -> Dict:
    device = config.device
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.num_epochs)

    best_val_ppl = float('inf')
    best_epoch = 0
    patience_counter = 0
    best_state = None

    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"{'='*60}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"{'='*60}")

    start_time = time.time()

    for epoch in range(config.num_epochs):
        epoch_start = time.time()

        train_loss = train_epoch(model, train_batches, optimizer, device, config.max_grad_norm)
        train_ppl = math.exp(train_loss)
        val_ppl = compute_ppl(model, val_batches, device)

        scheduler.step()
        epoch_time = time.time() - epoch_start

        print(f"Epoch {epoch+1:3d}/{config.num_epochs} | "
              f"Train Loss: {train_loss:.4f} | Train PPL: {train_ppl:8.2f} | "
              f"Val PPL: {val_ppl:8.2f} | Time: {epoch_time:.1f}s")

        if val_ppl < best_val_ppl:
            best_val_ppl = val_ppl
            best_epoch = epoch + 1
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                print(f"Early stopping at epoch {epoch+1} (patience={config.patience})")
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
        'num_params': sum(p.numel() for p in model.parameters()),
    }


# ============================
# Main Experiment
# ============================

def run_experiment(config: TrainingConfig) -> Dict:
    print("="*70)
    print("DCA-LLM Training Experiment")
    print("="*70)
    print(f"Device: {config.device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print("="*70)

    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)

    print("\nLoading WikiText-2 dataset...")
    train_batches, val_batches, vocab_size = create_wikitext_dataloaders(
        num_samples=config.num_samples,
        batch_size=config.batch_size,
        seq_len=config.seq_len,
        seed=config.seed,
    )
    print(f"Train batches: {len(train_batches)}")
    print(f"Val batches: {len(val_batches)}")
    print(f"Vocab size: {vocab_size}")

    results = {}

    # ===== Baseline (DCAなし) =====
    print("\n" + "="*70)
    print("Training Baseline (no DCA)")
    print("="*70)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    baseline_model = create_dca_llm_from_scratch(
        vocab_size=vocab_size, dim=config.dim, num_layers=config.num_layers,
        num_heads=config.num_heads, max_seq_len=config.max_seq_len,
        window_size=config.window_size, max_representatives=config.max_representatives,
        encoder_mode=config.encoder_mode, use_dca=False, device=config.device,
    )

    baseline_results = train_model(baseline_model, train_batches, val_batches, config, "Baseline (no DCA)")
    results['baseline'] = baseline_results

    del baseline_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ===== DCA Model =====
    print("\n" + "="*70)
    print("Training DCA-LLM")
    print("="*70)

    dca_model = create_dca_llm_from_scratch(
        vocab_size=vocab_size, dim=config.dim, num_layers=config.num_layers,
        num_heads=config.num_heads, max_seq_len=config.max_seq_len,
        window_size=config.window_size, max_representatives=config.max_representatives,
        encoder_mode=config.encoder_mode, use_dca=True, device=config.device,
    )

    dca_results = train_model(dca_model, train_batches, val_batches, config, "DCA-LLM")
    results['dca'] = dca_results

    # ===== 結果サマリ =====
    print("\n" + "="*70)
    print("EXPERIMENT SUMMARY")
    print("="*70)
    print(f"\n{'Model':<25} {'Train PPL':>12} {'Val PPL':>12} {'Params':>12} {'Time':>10}")
    print("-"*70)

    for name, res in results.items():
        print(f"{res['model_name']:<25} "
              f"{res['train_ppl']:>12.2f} "
              f"{res['val_ppl']:>12.2f} "
              f"{res['num_params']:>12,} "
              f"{res['total_time']:>9.1f}s")

    print("-"*70)

    baseline_val = results['baseline']['val_ppl']
    dca_val = results['dca']['val_ppl']
    diff = dca_val - baseline_val
    diff_pct = (diff / baseline_val) * 100

    print("\nDCA vs Baseline:")
    print(f"  Val PPL difference: {diff:+.2f} ({diff_pct:+.1f}%)")
    if diff < 0:
        print(f"  DCA improves perplexity by {abs(diff):.2f} ({abs(diff_pct):.1f}%)")
    else:
        print(f"  Baseline is better by {diff:.2f} ({diff_pct:.1f}%)")

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="DCA-LLM Training")
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--num_samples", type=int, default=5000)
    parser.add_argument("--num_epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=1)
    parser.add_argument("--window_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = TrainingConfig(
        dim=args.dim, num_layers=args.num_layers, num_heads=args.num_heads,
        batch_size=args.batch_size, seq_len=args.seq_len, num_samples=args.num_samples,
        num_epochs=args.num_epochs, learning_rate=args.lr, patience=args.patience,
        window_size=args.window_size, seed=args.seed,
    )

    run_experiment(config)


if __name__ == "__main__":
    main()
