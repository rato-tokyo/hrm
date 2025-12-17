#!/usr/bin/env python3
"""
DCA-LLM v2 訓練・評価スクリプト (Colab用スタンドアロン版)

改善点:
1. 長系列（1024トークン）での検証 - DCAの本来の価値を発揮
2. DCAをGPT-2内部に統合 - 後付けではなく置換
3. 公平なパラメータ比較 - Baseline 5層 vs DCA内蔵4層

Colab実行:
    !pip install transformers datasets
    !python dca_v2_training_colab.py
"""

import time
import math
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from transformers import GPT2Config, GPT2LMHeadModel, AutoTokenizer
from datasets import load_dataset


# ============================================================
# Training Configuration
# ============================================================

@dataclass
class TrainingConfig:
    """訓練設定。"""
    # モデル設定
    dim: int = 256
    num_heads: int = 4
    max_seq_len: int = 1024  # 長系列

    # レイヤー数（公平な比較のため）
    baseline_layers: int = 5   # Baseline: 5層
    dca_layers: int = 4        # DCA: 4層（パラメータ数を揃える）

    # DCA固有の設定
    window_size: int = 256         # L0ウィンドウサイズ
    compression_ratio: int = 4     # L1圧縮率

    # 訓練設定
    batch_size: int = 8        # 長系列なのでバッチサイズ小さめ
    seq_len: int = 1024        # 長系列
    num_samples: int = 5000
    num_epochs: int = 15
    learning_rate: float = 2.5e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # Early stopping
    patience: int = 1

    # その他
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# IntegratedDCA Model (DCAをTransformer内部に統合)
# ============================================================

@dataclass
class DCALLMOutput:
    """DCA-LLMの出力。"""
    logits: Tensor
    loss: Optional[Tensor] = None


class IntegratedDCABlock(nn.Module):
    """
    DCAを内蔵したTransformerブロック（L0/L1 2層構造）。

    L0: ローカルコンテキスト（ウィンドウ内の詳細なattention）
    L1: 圧縮コンテキスト（ウィンドウ外の要約情報）

    訓練時は長いシーケンスを分割してL0/L1を使用。
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int = 256,
        compression_ratio: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.compression_ratio = compression_ratio

        # Q projection (shared)
        self.q_proj = nn.Linear(dim, dim)

        # L0: Local context (within window)
        self.l0_k_proj = nn.Linear(dim, dim)
        self.l0_v_proj = nn.Linear(dim, dim)

        # L1: Compressed context (outside window)
        self.l1_k_proj = nn.Linear(dim, dim)
        self.l1_v_proj = nn.Linear(dim, dim)

        # Compression layer for L1 (average pooling + linear projection)
        self.l1_compressor = nn.Linear(dim, dim)

        # Output projection
        self.out_proj = nn.Linear(dim, dim)

        # Gating mechanism to balance L0 and L1
        self.gate = nn.Linear(dim, 2)

        self.scale = self.head_dim ** -0.5
        self.dropout = nn.Dropout(dropout)

        # FFN
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

    def _compress_context(self, hidden: Tensor) -> Tensor:
        """
        過去のコンテキストを圧縮してL1表現を作成。

        Args:
            hidden: (batch, past_len, dim)

        Returns:
            compressed: (batch, past_len // compression_ratio, dim)
        """
        batch_size, seq_len, dim = hidden.shape

        if seq_len == 0:
            return hidden

        # Pad to make divisible by compression_ratio
        pad_len = (self.compression_ratio - seq_len % self.compression_ratio) % self.compression_ratio
        if pad_len > 0:
            hidden = F.pad(hidden, (0, 0, 0, pad_len))

        # Reshape and average pool
        new_len = hidden.size(1) // self.compression_ratio
        hidden = hidden.view(batch_size, new_len, self.compression_ratio, dim)
        compressed = hidden.mean(dim=2)  # (batch, new_len, dim)

        # Project
        compressed = self.l1_compressor(compressed)
        return compressed

    def forward(
        self,
        hidden_states: Tensor,
        causal_mask: Tensor,
        past_context: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward pass with dual-context attention.

        Args:
            hidden_states: (batch, seq_len, dim) - 現在のウィンドウ
            causal_mask: (seq_len, seq_len) - causal mask for L0
            past_context: (batch, past_len, dim) - 過去のコンテキスト（L1用）

        Returns:
            output: (batch, seq_len, dim)
            current_context: hidden_states（次のブロック用）
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Pre-norm
        normed = self.ln1(hidden_states)

        # Query projection
        q = self.q_proj(normed)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # === L0: Local Context Attention ===
        l0_k = self.l0_k_proj(normed)
        l0_v = self.l0_v_proj(normed)
        l0_k = l0_k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        l0_v = l0_v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # L0 attention with causal mask
        attn_l0 = torch.matmul(q, l0_k.transpose(-2, -1)) * self.scale
        attn_l0 = attn_l0.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0) == 0, float('-inf'))
        attn_l0 = F.softmax(attn_l0, dim=-1)
        attn_l0 = self.dropout(attn_l0)
        out_l0 = torch.matmul(attn_l0, l0_v)  # (batch, heads, seq, head_dim)

        # === L1: Compressed Context Attention ===
        if past_context is not None and past_context.size(1) > 0:
            # Compress past context
            compressed = self._compress_context(past_context)
            comp_len = compressed.size(1)

            # L1 K/V projections
            l1_k = self.l1_k_proj(compressed)
            l1_v = self.l1_v_proj(compressed)
            l1_k = l1_k.view(batch_size, comp_len, self.num_heads, self.head_dim).transpose(1, 2)
            l1_v = l1_v.view(batch_size, comp_len, self.num_heads, self.head_dim).transpose(1, 2)

            # L1 attention (no causal mask - all past is visible)
            attn_l1 = torch.matmul(q, l1_k.transpose(-2, -1)) * self.scale
            attn_l1 = F.softmax(attn_l1, dim=-1)
            attn_l1 = self.dropout(attn_l1)
            out_l1 = torch.matmul(attn_l1, l1_v)  # (batch, heads, seq, head_dim)

            # Gating: learn to balance L0 and L1
            gate_input = normed.mean(dim=1)  # (batch, dim)
            gate_weights = F.softmax(self.gate(gate_input), dim=-1)  # (batch, 2)
            gate_l0 = gate_weights[:, 0].view(batch_size, 1, 1, 1)
            gate_l1 = gate_weights[:, 1].view(batch_size, 1, 1, 1)

            # Combine L0 and L1
            out = gate_l0 * out_l0 + gate_l1 * out_l1
        else:
            # No past context, use L0 only
            out = out_l0

        # Reshape and project
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)
        out = self.out_proj(out)

        # Residual connection
        hidden_states = hidden_states + out

        # FFN with pre-norm and residual
        hidden_states = hidden_states + self.ffn(self.ln2(hidden_states))

        return hidden_states, normed  # Return normed as context for next window


class IntegratedDCALLM(nn.Module):
    """
    DCAを内部に統合した言語モデル（L0/L1 2層構造）。

    長いシーケンスをウィンドウに分割し、各ウィンドウで:
    - L0: 現在のウィンドウ内でcausal attention
    - L1: 過去のウィンドウを圧縮してattention
    """

    def __init__(
        self,
        vocab_size: int,
        dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        max_seq_len: int = 1024,
        window_size: int = 256,
        compression_ratio: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.window_size = window_size
        self.compression_ratio = compression_ratio

        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.position_embedding = nn.Embedding(max_seq_len, dim)
        self.embed_dropout = nn.Dropout(dropout)

        # DCA Blocks
        self.blocks = nn.ModuleList([
            IntegratedDCABlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                compression_ratio=compression_ratio,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        # Output
        self.ln_f = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.token_embedding.weight

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights similar to GPT-2."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                torch.nn.init.ones_(module.weight)
                torch.nn.init.zeros_(module.bias)

    def reset_memory(self):
        """Compatibility method (no-op for training)."""
        pass

    def forward(
        self,
        input_ids: Tensor,
        labels: Optional[Tensor] = None,
    ) -> DCALLMOutput:
        """
        Forward pass with windowed DCA.

        長いシーケンスをwindow_sizeに分割し、各ウィンドウで:
        - L0: 現在のウィンドウ内のcausal attention
        - L1: 過去のウィンドウを圧縮したattention

        Args:
            input_ids: (batch, seq_len)
            labels: (batch, seq_len) for loss calculation

        Returns:
            DCALLMOutput
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Embeddings
        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        hidden_states = self.token_embedding(input_ids) + self.position_embedding(positions)
        hidden_states = self.embed_dropout(hidden_states)

        # Split into windows
        num_windows = (seq_len + self.window_size - 1) // self.window_size
        pad_len = num_windows * self.window_size - seq_len

        if pad_len > 0:
            hidden_states = F.pad(hidden_states, (0, 0, 0, pad_len))

        # Process windows
        all_outputs = []

        for window_idx in range(num_windows):
            start_idx = window_idx * self.window_size
            end_idx = start_idx + self.window_size
            window_hidden = hidden_states[:, start_idx:end_idx, :]

            # Causal mask for this window
            causal_mask = torch.tril(torch.ones(
                self.window_size, self.window_size, device=device
            ))

            # Collect past context from previous windows
            if window_idx > 0:
                past_context = hidden_states[:, :start_idx, :]
            else:
                past_context = None

            # Forward through all DCA blocks
            for block in self.blocks:
                window_hidden, _ = block(window_hidden, causal_mask, past_context)

            all_outputs.append(window_hidden)

        # Concatenate all window outputs
        hidden_states = torch.cat(all_outputs, dim=1)

        # Remove padding
        if pad_len > 0:
            hidden_states = hidden_states[:, :seq_len, :]

        # Output
        hidden_states = self.ln_f(hidden_states)
        logits = self.lm_head(hidden_states)

        # Loss
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


# ============================================================
# Data Loading
# ============================================================

def create_wikitext_dataloaders(
    num_samples: int = 5000,
    batch_size: int = 8,
    seq_len: int = 1024,
    seed: int = 42,
) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor]], List[Tuple[torch.Tensor, torch.Tensor]], int]:
    """WikiText-2データセットをロード。"""
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    train_text = "\n".join(dataset["train"]["text"])
    val_text = "\n".join(dataset["validation"]["text"])

    train_ids = tokenizer.encode(train_text, add_special_tokens=False)
    val_ids = tokenizer.encode(val_text, add_special_tokens=False)

    vocab_size = tokenizer.vocab_size

    def create_batches(token_ids: List[int], num_samples: int, batch_size: int, seq_len: int):
        batches = []
        total_len = len(token_ids)

        torch.manual_seed(seed)
        indices = torch.randperm(total_len - seq_len - 1)[:num_samples]

        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i+batch_size]
            if len(batch_indices) < batch_size:
                continue

            x_batch = []
            y_batch = []
            for idx in batch_indices:
                idx = idx.item()
                x_batch.append(token_ids[idx:idx+seq_len])
                y_batch.append(token_ids[idx:idx+seq_len])

            x = torch.tensor(x_batch, dtype=torch.long)
            y = torch.tensor(y_batch, dtype=torch.long)
            batches.append((x, y))

        return batches

    train_batches = create_batches(train_ids, num_samples, batch_size, seq_len)
    val_batches = create_batches(val_ids, min(num_samples // 5, 1000), batch_size, seq_len)

    print(f"Train batches: {len(train_batches)}, Val batches: {len(val_batches)}")
    print(f"Vocab size: {vocab_size}")

    return train_batches, val_batches, vocab_size


# ============================================================
# Training Functions
# ============================================================

def compute_ppl(model: nn.Module, batches: List[Tuple[torch.Tensor, torch.Tensor]], device: str) -> float:
    """Perplexityを計算。"""
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
    """1エポック訓練。"""
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
    config: TrainingConfig,
    model_name: str = "model",
) -> Dict:
    """モデルを訓練。"""
    device = config.device
    model = model.to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=config.num_epochs)

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

    for epoch in range(config.num_epochs):
        epoch_start = time.time()

        train_loss = train_epoch(model, train_batches, optimizer, device, config.max_grad_norm)
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

        print(f"Epoch {epoch+1:3d}/{config.num_epochs} | "
              f"Train PPL: {train_ppl:8.2f} | Val PPL: {val_ppl:8.2f} | "
              f"Time: {epoch_time:.1f}s{best_marker}")

        if patience_counter >= config.patience:
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
    """GPUメモリ使用量を取得。"""
    if torch.cuda.is_available():
        return {
            'allocated_mb': torch.cuda.memory_allocated() / 1024 / 1024,
            'reserved_mb': torch.cuda.memory_reserved() / 1024 / 1024,
        }
    return {'allocated_mb': 0, 'reserved_mb': 0}


def create_baseline_gpt2(vocab_size: int, config: TrainingConfig) -> nn.Module:
    """Baseline GPT-2モデルを作成。"""
    gpt2_config = GPT2Config(
        vocab_size=vocab_size,
        n_embd=config.dim,
        n_head=config.num_heads,
        n_layer=config.baseline_layers,
        n_positions=config.max_seq_len,
        loss_type="ForCausalLMLoss",  # 明示的に設定して警告を抑制
    )
    return GPT2LMHeadModel(gpt2_config)


def create_integrated_dca_llm(
    vocab_size: int,
    config: TrainingConfig,
) -> IntegratedDCALLM:
    """IntegratedDCALLMを作成。"""
    return IntegratedDCALLM(
        vocab_size=vocab_size,
        dim=config.dim,
        num_layers=config.dca_layers,
        num_heads=config.num_heads,
        max_seq_len=config.max_seq_len,
        window_size=config.window_size,
        compression_ratio=config.compression_ratio,
    )


# ============================================================
# Main Experiment
# ============================================================

def run_experiment(config: TrainingConfig) -> Dict:
    """実験を実行。"""
    print("="*70)
    print("DCA-LLM v2 Training Experiment")
    print("="*70)
    print(f"Device: {config.device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print("="*70)
    print("\nExperiment Settings:")
    print(f"  Sequence Length: {config.seq_len} (long sequence)")
    print(f"  Baseline: GPT-2 with {config.baseline_layers} layers")
    print(f"  DCA: IntegratedDCA with {config.dca_layers} layers")
    print("="*70)

    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)

    # データ準備
    print("\nLoading WikiText-2 dataset...")
    train_batches, val_batches, vocab_size = create_wikitext_dataloaders(
        num_samples=config.num_samples,
        batch_size=config.batch_size,
        seq_len=config.seq_len,
        seed=config.seed,
    )

    results = {}

    # ===== Baseline (GPT-2 5層) =====
    print("\n" + "="*70)
    print(f"Training Baseline (GPT-2 {config.baseline_layers} layers)")
    print("="*70)

    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    baseline_model = create_baseline_gpt2(vocab_size, config)
    baseline_results = train_model(
        baseline_model,
        train_batches,
        val_batches,
        config,
        model_name=f"Baseline (GPT-2 {config.baseline_layers}L)",
    )
    baseline_results['memory'] = get_memory_usage()
    results['baseline'] = baseline_results

    del baseline_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ===== IntegratedDCA (4層) =====
    print("\n" + "="*70)
    print(f"Training IntegratedDCA ({config.dca_layers} layers)")
    print("="*70)

    dca_model = create_integrated_dca_llm(vocab_size, config)
    dca_results = train_model(
        dca_model,
        train_batches,
        val_batches,
        config,
        model_name=f"IntegratedDCA ({config.dca_layers}L)",
    )
    dca_results['memory'] = get_memory_usage()
    results['dca'] = dca_results

    # ===== 結果サマリ =====
    print("\n" + "="*70)
    print("EXPERIMENT SUMMARY")
    print("="*70)
    print(f"\n{'Model':<30} {'Train PPL':>12} {'Val PPL':>12} {'Params':>12} {'Time':>10}")
    print("-"*70)

    for name, res in results.items():
        print(f"{res['model_name']:<30} "
              f"{res['train_ppl']:>12.2f} "
              f"{res['val_ppl']:>12.2f} "
              f"{res['num_params']:>12,} "
              f"{res['total_time']:>9.1f}s")

    print("-"*70)

    baseline_val = results['baseline']['val_ppl']
    dca_val = results['dca']['val_ppl']
    baseline_params = results['baseline']['num_params']
    dca_params = results['dca']['num_params']

    diff = dca_val - baseline_val
    diff_pct = (diff / baseline_val) * 100
    param_diff = dca_params - baseline_params
    param_diff_pct = (param_diff / baseline_params) * 100

    print("\nDCA vs Baseline:")
    print(f"  Val PPL difference: {diff:+.2f} ({diff_pct:+.1f}%)")
    print(f"  Parameter difference: {param_diff:+,} ({param_diff_pct:+.1f}%)")

    if diff < 0:
        print(f"  >>> DCA improves perplexity by {abs(diff):.2f} ({abs(diff_pct):.1f}%)")
    else:
        print(f"  >>> Baseline is better by {diff:.2f} ({diff_pct:.1f}%)")

    return results


if __name__ == "__main__":
    config = TrainingConfig()
    run_experiment(config)
