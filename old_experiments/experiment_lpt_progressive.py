"""
Experiment: LPT Progressive Growing

Train model progressively, adding layers one at a time,
but using LPT (Layer-wise Progressive Training) at each stage.

Key difference from Greedy Layer-wise:
- Greedy: Freeze previous layers, only train new layer
- LPT Progressive: Train ALL layers with LPT loss at each stage

Hypothesis: Since LPT trains each layer to produce "predictable" representations,
adding new layers should not break the model because existing layers already
output useful representations.
"""

import torch
import torch.nn as nn
import math
import random
import numpy as np
from typing import List, Tuple
import copy

from modules.transformer import TransformerBlock


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


class LPTGrowableTransformer(nn.Module):
    """
    Transformer that grows layer by layer with LPT training.
    All layers are always trained (no freezing).
    """

    def __init__(
        self,
        vocab_size: int,
        dim: int = 64,
        seq_len: int = 64,
        num_heads: int = 4,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.seq_len = seq_len
        self.num_heads = num_heads

        # Input embedding
        self.embedding = nn.Embedding(vocab_size, dim)

        # Start with no layers
        self.layers = nn.ModuleList()

        # Shared output head (used for all layers in LPT)
        self.output_head = nn.Linear(dim, vocab_size, bias=False)

        self._init_embedding()

    def _init_embedding(self) -> None:
        nn.init.trunc_normal_(self.embedding.weight, std=0.02)
        nn.init.trunc_normal_(self.output_head.weight, std=0.02)

    def add_layer(self) -> None:
        """Add a new layer (no freezing - all layers remain trainable)."""
        new_layer = TransformerBlock(self.dim, self.num_heads)
        for module in new_layer.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
        self.layers.append(new_layer)

    def forward_all_layers(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass returning outputs from all layers (for LPT training).
        """
        h = self.embedding(x)
        outputs = []

        for layer in self.layers:
            h = layer(h)
            y_hat = self.output_head(h)
            outputs.append(y_hat)

        return outputs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass (only final layer output)."""
        h = self.embedding(x)

        for layer in self.layers:
            h = layer(h)

        return self.output_head(h)


def compute_lpt_loss(
    outputs: List[torch.Tensor],
    target: torch.Tensor,
    vocab_size: int
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """Compute LPT loss (sum of all layer losses)."""
    layer_losses = []

    for y_hat in outputs:
        loss = nn.functional.cross_entropy(
            y_hat.view(-1, vocab_size), target.view(-1)
        )
        layer_losses.append(loss)

    total_loss = sum(layer_losses)
    return total_loss, layer_losses


def prepare_data(
    train_chars: int = 20000,
    val_chars: int = 5000,
    seq_len: int = 64,
    batch_size: int = 16
) -> Tuple:
    """Prepare WikiText-2 style character data."""
    try:
        with open('data/wikitext2_train.txt', 'r') as f:
            train_text = f.read()[:train_chars]
        with open('data/wikitext2_valid.txt', 'r') as f:
            val_text = f.read()[:val_chars]
    except FileNotFoundError:
        train_text = "The quick brown fox jumps over the lazy dog. " * (train_chars // 45)
        val_text = "A quick brown dog runs in the park. " * (val_chars // 35)

    chars = sorted(set(train_text + val_text))
    char_to_idx = {c: i for i, c in enumerate(chars)}
    vocab_size = len(chars)

    def text_to_batches(text: str) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        indices = [char_to_idx.get(c, 0) for c in text]
        batches = []
        for i in range(0, len(indices) - seq_len - 1, seq_len * batch_size):
            batch_x, batch_y = [], []
            for j in range(batch_size):
                start = i + j * seq_len
                if start + seq_len + 1 <= len(indices):
                    batch_x.append(indices[start:start + seq_len])
                    batch_y.append(indices[start + 1:start + seq_len + 1])
            if len(batch_x) == batch_size:
                batches.append((
                    torch.tensor(batch_x, dtype=torch.long),
                    torch.tensor(batch_y, dtype=torch.long)
                ))
        return batches

    train_batches = text_to_batches(train_text)
    val_batches = text_to_batches(val_text)

    return train_batches, val_batches, vocab_size


def train_epoch_lpt(
    model: LPTGrowableTransformer,
    batches: List[Tuple[torch.Tensor, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    vocab_size: int,
    device: str = 'cpu'
) -> Tuple[float, List[float]]:
    """Train one epoch with LPT loss."""
    model.train()
    total_loss = 0.0
    num_layers = len(model.layers)
    layer_losses_sum = [0.0] * num_layers

    for x, y in batches:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        outputs = model.forward_all_layers(x)
        loss, layer_losses = compute_lpt_loss(outputs, y, vocab_size)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        for i, ll in enumerate(layer_losses):
            layer_losses_sum[i] += ll.item()

    n = len(batches)
    return total_loss / n, [l / n for l in layer_losses_sum]


@torch.no_grad()
def evaluate(
    model: nn.Module,
    batches: List[Tuple[torch.Tensor, torch.Tensor]],
    vocab_size: int,
    device: str = 'cpu'
) -> float:
    """Evaluate using final layer output only."""
    model.eval()
    total_loss = 0.0

    for x, y in batches:
        x, y = x.to(device), y.to(device)
        y_hat = model(x)
        loss = nn.functional.cross_entropy(
            y_hat.view(-1, vocab_size), y.view(-1)
        )
        total_loss += loss.item()

    avg_loss = total_loss / len(batches)
    return np.exp(avg_loss)


def train_until_convergence_lpt(
    model: LPTGrowableTransformer,
    train_batches: List,
    val_batches: List,
    vocab_size: int,
    max_epochs: int = 30,
    patience: int = 5,
    lr: float = 1e-3,
    device: str = 'cpu',
    verbose: bool = True
) -> Tuple[float, int]:
    """Train with LPT until convergence."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    best_ppl = float('inf')
    best_epoch = 0
    patience_counter = 0

    for epoch in range(max_epochs):
        train_loss, layer_losses = train_epoch_lpt(
            model, train_batches, optimizer, vocab_size, device
        )
        val_ppl = evaluate(model, val_batches, vocab_size, device)

        if val_ppl < best_ppl:
            best_ppl = val_ppl
            best_epoch = epoch + 1
            patience_counter = 0
        else:
            patience_counter += 1

        if verbose:
            layer_str = " ".join([f"L{i+1}:{l:.2f}" for i, l in enumerate(layer_losses)])
            print(f"    Epoch {epoch + 1}: Loss={train_loss:.2f} [{layer_str}] Val PPL={val_ppl:.2f}")

        if patience_counter >= patience:
            if verbose:
                print(f"    Converged at epoch {epoch + 1}")
            break

    return best_ppl, best_epoch


def train_epoch_standard(
    model: nn.Module,
    batches: List[Tuple[torch.Tensor, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    vocab_size: int,
    device: str = 'cpu'
) -> float:
    """Train one epoch with standard loss (final layer only)."""
    model.train()
    total_loss = 0.0

    for x, y in batches:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        y_hat = model(x)
        loss = nn.functional.cross_entropy(
            y_hat.view(-1, vocab_size), y.view(-1)
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(batches)


def train_until_convergence_standard(
    model: nn.Module,
    train_batches: List,
    val_batches: List,
    vocab_size: int,
    max_epochs: int = 30,
    patience: int = 5,
    lr: float = 1e-3,
    device: str = 'cpu',
    verbose: bool = True
) -> Tuple[float, int]:
    """Train with standard loss until convergence."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    best_ppl = float('inf')
    best_epoch = 0
    patience_counter = 0

    for epoch in range(max_epochs):
        train_loss = train_epoch_standard(
            model, train_batches, optimizer, vocab_size, device
        )
        val_ppl = evaluate(model, val_batches, vocab_size, device)

        if val_ppl < best_ppl:
            best_ppl = val_ppl
            best_epoch = epoch + 1
            patience_counter = 0
        else:
            patience_counter += 1

        if verbose:
            print(f"    Epoch {epoch + 1}: Loss={train_loss:.4f} Val PPL={val_ppl:.2f}")

        if patience_counter >= patience:
            if verbose:
                print(f"    Converged at epoch {epoch + 1}")
            break

    return best_ppl, best_epoch


def main():
    set_seed(42)

    # Config
    seq_len = 64
    dim = 64
    num_heads = 4
    batch_size = 16
    device = 'cpu'
    train_chars = 20000
    val_chars = 5000
    target_layers = 4

    print("=" * 70)
    print("Experiment: LPT Progressive Growing")
    print("=" * 70)
    print(f"\nConfig:")
    print(f"  Target layers: {target_layers}")
    print(f"  Dimension: {dim}")
    print(f"  Sequence length: {seq_len}")

    # Prepare data
    train_batches, val_batches, vocab_size = prepare_data(
        train_chars, val_chars, seq_len, batch_size
    )
    print(f"  Vocab size: {vocab_size}")
    print(f"  Train batches: {len(train_batches)}")
    print(f"  Val batches: {len(val_batches)}")

    results = []

    # ===== LPT Progressive Growing =====
    print("\n" + "=" * 70)
    print("LPT Progressive Growing (all layers trained with LPT)")
    print("=" * 70)

    set_seed(42)
    lpt_model = LPTGrowableTransformer(
        vocab_size=vocab_size,
        dim=dim,
        seq_len=seq_len,
        num_heads=num_heads
    ).to(device)

    lpt_layer_results = []

    for layer_num in range(1, target_layers + 1):
        print(f"\n--- Adding Layer {layer_num} (LPT training) ---")
        lpt_model.add_layer()

        print(f"  Total params: {count_params(lpt_model):,}")

        best_ppl, best_epoch = train_until_convergence_lpt(
            lpt_model, train_batches, val_batches, vocab_size,
            max_epochs=30, patience=5, lr=1e-3, device=device
        )

        lpt_layer_results.append({
            'layer': layer_num,
            'params': count_params(lpt_model),
            'best_ppl': best_ppl,
            'best_epoch': best_epoch
        })

        print(f"  Layer {layer_num} done: Best PPL={best_ppl:.2f} at epoch {best_epoch}")

    results.append({
        'model': 'LPT Progressive (final)',
        'params': count_params(lpt_model),
        'best_ppl': lpt_layer_results[-1]['best_ppl'],
        'method': 'lpt_progressive'
    })

    # ===== Standard Progressive Growing (for comparison) =====
    print("\n" + "=" * 70)
    print("Standard Progressive Growing (all layers trained, final loss only)")
    print("=" * 70)

    set_seed(42)
    std_model = LPTGrowableTransformer(
        vocab_size=vocab_size,
        dim=dim,
        seq_len=seq_len,
        num_heads=num_heads
    ).to(device)

    std_layer_results = []

    for layer_num in range(1, target_layers + 1):
        print(f"\n--- Adding Layer {layer_num} (Standard training) ---")
        std_model.add_layer()

        print(f"  Total params: {count_params(std_model):,}")

        best_ppl, best_epoch = train_until_convergence_standard(
            std_model, train_batches, val_batches, vocab_size,
            max_epochs=30, patience=5, lr=1e-3, device=device
        )

        std_layer_results.append({
            'layer': layer_num,
            'params': count_params(std_model),
            'best_ppl': best_ppl,
            'best_epoch': best_epoch
        })

        print(f"  Layer {layer_num} done: Best PPL={best_ppl:.2f} at epoch {best_epoch}")

    results.append({
        'model': 'Standard Progressive (final)',
        'params': count_params(std_model),
        'best_ppl': std_layer_results[-1]['best_ppl'],
        'method': 'standard_progressive'
    })

    # ===== 4-layer from Scratch (Baseline) =====
    print("\n" + "=" * 70)
    print("4-layer from Scratch (Standard Training)")
    print("=" * 70)

    set_seed(42)
    scratch_model = LPTGrowableTransformer(
        vocab_size=vocab_size,
        dim=dim,
        seq_len=seq_len,
        num_heads=num_heads
    ).to(device)

    for _ in range(target_layers):
        scratch_model.add_layer()

    print(f"  Total params: {count_params(scratch_model):,}")

    scratch_ppl, scratch_epoch = train_until_convergence_standard(
        scratch_model, train_batches, val_batches, vocab_size,
        max_epochs=30, patience=5, lr=1e-3, device=device
    )

    results.append({
        'model': '4-layer Scratch (Standard)',
        'params': count_params(scratch_model),
        'best_ppl': scratch_ppl,
        'method': 'scratch_standard'
    })

    print(f"  Scratch: Best PPL={scratch_ppl:.2f} at epoch {scratch_epoch}")

    # ===== 4-layer LPT from Scratch =====
    print("\n" + "=" * 70)
    print("4-layer from Scratch (LPT Training)")
    print("=" * 70)

    set_seed(42)
    scratch_lpt_model = LPTGrowableTransformer(
        vocab_size=vocab_size,
        dim=dim,
        seq_len=seq_len,
        num_heads=num_heads
    ).to(device)

    for _ in range(target_layers):
        scratch_lpt_model.add_layer()

    print(f"  Total params: {count_params(scratch_lpt_model):,}")

    scratch_lpt_ppl, scratch_lpt_epoch = train_until_convergence_lpt(
        scratch_lpt_model, train_batches, val_batches, vocab_size,
        max_epochs=30, patience=5, lr=1e-3, device=device
    )

    results.append({
        'model': '4-layer Scratch (LPT)',
        'params': count_params(scratch_lpt_model),
        'best_ppl': scratch_lpt_ppl,
        'method': 'scratch_lpt'
    })

    print(f"  Scratch LPT: Best PPL={scratch_lpt_ppl:.2f} at epoch {scratch_lpt_epoch}")

    # ===== Summary =====
    print("\n" + "=" * 70)
    print("LAYER-BY-LAYER PROGRESS")
    print("=" * 70)

    print("\nLPT Progressive:")
    print(f"{'Layer':<8} {'Params':>10} {'Best PPL':>10} {'Epoch':>8}")
    print("-" * 38)
    for r in lpt_layer_results:
        print(f"{r['layer']:<8} {r['params']:>10,} {r['best_ppl']:>10.2f} {r['best_epoch']:>8}")

    print("\nStandard Progressive:")
    print(f"{'Layer':<8} {'Params':>10} {'Best PPL':>10} {'Epoch':>8}")
    print("-" * 38)
    for r in std_layer_results:
        print(f"{r['layer']:<8} {r['params']:>10,} {r['best_ppl']:>10.2f} {r['best_epoch']:>8}")

    print("\n" + "=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)
    print(f"\n{'Model':<30} {'Params':>10} {'Best PPL':>10}")
    print("-" * 52)
    for r in results:
        print(f"{r['model']:<30} {r['params']:>10,} {r['best_ppl']:>10.2f}")

    # Analysis
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    lpt_prog = results[0]['best_ppl']
    std_prog = results[1]['best_ppl']
    scratch_std = results[2]['best_ppl']
    scratch_lpt = results[3]['best_ppl']

    print(f"\n1. Progressive Growing comparison:")
    print(f"   LPT Progressive:      {lpt_prog:.2f}")
    print(f"   Standard Progressive: {std_prog:.2f}")
    diff = (std_prog - lpt_prog) / std_prog * 100
    print(f"   LPT advantage: {diff:+.1f}%")

    print(f"\n2. LPT Progressive vs Scratch:")
    print(f"   LPT Progressive: {lpt_prog:.2f}")
    print(f"   Scratch (LPT):   {scratch_lpt:.2f}")
    diff2 = (lpt_prog - scratch_lpt) / scratch_lpt * 100
    print(f"   Difference: {diff2:+.1f}%")

    print(f"\n3. Training method comparison (Scratch):")
    print(f"   Standard: {scratch_std:.2f}")
    print(f"   LPT:      {scratch_lpt:.2f}")
    diff3 = (scratch_std - scratch_lpt) / scratch_std * 100
    print(f"   LPT advantage: {diff3:+.1f}%")

    if lpt_prog < std_prog * 1.1:  # Within 10%
        print("\n=> LPT Progressive Growing WORKS!")
        if lpt_prog < scratch_lpt * 1.05:
            print("=> LPT Progressive matches Scratch performance!")
    else:
        print("\n=> LPT Progressive still worse than Standard Progressive")


if __name__ == "__main__":
    main()
