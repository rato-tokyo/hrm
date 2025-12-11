"""
Experiment: Layer Freezing vs Full Training

Compare two approaches when adding a new layer:
1. Full Training: Train all layers together
2. Frozen Training: Freeze existing layers, only train new layer

This tests whether pre-trained representations transfer well to new layers.
"""

import torch
import torch.nn as nn
import random
import numpy as np
from typing import List, Tuple, Dict
import copy

from modules.transformer import TransformerBlock


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class GrowableTransformer(nn.Module):
    """
    Transformer that can grow by adding layers.
    Supports freezing existing layers when adding new ones.
    """

    def __init__(
        self,
        vocab_size: int,
        dim: int = 64,
        seq_len: int = 64,
        num_layers: int = 3,
        num_heads: int = 4,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.seq_len = seq_len

        # Input embedding
        self.embedding = nn.Embedding(vocab_size, dim)

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(dim, num_heads) for _ in range(num_layers)
        ])

        # Output head
        self.output_head = nn.Linear(dim, vocab_size, bias=False)

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
            elif isinstance(module, nn.Embedding):
                nn.init.trunc_normal_(module.weight, std=0.02)

    def add_layer(self, num_heads: int = 4, freeze_existing: bool = False) -> None:
        """Add a new layer, optionally freezing existing layers."""
        # Add new layer
        new_layer = TransformerBlock(self.dim, num_heads)
        # Initialize new layer
        for module in new_layer.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
        self.layers.append(new_layer)

        if freeze_existing:
            # Freeze embedding and all existing layers (except the new one)
            self.embedding.requires_grad_(False)
            for i, layer in enumerate(self.layers[:-1]):
                layer.requires_grad_(False)
            # Keep output head trainable
            self.output_head.requires_grad_(True)

    def unfreeze_all(self) -> None:
        """Unfreeze all parameters."""
        for param in self.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embedding(x)

        for layer in self.layers:
            h = layer(h)

        return self.output_head(h)


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
        # Fallback to simple text
        train_text = "The quick brown fox jumps over the lazy dog. " * (train_chars // 45)
        val_text = "A quick brown dog runs in the park. " * (val_chars // 35)

    # Build vocabulary
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


def train_epoch(
    model: nn.Module,
    batches: List[Tuple[torch.Tensor, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    vocab_size: int,
    device: str = 'cpu'
) -> float:
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


@torch.no_grad()
def evaluate(
    model: nn.Module,
    batches: List[Tuple[torch.Tensor, torch.Tensor]],
    vocab_size: int,
    device: str = 'cpu'
) -> float:
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
    return np.exp(avg_loss)  # Perplexity


def train_model(
    model: nn.Module,
    train_batches: List,
    val_batches: List,
    vocab_size: int,
    max_epochs: int = 20,
    patience: int = 5,
    lr: float = 1e-3,
    device: str = 'cpu',
    verbose: bool = True
) -> Tuple[float, int]:
    """Train model and return best PPL and epoch."""
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr
    )

    best_ppl = float('inf')
    best_epoch = 0
    patience_counter = 0

    for epoch in range(max_epochs):
        train_loss = train_epoch(model, train_batches, optimizer, vocab_size, device)
        val_ppl = evaluate(model, val_batches, vocab_size, device)

        if val_ppl < best_ppl:
            best_ppl = val_ppl
            best_epoch = epoch + 1
            patience_counter = 0
        else:
            patience_counter += 1

        if verbose and (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch + 1}: Train Loss={train_loss:.4f}, Val PPL={val_ppl:.2f}")

        if patience_counter >= patience:
            if verbose:
                print(f"  Early stopping at epoch {epoch + 1}")
            break

    return best_ppl, best_epoch


def main():
    set_seed(42)

    # Config
    seq_len = 64
    dim = 64
    initial_layers = 3
    num_heads = 4
    batch_size = 16
    device = 'cpu'
    train_chars = 20000
    val_chars = 5000

    print("=" * 70)
    print("Experiment: Layer Freezing vs Full Training")
    print("=" * 70)
    print(f"\nConfig:")
    print(f"  Initial layers: {initial_layers}")
    print(f"  Final layers: {initial_layers + 1}")
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

    # ===== Phase 1: Train 3-layer model =====
    print("\n" + "=" * 70)
    print("Phase 1: Pre-training 3-layer model")
    print("=" * 70)

    set_seed(42)
    base_model = GrowableTransformer(
        vocab_size=vocab_size,
        dim=dim,
        seq_len=seq_len,
        num_layers=initial_layers,
        num_heads=num_heads
    ).to(device)

    print(f"Parameters: {count_params(base_model):,}")

    base_ppl, base_epoch = train_model(
        base_model, train_batches, val_batches, vocab_size,
        max_epochs=30, patience=5, lr=1e-3, device=device
    )
    print(f"\n3-layer model: Best PPL={base_ppl:.2f} at epoch {base_epoch}")

    results.append({
        'model': '3-layer (base)',
        'params': count_params(base_model),
        'trainable_params': count_params(base_model),
        'best_ppl': base_ppl,
        'best_epoch': base_epoch
    })

    # Save base model state for fair comparison
    base_state = copy.deepcopy(base_model.state_dict())

    # ===== Phase 2a: Add layer with FROZEN existing layers =====
    print("\n" + "=" * 70)
    print("Phase 2a: Add 4th layer (FREEZE existing layers)")
    print("=" * 70)

    set_seed(42)
    frozen_model = GrowableTransformer(
        vocab_size=vocab_size,
        dim=dim,
        seq_len=seq_len,
        num_layers=initial_layers,
        num_heads=num_heads
    ).to(device)
    frozen_model.load_state_dict(base_state)
    frozen_model.add_layer(num_heads=num_heads, freeze_existing=True)

    print(f"Total parameters: {count_params(frozen_model):,}")
    print(f"Trainable parameters: {count_trainable_params(frozen_model):,}")

    frozen_ppl, frozen_epoch = train_model(
        frozen_model, train_batches, val_batches, vocab_size,
        max_epochs=30, patience=5, lr=1e-3, device=device
    )
    print(f"\n4-layer (frozen): Best PPL={frozen_ppl:.2f} at epoch {frozen_epoch}")

    results.append({
        'model': '4-layer (frozen)',
        'params': count_params(frozen_model),
        'trainable_params': count_trainable_params(frozen_model),
        'best_ppl': frozen_ppl,
        'best_epoch': frozen_epoch
    })

    # ===== Phase 2b: Add layer with FULL training =====
    print("\n" + "=" * 70)
    print("Phase 2b: Add 4th layer (FULL training)")
    print("=" * 70)

    set_seed(42)
    full_model = GrowableTransformer(
        vocab_size=vocab_size,
        dim=dim,
        seq_len=seq_len,
        num_layers=initial_layers,
        num_heads=num_heads
    ).to(device)
    full_model.load_state_dict(base_state)
    full_model.add_layer(num_heads=num_heads, freeze_existing=False)

    print(f"Total parameters: {count_params(full_model):,}")
    print(f"Trainable parameters: {count_trainable_params(full_model):,}")

    full_ppl, full_epoch = train_model(
        full_model, train_batches, val_batches, vocab_size,
        max_epochs=30, patience=5, lr=1e-3, device=device
    )
    print(f"\n4-layer (full): Best PPL={full_ppl:.2f} at epoch {full_epoch}")

    results.append({
        'model': '4-layer (full)',
        'params': count_params(full_model),
        'trainable_params': count_trainable_params(full_model),
        'best_ppl': full_ppl,
        'best_epoch': full_epoch
    })

    # ===== Phase 2c: Train 4-layer from scratch (baseline) =====
    print("\n" + "=" * 70)
    print("Phase 2c: Train 4-layer from SCRATCH (baseline)")
    print("=" * 70)

    set_seed(42)
    scratch_model = GrowableTransformer(
        vocab_size=vocab_size,
        dim=dim,
        seq_len=seq_len,
        num_layers=initial_layers + 1,
        num_heads=num_heads
    ).to(device)

    print(f"Total parameters: {count_params(scratch_model):,}")
    print(f"Trainable parameters: {count_trainable_params(scratch_model):,}")

    scratch_ppl, scratch_epoch = train_model(
        scratch_model, train_batches, val_batches, vocab_size,
        max_epochs=30, patience=5, lr=1e-3, device=device
    )
    print(f"\n4-layer (scratch): Best PPL={scratch_ppl:.2f} at epoch {scratch_epoch}")

    results.append({
        'model': '4-layer (scratch)',
        'params': count_params(scratch_model),
        'trainable_params': count_trainable_params(scratch_model),
        'best_ppl': scratch_ppl,
        'best_epoch': scratch_epoch
    })

    # ===== Summary =====
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n{'Model':<20} {'Params':>10} {'Trainable':>12} {'Best PPL':>10} {'Epoch':>8}")
    print("-" * 62)
    for r in results:
        print(f"{r['model']:<20} {r['params']:>10,} {r['trainable_params']:>12,} {r['best_ppl']:>10.2f} {r['best_epoch']:>8}")

    # Analysis
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    base = results[0]['best_ppl']
    frozen = results[1]['best_ppl']
    full = results[2]['best_ppl']
    scratch = results[3]['best_ppl']

    print(f"\n1. Adding 4th layer improvement over 3-layer base ({base:.2f}):")
    print(f"   - Frozen:  {frozen:.2f} ({(base - frozen) / base * 100:+.1f}%)")
    print(f"   - Full:    {full:.2f} ({(base - full) / base * 100:+.1f}%)")
    print(f"   - Scratch: {scratch:.2f} ({(base - scratch) / base * 100:+.1f}%)")

    print(f"\n2. Frozen vs Full training difference:")
    diff = (frozen - full) / full * 100
    print(f"   - PPL difference: {frozen:.2f} vs {full:.2f} ({diff:+.1f}%)")

    print(f"\n3. Pre-training benefit (Full vs Scratch):")
    pretrain_diff = (scratch - full) / scratch * 100
    print(f"   - PPL difference: {full:.2f} vs {scratch:.2f} ({pretrain_diff:+.1f}%)")

    if frozen < full:
        print("\n=> Frozen training performed BETTER (unexpected)")
    elif abs(frozen - full) < 0.5:
        print("\n=> Frozen and Full training performed SIMILARLY")
    else:
        print("\n=> Full training performed BETTER (as expected)")


if __name__ == "__main__":
    main()
