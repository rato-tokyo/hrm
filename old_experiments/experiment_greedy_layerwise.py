"""
Experiment: Greedy Layer-wise Training

Train one layer at a time, freezing previous layers when adding new ones.
This tests whether building up a model layer-by-layer can match training from scratch.

Approach:
1. Train 1-layer model until convergence
2. Freeze layer 1, add layer 2, train until convergence
3. Freeze layers 1-2, add layer 3, train until convergence
4. Freeze layers 1-3, add layer 4, train until convergence
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


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class GreedyGrowableTransformer(nn.Module):
    """
    Transformer that grows layer by layer with greedy training.
    Each new layer is trained while previous layers are frozen.
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

        # Output head
        self.output_head = nn.Linear(dim, vocab_size, bias=False)

        self._init_embedding()

    def _init_embedding(self) -> None:
        nn.init.trunc_normal_(self.embedding.weight, std=0.02)
        nn.init.trunc_normal_(self.output_head.weight, std=0.02)

    def add_layer(self) -> None:
        """Add a new layer and freeze all previous layers."""
        # Freeze embedding if this is the first layer
        if len(self.layers) == 0:
            self.embedding.requires_grad_(False)

        # Freeze all existing layers
        for layer in self.layers:
            layer.requires_grad_(False)

        # Add new trainable layer
        new_layer = TransformerBlock(self.dim, self.num_heads)
        for module in new_layer.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
        self.layers.append(new_layer)

        # Output head is always trainable
        self.output_head.requires_grad_(True)

    def unfreeze_all(self) -> None:
        """Unfreeze all parameters for fine-tuning."""
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
    return np.exp(avg_loss)


def train_until_convergence(
    model: nn.Module,
    train_batches: List,
    val_batches: List,
    vocab_size: int,
    max_epochs: int = 30,
    patience: int = 5,
    lr: float = 1e-3,
    device: str = 'cpu',
    verbose: bool = True,
    layer_name: str = ""
) -> Tuple[float, int, List[float]]:
    """Train until convergence, return best PPL, epoch, and history."""
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr
    )

    best_ppl = float('inf')
    best_epoch = 0
    patience_counter = 0
    ppl_history = []

    for epoch in range(max_epochs):
        train_loss = train_epoch(model, train_batches, optimizer, vocab_size, device)
        val_ppl = evaluate(model, val_batches, vocab_size, device)
        ppl_history.append(val_ppl)

        if val_ppl < best_ppl:
            best_ppl = val_ppl
            best_epoch = epoch + 1
            patience_counter = 0
        else:
            patience_counter += 1

        if verbose:
            print(f"    Epoch {epoch + 1}: Train Loss={train_loss:.4f}, Val PPL={val_ppl:.2f}")

        if patience_counter >= patience:
            if verbose:
                print(f"    Converged at epoch {epoch + 1}")
            break

    return best_ppl, best_epoch, ppl_history


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
    print("Experiment: Greedy Layer-wise Training")
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

    # ===== Greedy Layer-wise Training =====
    print("\n" + "=" * 70)
    print("Greedy Layer-wise Training")
    print("=" * 70)

    set_seed(42)
    greedy_model = GreedyGrowableTransformer(
        vocab_size=vocab_size,
        dim=dim,
        seq_len=seq_len,
        num_heads=num_heads
    ).to(device)

    layer_results = []

    for layer_num in range(1, target_layers + 1):
        print(f"\n--- Adding Layer {layer_num} ---")
        greedy_model.add_layer()

        print(f"  Total params: {count_params(greedy_model):,}")
        print(f"  Trainable params: {count_trainable_params(greedy_model):,}")

        best_ppl, best_epoch, history = train_until_convergence(
            greedy_model, train_batches, val_batches, vocab_size,
            max_epochs=30, patience=5, lr=1e-3, device=device,
            layer_name=f"Layer {layer_num}"
        )

        layer_results.append({
            'layer': layer_num,
            'params': count_params(greedy_model),
            'trainable': count_trainable_params(greedy_model),
            'best_ppl': best_ppl,
            'best_epoch': best_epoch
        })

        print(f"  Layer {layer_num} done: Best PPL={best_ppl:.2f} at epoch {best_epoch}")

    results.append({
        'model': 'Greedy (final)',
        'params': count_params(greedy_model),
        'trainable': count_params(greedy_model),
        'best_ppl': layer_results[-1]['best_ppl'],
        'method': 'greedy'
    })

    # ===== Greedy + Fine-tune =====
    print("\n" + "=" * 70)
    print("Fine-tuning all layers after greedy training")
    print("=" * 70)

    greedy_model.unfreeze_all()
    print(f"  Total params: {count_params(greedy_model):,}")
    print(f"  Trainable params: {count_trainable_params(greedy_model):,}")

    finetune_ppl, finetune_epoch, _ = train_until_convergence(
        greedy_model, train_batches, val_batches, vocab_size,
        max_epochs=30, patience=5, lr=1e-4,  # Lower LR for fine-tuning
        device=device, layer_name="Fine-tune"
    )

    results.append({
        'model': 'Greedy + Fine-tune',
        'params': count_params(greedy_model),
        'trainable': count_params(greedy_model),
        'best_ppl': finetune_ppl,
        'method': 'greedy+finetune'
    })

    print(f"  Fine-tuned: Best PPL={finetune_ppl:.2f} at epoch {finetune_epoch}")

    # ===== Scratch Training (Baseline) =====
    print("\n" + "=" * 70)
    print("Training 4-layer from Scratch (Baseline)")
    print("=" * 70)

    set_seed(42)
    scratch_model = GreedyGrowableTransformer(
        vocab_size=vocab_size,
        dim=dim,
        seq_len=seq_len,
        num_heads=num_heads
    ).to(device)

    # Add all layers at once (no freezing)
    for _ in range(target_layers):
        # Add layer without freezing
        new_layer = TransformerBlock(dim, num_heads)
        for module in new_layer.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
        scratch_model.layers.append(new_layer)

    print(f"  Total params: {count_params(scratch_model):,}")
    print(f"  Trainable params: {count_trainable_params(scratch_model):,}")

    scratch_ppl, scratch_epoch, _ = train_until_convergence(
        scratch_model, train_batches, val_batches, vocab_size,
        max_epochs=30, patience=5, lr=1e-3, device=device,
        layer_name="Scratch"
    )

    results.append({
        'model': '4-layer Scratch',
        'params': count_params(scratch_model),
        'trainable': count_params(scratch_model),
        'best_ppl': scratch_ppl,
        'method': 'scratch'
    })

    print(f"  Scratch: Best PPL={scratch_ppl:.2f} at epoch {scratch_epoch}")

    # ===== Summary =====
    print("\n" + "=" * 70)
    print("LAYER-BY-LAYER PROGRESS (Greedy)")
    print("=" * 70)
    print(f"\n{'Layer':<8} {'Params':>10} {'Trainable':>12} {'Best PPL':>10} {'Epoch':>8}")
    print("-" * 50)
    for r in layer_results:
        print(f"{r['layer']:<8} {r['params']:>10,} {r['trainable']:>12,} {r['best_ppl']:>10.2f} {r['best_epoch']:>8}")

    print("\n" + "=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)
    print(f"\n{'Model':<25} {'Params':>10} {'Best PPL':>10}")
    print("-" * 47)
    for r in results:
        print(f"{r['model']:<25} {r['params']:>10,} {r['best_ppl']:>10.2f}")

    # Analysis
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    greedy_ppl = results[0]['best_ppl']
    finetune_ppl = results[1]['best_ppl']
    scratch_ppl = results[2]['best_ppl']

    print(f"\n1. Greedy vs Scratch:")
    diff = (greedy_ppl - scratch_ppl) / scratch_ppl * 100
    print(f"   Greedy: {greedy_ppl:.2f}, Scratch: {scratch_ppl:.2f} ({diff:+.1f}%)")

    print(f"\n2. Fine-tuning improvement:")
    improve = (greedy_ppl - finetune_ppl) / greedy_ppl * 100
    print(f"   Before: {greedy_ppl:.2f}, After: {finetune_ppl:.2f} ({improve:+.1f}%)")

    print(f"\n3. Fine-tuned vs Scratch:")
    diff2 = (finetune_ppl - scratch_ppl) / scratch_ppl * 100
    print(f"   Fine-tuned: {finetune_ppl:.2f}, Scratch: {scratch_ppl:.2f} ({diff2:+.1f}%)")

    if finetune_ppl < scratch_ppl:
        print("\n=> Greedy + Fine-tune BEATS Scratch!")
    elif abs(finetune_ppl - scratch_ppl) < 0.5:
        print("\n=> Greedy + Fine-tune matches Scratch")
    else:
        print("\n=> Scratch still better than Greedy + Fine-tune")


if __name__ == "__main__":
    main()
