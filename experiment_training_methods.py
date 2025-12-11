"""
Experiment: Comparison of Training Methods

Compares three training approaches:
1. Standard Training: Single forward pass, single loss (final layer only)
2. Iterative Refinement Training (IRT): Multiple forward passes, loss at each segment
3. Layer-wise Progressive Training (LPT): Single forward pass, loss at each layer

All models use the same architecture (multi-layer Transformer) but differ in how
they compute and propagate losses during training.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import math
import random
import numpy as np
from typing import List

from modules import (
    StandardTransformer,
    DeepSupervisionTransformer,
    LayerProgressiveTransformer,
    compute_lpt_loss,
)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def download_wikitext(split: str = 'train') -> str:
    import urllib.request
    import ssl
    import os

    cache_dir = os.path.expanduser('~/.cache/wikitext2_real')
    os.makedirs(cache_dir, exist_ok=True)

    split_map = {'train': 'train', 'validation': 'valid', 'test': 'test'}
    base_url = 'https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2'
    filename = f'{split_map[split]}.txt'
    url = f'{base_url}/{filename}'
    file_path = os.path.join(cache_dir, filename)

    if not os.path.exists(file_path):
        print(f"  Downloading {filename}...")
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

        try:
            with urllib.request.urlopen(url, context=ctx, timeout=30) as response:
                data = response.read().decode('utf-8')
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(data)
        except Exception as e:
            print(f"  Download failed: {e}")
            return generate_synthetic_wikitext(split)

    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


def generate_synthetic_wikitext(split: str = 'train') -> str:
    paragraphs = [
        "The history of artificial intelligence began in antiquity with myths and stories.",
        "Natural language processing is a subfield of linguistics and computer science.",
        "Machine learning provides systems the ability to learn from experience.",
        "Deep learning is based on artificial neural networks with representation learning.",
        "The Turing test is a test of a machine's ability to exhibit intelligent behaviour.",
    ] * 20
    random.seed(42 if split == 'train' else 123)
    random.shuffle(paragraphs)
    return '\n\n'.join(paragraphs)


class WikiTextDataset(Dataset):
    def __init__(self, split: str = 'train', seq_len: int = 64, max_chars: int = 50000):
        self.seq_len = seq_len
        text = download_wikitext(split)
        if max_chars and len(text) > max_chars:
            text = text[:max_chars]
        self.text = text

        self.chars = [chr(i) for i in range(32, 127)] + ['\n', '\t']
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.vocab_size = len(self.chars)
        self.unk_idx = 0

        self.data = [self.char_to_idx.get(ch, self.unk_idx) for ch in text]
        self.num_samples = max(1, (len(self.data) - 1) // seq_len)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len
        x = self.data[start:end]
        y = self.data[start + 1:end + 1]
        if len(x) < self.seq_len:
            x = x + [self.unk_idx] * (self.seq_len - len(x))
        if len(y) < self.seq_len:
            y = y + [self.unk_idx] * (self.seq_len - len(y))
        return torch.tensor(x), torch.tensor(y)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# ===== Standard Training =====

def compute_perplexity_standard(model: StandardTransformer, dataloader: DataLoader,
                                 device: str) -> float:
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            loss = nn.functional.cross_entropy(
                y_hat.view(-1, model.vocab_size), y.view(-1), reduction='sum'
            )
            total_loss += loss.item()
            total_tokens += y.numel()

    return math.exp(min(total_loss / total_tokens, 100))


def train_epoch_standard(model: StandardTransformer, dataloader: DataLoader,
                         optimizer: optim.Optimizer, device: str) -> float:
    model.train()
    total_loss = 0.0
    num_batches = 0

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)

        y_hat = model(x)
        loss = nn.functional.cross_entropy(
            y_hat.view(-1, model.vocab_size), y.view(-1)
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


# ===== Iterative Refinement Training (IRT) =====

def compute_perplexity_irt(model: DeepSupervisionTransformer, dataloader: DataLoader,
                           device: str, num_segments: int = 2) -> float:
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            y_hat, _ = model(x, num_segments=num_segments)
            loss = nn.functional.cross_entropy(
                y_hat.view(-1, model.vocab_size), y.view(-1), reduction='sum'
            )
            total_loss += loss.item()
            total_tokens += y.numel()

    return math.exp(min(total_loss / total_tokens, 100))


def train_epoch_irt(model: DeepSupervisionTransformer, dataloader: DataLoader,
                    optimizer: optim.Optimizer, device: str,
                    num_segments: int = 2) -> float:
    model.train()
    total_loss = 0.0
    num_batches = 0

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        state = None

        for _ in range(num_segments):
            state, y_hat = model.forward_pass(x, state)
            loss = nn.functional.cross_entropy(
                y_hat.view(-1, model.vocab_size), y.view(-1)
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            state = state.detach()
            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches


# ===== Layer-wise Progressive Training (LPT) =====

def compute_perplexity_lpt(model: LayerProgressiveTransformer, dataloader: DataLoader,
                           device: str) -> float:
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            y_hat = model(x)  # Final layer output
            loss = nn.functional.cross_entropy(
                y_hat.view(-1, model.vocab_size), y.view(-1), reduction='sum'
            )
            total_loss += loss.item()
            total_tokens += y.numel()

    return math.exp(min(total_loss / total_tokens, 100))


def train_epoch_lpt(model: LayerProgressiveTransformer, dataloader: DataLoader,
                    optimizer: optim.Optimizer, device: str,
                    mode: str = 'sum') -> tuple:
    """
    Train with Layer-wise Progressive Training.

    Returns:
        avg_loss: Average total loss
        layer_losses: Average loss per layer
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    layer_loss_sums: List[float] = [0.0] * model.num_layers

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)

        # Get outputs from all layers
        outputs = model.forward_all_layers(x)

        # Compute LPT loss
        loss, layer_losses = compute_lpt_loss(
            outputs, y, model.vocab_size, mode=mode
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        for i, ll in enumerate(layer_losses):
            layer_loss_sums[i] += ll.item()
        num_batches += 1

    avg_layer_losses = [s / num_batches for s in layer_loss_sums]
    return total_loss / num_batches, avg_layer_losses


def main():
    set_seed(42)

    # Config
    seq_len = 64
    dim = 64
    num_layers = 4  # Multiple layers to see layer-wise effects
    num_heads = 4
    num_segments = 2  # For IRT
    batch_size = 16
    max_epochs = 30
    patience = 5
    lr = 1e-3
    device = 'cpu'
    train_chars = 20000
    val_chars = 5000

    print("=" * 70)
    print("Training Methods Comparison Experiment")
    print("=" * 70)
    print("\nComparing three training approaches:")
    print("1. Standard: Single forward, loss at final layer only")
    print("2. IRT (Iterative Refinement): Multiple forward passes, loss at each")
    print("3. LPT (Layer-wise Progressive): Single forward, loss at each layer")

    # Load data
    print("\nLoading WikiText-2...")
    train_dataset = WikiTextDataset('train', seq_len=seq_len, max_chars=train_chars)
    val_dataset = WikiTextDataset('validation', seq_len=seq_len, max_chars=val_chars)
    vocab_size = train_dataset.vocab_size

    print(f"Vocab: {vocab_size}, Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    print(f"Config: {num_layers} layers, dim={dim}, heads={num_heads}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    results = []

    # ===== Model 1: Standard Training =====
    print("\n" + "=" * 70)
    print("Model 1: Standard Training (baseline)")
    print("=" * 70)

    set_seed(42)
    model_standard = StandardTransformer(
        vocab_size=vocab_size,
        dim=dim,
        seq_len=seq_len,
        num_layers=num_layers,
        num_heads=num_heads,
    ).to(device)

    param_count = count_params(model_standard)
    print(f"Parameters: {param_count:,}")

    optimizer = optim.AdamW(model_standard.parameters(), lr=lr, weight_decay=0.01)

    print(f"\n{'Epoch':>5} | {'Train Loss':>10} | {'Val PPL':>10}")
    print("-" * 40)

    best_ppl = float('inf')
    best_epoch = 0
    no_improve = 0

    for epoch in range(max_epochs):
        train_loss = train_epoch_standard(model_standard, train_loader, optimizer, device)
        val_ppl = compute_perplexity_standard(model_standard, val_loader, device)

        if val_ppl < best_ppl:
            best_ppl = val_ppl
            best_epoch = epoch + 1
            no_improve = 0
            marker = " *"
        else:
            no_improve += 1
            marker = ""

        print(f"{epoch+1:5d} | {train_loss:10.4f} | {val_ppl:10.2f}{marker}")

        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    results.append({
        'name': 'Standard',
        'params': param_count,
        'best_ppl': best_ppl,
        'best_epoch': best_epoch,
        'description': 'Single forward, final layer loss'
    })

    # ===== Model 2: IRT (Iterative Refinement Training) =====
    print("\n" + "=" * 70)
    print("Model 2: IRT (Iterative Refinement Training)")
    print("=" * 70)

    set_seed(42)
    model_irt = DeepSupervisionTransformer(
        vocab_size=vocab_size,
        dim=dim,
        seq_len=seq_len,
        num_layers=num_layers,
        num_heads=num_heads,
    ).to(device)

    param_count = count_params(model_irt)
    print(f"Parameters: {param_count:,}")
    print(f"Segments: {num_segments}")

    optimizer = optim.AdamW(model_irt.parameters(), lr=lr, weight_decay=0.01)

    print(f"\n{'Epoch':>5} | {'Train Loss':>10} | {'Val PPL':>10}")
    print("-" * 40)

    best_ppl = float('inf')
    best_epoch = 0
    no_improve = 0

    for epoch in range(max_epochs):
        train_loss = train_epoch_irt(model_irt, train_loader, optimizer, device, num_segments)
        val_ppl = compute_perplexity_irt(model_irt, val_loader, device, num_segments)

        if val_ppl < best_ppl:
            best_ppl = val_ppl
            best_epoch = epoch + 1
            no_improve = 0
            marker = " *"
        else:
            no_improve += 1
            marker = ""

        print(f"{epoch+1:5d} | {train_loss:10.4f} | {val_ppl:10.2f}{marker}")

        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    results.append({
        'name': 'IRT',
        'params': param_count,
        'best_ppl': best_ppl,
        'best_epoch': best_epoch,
        'description': f'{num_segments} segments, loss at each'
    })

    # ===== Model 3: LPT (Layer-wise Progressive Training) - Sum mode =====
    print("\n" + "=" * 70)
    print("Model 3: LPT (Layer-wise Progressive Training) - Sum")
    print("=" * 70)

    set_seed(42)
    model_lpt_sum = LayerProgressiveTransformer(
        vocab_size=vocab_size,
        dim=dim,
        seq_len=seq_len,
        num_layers=num_layers,
        num_heads=num_heads,
    ).to(device)

    param_count = count_params(model_lpt_sum)
    print(f"Parameters: {param_count:,}")
    print(f"Mode: sum (equal weight for all {num_layers} layers)")

    optimizer = optim.AdamW(model_lpt_sum.parameters(), lr=lr, weight_decay=0.01)

    print(f"\n{'Epoch':>5} | {'Train Loss':>10} | {'Val PPL':>10} | Layer Losses")
    print("-" * 70)

    best_ppl = float('inf')
    best_epoch = 0
    no_improve = 0

    for epoch in range(max_epochs):
        train_loss, layer_losses = train_epoch_lpt(
            model_lpt_sum, train_loader, optimizer, device, mode='sum'
        )
        val_ppl = compute_perplexity_lpt(model_lpt_sum, val_loader, device)

        if val_ppl < best_ppl:
            best_ppl = val_ppl
            best_epoch = epoch + 1
            no_improve = 0
            marker = " *"
        else:
            no_improve += 1
            marker = ""

        layer_str = " ".join([f"L{i+1}:{l:.2f}" for i, l in enumerate(layer_losses)])
        print(f"{epoch+1:5d} | {train_loss:10.4f} | {val_ppl:10.2f}{marker} | {layer_str}")

        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    results.append({
        'name': 'LPT (sum)',
        'params': param_count,
        'best_ppl': best_ppl,
        'best_epoch': best_epoch,
        'description': f'{num_layers} layers, loss at each (sum)'
    })

    # ===== Model 4: LPT - Weighted mode =====
    print("\n" + "=" * 70)
    print("Model 4: LPT (Layer-wise Progressive Training) - Weighted")
    print("=" * 70)

    set_seed(42)
    model_lpt_weighted = LayerProgressiveTransformer(
        vocab_size=vocab_size,
        dim=dim,
        seq_len=seq_len,
        num_layers=num_layers,
        num_heads=num_heads,
    ).to(device)

    param_count = count_params(model_lpt_weighted)
    print(f"Parameters: {param_count:,}")
    print(f"Mode: weighted (deeper layers have higher weight)")

    optimizer = optim.AdamW(model_lpt_weighted.parameters(), lr=lr, weight_decay=0.01)

    print(f"\n{'Epoch':>5} | {'Train Loss':>10} | {'Val PPL':>10} | Layer Losses")
    print("-" * 70)

    best_ppl = float('inf')
    best_epoch = 0
    no_improve = 0

    for epoch in range(max_epochs):
        train_loss, layer_losses = train_epoch_lpt(
            model_lpt_weighted, train_loader, optimizer, device, mode='weighted'
        )
        val_ppl = compute_perplexity_lpt(model_lpt_weighted, val_loader, device)

        if val_ppl < best_ppl:
            best_ppl = val_ppl
            best_epoch = epoch + 1
            no_improve = 0
            marker = " *"
        else:
            no_improve += 1
            marker = ""

        layer_str = " ".join([f"L{i+1}:{l:.2f}" for i, l in enumerate(layer_losses)])
        print(f"{epoch+1:5d} | {train_loss:10.4f} | {val_ppl:10.2f}{marker} | {layer_str}")

        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    results.append({
        'name': 'LPT (weighted)',
        'params': param_count,
        'best_ppl': best_ppl,
        'best_epoch': best_epoch,
        'description': f'{num_layers} layers, weighted loss'
    })

    # ===== Summary =====
    print("\n" + "=" * 70)
    print("EXPERIMENT RESULTS SUMMARY")
    print("=" * 70)
    print(f"\n{'Model':<18} | {'Params':>10} | {'Best PPL':>10} | {'Epoch':>6} | Description")
    print("-" * 80)
    for r in results:
        print(f"{r['name']:<18} | {r['params']:>10,} | {r['best_ppl']:>10.2f} | {r['best_epoch']:>6} | {r['description']}")

    print("\n" + "=" * 70)
    print("Analysis")
    print("=" * 70)

    baseline_ppl = results[0]['best_ppl']
    print(f"\nBaseline (Standard): {baseline_ppl:.2f} PPL")

    for r in results[1:]:
        diff = baseline_ppl - r['best_ppl']
        pct = (diff / baseline_ppl) * 100
        if diff > 0:
            print(f"  {r['name']}: {diff:.2f} PPL better ({pct:.1f}% improvement)")
        else:
            print(f"  {r['name']}: {-diff:.2f} PPL worse ({-pct:.1f}% degradation)")

    print("\n" + "=" * 70)
    print("Training Method Comparison")
    print("=" * 70)
    print("""
┌─────────────────────────────────────────────────────────────────────┐
│ Standard Training                                                    │
├─────────────────────────────────────────────────────────────────────┤
│ x → [L1] → [L2] → [L3] → [L4] → Output → Loss                       │
│                                            ↑                         │
│                                    Only final layer                  │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ IRT (Iterative Refinement Training)                                  │
├─────────────────────────────────────────────────────────────────────┤
│ Segment 1: x → [Model] → Output → Loss₁ → backward                  │
│                   ↓ state                                            │
│ Segment 2: x → [Model] → Output → Loss₂ → backward                  │
│            (+state)                                                  │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ LPT (Layer-wise Progressive Training)                                │
├─────────────────────────────────────────────────────────────────────┤
│ x → [L1] → Out₁ → Loss₁ ─┐                                          │
│       ↓                   │                                          │
│     [L2] → Out₂ → Loss₂ ─┼→ Total Loss → backward                   │
│       ↓                   │                                          │
│     [L3] → Out₃ → Loss₃ ─┤                                          │
│       ↓                   │                                          │
│     [L4] → Out₄ → Loss₄ ─┘                                          │
│                                                                      │
│ All layers supervised with shared output head                        │
└─────────────────────────────────────────────────────────────────────┘
""")


if __name__ == "__main__":
    main()
