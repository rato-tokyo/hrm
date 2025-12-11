"""
Experiment: HRM vs Deep Supervision Transformer vs Standard Transformer

This experiment compares:
1. Standard Transformer (single forward pass, single loss)
2. Deep Supervision Transformer (multiple forward passes, loss at each segment)
3. HRM (hierarchical structure with deep supervision)

The goal is to isolate the effect of:
- Deep supervision training alone
- Hierarchical structure (HRM-specific)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import math
import random
import numpy as np
from typing import Optional

from modules import HRM, DeepSupervisionTransformer, StandardTransformer


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def download_wikitext(split: str = 'train') -> str:
    """Download WikiText-2 raw text files."""
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
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}
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


# ===== Standard Transformer Training (single forward, single loss) =====

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


# ===== Deep Supervision Transformer Training =====

def compute_perplexity_deepsup(model: DeepSupervisionTransformer, dataloader: DataLoader,
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


def train_epoch_deepsup(model: DeepSupervisionTransformer, dataloader: DataLoader,
                        optimizer: optim.Optimizer, device: str,
                        num_segments: int = 2) -> float:
    """Train with deep supervision: compute loss at each segment."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        state = None

        # Multiple forward passes with loss at each segment
        for _ in range(num_segments):
            state, y_hat = model.forward_pass(x, state)
            loss = nn.functional.cross_entropy(
                y_hat.view(-1, model.vocab_size), y.view(-1)
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # Detach state for truncated BPTT
            state = state.detach()

            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches


# ===== HRM Training =====

def compute_perplexity_hrm(model: HRM, dataloader: DataLoader, device: str,
                           num_segments: int = 2) -> float:
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


def train_epoch_hrm(model: HRM, dataloader: DataLoader, optimizer: optim.Optimizer,
                    device: str, num_segments: int = 2) -> float:
    model.train()
    total_loss = 0.0
    num_batches = 0

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        states = None

        for _ in range(num_segments):
            states, y_hat, _ = model.forward_pass(x, states)
            loss = nn.functional.cross_entropy(
                y_hat.view(-1, model.vocab_size), y.view(-1)
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            states = [s.detach() for s in states]
            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def main():
    set_seed(42)

    # Config
    seq_len = 64
    dim = 64
    num_layers = 1  # Fair comparison: all models use 1 transformer layer
    num_heads = 4
    num_segments = 2
    batch_size = 16
    max_epochs = 30
    patience = 5  # More patience for fair comparison
    lr = 1e-3
    device = 'cpu'
    train_chars = 20000
    val_chars = 5000

    print("=" * 70)
    print("Deep Supervision Experiment: HRM vs Transformer")
    print("=" * 70)
    print("\nThis experiment compares:")
    print("1. Standard Transformer: Single forward pass, single loss")
    print("2. Deep Supervision Transformer: Multiple passes, loss at each segment")
    print("3. HRM: Hierarchical layers with deep supervision")

    # Load data
    print("\nLoading WikiText-2...")
    train_dataset = WikiTextDataset('train', seq_len=seq_len, max_chars=train_chars)
    val_dataset = WikiTextDataset('validation', seq_len=seq_len, max_chars=val_chars)
    vocab_size = train_dataset.vocab_size

    print(f"Vocab: {vocab_size}, Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    results = []

    # ===== Model 1: Standard Transformer =====
    print("\n" + "=" * 70)
    print("Model 1: Standard Transformer (baseline, no deep supervision)")
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
    initial_ppl = compute_perplexity_standard(model_standard, val_loader, device)
    print(f"Initial Val PPL: {initial_ppl:.2f}")

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
        'name': 'Standard Transformer',
        'params': param_count,
        'best_ppl': best_ppl,
        'best_epoch': best_epoch,
        'description': 'Single forward, single loss'
    })

    # ===== Model 2: Deep Supervision Transformer =====
    print("\n" + "=" * 70)
    print("Model 2: Deep Supervision Transformer")
    print("=" * 70)

    set_seed(42)
    model_deepsup = DeepSupervisionTransformer(
        vocab_size=vocab_size,
        dim=dim,
        seq_len=seq_len,
        num_layers=num_layers,
        num_heads=num_heads,
    ).to(device)

    param_count = count_params(model_deepsup)
    print(f"Parameters: {param_count:,}")
    print(f"Training with {num_segments} segments (deep supervision)")

    optimizer = optim.AdamW(model_deepsup.parameters(), lr=lr, weight_decay=0.01)
    initial_ppl = compute_perplexity_deepsup(model_deepsup, val_loader, device, num_segments)
    print(f"Initial Val PPL: {initial_ppl:.2f}")

    print(f"\n{'Epoch':>5} | {'Train Loss':>10} | {'Val PPL':>10}")
    print("-" * 40)

    best_ppl = float('inf')
    best_epoch = 0
    no_improve = 0

    for epoch in range(max_epochs):
        train_loss = train_epoch_deepsup(
            model_deepsup, train_loader, optimizer, device, num_segments
        )
        val_ppl = compute_perplexity_deepsup(model_deepsup, val_loader, device, num_segments)

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
        'name': 'DeepSup Transformer',
        'params': param_count,
        'best_ppl': best_ppl,
        'best_epoch': best_epoch,
        'description': f'{num_segments} segments, loss at each'
    })

    # ===== Model 3: HRM =====
    print("\n" + "=" * 70)
    print("Model 3: HRM (Hierarchical Reasoning Model)")
    print("=" * 70)

    # HRM with 2 hierarchical layers (L and H)
    # Use smaller dim to match parameter count with DeepSup Transformer
    hrm_dim = 48  # Smaller to roughly match DeepSup params
    set_seed(42)
    model_hrm = HRM(
        vocab_size=vocab_size,
        dim=hrm_dim,
        seq_len=seq_len,
        num_layers=num_layers,
        num_heads=num_heads,
        N=1,  # N segments internally
        T=2   # H updates every T steps
    ).to(device)

    param_count = count_params(model_hrm)
    print(f"Parameters: {param_count:,}")
    print(f"Training with {num_segments} segments (deep supervision)")

    optimizer = optim.AdamW(model_hrm.parameters(), lr=lr, weight_decay=0.01)
    initial_ppl = compute_perplexity_hrm(model_hrm, val_loader, device, num_segments)
    print(f"Initial Val PPL: {initial_ppl:.2f}")

    print(f"\n{'Epoch':>5} | {'Train Loss':>10} | {'Val PPL':>10}")
    print("-" * 40)

    best_ppl = float('inf')
    best_epoch = 0
    no_improve = 0

    for epoch in range(max_epochs):
        train_loss = train_epoch_hrm(
            model_hrm, train_loader, optimizer, device, num_segments
        )
        val_ppl = compute_perplexity_hrm(model_hrm, val_loader, device, num_segments)

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
        'name': 'HRM',
        'params': param_count,
        'best_ppl': best_ppl,
        'best_epoch': best_epoch,
        'description': 'Hierarchical + deep supervision'
    })

    # ===== Summary =====
    print("\n" + "=" * 70)
    print("EXPERIMENT RESULTS SUMMARY")
    print("=" * 70)
    print(f"\n{'Model':<22} | {'Params':>10} | {'Best PPL':>10} | {'Epoch':>6} | Description")
    print("-" * 80)
    for r in results:
        print(f"{r['name']:<22} | {r['params']:>10,} | {r['best_ppl']:>10.2f} | {r['best_epoch']:>6} | {r['description']}")

    print("\n" + "=" * 70)
    print("Analysis")
    print("=" * 70)

    baseline_ppl = results[0]['best_ppl']  # Standard Transformer
    print(f"\nBaseline (Standard Transformer): {baseline_ppl:.2f} PPL")

    for r in results[1:]:
        diff = baseline_ppl - r['best_ppl']
        pct = (diff / baseline_ppl) * 100
        if diff > 0:
            print(f"  {r['name']}: {diff:.2f} PPL better ({pct:.1f}% improvement)")
        else:
            print(f"  {r['name']}: {-diff:.2f} PPL worse ({-pct:.1f}% degradation)")

    # Additional comparison: Deep Supervision effect
    if len(results) >= 2:
        deepsup_ppl = results[1]['best_ppl']
        hrm_ppl = results[2]['best_ppl']
        diff = deepsup_ppl - hrm_ppl
        print(f"\nHierarchical structure benefit (HRM vs DeepSup Transformer):")
        if diff > 0:
            print(f"  HRM is {diff:.2f} PPL better ({(diff/deepsup_ppl)*100:.1f}% improvement)")
        else:
            print(f"  DeepSup Transformer is {-diff:.2f} PPL better")


if __name__ == "__main__":
    main()
