"""
Experiment: HRM vs Infini-HRM

Compares:
1. Standard HRM (no memory)
2. Infini-HRM (Infini-Attention input layer + standard HRM)

The key idea: Infini-Attention handles long-range context at the input,
while HRM layers handle deep reasoning.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import math
import random
import numpy as np
from typing import Optional

from modules import HRM, InfiniHRM


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


# ===== Standard HRM Training =====

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


# ===== Infini-HRM Training =====

def compute_perplexity_infini(model: InfiniHRM, dataloader: DataLoader,
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


def train_epoch_infini(model: InfiniHRM, dataloader: DataLoader,
                       optimizer: optim.Optimizer, device: str,
                       num_segments: int = 2, carry_memory: bool = True) -> float:
    """Train Infini-HRM for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    memory: Optional[torch.Tensor] = None
    norm: Optional[torch.Tensor] = None
    prev_batch_size: Optional[int] = None

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        current_batch_size = x.shape[0]
        states = None

        # Reset memory if batch size changed
        if carry_memory and prev_batch_size is not None and current_batch_size != prev_batch_size:
            memory = None
            norm = None

        for _ in range(num_segments):
            states, memory, norm, y_hat, _ = model.forward_pass(
                x, states, memory, norm
            )
            loss = nn.functional.cross_entropy(
                y_hat.view(-1, model.vocab_size), y.view(-1)
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            states = [s.detach() for s in states]
            if carry_memory and memory is not None:
                memory = memory.detach()
                norm = norm.detach()

            total_loss += loss.item()
            num_batches += 1

        if not carry_memory:
            memory = None
            norm = None

        prev_batch_size = current_batch_size

    return total_loss / num_batches


def main():
    set_seed(42)

    # Config
    seq_len = 64
    dim = 64
    num_layers = 1
    num_heads = 4
    N = 1
    T = 2
    num_segments = 2
    batch_size = 16
    max_epochs = 30
    patience = 3
    lr = 1e-3
    device = 'cpu'
    train_chars = 20000
    val_chars = 5000

    print("=" * 60)
    print("HRM vs Infini-HRM Comparison Experiment")
    print("=" * 60)

    # Load data
    print("\nLoading WikiText-2...")
    train_dataset = WikiTextDataset('train', seq_len=seq_len, max_chars=train_chars)
    val_dataset = WikiTextDataset('validation', seq_len=seq_len, max_chars=val_chars)
    vocab_size = train_dataset.vocab_size

    print(f"Vocab: {vocab_size}, Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    results = []

    # ===== Model 1: Standard HRM =====
    print("\n" + "=" * 60)
    print("Model 1: Standard HRM (baseline)")
    print("=" * 60)

    set_seed(42)
    model_hrm = HRM(
        vocab_size=vocab_size,
        dim=dim,
        seq_len=seq_len,
        num_layers=num_layers,
        num_heads=num_heads,
        N=N,
        T=T
    ).to(device)

    param_count = sum(p.numel() for p in model_hrm.parameters())
    print(f"Parameters: {param_count:,}")

    optimizer = optim.AdamW(model_hrm.parameters(), lr=lr, weight_decay=0.01)
    initial_ppl = compute_perplexity_hrm(model_hrm, val_loader, device, num_segments)
    print(f"Initial Val PPL: {initial_ppl:.2f}")

    print(f"\n{'Epoch':>5} | {'Train Loss':>10} | {'Val PPL':>10}")
    print("-" * 40)

    best_ppl = float('inf')
    best_epoch = 0
    no_improve = 0

    for epoch in range(max_epochs):
        train_loss = train_epoch_hrm(model_hrm, train_loader, optimizer, device, num_segments)
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
        'name': 'HRM (standard)',
        'params': param_count,
        'best_ppl': best_ppl,
        'best_epoch': best_epoch
    })

    # ===== Model 2: Infini-HRM (no carry) =====
    print("\n" + "=" * 60)
    print("Model 2: Infini-HRM (memory reset per batch)")
    print("=" * 60)

    set_seed(42)
    model_infini = InfiniHRM(
        vocab_size=vocab_size,
        dim=dim,
        seq_len=seq_len,
        num_layers=num_layers,
        num_heads=num_heads,
        N=N,
        T=T
    ).to(device)

    param_count = sum(p.numel() for p in model_infini.parameters())
    print(f"Parameters: {param_count:,}")

    optimizer = optim.AdamW(model_infini.parameters(), lr=lr, weight_decay=0.01)
    initial_ppl = compute_perplexity_infini(model_infini, val_loader, device, num_segments)
    print(f"Initial Val PPL: {initial_ppl:.2f}")

    print(f"\n{'Epoch':>5} | {'Train Loss':>10} | {'Val PPL':>10}")
    print("-" * 40)

    best_ppl = float('inf')
    best_epoch = 0
    no_improve = 0

    for epoch in range(max_epochs):
        train_loss = train_epoch_infini(
            model_infini, train_loader, optimizer, device, num_segments,
            carry_memory=False
        )
        val_ppl = compute_perplexity_infini(model_infini, val_loader, device, num_segments)

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
        'name': 'Infini-HRM (reset)',
        'params': param_count,
        'best_ppl': best_ppl,
        'best_epoch': best_epoch
    })

    # ===== Model 3: Infini-HRM (carry memory) =====
    print("\n" + "=" * 60)
    print("Model 3: Infini-HRM (carry memory across batches)")
    print("=" * 60)

    set_seed(42)
    model_infini_carry = InfiniHRM(
        vocab_size=vocab_size,
        dim=dim,
        seq_len=seq_len,
        num_layers=num_layers,
        num_heads=num_heads,
        N=N,
        T=T
    ).to(device)

    param_count = sum(p.numel() for p in model_infini_carry.parameters())
    print(f"Parameters: {param_count:,}")

    optimizer = optim.AdamW(model_infini_carry.parameters(), lr=lr, weight_decay=0.01)
    initial_ppl = compute_perplexity_infini(model_infini_carry, val_loader, device, num_segments)
    print(f"Initial Val PPL: {initial_ppl:.2f}")

    print(f"\n{'Epoch':>5} | {'Train Loss':>10} | {'Val PPL':>10}")
    print("-" * 40)

    best_ppl = float('inf')
    best_epoch = 0
    no_improve = 0

    for epoch in range(max_epochs):
        train_loss = train_epoch_infini(
            model_infini_carry, train_loader, optimizer, device, num_segments,
            carry_memory=True
        )
        val_ppl = compute_perplexity_infini(model_infini_carry, val_loader, device, num_segments)

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
        'name': 'Infini-HRM (carry)',
        'params': param_count,
        'best_ppl': best_ppl,
        'best_epoch': best_epoch
    })

    # ===== Summary =====
    print("\n" + "=" * 60)
    print("EXPERIMENT RESULTS SUMMARY")
    print("=" * 60)
    print(f"\n{'Model':<25} | {'Params':>10} | {'Best PPL':>10} | {'Epoch':>6}")
    print("-" * 60)
    for r in results:
        print(f"{r['name']:<25} | {r['params']:>10,} | {r['best_ppl']:>10.2f} | {r['best_epoch']:>6}")

    print("\n" + "=" * 60)
    print("Analysis")
    print("=" * 60)

    baseline_ppl = results[0]['best_ppl']
    for r in results[1:]:
        diff = baseline_ppl - r['best_ppl']
        pct = (diff / baseline_ppl) * 100
        if diff > 0:
            print(f"{r['name']}: {diff:.2f} PPL better ({pct:.1f}% improvement)")
        else:
            print(f"{r['name']}: {-diff:.2f} PPL worse ({-pct:.1f}% degradation)")


if __name__ == "__main__":
    main()
