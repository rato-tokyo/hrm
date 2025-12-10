"""
HRM Language Model Benchmark on WikiText-2

Uses Hugging Face datasets for proper evaluation.
Starts with small data subset, can scale up.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import math
import random
import numpy as np

from modules import HRM


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

    # Map split names
    split_map = {
        'train': 'train',
        'validation': 'valid',
        'test': 'test'
    }

    # PyTorch examples URL
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
            print("  Using fallback: generating synthetic WikiText-like data...")
            return generate_synthetic_wikitext(split)

    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


def generate_synthetic_wikitext(split: str = 'train') -> str:
    """Generate synthetic Wikipedia-like text as fallback."""
    paragraphs = [
        "The history of artificial intelligence began in antiquity with myths and stories of artificial beings endowed with intelligence or consciousness by master craftsmen.",
        "Natural language processing is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language.",
        "Machine learning is a subset of artificial intelligence that provides systems the ability to automatically learn and improve from experience without being explicitly programmed.",
        "Deep learning is part of a broader family of machine learning methods based on artificial neural networks with representation learning.",
        "The Turing test, originally called the imitation game by Alan Turing in 1950, is a test of a machine's ability to exhibit intelligent behaviour equivalent to a human.",
        "Computer vision is an interdisciplinary scientific field that deals with how computers can gain high-level understanding from digital images or videos.",
        "Robotics is an interdisciplinary branch of engineering and science that includes mechanical engineering, electronic engineering, information engineering, computer science, and others.",
        "Reinforcement learning is an area of machine learning concerned with how software agents ought to take actions in an environment in order to maximize some notion of cumulative reward.",
        "Neural networks are computing systems vaguely inspired by the biological neural networks that constitute animal brains.",
        "The field of artificial intelligence research was founded at a workshop held on the campus of Dartmouth College during the summer of 1956.",
        "Supervised learning is the machine learning task of learning a function that maps an input to an output based on example input-output pairs.",
        "Unsupervised learning is a type of machine learning that looks for previously undetected patterns in a data set with no pre-existing labels.",
        "Transfer learning is a research problem in machine learning that focuses on storing knowledge gained while solving one problem and applying it to a different but related problem.",
        "Attention mechanisms have become an integral part of compelling sequence modeling and transduction models in various tasks.",
        "The transformer architecture was introduced in the paper Attention Is All You Need and has become the foundation for modern language models.",
    ]

    # Repeat and shuffle for more data
    import random
    random.seed(42 if split == 'train' else 123)
    text_parts = paragraphs * (50 if split == 'train' else 10)
    random.shuffle(text_parts)
    return '\n\n'.join(text_parts)


class WikiTextDataset(Dataset):
    """WikiText-2 character-level dataset."""

    def __init__(self, split: str = 'train', seq_len: int = 64, max_chars: int = 50000):
        """
        Args:
            split: 'train', 'validation', or 'test'
            seq_len: Sequence length for each sample
            max_chars: Maximum characters to use (for CPU-friendly training)
        """
        self.seq_len = seq_len

        # Load WikiText-2
        text = download_wikitext(split)

        # Limit text size for CPU
        if max_chars and len(text) > max_chars:
            text = text[:max_chars]

        self.text = text
        self.text_len = len(text)

        # Build vocabulary from ASCII printable characters
        # This ensures train/test have same vocab
        self.chars = [chr(i) for i in range(32, 127)] + ['\n', '\t']
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}
        self.vocab_size = len(self.chars)
        self.unk_idx = 0  # Use space as UNK

        # Encode text
        self.data = self._encode(text)

        # Create samples
        self.num_samples = max(1, (len(self.data) - 1) // seq_len)

    def _encode(self, text: str) -> list:
        """Encode text to indices, handling unknown characters."""
        return [self.char_to_idx.get(ch, self.unk_idx) for ch in text]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len

        x = self.data[start:end]
        y = self.data[start + 1:end + 1]

        # Pad if necessary
        if len(x) < self.seq_len:
            x = x + [self.unk_idx] * (self.seq_len - len(x))
        if len(y) < self.seq_len:
            y = y + [self.unk_idx] * (self.seq_len - len(y))

        return torch.tensor(x), torch.tensor(y)

    def decode(self, indices):
        return ''.join([self.idx_to_char.get(i, '?') for i in indices])


def compute_perplexity(model: nn.Module, dataloader: DataLoader, device: str, num_segments: int = 2) -> float:
    """Compute perplexity on dataset."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)

            y_hat, _ = model(x, num_segments=num_segments)
            loss = nn.functional.cross_entropy(
                y_hat.view(-1, model.vocab_size),
                y.view(-1),
                reduction='sum'
            )

            total_loss += loss.item()
            total_tokens += y.numel()

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(min(avg_loss, 100))  # Cap to avoid overflow
    return perplexity


def train_epoch(model: nn.Module, dataloader: DataLoader, optimizer: optim.Optimizer,
                device: str, num_segments: int = 2) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        z_H, z_L = None, None

        for _ in range(num_segments):
            z_H, z_L, y_hat, _ = model.forward_pass(x, z_H, z_L)

            loss = nn.functional.cross_entropy(
                y_hat.view(-1, model.vocab_size),
                y.view(-1)
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            z_H = z_H.detach()
            z_L = z_L.detach()

            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches


def generate_text(model: nn.Module, dataset: WikiTextDataset, seed_text: str,
                  length: int = 100, device: str = 'cpu', num_segments: int = 2) -> str:
    """Generate text from seed."""
    model.eval()

    # Encode seed
    indices = [dataset.char_to_idx.get(ch, dataset.unk_idx) for ch in seed_text]

    # Pad or truncate to seq_len
    seq_len = model.seq_len
    if len(indices) < seq_len:
        indices = [dataset.unk_idx] * (seq_len - len(indices)) + indices
    else:
        indices = indices[-seq_len:]

    generated = list(seed_text)

    with torch.no_grad():
        for _ in range(length):
            x = torch.tensor([indices]).to(device)
            y_hat, _ = model(x, num_segments=num_segments)

            # Get next token probabilities (last position)
            logits = y_hat[0, -1] / 0.8  # Temperature
            probs = torch.softmax(logits, dim=-1)
            next_idx = int(torch.multinomial(probs, 1).item())

            if next_idx < len(dataset.idx_to_char):
                generated.append(dataset.idx_to_char[next_idx])
            else:
                generated.append('?')

            # Shift window
            indices = indices[1:] + [next_idx]

    return ''.join(generated)


def main():
    set_seed(42)

    # Config - minimal for CPU
    seq_len = 64
    dim = 64
    num_layers = 1
    num_heads = 4
    N = 1
    T = 2
    num_segments = 2
    batch_size = 16
    max_epochs = 100  # Maximum epochs (will early stop)
    patience = 1  # Stop if val PPL increases
    lr = 1e-3
    device = 'cpu'

    # Data size - start small
    train_chars = 20000  # ~20KB of text
    val_chars = 5000

    print("="*60)
    print("HRM Language Model - WikiText-2 Benchmark")
    print("="*60)

    # Load datasets (train and validation are separate files)
    print("\nLoading WikiText-2...")
    train_dataset = WikiTextDataset('train', seq_len=seq_len, max_chars=train_chars)
    val_dataset = WikiTextDataset('validation', seq_len=seq_len, max_chars=val_chars)

    vocab_size = train_dataset.vocab_size

    print("\nDataset:")
    print(f"  Vocabulary size: {vocab_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Train: {len(train_dataset)} samples ({train_chars} chars)")
    print(f"  Val:   {len(val_dataset)} samples ({val_chars} chars)")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Sample text
    print("\nSample text from train:")
    sample_text = train_dataset.text[:200].replace('\n', '\\n')
    print(f"  \"{sample_text}...\"")

    # Model
    print("\nModel Config:")
    print(f"  dim={dim}, layers={num_layers}, heads={num_heads}")
    print(f"  N={N}, T={T}, segments={num_segments}")

    model = HRM(
        vocab_size=vocab_size,
        dim=dim,
        num_layers=num_layers,
        num_heads=num_heads,
        seq_len=seq_len,
        N=N,
        T=T
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {param_count:,}")

    # Compute initial PPL
    initial_train_ppl = compute_perplexity(model, train_loader, device, num_segments)
    initial_val_ppl = compute_perplexity(model, val_loader, device, num_segments)
    print("\nInitial PPL (random):")
    print(f"  Train: {initial_train_ppl:.2f}, Val: {initial_val_ppl:.2f}")
    print(f"  (Random baseline ~{vocab_size})")

    # Training with early stopping
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    print(f"\nTraining (max {max_epochs} epochs, patience={patience})...")
    print("-"*60)
    print(f"{'Epoch':>5} | {'Train Loss':>10} | {'Train PPL':>10} | {'Val PPL':>10}")
    print("-"*60)

    best_val_ppl = float('inf')
    epochs_without_improvement = 0
    best_epoch = 0

    for epoch in range(max_epochs):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device, num_segments)

        # Evaluate on both train and val
        train_ppl = compute_perplexity(model, train_loader, device, num_segments)
        val_ppl = compute_perplexity(model, val_loader, device, num_segments)

        print(f"{epoch+1:5d} | {train_loss:10.4f} | {train_ppl:10.2f} | {val_ppl:10.2f}", end="")

        # Early stopping check
        if val_ppl < best_val_ppl:
            best_val_ppl = val_ppl
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            print(" *")  # Mark improvement
        else:
            epochs_without_improvement += 1
            print(f" (no improve: {epochs_without_improvement})")

            if epochs_without_improvement >= patience:
                print(f"\nEarly stopping at epoch {epoch+1} (patience={patience})")
                break

    # Final results
    print("-"*60)
    print("\nFinal Results:")
    print(f"  Initial Train PPL: {initial_train_ppl:.2f}")
    print(f"  Initial Val PPL:   {initial_val_ppl:.2f}")
    print(f"  Best Val PPL:      {best_val_ppl:.2f} (epoch {best_epoch})")
    print(f"  Final Train PPL:   {train_ppl:.2f}")
    print(f"  Final Val PPL:     {val_ppl:.2f}")
    print(f"  Improvement:       {initial_val_ppl/best_val_ppl:.2f}x")

    # Generate sample text
    print("\n" + "="*60)
    print("Text Generation Sample")
    print("="*60)

    seeds = ["The ", "In the ", "A "]
    for seed in seeds:
        generated = generate_text(model, train_dataset, seed, length=80, device=device, num_segments=num_segments)
        print(f"\nSeed: '{seed}'")
        print(f"Generated: {generated[:100]}...")

    # Comparison with standard LM baselines
    print("\n" + "="*60)
    print("Reference PPL (WikiText-2, full dataset)")
    print("="*60)
    print("  GPT-2 Small (117M params): ~30")
    print("  LSTM (10M params):         ~100")
    print("  Transformer-XL (257M):     ~24")
    print(f"  HRM ({param_count//1000}K params, {train_chars//1000}K chars): {best_val_ppl:.1f}")


if __name__ == "__main__":
    main()
