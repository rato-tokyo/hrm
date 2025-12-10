"""
Minimal LLM Benchmark for HRM

Measures perplexity (PPL) on simple text data.
Designed for CPU execution with minimal resources.
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


# Larger text corpus for testing
CORPUS = """
The cat sat on the mat and looked at the door.
The dog ran in the park and played with a ball.
A bird flew over the tree and sang a song.
The sun shines in the sky during the day.
Rain falls from the clouds and waters the plants.
Fish swim in the sea and eat small food.
The moon glows at night when the sun is gone.
Stars twinkle in the dark sky above us.
Wind blows through the leaves of the tall trees.
Snow covers the ground in the cold winter months.
The boy plays with a ball in the green yard.
The girl reads a book under the old tree.
A man walks down the street to his home.
A woman drives a car to work each day.
Children laugh and play in the sunny park.
Birds sing in the morning when the sun rises.
Flowers bloom in spring after the snow melts.
Leaves fall in autumn from the tall trees.
Ice forms in winter on the cold lake.
Grass grows in summer under the warm sun.
The teacher writes on the board in class.
Students read books and learn new things each day.
The farmer works in the field all day long.
Cows eat grass in the green meadow near the barn.
The baker makes bread in the hot oven.
People buy food at the store down the street.
The doctor helps sick people get well again.
Nurses work hard to care for all patients.
The pilot flies the plane high in the sky.
Ships sail across the wide blue ocean.
Trains run on tracks from town to town.
Cars drive on roads and highways each day.
The artist paints pictures with bright colors.
Music fills the room when the band plays.
Dancers move to the beat of the drums.
Actors perform on stage for the crowd.
The chef cooks food in the busy kitchen.
Waiters serve meals to hungry customers.
The gardener plants seeds in the rich soil.
Bees fly from flower to flower in the garden.
The carpenter builds houses made of wood.
Painters cover walls with fresh new paint.
The plumber fixes pipes under the sink.
Electricians install lights in every room.
Teachers help students learn to read and write.
Books contain stories about many different things.
Libraries have thousands of books to read.
Schools teach children many important skills.
""".strip()


class CharDataset(Dataset):
    """Character-level language modeling dataset."""

    def __init__(self, text: str, seq_len: int = 32):
        self.seq_len = seq_len

        # Build vocabulary
        self.chars = sorted(list(set(text)))
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}
        self.vocab_size = len(self.chars)

        # Encode text
        self.data = [self.char_to_idx[ch] for ch in text]

        # Create sequences
        self.samples = []
        for i in range(0, len(self.data) - seq_len, seq_len // 2):  # Overlapping
            x = self.data[i:i + seq_len]
            y = self.data[i + 1:i + seq_len + 1]
            if len(x) == seq_len and len(y) == seq_len:
                self.samples.append((torch.tensor(x), torch.tensor(y)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def decode(self, indices):
        return ''.join([self.idx_to_char[i] for i in indices])


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
    perplexity = math.exp(avg_loss)
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
            optimizer.step()

            z_H = z_H.detach()
            z_L = z_L.detach()

            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches


def generate_text(model: nn.Module, dataset: CharDataset, seed_text: str,
                  length: int = 50, device: str = 'cpu', num_segments: int = 2) -> str:
    """Generate text from seed."""
    model.eval()

    # Encode seed
    indices = [dataset.char_to_idx.get(ch, 0) for ch in seed_text]

    # Pad or truncate to seq_len
    seq_len = model.seq_len
    if len(indices) < seq_len:
        indices = [0] * (seq_len - len(indices)) + indices
    else:
        indices = indices[-seq_len:]

    generated = list(seed_text)

    with torch.no_grad():
        for _ in range(length):
            x = torch.tensor([indices]).to(device)
            y_hat, _ = model(x, num_segments=num_segments)

            # Get next token probabilities (last position)
            probs = torch.softmax(y_hat[0, -1], dim=-1)
            next_idx = int(torch.multinomial(probs, 1).item())

            # Handle out-of-vocabulary indices
            if next_idx in dataset.idx_to_char:
                generated.append(dataset.idx_to_char[next_idx])
            else:
                generated.append('?')

            # Shift window
            indices = indices[1:] + [next_idx]

    return ''.join(generated)


def main():
    set_seed(42)

    # Config - minimal for CPU
    seq_len = 32
    dim = 32  # Small dimension
    num_layers = 1  # Single layer
    num_heads = 2
    N = 1  # High-level cycles
    T = 2  # Low-level timesteps
    num_segments = 2
    batch_size = 8
    num_epochs = 50
    lr = 1e-3
    device = 'cpu'

    print("="*60)
    print("HRM Language Model - Minimal Benchmark")
    print("="*60)

    # Dataset
    dataset = CharDataset(CORPUS, seq_len=seq_len)
    vocab_size = dataset.vocab_size

    print("\nDataset:")
    print(f"  Vocabulary size: {vocab_size}")
    print(f"  Characters: {''.join(dataset.chars)}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Number of samples: {len(dataset)}")

    # Create separate train and test datasets (non-overlapping text)
    # Split corpus in half for true generalization test
    mid = len(CORPUS) // 2
    train_corpus = CORPUS[:mid]
    test_corpus = CORPUS[mid:]

    train_dataset = CharDataset(train_corpus, seq_len=seq_len)
    test_dataset = CharDataset(test_corpus, seq_len=seq_len)

    # Use same vocabulary for both
    test_dataset.char_to_idx = train_dataset.char_to_idx
    test_dataset.idx_to_char = train_dataset.idx_to_char

    print(f"  Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")

    train_data = train_dataset
    test_data = test_dataset
    dataset = train_dataset  # For generation

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size)

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

    # Compute initial PPL (random model)
    initial_ppl = compute_perplexity(model, test_loader, device, num_segments)
    print(f"\nInitial PPL (random): {initial_ppl:.2f}")
    print(f"  (Random baseline should be ~{vocab_size:.0f})")

    # Training
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    print(f"\nTraining for {num_epochs} epochs...")
    print("-"*50)

    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device, num_segments)

        if (epoch + 1) % 10 == 0:
            ppl = compute_perplexity(model, test_loader, device, num_segments)
            print(f"Epoch {epoch+1:3d} | Loss: {train_loss:.4f} | Test PPL: {ppl:.2f}")

    # Final evaluation
    final_ppl = compute_perplexity(model, test_loader, device, num_segments)
    print("-"*50)
    print("\nFinal Results:")
    print(f"  Initial PPL: {initial_ppl:.2f}")
    print(f"  Final PPL:   {final_ppl:.2f}")
    print(f"  Improvement: {initial_ppl/final_ppl:.2f}x")

    # Generate sample text
    print("\n" + "="*60)
    print("Text Generation Sample")
    print("="*60)

    seed = "The "
    generated = generate_text(model, dataset, seed, length=100, device=device, num_segments=num_segments)
    print(f"\nSeed: '{seed}'")
    print(f"Generated:\n{generated}")


if __name__ == "__main__":
    main()
