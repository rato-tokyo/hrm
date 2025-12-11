"""
Experiment: Confidence-Routed Transformer v2

Improvements:
1. More data: 100,000 chars (5x increase)
2. Fewer layers: 3 layers (Shallow: 1 layer, Deep: 3 layers)
3. Strict early stopping: Stop immediately when val PPL worsens

Architecture:
- Shallow path: L1 → Output (1 layer)
- Deep path: L1 → L2 → L3 → Output (3 layers)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
import numpy as np
from typing import List, Tuple, Dict

from modules.transformer import TransformerBlock


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


class ConfidenceRoutedTransformerV2(nn.Module):
    """
    Confidence-Routed Transformer with 3 layers.

    - Shallow path: L1 → Output (25% compute)
    - Deep path: L1 → L2 → L3 → Output (100% compute)
    """

    def __init__(
        self,
        vocab_size: int,
        dim: int = 64,
        seq_len: int = 64,
        num_heads: int = 4,
        routing_threshold: float = 0.8,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.seq_len = seq_len
        self.routing_threshold = routing_threshold

        self.embedding = nn.Embedding(vocab_size, dim)
        self.layer1 = TransformerBlock(dim, num_heads)
        self.layer2 = TransformerBlock(dim, num_heads)
        self.layer3 = TransformerBlock(dim, num_heads)
        self.output_head = nn.Linear(dim, vocab_size, bias=False)

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=1.0 / math.sqrt(self.dim))
            elif isinstance(module, nn.Embedding):
                nn.init.trunc_normal_(module.weight, std=0.02)

    def compute_confidence(self, h: torch.Tensor) -> torch.Tensor:
        logits = self.output_head(h)
        probs = F.softmax(logits, dim=-1)
        confidence = probs.max(dim=-1).values
        return confidence

    def forward_train(self, x: torch.Tensor) -> Dict:
        """Training: compute both paths."""
        h = self.embedding(x)
        h1 = self.layer1(h)

        # Shallow output (after L1)
        shallow_logits = self.output_head(h1)

        # Deep path (L2 → L3)
        h2 = self.layer2(h1)
        h3 = self.layer3(h2)
        deep_logits = self.output_head(h3)

        # Confidence at L1 for monitoring
        with torch.no_grad():
            confidence = self.compute_confidence(h1)

        return {
            'shallow_logits': shallow_logits,
            'deep_logits': deep_logits,
            'confidence': confidence,
            'shallow_ratio': (confidence >= self.routing_threshold).float().mean().item(),
        }

    def forward_inference(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Inference: hard routing based on L1 confidence."""
        batch_size, seq_len = x.shape

        h = self.embedding(x)
        h1 = self.layer1(h)

        confidence = self.compute_confidence(h1)
        shallow_logits = self.output_head(h1)

        h2 = self.layer2(h1)
        h3 = self.layer3(h2)
        deep_logits = self.output_head(h3)

        # Hard routing
        mask = (confidence >= self.routing_threshold).unsqueeze(-1)
        output = torch.where(mask, shallow_logits, deep_logits)

        # Compute cost: shallow=1/3, deep=3/3
        shallow_count = mask.sum().item()
        total_count = batch_size * seq_len
        deep_count = total_count - shallow_count
        compute_cost = (shallow_count * 1 + deep_count * 3) / (total_count * 3)

        stats = {
            'mean_confidence': confidence.mean().item(),
            'shallow_ratio': shallow_count / total_count,
            'compute_cost': compute_cost,
        }

        return output, stats


class StandardTransformer(nn.Module):
    """Standard transformer for comparison."""

    def __init__(
        self,
        vocab_size: int,
        dim: int = 64,
        seq_len: int = 64,
        num_heads: int = 4,
        num_layers: int = 3,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim

        self.embedding = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([
            TransformerBlock(dim, num_heads) for _ in range(num_layers)
        ])
        self.output_head = nn.Linear(dim, vocab_size, bias=False)

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=1.0 / math.sqrt(self.dim))
            elif isinstance(module, nn.Embedding):
                nn.init.trunc_normal_(module.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embedding(x)
        for layer in self.layers:
            h = layer(h)
        return self.output_head(h)


def prepare_data(
    train_chars: int = 100000,
    val_chars: int = 10000,
    seq_len: int = 64,
    batch_size: int = 32
) -> Tuple:
    """Prepare WikiText-2 data with more samples."""
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


def train_epoch_routed(
    model: ConfidenceRoutedTransformerV2,
    batches: List,
    optimizer: torch.optim.Optimizer,
    vocab_size: int,
    device: str = 'cpu',
    shallow_weight: float = 0.5
) -> Tuple[float, float, Dict]:
    """Train with both paths. Returns train_loss, train_ppl, stats."""
    model.train()
    total_loss = 0.0
    total_stats = {'shallow_ratio': 0, 'shallow_loss': 0, 'deep_loss': 0}

    for x, y in batches:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        outputs = model.forward_train(x)

        shallow_loss = F.cross_entropy(outputs['shallow_logits'].view(-1, vocab_size), y.view(-1))
        deep_loss = F.cross_entropy(outputs['deep_logits'].view(-1, vocab_size), y.view(-1))
        loss = shallow_weight * shallow_loss + (1 - shallow_weight) * deep_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        total_stats['shallow_ratio'] += outputs['shallow_ratio']
        total_stats['shallow_loss'] += shallow_loss.item()
        total_stats['deep_loss'] += deep_loss.item()

    n = len(batches)
    avg_loss = total_loss / n
    train_ppl = np.exp(avg_loss)
    for k in total_stats:
        total_stats[k] /= n

    return avg_loss, train_ppl, total_stats


def train_epoch_standard(
    model: nn.Module,
    batches: List,
    optimizer: torch.optim.Optimizer,
    vocab_size: int,
    device: str = 'cpu'
) -> Tuple[float, float]:
    """Train standard model. Returns train_loss, train_ppl."""
    model.train()
    total_loss = 0.0

    for x, y in batches:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        output = model(x)
        loss = F.cross_entropy(output.view(-1, vocab_size), y.view(-1))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(batches)
    train_ppl = np.exp(avg_loss)
    return avg_loss, train_ppl


@torch.no_grad()
def evaluate_routed(
    model: ConfidenceRoutedTransformerV2,
    batches: List,
    vocab_size: int,
    device: str = 'cpu'
) -> Dict:
    """Evaluate with separate shallow/deep metrics."""
    model.eval()

    total_shallow_loss = 0.0
    total_deep_loss = 0.0
    total_routed_loss = 0.0
    shallow_correct = 0
    deep_correct = 0
    routed_correct = 0
    total_tokens = 0
    total_stats = {'shallow_ratio': 0, 'compute_cost': 0}

    for x, y in batches:
        x, y = x.to(device), y.to(device)
        outputs = model.forward_train(x)

        shallow_logits = outputs['shallow_logits']
        deep_logits = outputs['deep_logits']
        confidence = outputs['confidence']

        # Shallow metrics
        total_shallow_loss += F.cross_entropy(shallow_logits.view(-1, vocab_size), y.view(-1)).item()
        shallow_preds = shallow_logits.argmax(dim=-1)
        shallow_correct += (shallow_preds == y).sum().item()

        # Deep metrics
        total_deep_loss += F.cross_entropy(deep_logits.view(-1, vocab_size), y.view(-1)).item()
        deep_preds = deep_logits.argmax(dim=-1)
        deep_correct += (deep_preds == y).sum().item()

        # Routed metrics
        mask = (confidence >= model.routing_threshold).unsqueeze(-1)
        routed_logits = torch.where(mask, shallow_logits, deep_logits)
        total_routed_loss += F.cross_entropy(routed_logits.view(-1, vocab_size), y.view(-1)).item()
        routed_preds = routed_logits.argmax(dim=-1)
        routed_correct += (routed_preds == y).sum().item()

        total_tokens += y.numel()

        # Compute cost
        shallow_count = mask.sum().item()
        total_count = x.shape[0] * x.shape[1]
        deep_count = total_count - shallow_count
        total_stats['shallow_ratio'] += shallow_count / total_count
        total_stats['compute_cost'] += (shallow_count * 1 + deep_count * 3) / (total_count * 3)

    n = len(batches)
    return {
        'shallow_ppl': np.exp(total_shallow_loss / n),
        'deep_ppl': np.exp(total_deep_loss / n),
        'routed_ppl': np.exp(total_routed_loss / n),
        'shallow_acc': shallow_correct / total_tokens,
        'deep_acc': deep_correct / total_tokens,
        'routed_acc': routed_correct / total_tokens,
        'shallow_ratio': total_stats['shallow_ratio'] / n,
        'compute_cost': total_stats['compute_cost'] / n,
    }


@torch.no_grad()
def evaluate_standard(
    model: nn.Module,
    batches: List,
    vocab_size: int,
    device: str = 'cpu'
) -> Tuple[float, float]:
    """Evaluate standard model. Returns ppl, accuracy."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0

    for x, y in batches:
        x, y = x.to(device), y.to(device)
        output = model(x)
        loss = F.cross_entropy(output.view(-1, vocab_size), y.view(-1))

        preds = output.argmax(dim=-1)
        total_correct += (preds == y).sum().item()
        total_tokens += y.numel()
        total_loss += loss.item()

    ppl = np.exp(total_loss / len(batches))
    acc = total_correct / total_tokens
    return ppl, acc


def main():
    set_seed(42)

    # Config
    seq_len = 64
    dim = 64
    num_heads = 4
    batch_size = 32
    device = 'cpu'
    train_chars = 100000  # 5x more data
    val_chars = 10000
    max_epochs = 50
    lr = 1e-3

    print("=" * 70)
    print("Experiment: Confidence-Routed Transformer v2")
    print("=" * 70)
    print("\nImprovements:")
    print("  1. Data: 100,000 chars (5x increase)")
    print("  2. Layers: 3 (Shallow: 1 layer, Deep: 3 layers)")
    print("  3. Early stopping: Stop when val PPL worsens")

    # Prepare data
    train_batches, val_batches, vocab_size = prepare_data(
        train_chars, val_chars, seq_len, batch_size
    )
    print(f"\nConfig:")
    print(f"  Vocab size: {vocab_size}")
    print(f"  Train batches: {len(train_batches)}")
    print(f"  Val batches: {len(val_batches)}")
    print(f"  Dimension: {dim}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Routing threshold: 0.8")

    results = []

    # ===== Confidence-Routed Transformer =====
    print("\n" + "=" * 70)
    print("1. Confidence-Routed Transformer (Shallow: 1L, Deep: 3L)")
    print("=" * 70)

    set_seed(42)
    routed_model = ConfidenceRoutedTransformerV2(
        vocab_size=vocab_size, dim=dim, seq_len=seq_len,
        num_heads=num_heads, routing_threshold=0.8
    ).to(device)

    print(f"  Params: {count_params(routed_model):,}")

    optimizer = torch.optim.AdamW(routed_model.parameters(), lr=lr)

    best_val_ppl = float('inf')
    best_stats = {}

    print("\n  Epoch | Train PPL | Val PPL (Routed) | Val PPL (Shallow) | Val PPL (Deep) | Shallow%")
    print("  " + "-" * 85)

    for epoch in range(max_epochs):
        train_loss, train_ppl, train_stats = train_epoch_routed(
            routed_model, train_batches, optimizer, vocab_size, device
        )
        val_stats = evaluate_routed(routed_model, val_batches, vocab_size, device)

        print(f"  {epoch+1:5} | {train_ppl:9.2f} | {val_stats['routed_ppl']:16.2f} | "
              f"{val_stats['shallow_ppl']:17.2f} | {val_stats['deep_ppl']:14.2f} | "
              f"{val_stats['shallow_ratio']*100:7.1f}%")

        # Strict early stopping: stop if val PPL worsens
        if val_stats['routed_ppl'] < best_val_ppl:
            best_val_ppl = val_stats['routed_ppl']
            best_stats = val_stats.copy()
            best_stats['epoch'] = epoch + 1
        else:
            print(f"  >>> Early stopping at epoch {epoch + 1} (val PPL worsened)")
            break

    results.append({
        'model': 'Confidence-Routed',
        'params': count_params(routed_model),
        'ppl': best_val_ppl,
        'accuracy': best_stats['routed_acc'],
        'shallow_ratio': best_stats['shallow_ratio'],
        'compute_cost': best_stats['compute_cost'],
        'epoch': best_stats['epoch'],
    })

    # ===== Standard 3-layer Transformer =====
    print("\n" + "=" * 70)
    print("2. Standard 3-layer Transformer")
    print("=" * 70)

    set_seed(42)
    standard_model = StandardTransformer(
        vocab_size=vocab_size, dim=dim, seq_len=seq_len,
        num_heads=num_heads, num_layers=3
    ).to(device)

    print(f"  Params: {count_params(standard_model):,}")

    optimizer = torch.optim.AdamW(standard_model.parameters(), lr=lr)

    best_val_ppl_std = float('inf')
    best_acc_std = 0
    best_epoch_std = 0

    print("\n  Epoch | Train PPL | Val PPL")
    print("  " + "-" * 30)

    for epoch in range(max_epochs):
        train_loss, train_ppl = train_epoch_standard(
            standard_model, train_batches, optimizer, vocab_size, device
        )
        val_ppl, val_acc = evaluate_standard(standard_model, val_batches, vocab_size, device)

        print(f"  {epoch+1:5} | {train_ppl:9.2f} | {val_ppl:7.2f}")

        if val_ppl < best_val_ppl_std:
            best_val_ppl_std = val_ppl
            best_acc_std = val_acc
            best_epoch_std = epoch + 1
        else:
            print(f"  >>> Early stopping at epoch {epoch + 1} (val PPL worsened)")
            break

    results.append({
        'model': 'Standard 3-layer',
        'params': count_params(standard_model),
        'ppl': best_val_ppl_std,
        'accuracy': best_acc_std,
        'shallow_ratio': 0.0,
        'compute_cost': 1.0,
        'epoch': best_epoch_std,
    })

    # ===== Standard 1-layer Transformer =====
    print("\n" + "=" * 70)
    print("3. Standard 1-layer Transformer")
    print("=" * 70)

    set_seed(42)
    shallow_model = StandardTransformer(
        vocab_size=vocab_size, dim=dim, seq_len=seq_len,
        num_heads=num_heads, num_layers=1
    ).to(device)

    print(f"  Params: {count_params(shallow_model):,}")

    optimizer = torch.optim.AdamW(shallow_model.parameters(), lr=lr)

    best_val_ppl_shallow = float('inf')
    best_acc_shallow = 0
    best_epoch_shallow = 0

    print("\n  Epoch | Train PPL | Val PPL")
    print("  " + "-" * 30)

    for epoch in range(max_epochs):
        train_loss, train_ppl = train_epoch_standard(
            shallow_model, train_batches, optimizer, vocab_size, device
        )
        val_ppl, val_acc = evaluate_standard(shallow_model, val_batches, vocab_size, device)

        print(f"  {epoch+1:5} | {train_ppl:9.2f} | {val_ppl:7.2f}")

        if val_ppl < best_val_ppl_shallow:
            best_val_ppl_shallow = val_ppl
            best_acc_shallow = val_acc
            best_epoch_shallow = epoch + 1
        else:
            print(f"  >>> Early stopping at epoch {epoch + 1} (val PPL worsened)")
            break

    results.append({
        'model': 'Standard 1-layer',
        'params': count_params(shallow_model),
        'ppl': best_val_ppl_shallow,
        'accuracy': best_acc_shallow,
        'shallow_ratio': 1.0,
        'compute_cost': 1/3,
        'epoch': best_epoch_shallow,
    })

    # ===== Summary =====
    print("\n" + "=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)

    print(f"\n{'Model':<25} {'Params':>10} {'PPL':>8} {'Acc':>8} {'Compute':>10} {'Epoch':>7}")
    print("-" * 75)
    for r in results:
        print(f"{r['model']:<25} {r['params']:>10,} {r['ppl']:>8.2f} "
              f"{r['accuracy']*100:>7.1f}% {r['compute_cost']*100:>9.1f}% {r['epoch']:>7}")

    # Analysis
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    routed = results[0]
    standard = results[1]
    shallow = results[2]

    print(f"\n1. Confidence-Routed vs Standard 3-layer:")
    ppl_diff = (routed['ppl'] - standard['ppl']) / standard['ppl'] * 100
    compute_save = (1 - routed['compute_cost']) * 100
    print(f"   PPL: {routed['ppl']:.2f} vs {standard['ppl']:.2f} ({ppl_diff:+.1f}%)")
    print(f"   Compute: {routed['compute_cost']*100:.1f}% vs 100% (saving: {compute_save:.1f}%)")

    print(f"\n2. Confidence-Routed vs Standard 1-layer:")
    ppl_diff2 = (routed['ppl'] - shallow['ppl']) / shallow['ppl'] * 100
    print(f"   PPL: {routed['ppl']:.2f} vs {shallow['ppl']:.2f} ({ppl_diff2:+.1f}%)")

    print(f"\n3. Path Performance (from final Confidence-Routed model):")
    print(f"   Shallow path PPL: {best_stats['shallow_ppl']:.2f}")
    print(f"   Deep path PPL: {best_stats['deep_ppl']:.2f}")
    print(f"   Routed PPL: {best_stats['routed_ppl']:.2f}")

    # Test different thresholds
    print("\n" + "=" * 70)
    print("THRESHOLD ANALYSIS (using best model)")
    print("=" * 70)

    print(f"\n{'Threshold':>10} {'PPL':>8} {'Accuracy':>10} {'Shallow%':>10} {'Compute%':>10}")
    print("-" * 55)

    for threshold in [0.3, 0.5, 0.7, 0.8, 0.9, 0.95]:
        routed_model.routing_threshold = threshold
        stats = evaluate_routed(routed_model, val_batches, vocab_size, device)
        print(f"{threshold:>10.2f} {stats['routed_ppl']:>8.2f} {stats['routed_acc']*100:>9.1f}% "
              f"{stats['shallow_ratio']*100:>9.1f}% {stats['compute_cost']*100:>9.1f}%")


if __name__ == "__main__":
    main()
