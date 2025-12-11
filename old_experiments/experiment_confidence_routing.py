"""
Experiment: Confidence-Routed Transformer

Routes tokens to different depth paths based on L2 confidence.
- High confidence tokens: 2-layer path (shallow)
- Low confidence tokens: 4-layer path (deep)

Training: Soft routing (weighted average of both paths)
Inference: Hard routing (threshold-based selection)
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


class ConfidenceRoutedTransformer(nn.Module):
    """
    Transformer with confidence-based routing.

    Architecture:
    - Layer 1-2: Shared layers
    - After L2: Router computes confidence-based routing weight
    - Shallow path: Output from L2
    - Deep path: L3 -> L4 -> Output

    Training: output = w * shallow + (1-w) * deep (soft routing)
    Inference: output = shallow if conf >= threshold else deep (hard routing)
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

        # Input embedding
        self.embedding = nn.Embedding(vocab_size, dim)

        # Shared layers (L1, L2)
        self.layer1 = TransformerBlock(dim, num_heads)
        self.layer2 = TransformerBlock(dim, num_heads)

        # Deep path layers (L3, L4)
        self.layer3 = TransformerBlock(dim, num_heads)
        self.layer4 = TransformerBlock(dim, num_heads)

        # Shared output head
        self.output_head = nn.Linear(dim, vocab_size, bias=False)

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=1.0 / math.sqrt(self.dim))
            elif isinstance(module, nn.Embedding):
                nn.init.trunc_normal_(module.weight, std=0.02)

    def compute_confidence(self, h: torch.Tensor) -> torch.Tensor:
        """
        Compute confidence (max probability) from hidden state.

        Args:
            h: Hidden state [batch, seq, dim]

        Returns:
            confidence: [batch, seq] max probability for each position
        """
        logits = self.output_head(h)  # [batch, seq, vocab]
        probs = F.softmax(logits, dim=-1)
        confidence = probs.max(dim=-1).values  # [batch, seq]
        return confidence

    def forward_train(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Training forward pass with auxiliary losses.

        Both paths are trained with their own loss.
        """
        # Shared layers
        h = self.embedding(x)
        h = self.layer1(h)
        h = self.layer2(h)

        # Compute confidence for monitoring
        with torch.no_grad():
            confidence = self.compute_confidence(h)  # [batch, seq]

        # Shallow path output (from L2)
        shallow_logits = self.output_head(h)  # [batch, seq, vocab]

        # Deep path (L3 -> L4)
        h_deep = self.layer3(h)
        h_deep = self.layer4(h_deep)
        deep_logits = self.output_head(h_deep)  # [batch, seq, vocab]

        # Return both outputs for separate loss computation
        # Stats for logging
        stats = {
            'mean_confidence': confidence.mean().item(),
            'shallow_ratio': (confidence >= self.routing_threshold).float().mean().item(),
            'shallow_logits': shallow_logits,
            'deep_logits': deep_logits,
        }

        # For backward compatibility, return deep as main output
        return deep_logits, stats

    def forward_inference(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Inference forward pass with hard routing.

        Uses threshold to decide shallow vs deep path.
        """
        batch_size, seq_len = x.shape

        # Shared layers
        h = self.embedding(x)
        h = self.layer1(h)
        h = self.layer2(h)

        # Compute confidence for routing decision
        confidence = self.compute_confidence(h)  # [batch, seq]

        # Shallow path output
        shallow_logits = self.output_head(h)  # [batch, seq, vocab]

        # Deep path
        h_deep = self.layer3(h)
        h_deep = self.layer4(h_deep)
        deep_logits = self.output_head(h_deep)  # [batch, seq, vocab]

        # Hard routing based on threshold
        mask = (confidence >= self.routing_threshold).unsqueeze(-1)  # [batch, seq, 1]
        output = torch.where(mask, shallow_logits, deep_logits)

        # Compute actual compute cost
        shallow_count = mask.sum().item()
        total_count = batch_size * seq_len
        # Shallow: 2 layers, Deep: 4 layers
        # Compute cost = (shallow_count * 2 + deep_count * 4) / (total * 4)
        deep_count = total_count - shallow_count
        compute_cost = (shallow_count * 2 + deep_count * 4) / (total_count * 4)

        stats = {
            'mean_confidence': confidence.mean().item(),
            'shallow_ratio': shallow_count / total_count,
            'compute_cost': compute_cost,
        }

        return output, stats

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward (uses training mode)."""
        if self.training:
            output, _ = self.forward_train(x)
        else:
            output, _ = self.forward_inference(x)
        return output


class StandardTransformer(nn.Module):
    """Standard 4-layer transformer for comparison."""

    def __init__(
        self,
        vocab_size: int,
        dim: int = 64,
        seq_len: int = 64,
        num_heads: int = 4,
        num_layers: int = 4,
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


def train_epoch_routed(
    model: ConfidenceRoutedTransformer,
    batches: List[Tuple[torch.Tensor, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    vocab_size: int,
    device: str = 'cpu',
    shallow_weight: float = 0.5
) -> Tuple[float, Dict]:
    """
    Train with both shallow and deep losses.

    This ensures both paths learn to predict well.
    """
    model.train()
    total_loss = 0.0
    total_shallow_loss = 0.0
    total_deep_loss = 0.0
    total_stats = {'mean_confidence': 0, 'shallow_ratio': 0}

    for x, y in batches:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        _, stats = model.forward_train(x)

        # Compute losses for both paths
        shallow_loss = F.cross_entropy(
            stats['shallow_logits'].view(-1, vocab_size), y.view(-1)
        )
        deep_loss = F.cross_entropy(
            stats['deep_logits'].view(-1, vocab_size), y.view(-1)
        )

        # Combined loss: train both paths
        loss = shallow_weight * shallow_loss + (1 - shallow_weight) * deep_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        total_shallow_loss += shallow_loss.item()
        total_deep_loss += deep_loss.item()
        total_stats['mean_confidence'] += stats['mean_confidence']
        total_stats['shallow_ratio'] += stats['shallow_ratio']

    n = len(batches)
    avg_loss = total_loss / n
    total_stats['mean_confidence'] /= n
    total_stats['shallow_ratio'] /= n
    total_stats['shallow_loss'] = total_shallow_loss / n
    total_stats['deep_loss'] = total_deep_loss / n

    return avg_loss, total_stats


def train_epoch_standard(
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

        output = model(x)
        loss = F.cross_entropy(output.view(-1, vocab_size), y.view(-1))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(batches)


@torch.no_grad()
def evaluate_routed(
    model: ConfidenceRoutedTransformer,
    batches: List[Tuple[torch.Tensor, torch.Tensor]],
    vocab_size: int,
    device: str = 'cpu'
) -> Tuple[float, Dict]:
    model.eval()
    total_loss = 0.0
    total_stats = {'mean_confidence': 0, 'shallow_ratio': 0, 'compute_cost': 0}
    total_correct = 0
    total_tokens = 0

    for x, y in batches:
        x, y = x.to(device), y.to(device)

        output, stats = model.forward_inference(x)
        loss = F.cross_entropy(output.view(-1, vocab_size), y.view(-1))

        # Accuracy
        preds = output.argmax(dim=-1)
        total_correct += (preds == y).sum().item()
        total_tokens += y.numel()

        total_loss += loss.item()
        for k in total_stats:
            total_stats[k] += stats[k]

    avg_loss = total_loss / len(batches)
    for k in total_stats:
        total_stats[k] /= len(batches)

    total_stats['accuracy'] = total_correct / total_tokens
    total_stats['ppl'] = np.exp(avg_loss)

    return avg_loss, total_stats


@torch.no_grad()
def evaluate_standard(
    model: nn.Module,
    batches: List[Tuple[torch.Tensor, torch.Tensor]],
    vocab_size: int,
    device: str = 'cpu'
) -> Tuple[float, float]:
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

    avg_loss = total_loss / len(batches)
    accuracy = total_correct / total_tokens
    ppl = np.exp(avg_loss)

    return ppl, accuracy


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
    max_epochs = 30
    patience = 5
    lr = 1e-3

    print("=" * 70)
    print("Experiment: Confidence-Routed Transformer")
    print("=" * 70)
    print(f"\nConfig:")
    print(f"  Dimension: {dim}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Routing threshold: 0.8")

    # Prepare data
    train_batches, val_batches, vocab_size = prepare_data(
        train_chars, val_chars, seq_len, batch_size
    )
    print(f"  Vocab size: {vocab_size}")
    print(f"  Train batches: {len(train_batches)}")
    print(f"  Val batches: {len(val_batches)}")

    results = []

    # ===== Confidence-Routed Transformer =====
    print("\n" + "=" * 70)
    print("Training Confidence-Routed Transformer")
    print("=" * 70)

    set_seed(42)
    routed_model = ConfidenceRoutedTransformer(
        vocab_size=vocab_size,
        dim=dim,
        seq_len=seq_len,
        num_heads=num_heads,
        routing_threshold=0.8
    ).to(device)

    print(f"  Total params: {count_params(routed_model):,}")

    optimizer = torch.optim.AdamW(routed_model.parameters(), lr=lr)

    best_ppl = float('inf')
    patience_counter = 0

    for epoch in range(max_epochs):
        train_loss, train_stats = train_epoch_routed(
            routed_model, train_batches, optimizer, vocab_size, device
        )
        val_loss, val_stats = evaluate_routed(
            routed_model, val_batches, vocab_size, device
        )

        if val_stats['ppl'] < best_ppl:
            best_ppl = val_stats['ppl']
            best_stats = val_stats.copy()
            patience_counter = 0
        else:
            patience_counter += 1

        print(f"  Epoch {epoch + 1}: Loss={train_loss:.4f}, "
              f"PPL={val_stats['ppl']:.2f}, "
              f"Shallow={val_stats['shallow_ratio']*100:.1f}%, "
              f"Compute={val_stats['compute_cost']*100:.1f}%, "
              f"ShallowL={train_stats.get('shallow_loss', 0):.3f}, "
              f"DeepL={train_stats.get('deep_loss', 0):.3f}")

        if patience_counter >= patience:
            print(f"  Converged at epoch {epoch + 1}")
            break

    results.append({
        'model': 'Confidence-Routed',
        'params': count_params(routed_model),
        'ppl': best_ppl,
        'accuracy': best_stats['accuracy'],
        'shallow_ratio': best_stats['shallow_ratio'],
        'compute_cost': best_stats['compute_cost'],
    })

    # ===== Standard 4-layer Transformer =====
    print("\n" + "=" * 70)
    print("Training Standard 4-layer Transformer")
    print("=" * 70)

    set_seed(42)
    standard_model = StandardTransformer(
        vocab_size=vocab_size,
        dim=dim,
        seq_len=seq_len,
        num_heads=num_heads,
        num_layers=4
    ).to(device)

    print(f"  Total params: {count_params(standard_model):,}")

    optimizer = torch.optim.AdamW(standard_model.parameters(), lr=lr)

    best_ppl_std = float('inf')
    patience_counter = 0

    for epoch in range(max_epochs):
        train_loss = train_epoch_standard(
            standard_model, train_batches, optimizer, vocab_size, device
        )
        val_ppl, val_acc = evaluate_standard(
            standard_model, val_batches, vocab_size, device
        )

        if val_ppl < best_ppl_std:
            best_ppl_std = val_ppl
            best_acc_std = val_acc
            patience_counter = 0
        else:
            patience_counter += 1

        print(f"  Epoch {epoch + 1}: Loss={train_loss:.4f}, PPL={val_ppl:.2f}")

        if patience_counter >= patience:
            print(f"  Converged at epoch {epoch + 1}")
            break

    results.append({
        'model': 'Standard 4-layer',
        'params': count_params(standard_model),
        'ppl': best_ppl_std,
        'accuracy': best_acc_std,
        'shallow_ratio': 0.0,
        'compute_cost': 1.0,
    })

    # ===== Standard 2-layer Transformer =====
    print("\n" + "=" * 70)
    print("Training Standard 2-layer Transformer")
    print("=" * 70)

    set_seed(42)
    shallow_model = StandardTransformer(
        vocab_size=vocab_size,
        dim=dim,
        seq_len=seq_len,
        num_heads=num_heads,
        num_layers=2
    ).to(device)

    print(f"  Total params: {count_params(shallow_model):,}")

    optimizer = torch.optim.AdamW(shallow_model.parameters(), lr=lr)

    best_ppl_shallow = float('inf')
    patience_counter = 0

    for epoch in range(max_epochs):
        train_loss = train_epoch_standard(
            shallow_model, train_batches, optimizer, vocab_size, device
        )
        val_ppl, val_acc = evaluate_standard(
            shallow_model, val_batches, vocab_size, device
        )

        if val_ppl < best_ppl_shallow:
            best_ppl_shallow = val_ppl
            best_acc_shallow = val_acc
            patience_counter = 0
        else:
            patience_counter += 1

        print(f"  Epoch {epoch + 1}: Loss={train_loss:.4f}, PPL={val_ppl:.2f}")

        if patience_counter >= patience:
            print(f"  Converged at epoch {epoch + 1}")
            break

    results.append({
        'model': 'Standard 2-layer',
        'params': count_params(shallow_model),
        'ppl': best_ppl_shallow,
        'accuracy': best_acc_shallow,
        'shallow_ratio': 1.0,
        'compute_cost': 0.5,
    })

    # ===== Test different thresholds =====
    print("\n" + "=" * 70)
    print("Testing Different Routing Thresholds")
    print("=" * 70)

    threshold_results = []
    for threshold in [0.5, 0.6, 0.7, 0.8, 0.9]:
        routed_model.routing_threshold = threshold
        _, stats = evaluate_routed(routed_model, val_batches, vocab_size, device)
        threshold_results.append({
            'threshold': threshold,
            'ppl': stats['ppl'],
            'accuracy': stats['accuracy'],
            'shallow_ratio': stats['shallow_ratio'],
            'compute_cost': stats['compute_cost'],
        })
        print(f"  Threshold {threshold}: PPL={stats['ppl']:.2f}, "
              f"Shallow={stats['shallow_ratio']*100:.1f}%, "
              f"Compute={stats['compute_cost']*100:.1f}%")

    # ===== Summary =====
    print("\n" + "=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)

    print(f"\n{'Model':<25} {'Params':>10} {'PPL':>8} {'Acc':>8} {'Compute':>10}")
    print("-" * 65)
    for r in results:
        print(f"{r['model']:<25} {r['params']:>10,} {r['ppl']:>8.2f} "
              f"{r['accuracy']*100:>7.1f}% {r['compute_cost']*100:>9.1f}%")

    print("\n" + "=" * 70)
    print("THRESHOLD ANALYSIS")
    print("=" * 70)

    print(f"\n{'Threshold':>10} {'PPL':>8} {'Accuracy':>10} {'Shallow%':>10} {'Compute%':>10}")
    print("-" * 50)
    for r in threshold_results:
        print(f"{r['threshold']:>10.2f} {r['ppl']:>8.2f} {r['accuracy']*100:>9.1f}% "
              f"{r['shallow_ratio']*100:>9.1f}% {r['compute_cost']*100:>9.1f}%")

    # ===== Analysis =====
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    routed = results[0]
    standard = results[1]
    shallow = results[2]

    print(f"\n1. Confidence-Routed vs Standard 4-layer:")
    ppl_diff = (routed['ppl'] - standard['ppl']) / standard['ppl'] * 100
    compute_save = (1 - routed['compute_cost']) * 100
    print(f"   PPL: {routed['ppl']:.2f} vs {standard['ppl']:.2f} ({ppl_diff:+.1f}%)")
    print(f"   Compute saving: {compute_save:.1f}%")

    print(f"\n2. Confidence-Routed vs Standard 2-layer:")
    ppl_diff2 = (routed['ppl'] - shallow['ppl']) / shallow['ppl'] * 100
    print(f"   PPL: {routed['ppl']:.2f} vs {shallow['ppl']:.2f} ({ppl_diff2:+.1f}%)")

    print(f"\n3. Routing Statistics:")
    print(f"   Shallow path ratio: {routed['shallow_ratio']*100:.1f}%")
    print(f"   Effective compute: {routed['compute_cost']*100:.1f}%")

    if routed['ppl'] < standard['ppl'] * 1.05:
        print(f"\n=> Confidence-Routed achieves comparable quality with {compute_save:.1f}% compute savings!")
    elif routed['ppl'] < shallow['ppl']:
        print(f"\n=> Confidence-Routed is better than 2-layer but worse than 4-layer")
    else:
        print(f"\n=> Confidence-Routed needs improvement")


if __name__ == "__main__":
    main()
