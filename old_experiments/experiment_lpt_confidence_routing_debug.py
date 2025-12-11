"""
Debug: Check shallow vs deep path accuracy separately
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


class LPTConfidenceRoutedTransformer(nn.Module):
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
        self.layer4 = TransformerBlock(dim, num_heads)
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

    def forward_lpt(self, x: torch.Tensor) -> Dict:
        h = self.embedding(x)
        h1 = self.layer1(h)
        h2 = self.layer2(h1)
        h3 = self.layer3(h2)
        h4 = self.layer4(h3)

        logits1 = self.output_head(h1)
        logits2 = self.output_head(h2)
        logits3 = self.output_head(h3)
        logits4 = self.output_head(h4)

        with torch.no_grad():
            confidence = self.compute_confidence(h2)

        return {
            'logits1': logits1,
            'logits2': logits2,
            'logits3': logits3,
            'logits4': logits4,
            'confidence': confidence,
        }


def prepare_data(train_chars=20000, val_chars=5000, seq_len=64, batch_size=16):
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

    def text_to_batches(text):
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


def train_epoch_standard(model, batches, optimizer, vocab_size, device='cpu'):
    model.train()
    total_loss = 0.0

    for x, y in batches:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        outputs = model.forward_lpt(x)
        shallow_loss = F.cross_entropy(outputs['logits2'].view(-1, vocab_size), y.view(-1))
        deep_loss = F.cross_entropy(outputs['logits4'].view(-1, vocab_size), y.view(-1))
        loss = 0.5 * shallow_loss + 0.5 * deep_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(batches)


@torch.no_grad()
def evaluate_detailed(model, batches, vocab_size, device='cpu'):
    """Evaluate with detailed path-specific metrics."""
    model.eval()

    # Separate tracking
    shallow_correct = 0
    shallow_total = 0
    deep_correct = 0
    deep_total = 0
    routed_correct = 0  # For routed output

    total_shallow_loss = 0.0
    total_deep_loss = 0.0
    total_routed_loss = 0.0

    for x, y in batches:
        x, y = x.to(device), y.to(device)
        outputs = model.forward_lpt(x)

        shallow_logits = outputs['logits2']
        deep_logits = outputs['logits4']
        confidence = outputs['confidence']

        # Shallow path metrics (all tokens)
        shallow_preds = shallow_logits.argmax(dim=-1)
        shallow_correct += (shallow_preds == y).sum().item()
        shallow_total += y.numel()
        total_shallow_loss += F.cross_entropy(shallow_logits.view(-1, vocab_size), y.view(-1)).item()

        # Deep path metrics (all tokens)
        deep_preds = deep_logits.argmax(dim=-1)
        deep_correct += (deep_preds == y).sum().item()
        deep_total += y.numel()
        total_deep_loss += F.cross_entropy(deep_logits.view(-1, vocab_size), y.view(-1)).item()

        # Routed output
        mask = (confidence >= model.routing_threshold).unsqueeze(-1)
        routed_logits = torch.where(mask, shallow_logits, deep_logits)
        routed_preds = routed_logits.argmax(dim=-1)
        routed_correct += (routed_preds == y).sum().item()
        total_routed_loss += F.cross_entropy(routed_logits.view(-1, vocab_size), y.view(-1)).item()

    n = len(batches)
    return {
        'shallow_acc': shallow_correct / shallow_total,
        'shallow_ppl': np.exp(total_shallow_loss / n),
        'deep_acc': deep_correct / deep_total,
        'deep_ppl': np.exp(total_deep_loss / n),
        'routed_acc': routed_correct / shallow_total,
        'routed_ppl': np.exp(total_routed_loss / n),
    }


def main():
    set_seed(42)

    seq_len = 64
    dim = 64
    num_heads = 4
    batch_size = 16
    device = 'cpu'
    train_chars = 20000
    val_chars = 5000
    max_epochs = 10
    lr = 1e-3

    print("=" * 70)
    print("Debug: Shallow vs Deep Path Accuracy Analysis")
    print("=" * 70)

    train_batches, val_batches, vocab_size = prepare_data(
        train_chars, val_chars, seq_len, batch_size
    )
    print(f"Vocab size: {vocab_size}")

    set_seed(42)
    model = LPTConfidenceRoutedTransformer(
        vocab_size=vocab_size, dim=dim, seq_len=seq_len,
        num_heads=num_heads, routing_threshold=0.8
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    print("\n" + "=" * 70)
    print("Epoch | Shallow_PPL | Deep_PPL | Routed_PPL | Shallow_Acc | Deep_Acc | Routed_Acc")
    print("-" * 70)

    for epoch in range(max_epochs):
        train_loss = train_epoch_standard(model, train_batches, optimizer, vocab_size, device)
        stats = evaluate_detailed(model, val_batches, vocab_size, device)

        print(f"  {epoch+1:2}  |   {stats['shallow_ppl']:6.2f}   |  {stats['deep_ppl']:6.2f}  |   {stats['routed_ppl']:6.2f}   |"
              f"    {stats['shallow_acc']*100:5.1f}%   |  {stats['deep_acc']*100:5.1f}%  |   {stats['routed_acc']*100:5.1f}%")

    print("\n" + "=" * 70)
    print("Analysis:")
    print("=" * 70)
    print(f"\nFinal Shallow PPL: {stats['shallow_ppl']:.2f}")
    print(f"Final Deep PPL:    {stats['deep_ppl']:.2f}")
    print(f"Final Routed PPL:  {stats['routed_ppl']:.2f}")

    if stats['shallow_ppl'] > stats['deep_ppl']:
        print(f"\n=> Shallow path is WORSE than Deep path!")
        print(f"   Difference: {stats['shallow_ppl'] - stats['deep_ppl']:.2f}")
        print(f"\n   This explains why routing more tokens to shallow path")
        print(f"   causes PPL degradation!")
    else:
        print(f"\n=> Shallow path is BETTER than Deep path!")


if __name__ == "__main__":
    main()
