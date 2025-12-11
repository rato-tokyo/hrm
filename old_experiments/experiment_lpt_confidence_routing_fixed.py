"""
Experiment: LPT + Confidence-Routed Transformer (Fixed)

Bug fixes:
1. Normalize LPT loss by dividing by number of layers
2. This ensures fair comparison of gradient magnitudes

Compare Confidence-Routed Transformer with and without LPT training.

Hypothesis:
- LPT makes L2 output more "prediction-ready"
- This should improve shallow path accuracy
- And increase the shallow ratio (more tokens exit early)
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


class LPTConfidenceRoutedTransformer(nn.Module):
    """
    Confidence-Routed Transformer with LPT training.

    Training modes:
    - 'standard': Only shallow and deep losses (current approach)
    - 'lpt': Add losses at L1, L2, L3, L4 (LPT style)
    - 'lpt_routing': LPT + Routing (L1, L2 for shallow, L3, L4 for deep)
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

        # All 4 layers
        self.layer1 = TransformerBlock(dim, num_heads)
        self.layer2 = TransformerBlock(dim, num_heads)
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
        """Compute confidence (max probability) from hidden state."""
        logits = self.output_head(h)
        probs = F.softmax(logits, dim=-1)
        confidence = probs.max(dim=-1).values
        return confidence

    def forward_lpt(self, x: torch.Tensor) -> Dict:
        """
        Forward pass returning all layer outputs for LPT training.
        """
        h = self.embedding(x)

        h1 = self.layer1(h)
        h2 = self.layer2(h1)
        h3 = self.layer3(h2)
        h4 = self.layer4(h3)

        # Get logits at each layer
        logits1 = self.output_head(h1)
        logits2 = self.output_head(h2)
        logits3 = self.output_head(h3)
        logits4 = self.output_head(h4)

        # Compute confidence at L2 for routing
        with torch.no_grad():
            confidence = self.compute_confidence(h2)

        return {
            'logits1': logits1,
            'logits2': logits2,  # shallow output
            'logits3': logits3,
            'logits4': logits4,  # deep output
            'confidence': confidence,
            'shallow_ratio': (confidence >= self.routing_threshold).float().mean().item(),
        }

    def forward_inference(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Inference with hard routing based on L2 confidence."""
        batch_size, seq_len = x.shape

        h = self.embedding(x)
        h = self.layer1(h)
        h = self.layer2(h)

        # Compute confidence for routing
        confidence = self.compute_confidence(h)

        # Shallow output
        shallow_logits = self.output_head(h)

        # Deep path
        h_deep = self.layer3(h)
        h_deep = self.layer4(h_deep)
        deep_logits = self.output_head(h_deep)

        # Hard routing
        mask = (confidence >= self.routing_threshold).unsqueeze(-1)
        output = torch.where(mask, shallow_logits, deep_logits)

        # Compute cost
        shallow_count = mask.sum().item()
        total_count = batch_size * seq_len
        deep_count = total_count - shallow_count
        compute_cost = (shallow_count * 2 + deep_count * 4) / (total_count * 4)

        stats = {
            'mean_confidence': confidence.mean().item(),
            'shallow_ratio': shallow_count / total_count,
            'compute_cost': compute_cost,
        }

        return output, stats

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            outputs = self.forward_lpt(x)
            return outputs['logits4']
        else:
            output, _ = self.forward_inference(x)
            return output


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


def train_epoch_standard(
    model: LPTConfidenceRoutedTransformer,
    batches: List,
    optimizer: torch.optim.Optimizer,
    vocab_size: int,
    device: str = 'cpu'
) -> Tuple[float, Dict]:
    """Train with standard routing loss (shallow + deep only)."""
    model.train()
    total_loss = 0.0
    total_stats = {'shallow_ratio': 0}

    for x, y in batches:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        outputs = model.forward_lpt(x)

        # Only shallow (L2) and deep (L4) losses
        shallow_loss = F.cross_entropy(outputs['logits2'].view(-1, vocab_size), y.view(-1))
        deep_loss = F.cross_entropy(outputs['logits4'].view(-1, vocab_size), y.view(-1))
        loss = 0.5 * shallow_loss + 0.5 * deep_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        total_stats['shallow_ratio'] += outputs['shallow_ratio']

    n = len(batches)
    return total_loss / n, {k: v / n for k, v in total_stats.items()}


def train_epoch_lpt(
    model: LPTConfidenceRoutedTransformer,
    batches: List,
    optimizer: torch.optim.Optimizer,
    vocab_size: int,
    device: str = 'cpu'
) -> Tuple[float, Dict]:
    """
    Train with full LPT loss (L1 + L2 + L3 + L4).

    FIX: Normalize by number of layers to ensure fair gradient comparison.
    """
    model.train()
    total_loss = 0.0
    total_stats = {'shallow_ratio': 0, 'l1_loss': 0, 'l2_loss': 0, 'l3_loss': 0, 'l4_loss': 0}

    for x, y in batches:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        outputs = model.forward_lpt(x)

        # LPT: loss at each layer
        l1_loss = F.cross_entropy(outputs['logits1'].view(-1, vocab_size), y.view(-1))
        l2_loss = F.cross_entropy(outputs['logits2'].view(-1, vocab_size), y.view(-1))
        l3_loss = F.cross_entropy(outputs['logits3'].view(-1, vocab_size), y.view(-1))
        l4_loss = F.cross_entropy(outputs['logits4'].view(-1, vocab_size), y.view(-1))

        # FIX: Average all losses instead of sum
        # This normalizes gradient magnitude to be comparable with standard routing
        loss = (l1_loss + l2_loss + l3_loss + l4_loss) / 4.0

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        total_stats['shallow_ratio'] += outputs['shallow_ratio']
        total_stats['l1_loss'] += l1_loss.item()
        total_stats['l2_loss'] += l2_loss.item()
        total_stats['l3_loss'] += l3_loss.item()
        total_stats['l4_loss'] += l4_loss.item()

    n = len(batches)
    return total_loss / n, {k: v / n for k, v in total_stats.items()}


def train_epoch_lpt_routing(
    model: LPTConfidenceRoutedTransformer,
    batches: List,
    optimizer: torch.optim.Optimizer,
    vocab_size: int,
    device: str = 'cpu'
) -> Tuple[float, Dict]:
    """
    Train with LPT + Routing aware loss.
    - L1, L2: Weighted towards shallow path
    - L3, L4: Weighted towards deep path

    FIX: Normalize by total weight to ensure fair gradient comparison.
    """
    model.train()
    total_loss = 0.0
    total_stats = {'shallow_ratio': 0}

    for x, y in batches:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        outputs = model.forward_lpt(x)

        # LPT losses
        l1_loss = F.cross_entropy(outputs['logits1'].view(-1, vocab_size), y.view(-1))
        l2_loss = F.cross_entropy(outputs['logits2'].view(-1, vocab_size), y.view(-1))
        l3_loss = F.cross_entropy(outputs['logits3'].view(-1, vocab_size), y.view(-1))
        l4_loss = F.cross_entropy(outputs['logits4'].view(-1, vocab_size), y.view(-1))

        # Routing-aware weighting:
        # - Emphasize L2 (shallow exit point) and L4 (deep exit point)
        # FIX: Normalize by total weight (0.5 + 1.0 + 0.5 + 1.0 = 3.0)
        raw_loss = 0.5 * l1_loss + 1.0 * l2_loss + 0.5 * l3_loss + 1.0 * l4_loss
        loss = raw_loss / 3.0  # Normalize

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        total_stats['shallow_ratio'] += outputs['shallow_ratio']

    n = len(batches)
    return total_loss / n, {k: v / n for k, v in total_stats.items()}


@torch.no_grad()
def evaluate(
    model: LPTConfidenceRoutedTransformer,
    batches: List,
    vocab_size: int,
    device: str = 'cpu'
) -> Dict:
    """Evaluate with hard routing."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    total_stats = {'shallow_ratio': 0, 'compute_cost': 0, 'mean_confidence': 0}

    for x, y in batches:
        x, y = x.to(device), y.to(device)

        output, stats = model.forward_inference(x)
        loss = F.cross_entropy(output.view(-1, vocab_size), y.view(-1))

        preds = output.argmax(dim=-1)
        total_correct += (preds == y).sum().item()
        total_tokens += y.numel()

        total_loss += loss.item()
        for k in total_stats:
            total_stats[k] += stats[k]

    n = len(batches)
    avg_loss = total_loss / n
    for k in total_stats:
        total_stats[k] /= n

    total_stats['ppl'] = np.exp(avg_loss)
    total_stats['accuracy'] = total_correct / total_tokens

    return total_stats


def train_model(
    model: LPTConfidenceRoutedTransformer,
    train_batches: List,
    val_batches: List,
    vocab_size: int,
    train_fn,
    max_epochs: int = 30,
    patience: int = 5,
    lr: float = 1e-3,
    device: str = 'cpu',
    verbose: bool = True,
    model_name: str = ""
) -> Tuple[float, Dict]:
    """Train model and return best results."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    best_ppl = float('inf')
    best_stats = {}
    patience_counter = 0

    for epoch in range(max_epochs):
        train_loss, train_stats = train_fn(
            model, train_batches, optimizer, vocab_size, device
        )
        val_stats = evaluate(model, val_batches, vocab_size, device)

        if val_stats['ppl'] < best_ppl:
            best_ppl = val_stats['ppl']
            best_stats = val_stats.copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if verbose:
            print(f"  Epoch {epoch + 1}: Loss={train_loss:.4f}, "
                  f"PPL={val_stats['ppl']:.2f}, "
                  f"Shallow={val_stats['shallow_ratio']*100:.1f}%, "
                  f"Compute={val_stats['compute_cost']*100:.1f}%")

        if patience_counter >= patience:
            if verbose:
                print(f"  Converged at epoch {epoch + 1}")
            break

    return best_ppl, best_stats


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
    print("Experiment: LPT + Confidence-Routed Transformer (FIXED)")
    print("=" * 70)
    print("\nFIX: LPT loss is now normalized (averaged instead of summed)")
    print("     This ensures fair gradient magnitude comparison.")
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

    # ===== 1. Standard Routing (current approach) =====
    print("\n" + "=" * 70)
    print("1. Standard Routing (Shallow + Deep Loss)")
    print("   Loss = 0.5 * L2 + 0.5 * L4")
    print("=" * 70)

    set_seed(42)
    model_standard = LPTConfidenceRoutedTransformer(
        vocab_size=vocab_size, dim=dim, seq_len=seq_len,
        num_heads=num_heads, routing_threshold=0.8
    ).to(device)

    print(f"  Params: {count_params(model_standard):,}")

    best_ppl, best_stats = train_model(
        model_standard, train_batches, val_batches, vocab_size,
        train_epoch_standard, max_epochs, patience, lr, device,
        model_name="Standard Routing"
    )

    results.append({
        'model': 'Standard Routing',
        'ppl': best_ppl,
        'accuracy': best_stats['accuracy'],
        'shallow_ratio': best_stats['shallow_ratio'],
        'compute_cost': best_stats['compute_cost'],
    })

    # ===== 2. Full LPT (L1 + L2 + L3 + L4) / 4 =====
    print("\n" + "=" * 70)
    print("2. Full LPT (L1 + L2 + L3 + L4) / 4")
    print("   Loss = (L1 + L2 + L3 + L4) / 4  [NORMALIZED]")
    print("=" * 70)

    set_seed(42)
    model_lpt = LPTConfidenceRoutedTransformer(
        vocab_size=vocab_size, dim=dim, seq_len=seq_len,
        num_heads=num_heads, routing_threshold=0.8
    ).to(device)

    print(f"  Params: {count_params(model_lpt):,}")

    best_ppl, best_stats = train_model(
        model_lpt, train_batches, val_batches, vocab_size,
        train_epoch_lpt, max_epochs, patience, lr, device,
        model_name="Full LPT"
    )

    results.append({
        'model': 'Full LPT (normalized)',
        'ppl': best_ppl,
        'accuracy': best_stats['accuracy'],
        'shallow_ratio': best_stats['shallow_ratio'],
        'compute_cost': best_stats['compute_cost'],
    })

    # ===== 3. LPT + Routing Aware (normalized) =====
    print("\n" + "=" * 70)
    print("3. LPT + Routing Aware (Emphasize L2 and L4)")
    print("   Loss = (0.5*L1 + 1.0*L2 + 0.5*L3 + 1.0*L4) / 3  [NORMALIZED]")
    print("=" * 70)

    set_seed(42)
    model_lpt_routing = LPTConfidenceRoutedTransformer(
        vocab_size=vocab_size, dim=dim, seq_len=seq_len,
        num_heads=num_heads, routing_threshold=0.8
    ).to(device)

    print(f"  Params: {count_params(model_lpt_routing):,}")

    best_ppl, best_stats = train_model(
        model_lpt_routing, train_batches, val_batches, vocab_size,
        train_epoch_lpt_routing, max_epochs, patience, lr, device,
        model_name="LPT + Routing"
    )

    results.append({
        'model': 'LPT + Routing (normalized)',
        'ppl': best_ppl,
        'accuracy': best_stats['accuracy'],
        'shallow_ratio': best_stats['shallow_ratio'],
        'compute_cost': best_stats['compute_cost'],
    })

    # ===== Test different thresholds for each model =====
    print("\n" + "=" * 70)
    print("Threshold Analysis")
    print("=" * 70)

    threshold_results = {
        'Standard Routing': [],
        'Full LPT (normalized)': [],
        'LPT + Routing (normalized)': []
    }

    models = [
        ('Standard Routing', model_standard),
        ('Full LPT (normalized)', model_lpt),
        ('LPT + Routing (normalized)', model_lpt_routing)
    ]

    for model_name, model in models:
        print(f"\n{model_name}:")
        for threshold in [0.5, 0.6, 0.7, 0.8, 0.9]:
            model.routing_threshold = threshold
            stats = evaluate(model, val_batches, vocab_size, device)
            threshold_results[model_name].append({
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
    print("FINAL COMPARISON (threshold=0.8)")
    print("=" * 70)

    print(f"\n{'Model':<30} {'PPL':>8} {'Acc':>8} {'Shallow%':>10} {'Compute%':>10}")
    print("-" * 70)
    for r in results:
        print(f"{r['model']:<30} {r['ppl']:>8.2f} {r['accuracy']*100:>7.1f}% "
              f"{r['shallow_ratio']*100:>9.1f}% {r['compute_cost']*100:>9.1f}%")

    # ===== Analysis =====
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    standard = results[0]
    full_lpt = results[1]
    lpt_routing = results[2]

    print(f"\n1. Standard vs Full LPT (normalized):")
    ppl_diff = (full_lpt['ppl'] - standard['ppl']) / standard['ppl'] * 100
    shallow_diff = full_lpt['shallow_ratio'] - standard['shallow_ratio']
    print(f"   PPL: {standard['ppl']:.2f} → {full_lpt['ppl']:.2f} ({ppl_diff:+.1f}%)")
    print(f"   Shallow ratio: {standard['shallow_ratio']*100:.1f}% → {full_lpt['shallow_ratio']*100:.1f}% ({shallow_diff*100:+.1f}pp)")

    print(f"\n2. Standard vs LPT + Routing (normalized):")
    ppl_diff2 = (lpt_routing['ppl'] - standard['ppl']) / standard['ppl'] * 100
    shallow_diff2 = lpt_routing['shallow_ratio'] - standard['shallow_ratio']
    print(f"   PPL: {standard['ppl']:.2f} → {lpt_routing['ppl']:.2f} ({ppl_diff2:+.1f}%)")
    print(f"   Shallow ratio: {standard['shallow_ratio']*100:.1f}% → {lpt_routing['shallow_ratio']*100:.1f}% ({shallow_diff2*100:+.1f}pp)")

    print(f"\n3. Best Model:")
    best = min(results, key=lambda x: x['ppl'])
    print(f"   {best['model']}: PPL={best['ppl']:.2f}, Compute={best['compute_cost']*100:.1f}%")

    # Compute savings comparison
    print(f"\n4. Compute Efficiency (at threshold=0.8):")
    for r in results:
        savings = (1 - r['compute_cost']) * 100
        print(f"   {r['model']}: {r['compute_cost']*100:.1f}% compute ({savings:.1f}% savings)")


if __name__ == "__main__":
    main()
