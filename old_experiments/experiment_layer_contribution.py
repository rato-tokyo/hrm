"""
Experiment: Layer Contribution Analysis

Hypothesis:
1. Tokens that are hard to predict require larger layer contributions (higher norms)
2. Tokens that are easy to predict have smaller layer contributions
3. Early layers may already produce good predictions for easy tokens

This script analyzes:
- Per-token loss at each layer
- Layer output norm (the residual added to x) per token
- Correlation between layer contribution norm and prediction difficulty
"""

import torch
import torch.nn as nn
import math
import random
import numpy as np
from typing import List, Tuple, Dict
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

from modules.transformer import TransformerBlock


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


class AnalyzableTransformer(nn.Module):
    """
    Transformer that can analyze per-layer, per-token contributions.
    """

    def __init__(
        self,
        vocab_size: int,
        dim: int = 64,
        seq_len: int = 64,
        num_layers: int = 4,
        num_heads: int = 4,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.seq_len = seq_len
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([
            TransformerBlock(dim, num_heads) for _ in range(num_layers)
        ])
        self.output_head = nn.Linear(dim, vocab_size, bias=False)

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
            elif isinstance(module, nn.Embedding):
                nn.init.trunc_normal_(module.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embedding(x)
        for layer in self.layers:
            h = layer(h)
        return self.output_head(h)

    def forward_with_analysis(self, x: torch.Tensor) -> Dict:
        """
        Forward pass with detailed analysis of each layer's contribution.

        Returns:
            Dict containing:
            - layer_outputs: List of outputs at each layer [batch, seq, vocab]
            - layer_residuals: List of residuals added by each layer [batch, seq, dim]
            - residual_norms: List of per-token norms [batch, seq]
            - cumulative_h: Hidden state after each layer [batch, seq, dim]
        """
        batch_size, seq_len = x.shape
        h = self.embedding(x)  # [batch, seq, dim]

        layer_outputs = []
        layer_residuals = []
        residual_norms = []
        cumulative_h = [h.clone()]

        for i, layer in enumerate(self.layers):
            h_before = h.clone()
            h = layer(h)

            # The residual is what the layer added to the input
            # Note: TransformerBlock uses post-norm, so h = norm(h_before + attn + ffn)
            # We approximate the "contribution" as the change in h
            residual = h - h_before  # [batch, seq, dim]

            layer_residuals.append(residual)

            # Compute per-token norm of the residual
            norm = torch.norm(residual, dim=-1)  # [batch, seq]
            residual_norms.append(norm)

            # Output prediction at this layer
            y_hat = self.output_head(h)  # [batch, seq, vocab]
            layer_outputs.append(y_hat)

            cumulative_h.append(h.clone())

        return {
            'layer_outputs': layer_outputs,
            'layer_residuals': layer_residuals,
            'residual_norms': residual_norms,
            'cumulative_h': cumulative_h,
            'final_output': layer_outputs[-1]
        }


def prepare_data(
    train_chars: int = 20000,
    val_chars: int = 5000,
    seq_len: int = 64,
    batch_size: int = 16
) -> Tuple:
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
    idx_to_char = {i: c for c, i in char_to_idx.items()}
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

    return train_batches, val_batches, vocab_size, char_to_idx, idx_to_char


def train_model(
    model: nn.Module,
    train_batches: List,
    val_batches: List,
    vocab_size: int,
    max_epochs: int = 30,
    patience: int = 5,
    lr: float = 1e-3,
    device: str = 'cpu'
) -> float:
    """Train model and return best validation PPL."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    best_ppl = float('inf')
    patience_counter = 0

    for epoch in range(max_epochs):
        # Train
        model.train()
        for x, y in train_batches:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            y_hat = model(x)
            loss = nn.functional.cross_entropy(y_hat.view(-1, vocab_size), y.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # Validate
        model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for x, y in val_batches:
                x, y = x.to(device), y.to(device)
                y_hat = model(x)
                loss = nn.functional.cross_entropy(y_hat.view(-1, vocab_size), y.view(-1))
                total_loss += loss.item()

        val_ppl = np.exp(total_loss / len(val_batches))

        if val_ppl < best_ppl:
            best_ppl = val_ppl
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch + 1}: Val PPL = {val_ppl:.2f}")

        if patience_counter >= patience:
            print(f"  Converged at epoch {epoch + 1}")
            break

    return best_ppl


@torch.no_grad()
def analyze_layer_contributions(
    model: AnalyzableTransformer,
    batches: List[Tuple[torch.Tensor, torch.Tensor]],
    vocab_size: int,
    device: str = 'cpu'
) -> Dict:
    """
    Analyze layer contributions across all batches.

    Returns statistics about:
    - Per-layer loss for each token
    - Per-layer residual norm for each token
    - Correlation between difficulty and layer contribution
    """
    model.eval()

    all_token_losses = []  # [num_layers, total_tokens]
    all_residual_norms = []  # [num_layers, total_tokens]
    all_final_losses = []  # [total_tokens]
    all_targets = []  # [total_tokens]

    num_layers = len(model.layers)

    for x, y in batches:
        x, y = x.to(device), y.to(device)
        batch_size, seq_len = x.shape

        analysis = model.forward_with_analysis(x)

        # Per-layer, per-token loss
        for layer_idx in range(num_layers):
            y_hat = analysis['layer_outputs'][layer_idx]  # [batch, seq, vocab]
            # Compute per-token loss
            log_probs = nn.functional.log_softmax(y_hat, dim=-1)
            token_losses = -log_probs.gather(dim=-1, index=y.unsqueeze(-1)).squeeze(-1)  # [batch, seq]
            all_token_losses.append(token_losses.cpu().flatten())

        # Per-layer, per-token residual norm
        for layer_idx in range(num_layers):
            norms = analysis['residual_norms'][layer_idx]  # [batch, seq]
            all_residual_norms.append(norms.cpu().flatten())

        # Final layer loss (for difficulty measure)
        final_y_hat = analysis['final_output']
        final_log_probs = nn.functional.log_softmax(final_y_hat, dim=-1)
        final_token_losses = -final_log_probs.gather(dim=-1, index=y.unsqueeze(-1)).squeeze(-1)
        all_final_losses.append(final_token_losses.cpu().flatten())

        all_targets.append(y.cpu().flatten())

    # Reorganize into per-layer arrays
    total_tokens = sum(t.shape[0] for t in all_final_losses)

    layer_losses = []
    layer_norms = []

    for layer_idx in range(num_layers):
        losses = torch.cat([all_token_losses[i * num_layers + layer_idx]
                          for i in range(len(batches))])
        norms = torch.cat([all_residual_norms[i * num_layers + layer_idx]
                         for i in range(len(batches))])
        layer_losses.append(losses)
        layer_norms.append(norms)

    final_losses = torch.cat(all_final_losses)
    targets = torch.cat(all_targets)

    return {
        'layer_losses': layer_losses,  # List of [total_tokens] for each layer
        'layer_norms': layer_norms,    # List of [total_tokens] for each layer
        'final_losses': final_losses,  # [total_tokens]
        'targets': targets,            # [total_tokens]
        'num_layers': num_layers
    }


def compute_statistics(analysis: Dict) -> Dict:
    """Compute statistics from the analysis."""
    num_layers = analysis['num_layers']
    final_losses = analysis['final_losses'].numpy()

    # Categorize tokens by difficulty (final loss)
    easy_threshold = np.percentile(final_losses, 25)  # Bottom 25%
    hard_threshold = np.percentile(final_losses, 75)  # Top 25%

    easy_mask = final_losses <= easy_threshold
    hard_mask = final_losses >= hard_threshold
    medium_mask = ~easy_mask & ~hard_mask

    stats = {
        'num_layers': num_layers,
        'easy_threshold': easy_threshold,
        'hard_threshold': hard_threshold,
        'num_easy': easy_mask.sum(),
        'num_hard': hard_mask.sum(),
        'num_medium': medium_mask.sum(),
        'layer_stats': []
    }

    print(f"\nToken Difficulty Distribution:")
    print(f"  Easy (loss <= {easy_threshold:.2f}): {easy_mask.sum()} tokens")
    print(f"  Medium: {medium_mask.sum()} tokens")
    print(f"  Hard (loss >= {hard_threshold:.2f}): {hard_mask.sum()} tokens")

    for layer_idx in range(num_layers):
        losses = analysis['layer_losses'][layer_idx].numpy()
        norms = analysis['layer_norms'][layer_idx].numpy()

        layer_stat = {
            'layer': layer_idx + 1,
            'mean_loss': losses.mean(),
            'mean_norm': norms.mean(),
            'easy_loss': losses[easy_mask].mean(),
            'easy_norm': norms[easy_mask].mean(),
            'hard_loss': losses[hard_mask].mean(),
            'hard_norm': norms[hard_mask].mean(),
            'medium_loss': losses[medium_mask].mean(),
            'medium_norm': norms[medium_mask].mean(),
        }

        # Correlation between norm and final difficulty
        correlation = np.corrcoef(norms, final_losses)[0, 1]
        layer_stat['norm_difficulty_correlation'] = correlation

        stats['layer_stats'].append(layer_stat)

    return stats


def print_results(stats: Dict):
    """Print analysis results."""
    print("\n" + "=" * 70)
    print("Layer-wise Analysis Results")
    print("=" * 70)

    print("\n1. Per-Layer Mean Loss and Norm:")
    print(f"{'Layer':<8} {'Mean Loss':>12} {'Mean Norm':>12}")
    print("-" * 34)
    for s in stats['layer_stats']:
        print(f"L{s['layer']:<7} {s['mean_loss']:>12.4f} {s['mean_norm']:>12.4f}")

    print("\n2. Loss by Token Difficulty:")
    print(f"{'Layer':<8} {'Easy':>12} {'Medium':>12} {'Hard':>12}")
    print("-" * 46)
    for s in stats['layer_stats']:
        print(f"L{s['layer']:<7} {s['easy_loss']:>12.4f} {s['medium_loss']:>12.4f} {s['hard_loss']:>12.4f}")

    print("\n3. Residual Norm by Token Difficulty:")
    print(f"{'Layer':<8} {'Easy':>12} {'Medium':>12} {'Hard':>12}")
    print("-" * 46)
    for s in stats['layer_stats']:
        print(f"L{s['layer']:<7} {s['easy_norm']:>12.4f} {s['medium_norm']:>12.4f} {s['hard_norm']:>12.4f}")

    print("\n4. Correlation: Layer Norm vs Final Difficulty:")
    print(f"{'Layer':<8} {'Correlation':>12}")
    print("-" * 22)
    for s in stats['layer_stats']:
        print(f"L{s['layer']:<7} {s['norm_difficulty_correlation']:>12.4f}")

    # Hypothesis verification
    print("\n" + "=" * 70)
    print("Hypothesis Verification")
    print("=" * 70)

    # Check if hard tokens have higher norms
    easy_norms = [s['easy_norm'] for s in stats['layer_stats']]
    hard_norms = [s['hard_norm'] for s in stats['layer_stats']]

    print("\n1. Do hard tokens have higher residual norms?")
    for i, s in enumerate(stats['layer_stats']):
        diff = s['hard_norm'] - s['easy_norm']
        ratio = s['hard_norm'] / s['easy_norm'] if s['easy_norm'] > 0 else 0
        status = "YES" if diff > 0 else "NO"
        print(f"   L{s['layer']}: Hard={s['hard_norm']:.4f}, Easy={s['easy_norm']:.4f}, "
              f"Diff={diff:+.4f}, Ratio={ratio:.2f}x [{status}]")

    # Check if easy tokens are already well-predicted early
    print("\n2. Are easy tokens well-predicted at early layers?")
    for s in stats['layer_stats']:
        print(f"   L{s['layer']}: Easy Loss={s['easy_loss']:.4f}, Hard Loss={s['hard_loss']:.4f}")

    # Check layer contribution pattern
    print("\n3. Layer contribution pattern (norm decrease through layers):")
    for difficulty in ['easy', 'hard']:
        norms = [s[f'{difficulty}_norm'] for s in stats['layer_stats']]
        print(f"   {difficulty.capitalize()}: {' -> '.join([f'{n:.3f}' for n in norms])}")


def create_visualization(analysis: Dict, stats: Dict, save_path: str = 'layer_contribution_analysis.png'):
    """Create visualization of the analysis."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    num_layers = stats['num_layers']
    layers = [f'L{i+1}' for i in range(num_layers)]

    # 1. Loss by difficulty
    ax1 = axes[0, 0]
    easy_losses = [s['easy_loss'] for s in stats['layer_stats']]
    medium_losses = [s['medium_loss'] for s in stats['layer_stats']]
    hard_losses = [s['hard_loss'] for s in stats['layer_stats']]

    x = np.arange(num_layers)
    width = 0.25
    ax1.bar(x - width, easy_losses, width, label='Easy', color='green', alpha=0.7)
    ax1.bar(x, medium_losses, width, label='Medium', color='orange', alpha=0.7)
    ax1.bar(x + width, hard_losses, width, label='Hard', color='red', alpha=0.7)
    ax1.set_xlabel('Layer')
    ax1.set_ylabel('Loss')
    ax1.set_title('Per-Layer Loss by Token Difficulty')
    ax1.set_xticks(x)
    ax1.set_xticklabels(layers)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Residual norm by difficulty
    ax2 = axes[0, 1]
    easy_norms = [s['easy_norm'] for s in stats['layer_stats']]
    medium_norms = [s['medium_norm'] for s in stats['layer_stats']]
    hard_norms = [s['hard_norm'] for s in stats['layer_stats']]

    ax2.bar(x - width, easy_norms, width, label='Easy', color='green', alpha=0.7)
    ax2.bar(x, medium_norms, width, label='Medium', color='orange', alpha=0.7)
    ax2.bar(x + width, hard_norms, width, label='Hard', color='red', alpha=0.7)
    ax2.set_xlabel('Layer')
    ax2.set_ylabel('Residual Norm')
    ax2.set_title('Per-Layer Residual Norm by Token Difficulty')
    ax2.set_xticks(x)
    ax2.set_xticklabels(layers)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Correlation between norm and difficulty
    ax3 = axes[1, 0]
    correlations = [s['norm_difficulty_correlation'] for s in stats['layer_stats']]
    colors = ['green' if c > 0 else 'red' for c in correlations]
    ax3.bar(layers, correlations, color=colors, alpha=0.7)
    ax3.set_xlabel('Layer')
    ax3.set_ylabel('Correlation')
    ax3.set_title('Correlation: Layer Norm vs Token Difficulty')
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax3.grid(True, alpha=0.3)

    # 4. Scatter plot: Layer 4 norm vs final loss (sample)
    ax4 = axes[1, 1]
    final_losses = analysis['final_losses'].numpy()
    layer4_norms = analysis['layer_norms'][-1].numpy()

    # Sample for visibility
    sample_idx = np.random.choice(len(final_losses), min(1000, len(final_losses)), replace=False)
    ax4.scatter(layer4_norms[sample_idx], final_losses[sample_idx], alpha=0.3, s=10)
    ax4.set_xlabel('Layer 4 Residual Norm')
    ax4.set_ylabel('Final Loss (Difficulty)')
    ax4.set_title('Layer 4 Norm vs Token Difficulty (sampled)')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\nVisualization saved to: {save_path}")
    plt.close()


def main():
    set_seed(42)

    # Config
    seq_len = 64
    dim = 64
    num_layers = 4
    num_heads = 4
    batch_size = 16
    device = 'cpu'
    train_chars = 20000
    val_chars = 5000

    print("=" * 70)
    print("Experiment: Layer Contribution Analysis")
    print("=" * 70)
    print(f"\nHypothesis:")
    print("  1. Hard-to-predict tokens require larger layer contributions (higher norms)")
    print("  2. Easy-to-predict tokens have smaller layer contributions")
    print("  3. Easy tokens may be well-predicted at early layers")

    print(f"\nConfig:")
    print(f"  Layers: {num_layers}")
    print(f"  Dimension: {dim}")
    print(f"  Sequence length: {seq_len}")

    # Prepare data
    train_batches, val_batches, vocab_size, char_to_idx, idx_to_char = prepare_data(
        train_chars, val_chars, seq_len, batch_size
    )
    print(f"  Vocab size: {vocab_size}")
    print(f"  Train batches: {len(train_batches)}")
    print(f"  Val batches: {len(val_batches)}")

    # Create and train model
    print("\n" + "=" * 70)
    print("Training Model")
    print("=" * 70)

    model = AnalyzableTransformer(
        vocab_size=vocab_size,
        dim=dim,
        seq_len=seq_len,
        num_layers=num_layers,
        num_heads=num_heads
    ).to(device)

    print(f"Parameters: {count_params(model):,}")

    best_ppl = train_model(
        model, train_batches, val_batches, vocab_size,
        max_epochs=30, patience=5, lr=1e-3, device=device
    )
    print(f"\nBest Val PPL: {best_ppl:.2f}")

    # Analyze layer contributions
    print("\n" + "=" * 70)
    print("Analyzing Layer Contributions")
    print("=" * 70)

    analysis = analyze_layer_contributions(model, val_batches, vocab_size, device)
    stats = compute_statistics(analysis)
    print_results(stats)

    # Create visualization
    create_visualization(analysis, stats)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    # Check hypothesis 1: Hard tokens have higher norms
    h1_support = sum(1 for s in stats['layer_stats'] if s['hard_norm'] > s['easy_norm'])
    print(f"\nHypothesis 1 (Hard tokens → Higher norms):")
    print(f"  Supported in {h1_support}/{num_layers} layers")

    # Check hypothesis 2: Correlation between norm and difficulty
    avg_corr = np.mean([s['norm_difficulty_correlation'] for s in stats['layer_stats']])
    print(f"\nHypothesis 2 (Norm correlates with difficulty):")
    print(f"  Average correlation: {avg_corr:.4f}")
    if avg_corr > 0.1:
        print(f"  SUPPORTED: Positive correlation found")
    elif avg_corr < -0.1:
        print(f"  OPPOSITE: Negative correlation found")
    else:
        print(f"  WEAK: No strong correlation")

    # Check hypothesis 3: Easy tokens well-predicted early
    l1_easy = stats['layer_stats'][0]['easy_loss']
    l4_easy = stats['layer_stats'][-1]['easy_loss']
    l1_hard = stats['layer_stats'][0]['hard_loss']
    l4_hard = stats['layer_stats'][-1]['hard_loss']

    print(f"\nHypothesis 3 (Easy tokens predicted early):")
    print(f"  Easy tokens: L1 loss={l1_easy:.4f}, L4 loss={l4_easy:.4f}")
    print(f"  Hard tokens: L1 loss={l1_hard:.4f}, L4 loss={l4_hard:.4f}")
    easy_improvement = (l1_easy - l4_easy) / l1_easy * 100
    hard_improvement = (l1_hard - l4_hard) / l1_hard * 100
    print(f"  Easy improvement L1→L4: {easy_improvement:.1f}%")
    print(f"  Hard improvement L1→L4: {hard_improvement:.1f}%")


if __name__ == "__main__":
    main()
