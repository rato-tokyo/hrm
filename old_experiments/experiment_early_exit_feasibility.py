"""
Experiment: Early Exit Feasibility Analysis

Can we identify "easy" tokens at Layer 1 and exit early?

Analysis:
1. How well does L1 loss predict final (L4) difficulty?
2. Can we use L1 confidence (entropy/max_prob) to predict difficulty?
3. What would be the accuracy of early exit decisions?
"""

import torch
import torch.nn as nn
import math
import random
import numpy as np
from typing import List, Tuple, Dict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from modules.transformer import TransformerBlock


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class AnalyzableTransformer(nn.Module):
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

    def forward_all_layers(self, x: torch.Tensor) -> List[torch.Tensor]:
        h = self.embedding(x)
        outputs = []
        for layer in self.layers:
            h = layer(h)
            outputs.append(self.output_head(h))
        return outputs


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

    return text_to_batches(train_text), text_to_batches(val_text), vocab_size


def train_model(model, train_batches, val_batches, vocab_size, max_epochs=30, patience=5, lr=1e-3):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    best_ppl = float('inf')
    patience_counter = 0

    for epoch in range(max_epochs):
        model.train()
        for x, y in train_batches:
            optimizer.zero_grad()
            y_hat = model(x)
            loss = nn.functional.cross_entropy(y_hat.view(-1, vocab_size), y.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for x, y in val_batches:
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
def analyze_early_exit(model, batches, vocab_size):
    """Analyze if early exit is feasible based on L1 predictions."""
    model.eval()

    all_l1_losses = []
    all_l4_losses = []
    all_l1_correct = []
    all_l4_correct = []
    all_l1_max_probs = []
    all_l1_entropies = []

    for x, y in batches:
        outputs = model.forward_all_layers(x)

        # L1 analysis
        l1_output = outputs[0]
        l1_probs = torch.softmax(l1_output, dim=-1)
        l1_log_probs = torch.log_softmax(l1_output, dim=-1)

        # Per-token L1 loss
        l1_token_losses = -l1_log_probs.gather(dim=-1, index=y.unsqueeze(-1)).squeeze(-1)
        all_l1_losses.append(l1_token_losses.flatten())

        # L1 predictions
        l1_preds = l1_output.argmax(dim=-1)
        l1_correct = (l1_preds == y).float()
        all_l1_correct.append(l1_correct.flatten())

        # L1 confidence metrics
        l1_max_prob = l1_probs.max(dim=-1).values
        all_l1_max_probs.append(l1_max_prob.flatten())

        # L1 entropy
        l1_entropy = -(l1_probs * l1_log_probs).sum(dim=-1)
        all_l1_entropies.append(l1_entropy.flatten())

        # L4 (final) analysis
        l4_output = outputs[-1]
        l4_log_probs = torch.log_softmax(l4_output, dim=-1)
        l4_token_losses = -l4_log_probs.gather(dim=-1, index=y.unsqueeze(-1)).squeeze(-1)
        all_l4_losses.append(l4_token_losses.flatten())

        l4_preds = l4_output.argmax(dim=-1)
        l4_correct = (l4_preds == y).float()
        all_l4_correct.append(l4_correct.flatten())

    return {
        'l1_losses': torch.cat(all_l1_losses).numpy(),
        'l4_losses': torch.cat(all_l4_losses).numpy(),
        'l1_correct': torch.cat(all_l1_correct).numpy(),
        'l4_correct': torch.cat(all_l4_correct).numpy(),
        'l1_max_probs': torch.cat(all_l1_max_probs).numpy(),
        'l1_entropies': torch.cat(all_l1_entropies).numpy(),
    }


def analyze_early_exit_decisions(data):
    """Analyze potential early exit decisions."""
    l1_losses = data['l1_losses']
    l4_losses = data['l4_losses']
    l1_correct = data['l1_correct']
    l4_correct = data['l4_correct']
    l1_max_probs = data['l1_max_probs']
    l1_entropies = data['l1_entropies']

    total_tokens = len(l1_losses)

    print("\n" + "=" * 70)
    print("Early Exit Feasibility Analysis")
    print("=" * 70)

    # 1. Correlation between L1 and L4 losses
    corr = np.corrcoef(l1_losses, l4_losses)[0, 1]
    print(f"\n1. Correlation between L1 loss and L4 loss: {corr:.4f}")

    # 2. L1 accuracy vs L4 accuracy
    l1_acc = l1_correct.mean() * 100
    l4_acc = l4_correct.mean() * 100
    print(f"\n2. Overall Accuracy:")
    print(f"   L1: {l1_acc:.1f}%")
    print(f"   L4: {l4_acc:.1f}%")

    # 3. Can L1 confidence predict L4 correctness?
    print(f"\n3. L1 Confidence vs L4 Correctness:")

    # Define confidence thresholds
    thresholds = [0.5, 0.7, 0.8, 0.9, 0.95, 0.99]
    print(f"\n   Using L1 max probability as confidence:")
    print(f"   {'Threshold':>10} {'Tokens':>10} {'L1 Acc':>10} {'L4 Acc':>10} {'L1==L4':>10}")
    print("   " + "-" * 52)

    for thresh in thresholds:
        mask = l1_max_probs >= thresh
        n_tokens = mask.sum()
        if n_tokens > 0:
            l1_acc_subset = l1_correct[mask].mean() * 100
            l4_acc_subset = l4_correct[mask].mean() * 100
            # Agreement: both L1 and L4 give same prediction
            agreement = ((l1_correct == l4_correct) & mask).sum() / n_tokens * 100
            print(f"   {thresh:>10.2f} {n_tokens:>10} {l1_acc_subset:>10.1f}% {l4_acc_subset:>10.1f}% {agreement:>10.1f}%")

    # 4. Using entropy as confidence
    print(f"\n   Using L1 entropy as confidence (lower = more confident):")
    entropy_thresholds = [0.5, 1.0, 1.5, 2.0, 2.5]
    print(f"   {'Max Entropy':>12} {'Tokens':>10} {'L1 Acc':>10} {'L4 Acc':>10} {'L1==L4':>10}")
    print("   " + "-" * 54)

    for thresh in entropy_thresholds:
        mask = l1_entropies <= thresh
        n_tokens = mask.sum()
        if n_tokens > 0:
            l1_acc_subset = l1_correct[mask].mean() * 100
            l4_acc_subset = l4_correct[mask].mean() * 100
            agreement = ((l1_correct == l4_correct) & mask).sum() / n_tokens * 100
            print(f"   {thresh:>12.1f} {n_tokens:>10} {l1_acc_subset:>10.1f}% {l4_acc_subset:>10.1f}% {agreement:>10.1f}%")

    # 5. Early Exit Decision Analysis
    print(f"\n4. Early Exit Decision Analysis:")
    print(f"   Question: If we exit early when L1 is confident, how often is this correct?")

    # Define "exit early" as L1 max_prob >= threshold
    # Define "correct decision" as:
    #   - Exit early + L1 correct = Good (saved compute, correct answer)
    #   - Exit early + L1 wrong = Bad (saved compute, wrong answer)
    #   - Continue + L4 correct = Good (used compute, correct answer)
    #   - Continue + L4 wrong = Neutral (used compute, still wrong)

    print(f"\n   {'Threshold':>10} {'Exit %':>10} {'Exit Correct':>12} {'Continue':>10} {'Overall Acc':>12}")
    print("   " + "-" * 56)

    best_threshold = None
    best_score = 0

    for thresh in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]:
        exit_mask = l1_max_probs >= thresh
        continue_mask = ~exit_mask

        n_exit = exit_mask.sum()
        n_continue = continue_mask.sum()

        exit_correct = l1_correct[exit_mask].sum() if n_exit > 0 else 0
        continue_correct = l4_correct[continue_mask].sum() if n_continue > 0 else 0

        total_correct = exit_correct + continue_correct
        overall_acc = total_correct / total_tokens * 100

        exit_pct = n_exit / total_tokens * 100
        exit_acc = (exit_correct / n_exit * 100) if n_exit > 0 else 0

        print(f"   {thresh:>10.2f} {exit_pct:>10.1f}% {exit_acc:>12.1f}% {n_continue:>10} {overall_acc:>12.1f}%")

        # Track best threshold based on overall accuracy
        if overall_acc > best_score:
            best_score = overall_acc
            best_threshold = thresh

    # 6. Compute savings vs accuracy tradeoff
    print(f"\n5. Compute Savings vs Accuracy Tradeoff:")
    print(f"   Baseline (always use L4): {l4_acc:.1f}% accuracy, 100% compute")

    print(f"\n   {'Strategy':>30} {'Accuracy':>12} {'Compute':>12}")
    print("   " + "-" * 56)

    # Always L1
    print(f"   {'Always L1':>30} {l1_acc:>12.1f}% {25:>12}%")

    # Always L4
    print(f"   {'Always L4':>30} {l4_acc:>12.1f}% {100:>12}%")

    # Adaptive with best threshold
    if best_threshold:
        exit_mask = l1_max_probs >= best_threshold
        n_exit = exit_mask.sum()
        exit_correct = l1_correct[exit_mask].sum()
        continue_correct = l4_correct[~exit_mask].sum()
        adaptive_acc = (exit_correct + continue_correct) / total_tokens * 100

        # Compute savings: exit tokens use 1/4 compute, continue use 4/4
        compute_used = (n_exit * 0.25 + (total_tokens - n_exit) * 1.0) / total_tokens * 100

        print(f"   {f'Adaptive (thresh={best_threshold:.2f})':>30} {adaptive_acc:>12.1f}% {compute_used:>12.1f}%")

    # 7. The key question: Can L1 identify easy tokens?
    print(f"\n6. Key Question: Can L1 identify tokens that L4 will get correct?")

    # L4 correct tokens
    l4_correct_mask = l4_correct == 1
    l4_wrong_mask = l4_correct == 0

    l1_conf_when_l4_correct = l1_max_probs[l4_correct_mask].mean()
    l1_conf_when_l4_wrong = l1_max_probs[l4_wrong_mask].mean()

    print(f"   L1 confidence when L4 is correct: {l1_conf_when_l4_correct:.4f}")
    print(f"   L1 confidence when L4 is wrong:   {l1_conf_when_l4_wrong:.4f}")
    print(f"   Difference: {l1_conf_when_l4_correct - l1_conf_when_l4_wrong:.4f}")

    # Is L1 already correct for tokens that L4 gets correct?
    l1_correct_when_l4_correct = l1_correct[l4_correct_mask].mean() * 100
    print(f"\n   L1 accuracy on tokens L4 gets correct: {l1_correct_when_l4_correct:.1f}%")

    return {
        'correlation': corr,
        'l1_acc': l1_acc,
        'l4_acc': l4_acc,
        'l1_conf_l4_correct': l1_conf_when_l4_correct,
        'l1_conf_l4_wrong': l1_conf_when_l4_wrong,
        'l1_acc_on_l4_correct': l1_correct_when_l4_correct,
    }


def create_visualization(data, save_path='early_exit_analysis.png'):
    """Create visualization."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. L1 loss vs L4 loss scatter
    ax1 = axes[0, 0]
    sample_idx = np.random.choice(len(data['l1_losses']), min(1000, len(data['l1_losses'])), replace=False)
    ax1.scatter(data['l1_losses'][sample_idx], data['l4_losses'][sample_idx], alpha=0.3, s=10)
    ax1.set_xlabel('L1 Loss')
    ax1.set_ylabel('L4 Loss')
    ax1.set_title('L1 Loss vs L4 Loss (sampled)')
    ax1.plot([0, 10], [0, 10], 'r--', alpha=0.5, label='y=x')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. L1 confidence distribution by L4 correctness
    ax2 = axes[0, 1]
    l4_correct_mask = data['l4_correct'] == 1
    ax2.hist(data['l1_max_probs'][l4_correct_mask], bins=50, alpha=0.5, label='L4 Correct', density=True)
    ax2.hist(data['l1_max_probs'][~l4_correct_mask], bins=50, alpha=0.5, label='L4 Wrong', density=True)
    ax2.set_xlabel('L1 Max Probability (Confidence)')
    ax2.set_ylabel('Density')
    ax2.set_title('L1 Confidence Distribution by L4 Correctness')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Accuracy vs Exit Threshold
    ax3 = axes[1, 0]
    thresholds = np.arange(0.1, 1.0, 0.05)
    exit_rates = []
    overall_accs = []
    l1_only_accs = []

    total_tokens = len(data['l1_losses'])

    for thresh in thresholds:
        exit_mask = data['l1_max_probs'] >= thresh
        n_exit = exit_mask.sum()

        exit_correct = data['l1_correct'][exit_mask].sum() if n_exit > 0 else 0
        continue_correct = data['l4_correct'][~exit_mask].sum() if (total_tokens - n_exit) > 0 else 0

        overall_acc = (exit_correct + continue_correct) / total_tokens * 100
        exit_rate = n_exit / total_tokens * 100

        exit_rates.append(exit_rate)
        overall_accs.append(overall_acc)

    ax3.plot(thresholds, overall_accs, 'b-', label='Adaptive Accuracy', linewidth=2)
    ax3.axhline(y=data['l4_correct'].mean() * 100, color='r', linestyle='--', label='L4 Only')
    ax3.axhline(y=data['l1_correct'].mean() * 100, color='g', linestyle='--', label='L1 Only')
    ax3.set_xlabel('Exit Threshold (L1 max prob)')
    ax3.set_ylabel('Accuracy (%)')
    ax3.set_title('Accuracy vs Exit Threshold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Exit rate vs Threshold
    ax4 = axes[1, 1]
    ax4.plot(thresholds, exit_rates, 'b-', linewidth=2)
    ax4.set_xlabel('Exit Threshold (L1 max prob)')
    ax4.set_ylabel('Exit Rate (%)')
    ax4.set_title('Early Exit Rate vs Threshold')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\nVisualization saved to: {save_path}")
    plt.close()


def main():
    set_seed(42)

    print("=" * 70)
    print("Experiment: Early Exit Feasibility Analysis")
    print("=" * 70)
    print("\nQuestion: Can we identify 'easy' tokens at L1 and exit early?")

    # Prepare data and model
    train_batches, val_batches, vocab_size = prepare_data()
    print(f"\nVocab size: {vocab_size}")

    model = AnalyzableTransformer(
        vocab_size=vocab_size,
        dim=64,
        seq_len=64,
        num_layers=4,
        num_heads=4
    )

    print("\nTraining model...")
    train_model(model, train_batches, val_batches, vocab_size)

    print("\nAnalyzing early exit feasibility...")
    data = analyze_early_exit(model, val_batches, vocab_size)
    results = analyze_early_exit_decisions(data)

    create_visualization(data)

    # Summary
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)

    conf_diff = results['l1_conf_l4_correct'] - results['l1_conf_l4_wrong']

    if conf_diff > 0.1:
        print("\nEarly Exit is FEASIBLE:")
        print(f"  - L1 confidence is higher for tokens L4 gets correct")
        print(f"  - Confidence difference: {conf_diff:.4f}")
        print(f"  - L1 already gets {results['l1_acc_on_l4_correct']:.1f}% of L4-correct tokens right")
    elif conf_diff > 0.05:
        print("\nEarly Exit is MARGINALLY FEASIBLE:")
        print(f"  - L1 confidence is slightly higher for easy tokens")
        print(f"  - May achieve modest compute savings with small accuracy loss")
    else:
        print("\nEarly Exit is DIFFICULT:")
        print(f"  - L1 confidence does not reliably predict L4 correctness")
        print(f"  - Confidence difference: {conf_diff:.4f}")
        print(f"  - L1 cannot distinguish easy from hard tokens well")


if __name__ == "__main__":
    main()
