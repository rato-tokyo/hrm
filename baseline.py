"""
Baseline - Standard 4-layer Transformer LLM

For comparison with LEGO Framework.
Same total layers (4), same architecture, but no early exit.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from lego import (
    TransformerBlock,
    ExperimentConfig,
    set_seed,
    get_device,
    create_wikitext_dataloaders,
)


class BaselineLLM(nn.Module):
    """Standard Transformer LLM without early exit."""

    def __init__(
        self,
        vocab_size: int,
        dim: int,
        transformer: TransformerBlock,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        self.transformer = transformer
        self.output_head = nn.Linear(dim, vocab_size, bias=False)
        # Weight tying
        self.output_head.weight = self.embedding.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input token ids (batch_size, seq_len)

        Returns:
            logits: (batch_size, seq_len, vocab_size)
        """
        h = self.embedding(x)
        h = self.transformer(h)
        return self.output_head(h)

    @property
    def num_layers(self) -> int:
        return self.transformer.num_layers


def main() -> None:
    """Run baseline training."""
    import numpy as np

    # Configuration - same as example.py
    config = ExperimentConfig(
        dim=64,
        num_heads=4,
        ffn_dim=256,
        max_seq_len=1024,
        causal=True,
        eps=1e-6,
        seq_len=32,
        num_samples=10000,
        block_layers=(2, 2),  # Total 4 layers
    )

    # Training config
    batch_size = 64
    max_epochs = 50
    patience = 3
    grad_clip = 1.0
    val_ratio = 0.2
    lr = 1e-3

    device = get_device()

    print("=" * 60)
    print("Baseline - Standard 4-layer Transformer LLM")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Model: dim={config.dim}, heads={config.num_heads}")
    print(f"Total layers: {sum(config.block_layers)}")

    # Setup
    set_seed(42)
    train_batches, val_batches, vocab_size = create_wikitext_dataloaders(
        config.num_samples, batch_size, config.seq_len, seed=42
    )

    # Create model with 4 layers (same as LEGO 2+2)
    total_layers = sum(config.block_layers)
    transformer = TransformerBlock(
        config.dim, config.num_heads, total_layers,
        config.ffn_dim, config.max_seq_len, config.causal, config.eps
    )
    model = BaselineLLM(vocab_size, config.dim, transformer).to(device)

    print(f"Layers: {model.num_layers}")

    # Training
    print(f"\n{'=' * 60}")
    print("Training")
    print("=" * 60)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    best_ppl = float('inf')
    best_state = None
    patience_counter = 0
    best_epoch = 0

    train_ppls = []
    val_ppls = []

    for epoch in range(max_epochs):
        # Training
        model.train()
        total_loss = 0.0
        total_tokens = 0

        for x, y in train_batches:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            logits = model(x)
            batch_size_actual, seq_len, vocab_size_actual = logits.shape
            loss = F.cross_entropy(
                logits.view(-1, vocab_size_actual),
                y.view(-1),
                reduction='sum'
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            total_loss += loss.item()
            total_tokens += batch_size_actual * seq_len

        train_ppl = float(np.exp(total_loss / total_tokens))
        train_ppls.append(train_ppl)

        # Validation
        model.eval()
        val_loss = 0.0
        val_tokens = 0

        with torch.no_grad():
            for x, y in val_batches:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                batch_size_actual, seq_len, vocab_size_actual = logits.shape
                loss = F.cross_entropy(
                    logits.view(-1, vocab_size_actual),
                    y.view(-1),
                    reduction='sum'
                )
                val_loss += loss.item()
                val_tokens += batch_size_actual * seq_len

        val_ppl = float(np.exp(val_loss / val_tokens))
        val_ppls.append(val_ppl)

        # Early stopping
        is_best = val_ppl < best_ppl
        if is_best:
            best_ppl = val_ppl
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            best_epoch = epoch
        else:
            patience_counter += 1

        status = "best" if is_best else f"{patience_counter}/{patience}"
        print(f"  Epoch {epoch+1}/{max_epochs}: train_ppl={train_ppl:.2f}, val_ppl={val_ppl:.2f} [{status}]")

        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    # Restore best model
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    print(f"\nBest PPL: {best_ppl:.2f} (epoch {best_epoch+1})")

    # Evaluation
    print(f"\n{'=' * 60}")
    print("Evaluation")
    print("=" * 60)

    model.eval()
    total_loss = 0.0
    total_tokens = 0
    correct = 0

    with torch.no_grad():
        for x, y in val_batches:
            x, y = x.to(device), y.to(device)
            logits = model(x)

            logits_flat = logits.view(-1, logits.size(-1))
            y_flat = y.view(-1)
            total_loss += F.cross_entropy(logits_flat, y_flat, reduction='sum').item()
            total_tokens += y_flat.numel()
            correct += (logits_flat.argmax(dim=-1) == y_flat).sum().item()

    ppl = float(np.exp(total_loss / total_tokens))
    acc = correct / total_tokens

    print(f"\nFinal Results:")
    print(f"  Accuracy: {acc*100:.2f}%")
    print(f"  PPL: {ppl:.2f}")
    print(f"  Compute cost: 100.0% (no early exit)")

    print("\n" + "=" * 60)
    print("Baseline completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
