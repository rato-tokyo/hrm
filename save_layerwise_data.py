"""
Save Layer-wise Hidden States for Saturation Analysis

Saves hidden states after each layer (1, 2, 3, 4) along with:
- per_token_loss
- output_head weight matrix W (for low-rank experiments)

Output: layerwise_hidden_states.npz
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path

from lego import (
    ExperimentConfig,
    TrainerConfig,
    set_seed,
    get_device,
    create_wikitext_dataloaders,
)
from lego.modules.transformer import TransformerLayer


class LayerwiseTransformer(nn.Module):
    """Transformer that outputs hidden states at each layer."""

    def __init__(
        self,
        vocab_size: int,
        dim: int,
        num_heads: int,
        num_layers: int,
        ffn_dim: int,
        max_seq_len: int,
        causal: bool,
        eps: float,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([
            TransformerLayer(dim, num_heads, ffn_dim, max_seq_len, causal, eps)
            for _ in range(num_layers)
        ])
        self.output_head = nn.Linear(dim, vocab_size, bias=False)
        self.dim = dim
        self.num_layers = num_layers

    def forward(self, x: torch.Tensor) -> tuple[list[torch.Tensor], torch.Tensor]:
        """
        Returns:
            layer_hidden_states: list of (batch, seq, dim) for each layer
            final_logits: (batch, seq, vocab_size)
        """
        h = self.embedding(x)
        layer_hidden_states = []

        for layer in self.layers:
            h = layer(h)
            layer_hidden_states.append(h)

        logits = self.output_head(h)
        return layer_hidden_states, logits


def main():
    config = ExperimentConfig(
        dim=64,
        num_heads=4,
        ffn_dim=256,
        max_seq_len=1024,
        causal=True,
        eps=1e-6,
        seq_len=32,
        num_samples=10000,
        block_layers=(4,),  # 4 layers
        exit_hidden_dim=128,
    )

    trainer_config = TrainerConfig(
        batch_size=64,
        max_epochs=50,
        patience=3,
        grad_clip=1.0,
        val_ratio=0.2,
        hard_ratio=0.5,
        lr=1e-3,
        verbose=True,
    )

    device = get_device()

    print("=" * 60)
    print("Save Layer-wise Hidden States (4-layer model)")
    print("=" * 60)
    print(f"Device: {device}")

    # Setup
    set_seed(42)
    train_batches, val_batches, vocab_size = create_wikitext_dataloaders(
        config.num_samples, trainer_config.batch_size, config.seq_len, seed=42
    )

    # Create model
    model = LayerwiseTransformer(
        vocab_size=vocab_size,
        dim=config.dim,
        num_heads=config.num_heads,
        num_layers=4,
        ffn_dim=config.ffn_dim,
        max_seq_len=config.max_seq_len,
        causal=config.causal,
        eps=config.eps,
    ).to(device)

    print(f"Model: {sum(p.numel() for p in model.parameters()):,} parameters")

    # Train model
    print("\n" + "=" * 60)
    print("Training Model")
    print("=" * 60)

    optimizer = torch.optim.AdamW(model.parameters(), lr=trainer_config.lr)
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(trainer_config.max_epochs):
        # Train
        model.train()
        train_loss = 0.0
        train_tokens = 0

        for x, y in train_batches:
            x, y = x.to(device), y.to(device)
            _, logits = model(x)

            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                y.view(-1),
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), trainer_config.grad_clip)
            optimizer.step()

            train_loss += loss.item() * y.numel()
            train_tokens += y.numel()

        # Validate
        model.eval()
        val_loss = 0.0
        val_tokens = 0

        with torch.no_grad():
            for x, y in val_batches:
                x, y = x.to(device), y.to(device)
                _, logits = model(x)

                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    y.view(-1),
                )
                val_loss += loss.item() * y.numel()
                val_tokens += y.numel()

        train_ppl = np.exp(train_loss / train_tokens)
        val_ppl = np.exp(val_loss / val_tokens)

        print(f"Epoch {epoch + 1}: train_ppl={train_ppl:.1f}, val_ppl={val_ppl:.1f}")

        if val_loss / val_tokens < best_val_loss:
            best_val_loss = val_loss / val_tokens
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= trainer_config.patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    print(f"\nBest val PPL: {np.exp(best_val_loss):.1f}")

    # Collect layer-wise hidden states on validation set
    print("\n" + "=" * 60)
    print("Collecting Layer-wise Hidden States")
    print("=" * 60)

    model.eval()

    # Initialize lists for each layer
    all_layer_hidden: list[list[torch.Tensor]] = [[] for _ in range(4)]
    all_targets: list[torch.Tensor] = []
    all_per_token_loss: list[torch.Tensor] = []

    with torch.no_grad():
        for x, y in val_batches:
            x, y = x.to(device), y.to(device)
            layer_hidden_states, logits = model(x)

            # Per-token loss
            per_token_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                y.view(-1),
                reduction='none'
            ).view(x.size(0), x.size(1))

            # Store
            for i, h in enumerate(layer_hidden_states):
                all_layer_hidden[i].append(h.cpu())
            all_targets.append(y.cpu())
            all_per_token_loss.append(per_token_loss.cpu())

    # Concatenate and flatten
    layer_hidden_flat = []
    for i in range(4):
        h_cat = torch.cat(all_layer_hidden[i])  # (num_seq, seq_len, dim)
        h_flat = h_cat.view(-1, config.dim).numpy()  # (num_tokens, dim)
        layer_hidden_flat.append(h_flat)
        print(f"Layer {i + 1} hidden states: {h_flat.shape}")

    targets_flat = torch.cat(all_targets).view(-1).numpy()
    per_token_loss_flat = torch.cat(all_per_token_loss).view(-1).numpy()

    print(f"Targets: {targets_flat.shape}")
    print(f"Per-token loss: {per_token_loss_flat.shape}")
    print(f"Loss: mean={per_token_loss_flat.mean():.4f}, std={per_token_loss_flat.std():.4f}")

    # Get output head weight
    W = model.output_head.weight.detach().cpu().numpy()
    print(f"\nOutput head W: {W.shape} (vocab_size x dim)")

    # Save
    output_path = Path("layerwise_hidden_states.npz")
    print(f"\nSaving to {output_path}...")

    np.savez(
        output_path,
        layer1_hidden=layer_hidden_flat[0].astype(np.float32),
        layer2_hidden=layer_hidden_flat[1].astype(np.float32),
        layer3_hidden=layer_hidden_flat[2].astype(np.float32),
        layer4_hidden=layer_hidden_flat[3].astype(np.float32),
        targets=targets_flat.astype(np.int32),
        per_token_loss=per_token_loss_flat.astype(np.float32),
        output_head_W=W.astype(np.float32),
        dim=config.dim,
        vocab_size=vocab_size,
        num_layers=4,
    )

    print(f"Saved: {output_path}")
    print("Done!")


if __name__ == "__main__":
    main()
