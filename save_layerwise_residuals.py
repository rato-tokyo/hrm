"""
Save Layer-wise Residuals for Saturation Analysis

残差接続の変化量（delta）とtop-1変化の相関を分析するためのデータ収集。

各層で保存するデータ:
- h_in: 層への入力
- h_out: 層からの出力
- delta = h_out - h_in: 残差（Attention + FFN の出力）

Output: layerwise_residuals.npz
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path

from lego import (
    TrainerConfig,
    set_seed,
    get_device,
    create_wikitext_dataloaders,
)
from lego.modules.transformer import TransformerLayer


class ResidualTransformer(nn.Module):
    """Transformer that outputs residuals at each layer."""

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

    def forward_with_residuals(
        self, x: torch.Tensor
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], torch.Tensor]:
        """
        Returns:
            h_ins: list of (batch, seq, dim) - 各層への入力
            h_outs: list of (batch, seq, dim) - 各層からの出力
            deltas: list of (batch, seq, dim) - 残差 (h_out - h_in)
            final_logits: (batch, seq, vocab_size)
        """
        h = self.embedding(x)

        h_ins = []
        h_outs = []
        deltas = []

        for layer in self.layers:
            h_in = h
            h_out = layer(h)
            delta = h_out - h_in

            h_ins.append(h_in)
            h_outs.append(h_out)
            deltas.append(delta)

            h = h_out

        logits = self.output_head(h)
        return h_ins, h_outs, deltas, logits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward for training."""
        h = self.embedding(x)
        for layer in self.layers:
            h = layer(h)
        return self.output_head(h)


def main():
    # Model configuration (small model for fast convergence)
    dim = 64
    num_heads = 4
    ffn_dim = 256
    num_layers = 4
    max_seq_len = 1024
    seq_len = 32
    num_samples = 10000

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
    print("Save Layer-wise Residuals for Saturation Analysis")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Config: dim={dim}, heads={num_heads}, ffn={ffn_dim}, layers={num_layers}")

    # Setup
    set_seed(42)
    train_batches, val_batches, vocab_size = create_wikitext_dataloaders(
        num_samples, trainer_config.batch_size, seq_len, seed=42
    )

    # Create model
    model = ResidualTransformer(
        vocab_size=vocab_size,
        dim=dim,
        num_heads=num_heads,
        num_layers=num_layers,
        ffn_dim=ffn_dim,
        max_seq_len=max_seq_len,
        causal=True,
        eps=1e-6,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {num_params:,} parameters ({num_params / 1e6:.1f}M)")

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
            logits = model(x)

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
                logits = model(x)

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

    # Collect layer-wise residuals on validation set
    print("\n" + "=" * 60)
    print("Collecting Layer-wise Residuals")
    print("=" * 60)

    model.eval()

    # Initialize lists for each layer
    all_h_ins: list[list[torch.Tensor]] = [[] for _ in range(num_layers)]
    all_h_outs: list[list[torch.Tensor]] = [[] for _ in range(num_layers)]
    all_deltas: list[list[torch.Tensor]] = [[] for _ in range(num_layers)]
    all_targets: list[torch.Tensor] = []

    with torch.no_grad():
        for x, y in val_batches:
            x, y = x.to(device), y.to(device)
            h_ins, h_outs, deltas, logits = model.forward_with_residuals(x)

            # Store
            for i in range(num_layers):
                all_h_ins[i].append(h_ins[i].cpu())
                all_h_outs[i].append(h_outs[i].cpu())
                all_deltas[i].append(deltas[i].cpu())
            all_targets.append(y.cpu())

    # Concatenate and flatten
    print("\nProcessing collected data...")

    save_dict = {
        'dim': dim,
        'vocab_size': vocab_size,
        'num_layers': num_layers,
    }

    for i in range(num_layers):
        h_in_cat = torch.cat(all_h_ins[i]).view(-1, dim).numpy().astype(np.float32)
        h_out_cat = torch.cat(all_h_outs[i]).view(-1, dim).numpy().astype(np.float32)
        delta_cat = torch.cat(all_deltas[i]).view(-1, dim).numpy().astype(np.float32)

        save_dict[f'layer{i+1}_h_in'] = h_in_cat
        save_dict[f'layer{i+1}_h_out'] = h_out_cat
        save_dict[f'layer{i+1}_delta'] = delta_cat

        delta_norm = np.linalg.norm(delta_cat, axis=-1)
        print(f"Layer {i + 1}:")
        print(f"  h_in: {h_in_cat.shape}, norm mean={np.linalg.norm(h_in_cat, axis=-1).mean():.2f}")
        print(f"  h_out: {h_out_cat.shape}, norm mean={np.linalg.norm(h_out_cat, axis=-1).mean():.2f}")
        print(f"  delta: {delta_cat.shape}, norm mean={delta_norm.mean():.4f}, std={delta_norm.std():.4f}")

    targets_flat = torch.cat(all_targets).view(-1).numpy().astype(np.int32)
    save_dict['targets'] = targets_flat
    print(f"\nTargets: {targets_flat.shape}")

    # Get output head weight
    W = model.output_head.weight.detach().cpu().numpy().astype(np.float32)
    save_dict['output_head_W'] = W
    print(f"Output head W: {W.shape} (vocab_size x dim)")

    # Save
    output_path = Path("layerwise_residuals.npz")
    total_size = sum(v.nbytes for v in save_dict.values() if isinstance(v, np.ndarray))
    print(f"\nSaving to {output_path} ({total_size / (1024 * 1024):.1f} MB)...")

    np.savez(output_path, **save_dict)

    print(f"Saved: {output_path}")
    print("Done!")


if __name__ == "__main__":
    main()
