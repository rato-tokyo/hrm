"""
Maze Task Training: HRM (Hierarchical Reasoning Model)

Train HRM to find shortest paths in mazes.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import random
import numpy as np

from datasets import MazeDataset
from modules import HRM


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def compute_maze_accuracy(preds: torch.Tensor, targets: torch.Tensor) -> dict:
    """
    Compute maze-specific accuracy metrics.

    Args:
        preds: Predicted path mask [batch, seq_len] (binary)
        targets: Ground truth path mask [batch, seq_len] (binary)

    Returns:
        dict with cell_acc (per-cell), path_acc (exact match), path_f1
    """
    # Cell-level accuracy
    cell_correct = (preds == targets).float().mean().item()

    # Exact path match
    exact_match = (preds == targets).all(dim=-1).float().mean().item()

    # F1 score for path cells
    tp = ((preds == 1) & (targets == 1)).float().sum(dim=-1)
    fp = ((preds == 1) & (targets == 0)).float().sum(dim=-1)
    fn = ((preds == 0) & (targets == 1)).float().sum(dim=-1)

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    f1_mean = f1.mean().item()

    return {
        'cell_acc': cell_correct,
        'exact_match': exact_match,
        'path_f1': f1_mean
    }


@torch.no_grad()
def evaluate_hrm(model: HRM, test_loader: DataLoader, num_segments: int, device: str) -> dict:
    """Evaluate HRM model on maze task"""
    model.eval()

    all_preds = []
    all_targets = []

    for x, y in test_loader:
        x, y = x.to(device), y.to(device)

        y_hat, _ = model(x, num_segments=num_segments)
        preds = (y_hat.squeeze(-1) > 0).long()

        all_preds.append(preds)
        all_targets.append(y)

    all_preds_cat = torch.cat(all_preds, dim=0)
    all_targets_cat = torch.cat(all_targets, dim=0)

    return compute_maze_accuracy(all_preds_cat, all_targets_cat)


def train_hrm(
    model: HRM,
    train_loader: DataLoader,
    test_loader: DataLoader,
    num_epochs: int = 50,
    num_segments: int = 2,
    lr: float = 1e-3,
    device: str = 'cpu'
) -> list:
    """Train HRM model with deep supervision"""
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    results = []

    print(f"\nTraining HRM (Deep Supervision) for {num_epochs} epochs...")

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device).float()
            z_H, z_L = None, None

            for _ in range(num_segments):
                z_H, z_L, y_hat, _ = model.forward_pass(x, z_H, z_L)

                output = y_hat.squeeze(-1)
                loss = nn.functional.binary_cross_entropy_with_logits(output, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                z_H = z_H.detach()
                z_L = z_L.detach()

                total_loss += loss.item()
                num_batches += 1

        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / num_batches
            metrics = evaluate_hrm(model, test_loader, num_segments, device)
            print(f"  Epoch {epoch+1:3d} | Loss: {avg_loss:.4f} | "
                  f"Cell: {metrics['cell_acc']:.4f} | Exact: {metrics['exact_match']:.4f} | F1: {metrics['path_f1']:.4f}")
            results.append(metrics)

    return results


def main() -> None:
    set_seed(42)

    # Config
    grid_size = 10
    min_path_length = 10
    dim = 64
    num_heads = 4
    batch_size = 32
    device = 'cpu'

    # HRM specific
    num_layers = 2
    N = 2
    T = 4
    hrm_epochs = 50
    num_segments = 2

    vocab_size = 4
    seq_len = grid_size * grid_size

    print(f"Device: {device}")
    print(f"Maze Config: {grid_size}x{grid_size}, min_path_length={min_path_length}")
    print(f"Model Config: dim={dim}, num_heads={num_heads}")

    # Dataset
    print("\nGenerating maze datasets...")
    train_data = MazeDataset(
        num_samples=500,
        grid_size=grid_size,
        min_path_length=min_path_length,
        seed=42
    )
    test_data = MazeDataset(
        num_samples=100,
        grid_size=grid_size,
        min_path_length=min_path_length,
        seed=123
    )

    # Visualize a sample
    print("\nSample maze:")
    print(train_data.visualize(0))

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    # Train HRM
    print("\n" + "="*60)
    print("HRM (Hierarchical Reasoning Model)")
    print("="*60)

    hrm_model = HRM(
        vocab_size=vocab_size,
        dim=dim,
        num_layers=num_layers,
        num_heads=num_heads,
        seq_len=seq_len,
        N=N,
        T=T
    ).to(device)

    # Modify output head for binary classification
    hrm_model.output_head = nn.Linear(dim, 1, bias=False).to(device)

    param_count = sum(p.numel() for p in hrm_model.parameters())
    print(f"HRM Parameters: {param_count:,}")

    train_hrm(
        hrm_model, train_loader, test_loader,
        num_epochs=hrm_epochs,
        num_segments=num_segments,
        device=device
    )

    hrm_final = evaluate_hrm(hrm_model, test_loader, num_segments, device)

    # Final Results
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"{'Metric':<20} {'Value':<12}")
    print("-"*32)
    print(f"{'Cell Accuracy':<20} {hrm_final['cell_acc']:.4f}")
    print(f"{'Exact Match':<20} {hrm_final['exact_match']:.4f}")
    print(f"{'Path F1':<20} {hrm_final['path_f1']:.4f}")


if __name__ == "__main__":
    main()
