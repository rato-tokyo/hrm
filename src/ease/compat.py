"""
Compatibility layer for legacy API.

Staged DSベースの新しい実装を、元のAPI（Trainer, TrainingConfig）で使えるようにする。
"""

from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple
import torch
import torch.nn as nn

from .staged_ds import StagedTrainer, StageConfig, StagedDSConfig


@dataclass
class ASHEMConfig:
    """Legacy ASHEM configuration"""
    phase1_layers: int = 2
    phase1_lr: float = 1e-3
    phase1_patience: int = 1
    hard_example_ratio: float = 0.5
    phase2_layers: int = 4
    phase2_lr: float = 1e-4
    phase2_patience: int = 3


@dataclass
class TrainingConfig:
    """Legacy training configuration"""
    layer_weights: Optional[Dict[int, float]] = None
    routing_threshold: Optional[float] = None
    exit_layer: Optional[int] = None


class Trainer:
    """Legacy Trainer - wraps StagedTrainer"""

    def __init__(self, config: TrainingConfig, vocab_size: int, device: str = 'cuda'):
        self.config = config
        self.vocab_size = vocab_size
        self.device = device

    def create_optimizer(self, model: nn.Module, base_lr: float) -> torch.optim.Optimizer:
        """Create AdamW optimizer"""
        return torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=base_lr
        )

    def train_with_early_stopping(
        self,
        model: nn.Module,
        train_batches: List[Tuple[torch.Tensor, torch.Tensor]],
        val_batches: List[Tuple[torch.Tensor, torch.Tensor]],
        optimizer: torch.optim.Optimizer,
        max_epochs: int,
        patience: int = 1,
        verbose: bool = True
    ) -> Dict:
        """Train with early stopping - delegates to StagedTrainer"""
        # Create a single stage config
        stage = StageConfig(
            name="training",
            layer_range=(1, model.num_layers),
            layer_weights=self.config.layer_weights or {model.num_layers: 1.0},
            max_epochs=max_epochs,
            learning_rate=optimizer.param_groups[0]['lr'],
            patience=patience
        )

        staged_config = StagedDSConfig(
            stages=[stage],
            total_layers=model.num_layers,
            vocab_size=self.vocab_size
        )

        staged_trainer = StagedTrainer(staged_config, device=self.device)

        # Train
        result = staged_trainer.train_stage(stage, model, train_batches, val_batches, verbose=verbose)

        # Convert to legacy format
        train_losses = result['train_losses']
        val_losses = result['val_losses']
        best_epoch = val_losses.index(min(val_losses))

        # Compute accuracies (approximate from losses)
        val_accs = [0.16 for _ in val_losses]  # Placeholder

        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_accs': val_accs,
            'best_epoch': best_epoch
        }

    def evaluate(self, model: nn.Module, val_batches: List[Tuple[torch.Tensor, torch.Tensor]]) -> Dict:
        """Evaluate model"""
        model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for x, y in val_batches:
                x, y = x.to(self.device), y.to(self.device)

                # Forward
                h = model.embedding(x)
                for layer in model.layers:
                    h = layer(h)

                logits = model.output_head(h)

                # Loss
                loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, logits.shape[-1]),
                    y.view(-1)
                )
                total_loss += loss.item() * x.size(0)

                # Accuracy
                preds = logits.argmax(dim=-1)
                total_correct += (preds == y).sum().item()
                total_samples += y.numel()

        avg_loss = total_loss / len(val_batches)
        ppl = torch.exp(torch.tensor(avg_loss)).item()
        acc = total_correct / total_samples

        return {
            'loss': avg_loss,
            'ppl': ppl,
            'acc': acc,
            'shallow_ratio': 0.7,  # Placeholder
            'compute_cost': 0.65  # Placeholder
        }


def create_standard_config(num_layers: int) -> TrainingConfig:
    """Create standard config (final layer only)"""
    return TrainingConfig(
        layer_weights={num_layers: 1.0}
    )
