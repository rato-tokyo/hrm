"""
LEGO Framework - Training Configuration

Training Strategies:
1. Standard: Final layer loss only
2. Hard Example Mining: 2-phase training with hard example focus

Core Options:
- routing_threshold: Early Exit at inference

References:
- LEGO: Layered Ensemble with Gradual Optimization
- Early Exit: Teerapittayanon et al., 2016
- Hard Example Mining: Similar to HAM (IEEE TIFS 2025), HSM (2025)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass


@dataclass
class TrainingConfig:
    """
    Training configuration for LEGO framework.

    Args:
        routing_threshold: Confidence threshold for early exit (0 = disabled)
        exit_layer: Which layer to use for early exit (1-indexed)
    """
    routing_threshold: float = 0.0
    exit_layer: int = 1

    @property
    def has_routing(self) -> bool:
        """Returns True if early exit is enabled."""
        return self.routing_threshold > 0


class Trainer:
    """
    Trainer for LEGO models.

    Supports routing_threshold for early exit evaluation.
    """

    def __init__(self, config: TrainingConfig, vocab_size: int, device: str = 'cpu'):
        self.config = config
        self.vocab_size = vocab_size
        self.device = device

    def compute_loss(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute loss at the specified output layer.

        Uses forward() for final layer (fast path).
        """
        output = model(x)
        return F.cross_entropy(output.view(-1, self.vocab_size), y.view(-1))

    def create_optimizer(self, model: nn.Module, base_lr: float) -> torch.optim.Optimizer:
        """Create optimizer with uniform learning rate."""
        return torch.optim.AdamW(model.parameters(), lr=base_lr)

    @torch.no_grad()
    def evaluate(self, model: nn.Module, val_batches: List[Tuple[torch.Tensor, torch.Tensor]]) -> Dict[str, float]:
        """Evaluate model with optional early exit routing."""
        model.eval()

        total_loss = 0.0
        total_correct = 0
        total_tokens = 0
        total_shallow = 0.0
        total_compute = 0.0

        use_routing = self.config.has_routing
        threshold = self.config.routing_threshold
        exit_layer = self.config.exit_layer

        for x, y in val_batches:
            x, y = x.to(self.device), y.to(self.device)

            if use_routing:
                outputs = model.forward_train(x)
                confidence = outputs['confidence']
                mask = (confidence >= threshold).unsqueeze(-1)
                logits = torch.where(mask, outputs['shallow_logits'], outputs['deep_logits'])

                batch_size, seq_len = x.shape
                total_count = batch_size * seq_len
                shallow_count = mask.sum().item()
                total_shallow += shallow_count
                deep_count = total_count - shallow_count
                num_layers = model.num_layers
                total_compute += (shallow_count * exit_layer + deep_count * num_layers) / (total_count * num_layers)
            else:
                logits = model(x)

            loss = F.cross_entropy(logits.view(-1, self.vocab_size), y.view(-1), reduction='sum')
            total_loss += loss.item()
            total_correct += (logits.argmax(dim=-1) == y).sum().item()
            total_tokens += y.numel()

        avg_loss = total_loss / total_tokens
        total_all_tokens = sum(x.shape[0] * x.shape[1] for x, _ in val_batches)

        return {
            'ppl': float(np.exp(avg_loss)),
            'acc': total_correct / total_tokens,
            'shallow_ratio': total_shallow / total_all_tokens if use_routing else 0.0,
            'compute_cost': total_compute / len(val_batches) if use_routing else 1.0,
        }

    def train_epoch(self, model: nn.Module, train_batches: List[Tuple[torch.Tensor, torch.Tensor]],
                    optimizer: torch.optim.Optimizer, grad_clip: float = 1.0) -> float:
        """Train for one epoch."""
        model.train()
        total_loss = 0.0

        for x, y in train_batches:
            x, y = x.to(self.device), y.to(self.device)
            optimizer.zero_grad()

            loss = self.compute_loss(model, x, y)
            loss.backward()  # type: ignore[no-untyped-call]

            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_batches)

    def _early_stopping_loop(
        self,
        model: nn.Module,
        val_batches: List[Tuple[torch.Tensor, torch.Tensor]],
        train_fn: Any,  # Callable[[], float]
        max_epochs: int,
        patience: int,
        verbose: bool,
        extra_eval_fn: Any = None  # Optional[Callable[[], float]]
    ) -> Dict[str, Any]:
        """Core early stopping loop shared by training methods."""
        best_val_ppl = float('inf')
        best_model_state = None
        patience_counter = 0
        best_epoch = 0

        train_ppls: List[float] = []
        val_ppls: List[float] = []
        val_accs: List[float] = []
        hard_ppls: List[float] = []

        for epoch in range(max_epochs):
            train_loss = train_fn()
            train_ppl = float(np.exp(train_loss))
            train_ppls.append(train_ppl)

            val_stats = self.evaluate(model, val_batches)
            val_ppl = val_stats['ppl']
            val_acc = val_stats['acc']
            val_ppls.append(val_ppl)
            val_accs.append(val_acc)

            hard_ppl = extra_eval_fn() if extra_eval_fn else None
            if hard_ppl is not None:
                hard_ppls.append(hard_ppl)

            if verbose:
                msg = (f"Epoch {epoch+1}/{max_epochs} - "
                       f"Train PPL: {train_ppl:.4f} | "
                       f"Val PPL: {val_ppl:.2f} | "
                       f"Val Acc: {val_acc*100:.2f}%")
                if hard_ppl is not None:
                    msg += f" | Hard PPL: {hard_ppl:.2f}"
                print(msg)

            if val_ppl < best_val_ppl:
                best_val_ppl = val_ppl
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
                best_epoch = epoch
                if verbose:
                    print(f"  → New best (val_ppl: {val_ppl:.2f})")
            else:
                patience_counter += 1
                if verbose:
                    print(f"  → No improvement ({patience_counter}/{patience})")

            if patience_counter >= patience:
                if verbose:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                    print(f"Best model was at epoch {best_epoch+1}")
                break

        if best_model_state is not None:
            model.load_state_dict({k: v.to(self.device) for k, v in best_model_state.items()})
            if verbose:
                print(f"\nRestored best model from epoch {best_epoch+1}")

        result: Dict[str, Any] = {
            'train_losses': train_ppls,
            'val_losses': val_ppls,
            'val_accs': val_accs,
            'best_epoch': best_epoch,
            'best_val_ppl': best_val_ppl,
            'total_epochs': epoch + 1,
            'stopped_early': patience_counter >= patience
        }
        if hard_ppls:
            result['hard_ppls'] = hard_ppls
        return result

    def train_with_early_stopping(
        self,
        model: nn.Module,
        train_batches: List[Tuple[torch.Tensor, torch.Tensor]],
        val_batches: List[Tuple[torch.Tensor, torch.Tensor]],
        optimizer: torch.optim.Optimizer,
        max_epochs: int = 100,
        patience: int = 5,
        min_delta: float = 0.0,
        grad_clip: float = 1.0,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """Train with early stopping."""
        def train_fn() -> float:
            return self.train_epoch(model, train_batches, optimizer, grad_clip)

        return self._early_stopping_loop(
            model, val_batches, train_fn, max_epochs, patience, verbose
        )

    def train_upper_layers_with_early_stopping(
        self,
        model: nn.Module,
        hard_batches: List[Tuple[torch.Tensor, torch.Tensor]],
        val_batches: List[Tuple[torch.Tensor, torch.Tensor]],
        hard_examples: Dict[str, torch.Tensor],
        optimizer: torch.optim.Optimizer,
        num_lower_layers: int,
        max_epochs: int = 50,
        patience: int = 3,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """Train upper layers on hard examples with early stopping."""
        from .utils import train_upper_layers, evaluate_on_hard_examples

        def train_fn() -> float:
            return train_upper_layers(
                model, hard_batches, optimizer,
                self.vocab_size, self.device, num_lower_layers
            )

        def extra_eval_fn() -> float:
            return evaluate_on_hard_examples(
                model, hard_examples, self.vocab_size, self.device,
                batch_size=64, num_lower_layers=num_lower_layers
            )

        result = self._early_stopping_loop(
            model, val_batches, train_fn, max_epochs, patience, verbose, extra_eval_fn
        )
        # Rename for backward compatibility
        result['train_ppls'] = result.pop('train_losses')
        result['val_ppls'] = result.pop('val_losses')
        return result
