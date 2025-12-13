"""
LEGO Framework - Trainer

Training and evaluation for LEGO models.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Callable, Dict, List, Optional, Tuple, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .data import TrainingData

class Trainer:
    """
    Trainer for LEGO models.

    Args:
        vocab_size: Vocabulary size
        device: Device to run on ('cpu' or 'cuda')
    """

    def __init__(self, vocab_size: int, device: str = 'cpu'):
        self.vocab_size = vocab_size
        self.device = device

    @torch.no_grad()
    def evaluate(
        self,
        model: nn.Module,
        val_batches: List[Tuple[torch.Tensor, torch.Tensor]],
        use_routing: bool = False
    ) -> Dict[str, float]:
        """Evaluate model with optional early exit routing.

        Args:
            model: Model to evaluate
            val_batches: Validation data batches
            use_routing: If True, use model's block thresholds for routing
        """
        model.eval()

        total_loss = 0.0
        total_correct: float = 0
        total_tokens = 0
        total_shallow = 0.0
        total_compute = 0.0

        for x, y in val_batches:
            x, y = x.to(self.device), y.to(self.device)

            if use_routing:
                # Use model's forward_with_routing method (uses block thresholds)
                logits, stats = model.forward_with_routing(x)
                total_shallow += stats['shallow_ratio'] * x.numel()
                total_compute += stats['compute_cost']
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

    def _log_epoch(
        self,
        epoch: int,
        max_epochs: int,
        train_ppl: float,
        val_ppl: float,
        val_acc: Optional[float],
        is_best: bool,
        patience_counter: int,
        patience: int
    ) -> None:
        """Log training progress for one epoch."""
        msg = f"Epoch {epoch+1}/{max_epochs} - Train PPL: {train_ppl:.4f} | Val PPL: {val_ppl:.2f}"
        if val_acc is not None:
            msg += f" | Val Acc: {val_acc*100:.2f}%"
        print(msg)

        if is_best:
            print(f"  → New best (val_ppl: {val_ppl:.2f})")
        else:
            print(f"  → No improvement ({patience_counter}/{patience})")

    def _build_training_result(
        self,
        train_ppls: List[float],
        val_ppls: List[float],
        val_accs: List[float],
        best_epoch: int,
        best_val_ppl: float,
        total_epochs: int,
        stopped_early: bool
    ) -> Dict[str, Any]:
        """Build training result dictionary."""
        return {
            'train_ppls': train_ppls,
            'val_ppls': val_ppls,
            'val_accs': val_accs,
            'best_epoch': best_epoch,
            'best_val_ppl': best_val_ppl,
            'total_epochs': total_epochs,
            'stopped_early': stopped_early
        }

    def _early_stopping_loop(
        self,
        model: nn.Module,
        train_fn: Callable[[], float],
        val_fn: Callable[[], Tuple[float, Optional[float]]],
        max_epochs: int,
        patience: int,
        verbose: bool
    ) -> Dict[str, Any]:
        """Core early stopping loop shared by training methods.

        Args:
            train_fn: Returns average training loss
            val_fn: Returns (val_ppl, val_acc or None)
        """
        best_ppl = float('inf')
        best_model_state = None
        patience_counter = 0
        best_epoch = 0

        train_ppls: List[float] = []
        val_ppls: List[float] = []
        val_accs: List[float] = []

        for epoch in range(max_epochs):
            # Train
            train_ppl = float(np.exp(train_fn()))
            train_ppls.append(train_ppl)

            # Validate
            val_ppl, val_acc = val_fn()
            val_ppls.append(val_ppl)
            if val_acc is not None:
                val_accs.append(val_acc)

            # Check for improvement
            is_best = val_ppl < best_ppl
            if is_best:
                best_ppl = val_ppl
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
                best_epoch = epoch
            else:
                patience_counter += 1

            if verbose:
                self._log_epoch(epoch, max_epochs, train_ppl, val_ppl, val_acc,
                               is_best, patience_counter, patience)

            # Early stopping check
            if patience_counter >= patience:
                if verbose:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                    print(f"Best model was at epoch {best_epoch+1}")
                break

        # Restore best model
        if best_model_state is not None:
            model.load_state_dict({k: v.to(self.device) for k, v in best_model_state.items()})
            if verbose:
                print(f"\nRestored best model from epoch {best_epoch+1}")

        return self._build_training_result(
            train_ppls, val_ppls, val_accs,
            best_epoch, best_ppl, epoch + 1, patience_counter >= patience
        )

    def train_with_early_stopping(
        self,
        model: nn.Module,
        train_batches: List[Tuple[torch.Tensor, torch.Tensor]],
        val_batches: List[Tuple[torch.Tensor, torch.Tensor]],
        optimizer: torch.optim.Optimizer,
        max_epochs: int = 100,
        patience: int = 5,
        grad_clip: float = 1.0,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """Phase 1: Train model on full data with early stopping."""
        def train_fn() -> float:
            model.train()
            total_loss = 0.0
            for x, y in train_batches:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                output = model(x)
                loss = F.cross_entropy(output.view(-1, self.vocab_size), y.view(-1))
                loss.backward()  # type: ignore[no-untyped-call]
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                total_loss += loss.item()
            return total_loss / len(train_batches)

        def val_fn() -> Tuple[float, Optional[float]]:
            stats = self.evaluate(model, val_batches)
            return stats['ppl'], stats['acc']

        return self._early_stopping_loop(
            model, train_fn, val_fn, max_epochs, patience, verbose
        )

    def train_block(
        self,
        model: nn.Module,
        train_data: "TrainingData",
        val_data: "TrainingData",
        optimizer: torch.optim.Optimizer,
        start_block_idx: int,
        batch_size: int = 64,
        max_epochs: int = 50,
        patience: int = 3,
        grad_clip: float = 1.0,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """Phase 2+: Train new block on hard examples only.

        Both training and validation are done within hard examples.
        This ensures each block's training is independent.

        Note: Consider using LEGOBlock.train_block() instead for simpler API.

        Args:
            model: Extended model with new block
            train_data: TrainingData for training
            val_data: TrainingData for validation
            optimizer: Optimizer for trainable parameters
            start_block_idx: Index of the new block to train
            batch_size: Batch size for training/validation
            max_epochs: Maximum training epochs
            patience: Early stopping patience
            grad_clip: Gradient clipping value
            verbose: Print progress
        """
        train_batches = train_data.batches(batch_size)

        def train_fn() -> float:
            model.train()
            total_loss = 0.0
            for h, y in train_batches:
                h, y = h.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                logits = model.forward_from_block(h, start_block_idx).squeeze(1)
                loss = F.cross_entropy(logits, y)
                loss.backward()  # type: ignore[no-untyped-call]
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                total_loss += loss.item()
            return total_loss / len(train_batches)

        def val_fn() -> Tuple[float, Optional[float]]:
            model.eval()
            total_loss = 0.0
            total_samples = 0
            val_batches = val_data.batches(batch_size, shuffle=False)
            with torch.no_grad():
                for h, y in val_batches:
                    h, y = h.to(self.device), y.to(self.device)
                    logits = model.forward_from_block(h, start_block_idx).squeeze(1)
                    loss = F.cross_entropy(logits, y, reduction='sum')
                    total_loss += loss.item()
                    total_samples += len(y)
            avg_loss = total_loss / total_samples
            ppl = float(np.exp(avg_loss))
            return ppl, None  # No accuracy for hard example training

        return self._early_stopping_loop(
            model, train_fn, val_fn, max_epochs, patience, verbose
        )
