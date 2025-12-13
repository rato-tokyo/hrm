"""
LEGO Framework - Trainer

Training and evaluation for LEGO models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Any


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

    def compute_loss(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute cross-entropy loss."""
        output = model(x)
        return F.cross_entropy(output.view(-1, self.vocab_size), y.view(-1))

    def create_optimizer(self, model: nn.Module, base_lr: float) -> torch.optim.Optimizer:
        """Create optimizer with uniform learning rate."""
        return torch.optim.AdamW(model.parameters(), lr=base_lr)

    @torch.no_grad()
    def evaluate(
        self,
        model: nn.Module,
        val_batches: List[Tuple[torch.Tensor, torch.Tensor]],
        routing_threshold: float = 0.0,
        exit_layer: int = 1
    ) -> Dict[str, float]:
        """
        Evaluate model with optional early exit routing.

        Args:
            model: Model to evaluate
            val_batches: Validation data batches
            routing_threshold: Confidence threshold for early exit (0 = disabled)
            exit_layer: Layer to use for early exit
        """
        model.eval()

        total_loss = 0.0
        total_correct: float = 0
        total_tokens = 0
        total_shallow = 0.0
        total_compute = 0.0

        use_routing = routing_threshold > 0

        for x, y in val_batches:
            x, y = x.to(self.device), y.to(self.device)

            if use_routing:
                outputs = model.forward_train(x)
                confidence = outputs['confidence']
                mask = (confidence >= routing_threshold).unsqueeze(-1)
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

    def train_epoch(
        self,
        model: nn.Module,
        train_batches: List[Tuple[torch.Tensor, torch.Tensor]],
        optimizer: torch.optim.Optimizer,
        grad_clip: float = 1.0
    ) -> float:
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
        train_fn: Any,
        max_epochs: int,
        patience: int,
        verbose: bool,
        routing_threshold: float = 0.0,
        exit_layer: int = 1,
        extra_eval_fn: Any = None
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

            val_stats = self.evaluate(model, val_batches, routing_threshold, exit_layer)
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
        routing_threshold: float = 0.0,
        exit_layer: int = 1,
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
            model, val_batches, train_fn, max_epochs, patience, verbose,
            routing_threshold, exit_layer, extra_eval_fn
        )
        # Rename for backward compatibility
        result['train_ppls'] = result.pop('train_losses')
        result['val_ppls'] = result.pop('val_losses')
        return result

    def collect_hard_examples(
        self,
        model: nn.Module,
        val_batches: List[Tuple[torch.Tensor, torch.Tensor]],
        target_ratio: float = 0.5,
        batch_size: int = 64
    ) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor]], Dict[str, torch.Tensor], float]:
        """
        Collect hard examples in one step.

        Combines threshold computation, hard example collection, and batching.

        Args:
            model: Trained model
            val_batches: Validation data batches
            target_ratio: Ratio of examples to classify as hard (default: 0.5)
            batch_size: Batch size for hard example loader

        Returns:
            Tuple of (hard_batches, hard_examples, threshold)
        """
        from .utils import (
            compute_confidence_threshold,
            collect_hard_examples as _collect_hard_examples,
            create_hard_example_loader,
        )

        threshold = compute_confidence_threshold(model, val_batches, target_ratio, self.device)
        hard_examples = _collect_hard_examples(model, val_batches, threshold, self.device)
        hard_batches = create_hard_example_loader(hard_examples, batch_size)

        return hard_batches, hard_examples, threshold
