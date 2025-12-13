"""
LEGO Framework - Trainer

Training and evaluation for LEGO models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Callable, Dict, List, Optional, Tuple, Any

from .utils import compute_routing_cost


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

    def _forward_with_routing(
        self,
        model: nn.Module,
        x: torch.Tensor,
        routing_threshold: float
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Forward pass with routing statistics (for evaluation).

        Computes both shallow and deep outputs, then routes based on confidence.
        This is used for evaluation metrics only - for actual inference,
        use model.generate() for true computation savings.
        """
        batch_size, seq_len = x.shape
        exit_layer = getattr(model, 'exit_layer', model.num_layers)

        h = model.embedding(x)

        # Process up to exit layer
        for i in range(exit_layer):
            h = model.layers[i](h)

        # Shallow output and confidence (using model's compute_confidence)
        shallow_logits, confidence = model.compute_confidence(h)

        # Continue to deep output
        h_deep = h
        for i in range(exit_layer, model.num_layers):
            h_deep = model.layers[i](h_deep)
        deep_logits = model.output_head(h_deep)

        # Hard routing
        mask = (confidence >= routing_threshold).unsqueeze(-1)
        output = torch.where(mask, shallow_logits, deep_logits)

        # Compute statistics
        shallow_count = int(mask.sum().item())
        total_count = batch_size * seq_len
        deep_count = total_count - shallow_count

        cost = compute_routing_cost(shallow_count, deep_count, exit_layer, model.num_layers)

        stats = {
            'mean_confidence': confidence.mean().item(),
            'shallow_ratio': shallow_count / total_count,
            'compute_cost': cost,
        }

        return output, stats

    @torch.no_grad()
    def evaluate(
        self,
        model: nn.Module,
        val_batches: List[Tuple[torch.Tensor, torch.Tensor]],
        routing_threshold: float = 0.0
    ) -> Dict[str, float]:
        """Evaluate model with optional early exit routing."""
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
                logits, stats = self._forward_with_routing(model, x, routing_threshold)
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
        val_acc: float,
        hard_ppl: Optional[float],
        is_best: bool,
        patience_counter: int,
        patience: int
    ) -> None:
        """Log training progress for one epoch."""
        msg = (f"Epoch {epoch+1}/{max_epochs} - "
               f"Train PPL: {train_ppl:.4f} | "
               f"Val PPL: {val_ppl:.2f} | "
               f"Val Acc: {val_acc*100:.2f}%")
        if hard_ppl is not None:
            msg += f" | Hard PPL: {hard_ppl:.2f}"
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
        hard_ppls: List[float],
        best_epoch: int,
        best_val_ppl: float,
        total_epochs: int,
        stopped_early: bool
    ) -> Dict[str, Any]:
        """Build training result dictionary."""
        result: Dict[str, Any] = {
            'train_ppls': train_ppls,
            'val_ppls': val_ppls,
            'val_accs': val_accs,
            'best_epoch': best_epoch,
            'best_val_ppl': best_val_ppl,
            'total_epochs': total_epochs,
            'stopped_early': stopped_early
        }
        if hard_ppls:
            result['hard_ppls'] = hard_ppls
        return result

    def _early_stopping_loop(
        self,
        model: nn.Module,
        val_batches: List[Tuple[torch.Tensor, torch.Tensor]],
        train_fn: Callable[[], float],
        max_epochs: int,
        patience: int,
        verbose: bool,
        routing_threshold: float = 0.0,
        extra_eval_fn: Optional[Callable[[], float]] = None
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
            # Train and evaluate
            train_ppl = float(np.exp(train_fn()))
            train_ppls.append(train_ppl)

            val_stats = self.evaluate(model, val_batches, routing_threshold)
            val_ppl, val_acc = val_stats['ppl'], val_stats['acc']
            val_ppls.append(val_ppl)
            val_accs.append(val_acc)

            hard_ppl = extra_eval_fn() if extra_eval_fn else None
            if hard_ppl is not None:
                hard_ppls.append(hard_ppl)

            # Check for improvement
            is_best = val_ppl < best_val_ppl
            if is_best:
                best_val_ppl = val_ppl
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
                best_epoch = epoch
            else:
                patience_counter += 1

            if verbose:
                self._log_epoch(epoch, max_epochs, train_ppl, val_ppl, val_acc,
                               hard_ppl, is_best, patience_counter, patience)

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
            train_ppls, val_ppls, val_accs, hard_ppls,
            best_epoch, best_val_ppl, epoch + 1, patience_counter >= patience
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
        """Train with early stopping."""
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
        max_epochs: int = 50,
        patience: int = 3,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """Train upper layers on hard examples with early stopping."""
        from .utils import train_upper_layers, evaluate_on_hard_examples

        def train_fn() -> float:
            return train_upper_layers(
                model, hard_batches, optimizer,
                self.device, num_lower_layers
            )

        def extra_eval_fn() -> float:
            return evaluate_on_hard_examples(
                model, hard_examples, self.device,
                batch_size=64, num_lower_layers=num_lower_layers
            )

        return self._early_stopping_loop(
            model, val_batches, train_fn, max_epochs, patience, verbose,
            routing_threshold, extra_eval_fn
        )
