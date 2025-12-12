"""
LASH Framework - Training Configuration

Layered Adaptive Supervision Hierarchy: 層を組み合わせる柔軟なフレームワーク

Base Models:
- Standard: Final layer loss only
- Deep Supervision: Loss at all layers

Training Strategies:
1. Standard: Final layer loss only
2. Deep Supervision: Loss at all layers
3. ASHEM: Hard example mining with selective layer expansion

Core Options (2つのコアオプション):
- layer_weights: Layer-wise loss weights
- routing_threshold: Early Exit at inference

References:
- LASH: Layered Adaptive Supervision Hierarchy
- Deep Supervision: Lee et al., 2015
- Early Exit: Teerapittayanon et al., 2016
- ASHEM: Adaptive Supervision via Hard Example Mining
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass


@dataclass
class TrainingConfig:
    """
    Training configuration for LASH framework.

    LASH (Layered Adaptive Supervision Hierarchy) uses 2 core options:
    1. layer_weights: Which layers to train (loss weights)
    2. routing_threshold: When to exit early (inference efficiency)

    Args:
        layer_weights: Loss weight for each layer (1-indexed).
                      e.g., {1: 0.7, 2: 0, 3: 0.3} for asymmetric loss
        routing_threshold: Confidence threshold for early exit (0 = disabled).
        exit_layer: Which layer to use for early exit (1-indexed).
    """
    layer_weights: Dict[int, float]
    routing_threshold: float = 0.0
    exit_layer: int = 1

    @property
    def has_routing(self) -> bool:
        """Returns True if early exit is enabled."""
        return self.routing_threshold > 0

    def describe(self) -> str:
        """Human-readable description."""
        weights_str = ", ".join([f"L{k}:{v:.2f}" for k, v in sorted(self.layer_weights.items())])
        desc = f"Weights: [{weights_str}]"

        if self.has_routing:
            desc += f", Early Exit: threshold={self.routing_threshold}"

        return desc


def create_standard_config(num_layers: int = 3) -> TrainingConfig:
    """Create Standard LLM config (final layer loss only)."""
    weights = {i: 0.0 for i in range(1, num_layers + 1)}
    weights[num_layers] = 1.0
    return TrainingConfig(layer_weights=weights)


def create_deep_supervision_config(num_layers: int = 3) -> TrainingConfig:
    """Create Deep Supervision config (equal loss on all layers)."""
    weight = 1.0 / num_layers
    weights = {i: weight for i in range(1, num_layers + 1)}
    return TrainingConfig(layer_weights=weights)


class Trainer:
    """
    Trainer for LASH models.

    Supports LASH's 2 core options:
    - layer_weights: Layer-wise loss weighting
    - routing_threshold: Early exit evaluation
    """

    def __init__(self, config: TrainingConfig, vocab_size: int, device: str = 'cpu'):
        self.config = config
        self.vocab_size = vocab_size
        self.device = device

    def compute_loss(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute weighted loss across layers (optimized).

        Optimization: Use fast path (forward()) when only final layer is needed.
        Maintains full compatibility with both core options:
        - layer_weights: Determines which layers to compute
        - routing_threshold: Independent (used in evaluation only)
        """
        # Determine which layers need loss computation
        num_layers = model.num_layers if hasattr(model, 'num_layers') else len(model.layers)
        active_layers = [(idx, weight) for idx, weight in self.config.layer_weights.items()
                        if weight > 0 and idx <= num_layers]

        # Fast path: Only final layer needed (Standard Transformer pattern)
        if len(active_layers) == 1 and active_layers[0][0] == num_layers:
            output = model(x)  # Use forward() instead of forward_all_layers()
            weight = active_layers[0][1]
            loss = F.cross_entropy(output.view(-1, self.vocab_size), y.view(-1))
            return weight * loss if weight != 1.0 else loss

        # Fallback: No active layers, use final layer
        if not active_layers:
            output = model(x)
            return F.cross_entropy(output.view(-1, self.vocab_size), y.view(-1))

        # General path: Multiple layers or non-final layer (Deep Supervision pattern)
        all_outputs = model.forward_all_layers(x)
        total_loss: torch.Tensor = torch.tensor(0.0, device=x.device)

        for layer_idx, weight in active_layers:
            output = all_outputs[layer_idx - 1]
            layer_loss = F.cross_entropy(
                output.view(-1, self.vocab_size),
                y.view(-1)
            )
            total_loss = total_loss + weight * layer_loss

        return total_loss

    def create_optimizer(self, model: nn.Module, base_lr: float) -> torch.optim.Optimizer:
        """Create optimizer with uniform learning rate."""
        return torch.optim.AdamW(model.parameters(), lr=base_lr)

    @torch.no_grad()
    def evaluate(self, model: nn.Module, val_batches: List[Tuple[torch.Tensor, torch.Tensor]]) -> Dict[str, float]:
        """Evaluate model with optional early exit."""
        model.eval()

        if not self.config.has_routing:
            return self._evaluate_standard(model, val_batches)
        else:
            return self._evaluate_routing(model, val_batches)

    def _evaluate_standard(self, model: nn.Module, val_batches: List[Tuple[torch.Tensor, torch.Tensor]]) -> Dict[str, float]:
        """Standard evaluation using final layer output."""
        total_loss = 0.0
        total_correct = 0
        total_tokens = 0

        for x, y in val_batches:
            x, y = x.to(self.device), y.to(self.device)
            output = model(x)
            loss = F.cross_entropy(output.view(-1, self.vocab_size), y.view(-1), reduction='sum')
            preds = output.argmax(dim=-1)

            total_loss += loss.item()
            total_correct += (preds == y).sum().item()
            total_tokens += y.numel()

        avg_loss = total_loss / total_tokens
        return {
            'ppl': float(np.exp(avg_loss)),
            'acc': total_correct / total_tokens,
            'shallow_ratio': 0.0,
            'compute_cost': 1.0,
        }

    def _evaluate_routing(self, model: nn.Module, val_batches: List[Tuple[torch.Tensor, torch.Tensor]]) -> Dict[str, float]:
        """Evaluation with early exit routing."""
        total_loss = 0.0
        total_correct = 0
        total_tokens = 0
        total_shallow = 0.0
        total_compute = 0.0

        threshold = self.config.routing_threshold
        exit_layer = self.config.exit_layer

        for x, y in val_batches:
            x, y = x.to(self.device), y.to(self.device)

            outputs = model.forward_train(x)
            shallow_logits = outputs['shallow_logits']
            deep_logits = outputs['deep_logits']
            confidence = outputs['confidence']

            mask = (confidence >= threshold).unsqueeze(-1)
            routed_logits = torch.where(mask, shallow_logits, deep_logits)

            loss = F.cross_entropy(routed_logits.view(-1, self.vocab_size), y.view(-1), reduction='sum')
            preds = routed_logits.argmax(dim=-1)

            total_loss += loss.item()
            total_correct += int((preds == y).sum().item())
            total_tokens += y.numel()

            batch_size, seq_len = x.shape
            total_count = batch_size * seq_len
            shallow_count = mask.sum().item()
            deep_count = total_count - shallow_count

            total_shallow += shallow_count
            num_layers = model.num_layers
            compute = (shallow_count * exit_layer + deep_count * num_layers) / (total_count * num_layers)
            total_compute += compute

        total_all_tokens = sum(x.shape[0] * x.shape[1] for x, _ in val_batches)
        avg_loss = total_loss / total_tokens

        return {
            'ppl': float(np.exp(avg_loss)),
            'acc': total_correct / total_tokens,
            'shallow_ratio': total_shallow / total_all_tokens,
            'compute_cost': total_compute / len(val_batches),
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
        """
        Train with early stopping.

        Args:
            model: Model to train
            train_batches: Training data batches
            val_batches: Validation data batches
            optimizer: Optimizer
            max_epochs: Maximum number of epochs
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            grad_clip: Gradient clipping value
            verbose: Print progress

        Returns:
            Dictionary containing training history and best model state
        """
        best_val_loss = float('inf')
        best_epoch = 0
        patience_counter = 0
        best_model_state = None

        train_losses = []
        val_losses = []
        val_accs = []

        for epoch in range(max_epochs):
            # Training
            train_loss = self.train_epoch(model, train_batches, optimizer, grad_clip)
            train_losses.append(train_loss)

            # Validation
            val_stats = self.evaluate(model, val_batches)
            val_ppl = val_stats['ppl']
            val_acc = val_stats['acc']
            val_loss = np.log(val_ppl)  # Convert PPL back to loss

            val_losses.append(val_ppl)
            val_accs.append(val_acc)

            if verbose:
                train_ppl = np.exp(train_loss)
                print(f"Epoch {epoch+1}/{max_epochs} - "
                      f"Train PPL: {train_ppl:.4f} | "
                      f"Val PPL: {val_ppl:.4f} | "
                      f"Val Acc: {val_acc*100:.2f}%")

            # Early stopping check
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                best_epoch = epoch
                patience_counter = 0
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                if verbose:
                    print(f"  → New best model (val_loss: {val_loss:.4f})")
            else:
                patience_counter += 1
                if verbose and patience_counter > 0:
                    print(f"  → No improvement ({patience_counter}/{patience})")

            # Check if should stop
            if patience_counter >= patience:
                if verbose:
                    print(f"\nEarly stopping triggered at epoch {epoch+1}")
                    print(f"Best model was at epoch {best_epoch+1}")
                break

        # Restore best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            if verbose:
                print(f"\nRestored best model from epoch {best_epoch+1}")

        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_accs': val_accs,
            'best_epoch': best_epoch,
            'total_epochs': epoch + 1,
            'stopped_early': patience_counter >= patience
        }
