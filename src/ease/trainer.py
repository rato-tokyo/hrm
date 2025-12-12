"""
EASE Framework - Training Configuration

Simple training framework with two base models and three options:

Base Models:
- Standard: Final layer loss only
- Deep Supervision: Loss at all layers

Options:
- layer_weights: Layer-wise loss weights
- layer_lr_scales: Layer-wise learning rates (Discriminative Fine-Tuning)
- routing_threshold: Early Exit at inference

References:
- Deep Supervision: Lee et al., 2015
- Discriminative Fine-Tuning: Howard & Ruder, 2018
- Early Exit: Teerapittayanon et al., 2016
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class TrainingConfig:
    """
    Training configuration.

    Args:
        layer_weights: Loss weight for each layer (1-indexed).
                      e.g., {1: 0.7, 2: 0, 3: 0.3} for asymmetric loss
        layer_lr_scales: Learning rate scale for each layer (optional).
                        e.g., {1: 1.0, 2: 0.5, 3: 0.1} for decreasing LR
        routing_threshold: Confidence threshold for early exit (0 = disabled).
        exit_layer: Which layer to use for early exit (1-indexed).
    """
    layer_weights: Dict[int, float]
    layer_lr_scales: Optional[Dict[int, float]] = None
    routing_threshold: float = 0.0
    exit_layer: int = 1

    @property
    def has_routing(self) -> bool:
        """Returns True if early exit is enabled."""
        return self.routing_threshold > 0

    @property
    def has_layer_lr(self) -> bool:
        """Returns True if layer-wise LR is configured."""
        return self.layer_lr_scales is not None

    def describe(self) -> str:
        """Human-readable description."""
        weights_str = ", ".join([f"L{k}:{v:.2f}" for k, v in sorted(self.layer_weights.items())])
        desc = f"Weights: [{weights_str}]"

        if self.has_routing:
            desc += f", Early Exit: threshold={self.routing_threshold}"

        if self.has_layer_lr and self.layer_lr_scales is not None:
            lr_str = ", ".join([f"L{k}:{v:.2f}x" for k, v in sorted(self.layer_lr_scales.items())])
            desc += f", LR scales: [{lr_str}]"

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
    Trainer for EASE models.

    Supports:
    - Layer-wise loss weighting
    - Layer-wise learning rates
    - Early exit evaluation
    """

    def __init__(self, config: TrainingConfig, vocab_size: int, device: str = 'cpu'):
        self.config = config
        self.vocab_size = vocab_size
        self.device = device

    def compute_loss(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute weighted loss across layers."""
        all_outputs = model.forward_all_layers(x)
        num_layers = len(all_outputs)

        total_loss: torch.Tensor = torch.tensor(0.0, device=x.device)
        active_weights = 0.0

        for layer_idx, weight in self.config.layer_weights.items():
            if weight > 0 and layer_idx <= num_layers:
                output = all_outputs[layer_idx - 1]
                layer_loss = F.cross_entropy(
                    output.view(-1, self.vocab_size),
                    y.view(-1)
                )
                total_loss = total_loss + weight * layer_loss
                active_weights += weight

        # Fallback to final layer if no weights
        if active_weights == 0:
            output = all_outputs[-1]
            total_loss = F.cross_entropy(output.view(-1, self.vocab_size), y.view(-1))

        return total_loss

    def create_optimizer(self, model: nn.Module, base_lr: float) -> torch.optim.Optimizer:
        """Create optimizer with optional layer-wise learning rates."""
        if not self.config.has_layer_lr:
            return torch.optim.AdamW(model.parameters(), lr=base_lr)

        param_groups = []
        layer_lr_scales = self.config.layer_lr_scales or {}

        # Get layers from model
        if hasattr(model, 'layers'):
            layers = model.layers
        elif hasattr(model, 'transformer_layers'):
            layers = model.transformer_layers
        else:
            return torch.optim.AdamW(model.parameters(), lr=base_lr)

        assigned_params = set()
        for i, layer in enumerate(layers):
            layer_idx = i + 1
            lr_scale = layer_lr_scales.get(layer_idx, 1.0)
            layer_params = list(layer.parameters())
            for p in layer_params:
                assigned_params.add(id(p))
            param_groups.append({
                'params': layer_params,
                'lr': base_lr * lr_scale,
                'name': f'layer_{layer_idx}'
            })

        # Other parameters (embedding, output head)
        other_params = [p for p in model.parameters() if id(p) not in assigned_params]
        if other_params:
            param_groups.append({
                'params': other_params,
                'lr': base_lr,
                'name': 'other'
            })

        return torch.optim.AdamW(param_groups)

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
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_batches)
