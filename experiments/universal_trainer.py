"""
Universal Training Framework

すべての学習方法を統一的に表現できるフレームワーク。
各層への重み付けと推論時のRouting設定を分離して管理。

Examples:
    # Standard LLM (最終層のみ)
    weights = {1: 0, 2: 0, 3: 1}
    routing_threshold = 0  # 全トークンがDeep path

    # LPT (均等重み)
    weights = {1: 1/3, 2: 1/3, 3: 1/3}

    # Asymmetric (best)
    weights = {1: 0.7, 2: 0, 3: 0.3}
    routing_threshold = 0.95

    # Standard Routing
    weights = {1: 0.5, 2: 0, 3: 0.5}
    routing_threshold = 0.95

    # Dynamic alpha (curriculum learning)
    alpha_schedule = AlphaSchedule('linear', start=0.9, end=0.5)

    # Layer-wise learning rate
    layer_lr_scales = {1: 1.0, 2: 0.5, 3: 0.1}
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field


# =============================================================================
# Alpha Schedule (動的α)
# =============================================================================

@dataclass
class AlphaSchedule:
    """
    Dynamic alpha schedule for curriculum learning.

    Schedules:
        - 'constant': Fixed alpha throughout training
        - 'linear': Linear interpolation from start to end
        - 'cosine': Cosine annealing from start to end
        - 'step': Step decay at specified epochs

    Examples:
        # Constant (default behavior)
        schedule = AlphaSchedule('constant', start=0.7)

        # Linear decay: 0.9 -> 0.5 over training
        schedule = AlphaSchedule('linear', start=0.9, end=0.5)

        # Cosine annealing
        schedule = AlphaSchedule('cosine', start=0.9, end=0.5)

        # Step decay
        schedule = AlphaSchedule('step', start=0.9, end=0.5, steps=[10, 20, 30])
    """
    schedule_type: str = 'constant'
    start: float = 0.7
    end: float = 0.5
    steps: List[int] = field(default_factory=list)  # For step schedule

    def get_alpha(self, epoch: int, max_epochs: int) -> float:
        """Get alpha value for given epoch."""
        if self.schedule_type == 'constant':
            return self.start

        progress = min(1.0, epoch / max(1, max_epochs - 1))

        if self.schedule_type == 'linear':
            return self.start + (self.end - self.start) * progress

        elif self.schedule_type == 'cosine':
            return self.end + (self.start - self.end) * (1 + np.cos(np.pi * progress)) / 2

        elif self.schedule_type == 'step':
            alpha = self.start
            step_size = (self.start - self.end) / max(1, len(self.steps))
            for step_epoch in self.steps:
                if epoch >= step_epoch:
                    alpha -= step_size
            return max(self.end, alpha)

        else:
            raise ValueError(f"Unknown schedule type: {self.schedule_type}")

    def describe(self) -> str:
        if self.schedule_type == 'constant':
            return f"α={self.start} (constant)"
        elif self.schedule_type == 'step':
            return f"α: {self.start}→{self.end} (step at {self.steps})"
        else:
            return f"α: {self.start}→{self.end} ({self.schedule_type})"


@dataclass
class UniversalConfig:
    """Universal training configuration."""
    layer_weights: Dict[int, float]  # {layer_idx: weight} (1-indexed)
    exit_layer: int = 1              # Early exit layer for routing
    routing_threshold: float = 0.95  # Confidence threshold (0 = no routing)
    normalize_weights: bool = False  # Normalize weights to sum to 1

    # Dynamic alpha schedule (optional)
    alpha_schedule: Optional[AlphaSchedule] = None

    # Layer-wise learning rate scales (optional)
    # {layer_idx: lr_scale} where layer_idx is 1-indexed
    # Final LR = base_lr * lr_scale
    layer_lr_scales: Optional[Dict[int, float]] = None

    def __post_init__(self):
        if self.normalize_weights:
            total = sum(self.layer_weights.values())
            if total > 0:
                self.layer_weights = {k: v/total for k, v in self.layer_weights.items()}

    @property
    def has_routing(self) -> bool:
        """Returns True if routing is enabled (threshold > 0)."""
        return self.routing_threshold > 0

    @property
    def has_dynamic_alpha(self) -> bool:
        """Returns True if dynamic alpha is enabled."""
        return self.alpha_schedule is not None and self.alpha_schedule.schedule_type != 'constant'

    @property
    def has_layer_lr(self) -> bool:
        """Returns True if layer-wise LR is configured."""
        return self.layer_lr_scales is not None

    def get_layer_weights(self, epoch: int = 0, max_epochs: int = 1) -> Dict[int, float]:
        """
        Get layer weights for given epoch.
        If alpha_schedule is set, dynamically compute weights.
        """
        if self.alpha_schedule is None:
            return self.layer_weights

        alpha = self.alpha_schedule.get_alpha(epoch, max_epochs)

        # Apply dynamic alpha to asymmetric weights
        # Assumes layer_weights has format {1: α, 2: 0, 3: 1-α}
        new_weights = {}
        for layer_idx, weight in self.layer_weights.items():
            if layer_idx == 1:
                new_weights[layer_idx] = alpha
            elif layer_idx == max(self.layer_weights.keys()):
                new_weights[layer_idx] = 1 - alpha
            else:
                new_weights[layer_idx] = weight  # Keep intermediate layers unchanged
        return new_weights

    def describe(self) -> str:
        """Human-readable description."""
        weights_str = ", ".join([f"L{k}:{v:.2f}" for k, v in sorted(self.layer_weights.items())])
        routing_str = f"threshold={self.routing_threshold}" if self.has_routing else "disabled"
        desc = f"Weights: [{weights_str}], Routing: {routing_str}"

        if self.has_dynamic_alpha and self.alpha_schedule is not None:
            desc += f", {self.alpha_schedule.describe()}"

        if self.has_layer_lr and self.layer_lr_scales is not None:
            lr_str = ", ".join([f"L{k}:{v:.2f}x" for k, v in sorted(self.layer_lr_scales.items())])
            desc += f", LR scales: [{lr_str}]"

        return desc


# Preset configurations
PRESETS = {
    'standard_llm': UniversalConfig(
        layer_weights={1: 0, 2: 0, 3: 1},
        routing_threshold=0,  # No routing
    ),
    'lpt': UniversalConfig(
        layer_weights={1: 1/3, 2: 1/3, 3: 1/3},
        routing_threshold=0,  # No routing
    ),
    'lpt_routing': UniversalConfig(
        layer_weights={1: 1/3, 2: 1/3, 3: 1/3},
        routing_threshold=0.7,
    ),
    'standard_routing': UniversalConfig(
        layer_weights={1: 0.5, 2: 0, 3: 0.5},
        routing_threshold=0.95,
    ),
    'asymmetric_best': UniversalConfig(
        layer_weights={1: 0.7, 2: 0, 3: 0.3},
        routing_threshold=0.95,
    ),
    'asymmetric_with_l2': UniversalConfig(
        layer_weights={1: 0.7, 2: 1.0, 3: 0.3},
        routing_threshold=0.95,
    ),
    'deep_focus': UniversalConfig(
        layer_weights={1: 0.3, 2: 0, 3: 0.7},
        routing_threshold=0.95,
    ),
}


class UniversalTrainer:
    """
    Universal trainer that can reproduce any training method.

    Training: Loss = Σ weights[i] * L_i_loss
    Inference: Route based on L1 confidence if threshold > 0

    New features:
        - Dynamic alpha: Change layer weights during training
        - Layer-wise LR: Different learning rates per layer
    """

    def __init__(self, config: UniversalConfig, vocab_size: int, device: str = 'cpu'):
        self.config = config
        self.vocab_size = vocab_size
        self.device = device
        self.current_epoch = 0
        self.max_epochs = 1

    def set_training_state(self, epoch: int, max_epochs: int):
        """Set current training state for dynamic alpha."""
        self.current_epoch = epoch
        self.max_epochs = max_epochs

    def get_current_weights(self) -> Dict[int, float]:
        """Get layer weights for current epoch (supports dynamic alpha)."""
        return self.config.get_layer_weights(self.current_epoch, self.max_epochs)

    def compute_loss(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor,
                     weights: Optional[Dict[int, float]] = None) -> torch.Tensor:
        """
        Compute weighted loss across layers.

        Args:
            model: Model with forward_all_layers() method
            x: Input tensor
            y: Target tensor
            weights: Optional override for layer weights (for dynamic alpha)

        Returns:
            Weighted loss tensor
        """
        all_outputs = model.forward_all_layers(x)
        num_layers = len(all_outputs)

        # Use provided weights or get current weights (dynamic alpha support)
        layer_weights = weights if weights is not None else self.get_current_weights()

        total_loss: torch.Tensor = torch.tensor(0.0, device=x.device)
        active_weights = 0.0

        for layer_idx, weight in layer_weights.items():
            if weight > 0 and layer_idx <= num_layers:
                output = all_outputs[layer_idx - 1]  # Convert to 0-indexed
                layer_loss = F.cross_entropy(
                    output.view(-1, self.vocab_size),
                    y.view(-1)
                )
                total_loss = total_loss + weight * layer_loss
                active_weights += weight

        # Avoid division by zero
        if active_weights == 0:
            # Fallback to final layer only
            output = all_outputs[-1]
            total_loss = F.cross_entropy(output.view(-1, self.vocab_size), y.view(-1))

        return total_loss

    def create_optimizer(self, model: nn.Module, base_lr: float) -> torch.optim.Optimizer:
        """
        Create optimizer with optional layer-wise learning rates.

        Args:
            model: Model with layers attribute
            base_lr: Base learning rate

        Returns:
            Configured optimizer
        """
        if not self.config.has_layer_lr:
            # Standard optimizer
            return torch.optim.AdamW(model.parameters(), lr=base_lr)

        # Layer-wise learning rate
        param_groups = []
        layer_lr_scales = self.config.layer_lr_scales or {}

        # Get model layers
        if hasattr(model, 'layers'):
            layers = model.layers
        elif hasattr(model, 'transformer_layers'):
            layers = model.transformer_layers
        else:
            # Fallback to standard optimizer
            return torch.optim.AdamW(model.parameters(), lr=base_lr)

        # Collect layer parameters with scaled LR
        assigned_params = set()
        for i, layer in enumerate(layers):
            layer_idx = i + 1  # 1-indexed
            lr_scale = layer_lr_scales.get(layer_idx, 1.0)
            layer_params = list(layer.parameters())
            for p in layer_params:
                assigned_params.add(id(p))
            param_groups.append({
                'params': layer_params,
                'lr': base_lr * lr_scale,
                'name': f'layer_{layer_idx}'
            })

        # Add remaining parameters (embeddings, output heads, etc.)
        other_params = [p for p in model.parameters() if id(p) not in assigned_params]
        if other_params:
            param_groups.append({
                'params': other_params,
                'lr': base_lr,
                'name': 'other'
            })

        return torch.optim.AdamW(param_groups)

    @torch.no_grad()
    def evaluate(self, model: nn.Module, val_batches: List) -> Dict:
        """
        Evaluate model with optional routing.

        Returns:
            Dictionary with ppl, acc, shallow_ratio, compute_cost
        """
        model.eval()

        if not self.config.has_routing:
            # Standard evaluation (no routing)
            return self._evaluate_standard(model, val_batches)
        else:
            # Routing evaluation
            return self._evaluate_routing(model, val_batches)

    def _evaluate_standard(self, model: nn.Module, val_batches: List) -> Dict:
        """Standard evaluation using final layer output."""
        total_loss = 0.0
        total_correct = 0
        total_tokens = 0

        for x, y in val_batches:
            x, y = x.to(self.device), y.to(self.device)
            output = model(x)  # Final layer output
            loss = F.cross_entropy(output.view(-1, self.vocab_size), y.view(-1))
            preds = output.argmax(dim=-1)

            total_loss += loss.item()
            total_correct += (preds == y).sum().item()
            total_tokens += y.numel()

        n = len(val_batches)
        return {
            'ppl': np.exp(total_loss / n),
            'acc': total_correct / total_tokens,
            'shallow_ratio': 0.0,
            'compute_cost': 1.0,
        }

    def _evaluate_routing(self, model: nn.Module, val_batches: List) -> Dict:
        """Routing evaluation based on confidence."""
        total_loss = 0.0
        total_correct = 0
        total_tokens = 0
        total_shallow = 0
        total_compute = 0.0

        threshold = self.config.routing_threshold
        exit_layer = self.config.exit_layer

        for x, y in val_batches:
            x, y = x.to(self.device), y.to(self.device)

            # Get outputs for routing decision
            outputs = model.forward_train(x)
            shallow_logits = outputs['shallow_logits']
            deep_logits = outputs['deep_logits']
            confidence = outputs['confidence']

            # Route based on confidence
            mask = (confidence >= threshold).unsqueeze(-1)
            routed_logits = torch.where(mask, shallow_logits, deep_logits)

            # Compute loss and accuracy
            loss = F.cross_entropy(routed_logits.view(-1, self.vocab_size), y.view(-1))
            preds = routed_logits.argmax(dim=-1)

            total_loss += loss.item()
            total_correct += (preds == y).sum().item()
            total_tokens += y.numel()

            # Compute routing statistics
            batch_size, seq_len = x.shape
            total_count = batch_size * seq_len
            shallow_count = mask.sum().item()
            deep_count = total_count - shallow_count

            total_shallow += shallow_count
            num_layers = model.num_layers
            compute = (shallow_count * exit_layer + deep_count * num_layers) / (total_count * num_layers)
            total_compute += compute

        n = len(val_batches)
        total_all_tokens = sum(x.shape[0] * x.shape[1] for x, _ in val_batches)

        return {
            'ppl': np.exp(total_loss / n),
            'acc': total_correct / total_tokens,
            'shallow_ratio': total_shallow / total_all_tokens,
            'compute_cost': total_compute / n,
        }

    def train_epoch(self, model: nn.Module, train_batches: List,
                    optimizer: torch.optim.Optimizer, grad_clip: float = 1.0,
                    epoch: int = 0, max_epochs: int = 1) -> Tuple[float, Dict[int, float]]:
        """
        Train for one epoch.

        Args:
            model: Model to train
            train_batches: List of (x, y) batches
            optimizer: Optimizer
            grad_clip: Gradient clipping value
            epoch: Current epoch (for dynamic alpha)
            max_epochs: Total epochs (for dynamic alpha)

        Returns:
            Tuple of (average training loss, current layer weights)
        """
        model.train()
        total_loss = 0.0

        # Update training state for dynamic alpha
        self.set_training_state(epoch, max_epochs)
        current_weights = self.get_current_weights()

        for x, y in train_batches:
            x, y = x.to(self.device), y.to(self.device)
            optimizer.zero_grad()

            loss = self.compute_loss(model, x, y, weights=current_weights)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_batches), current_weights


def compare_presets():
    """Display all preset configurations."""
    print("=" * 70)
    print("UNIVERSAL TRAINING FRAMEWORK - PRESET CONFIGURATIONS")
    print("=" * 70)

    for name, config in PRESETS.items():
        print(f"\n{name}:")
        print(f"  {config.describe()}")

        # Show equivalent model
        weights = config.layer_weights
        if weights.get(2, 0) == 0:
            if weights.get(1, 0) == 0:
                equiv = "Standard LLM"
            elif weights.get(1, 0) == weights.get(3, 0):
                equiv = "Standard Routing"
            elif weights.get(1, 0) > weights.get(3, 0):
                equiv = "Asymmetric (Shallow-focused)"
            else:
                equiv = "Asymmetric (Deep-focused)"
        else:
            if all(w == 1/3 for w in weights.values()):
                equiv = "LPT"
            else:
                equiv = "Custom LPT variant"

        routing = "with Routing" if config.has_routing else "no Routing"
        print(f"  Equivalent: {equiv} ({routing})")


if __name__ == "__main__":
    compare_presets()
