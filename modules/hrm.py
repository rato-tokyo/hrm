"""
Hierarchical Reasoning Model (HRM) - Dynamic Multi-Layer Architecture

Supports arbitrary number of hierarchical layers with configurable update frequencies.

Example usage:
    model = HRM(
        vocab_size=100,
        dim=64,
        seq_len=64,
        layer_configs=[
            HRMLayerConfig(update_freq=1, num_layers=1, num_heads=4),  # Fast layer (L)
            HRMLayerConfig(update_freq=2, num_layers=1, num_heads=4),  # Slow layer (H)
            HRMLayerConfig(update_freq=4, num_layers=1, num_heads=4),  # Slower layer
        ]
    )
"""

import torch
import torch.nn as nn
import math
from typing import List, Tuple, Optional
from dataclasses import dataclass

from .transformer import RecurrentModule


@dataclass
class HRMLayerConfig:
    """Configuration for a single HRM layer"""
    update_freq: int = 1       # Update every N steps (1 = every step, 2 = every 2 steps, etc.)
    num_layers: int = 1        # Number of transformer layers in this module
    num_heads: int = 4         # Number of attention heads


class HRMLayer(nn.Module):
    """
    Single HRM Layer with configurable update frequency.

    Each layer receives:
    - Its own previous state
    - States from all other layers (summed)
    - Input embedding (for the fastest layer)
    """

    def __init__(self, dim: int, config: HRMLayerConfig):
        super().__init__()
        self.update_freq = config.update_freq
        self.recurrent = RecurrentModule(dim, config.num_layers, config.num_heads)

    def forward(self, prev_state: torch.Tensor, other_states_sum: torch.Tensor) -> torch.Tensor:
        """
        Args:
            prev_state: Previous state of this layer [batch, seq_len, dim]
            other_states_sum: Sum of states from other layers + input embedding [batch, seq_len, dim]

        Returns:
            Updated state [batch, seq_len, dim]
        """
        combined = prev_state + other_states_sum
        return self.recurrent(combined)


class HRM(nn.Module):
    """
    Hierarchical Reasoning Model with Dynamic Multi-Layer Architecture

    Args:
        vocab_size: Size of vocabulary for input/output tokens
        dim: Hidden dimension (shared across all layers)
        seq_len: Sequence length
        layer_configs: List of HRMLayerConfig defining each hierarchical layer
        total_steps: Total number of recurrent steps per forward pass

    Example:
        # 2-layer HRM (like original paper)
        model = HRM(
            vocab_size=100,
            dim=64,
            seq_len=64,
            layer_configs=[
                HRMLayerConfig(update_freq=1),  # L-module: every step
                HRMLayerConfig(update_freq=2),  # H-module: every 2 steps
            ],
            total_steps=8
        )

        # 3-layer HRM
        model = HRM(
            vocab_size=100,
            dim=64,
            seq_len=64,
            layer_configs=[
                HRMLayerConfig(update_freq=1),  # Fast
                HRMLayerConfig(update_freq=2),  # Medium
                HRMLayerConfig(update_freq=4),  # Slow
            ],
            total_steps=8
        )
    """

    def __init__(
        self,
        vocab_size: int,
        dim: int = 256,
        seq_len: int = 64,
        layer_configs: Optional[List[HRMLayerConfig]] = None,
        total_steps: int = 8,
        # Legacy parameters for backward compatibility
        num_layers: int = 1,
        num_heads: int = 4,
        N: int = 2,
        T: int = 4
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.seq_len = seq_len

        # If layer_configs not provided, create default 2-layer config (backward compatible)
        if layer_configs is None:
            layer_configs = [
                HRMLayerConfig(update_freq=1, num_layers=num_layers, num_heads=num_heads),  # L
                HRMLayerConfig(update_freq=T, num_layers=num_layers, num_heads=num_heads),  # H
            ]
            total_steps = N * T

        self.layer_configs = layer_configs
        self.total_steps = total_steps
        self.num_hrm_layers = len(layer_configs)

        # Input embedding
        self.input_embedding = nn.Embedding(vocab_size, dim)

        # Initial hidden states (learnable) for each layer
        self.init_states = nn.ParameterList([
            nn.Parameter(torch.randn(1, seq_len, dim) * 0.02)
            for _ in range(self.num_hrm_layers)
        ])

        # HRM layers
        self.hrm_layers = nn.ModuleList([
            HRMLayer(dim, config) for config in layer_configs
        ])

        # Output head (uses the slowest layer's output)
        self.output_head = nn.Linear(dim, vocab_size, bias=False)

        # Q-head for ACT (optional, uses slowest layer)
        self.q_head = nn.Linear(dim, 2, bias=False)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights using truncated LeCun Normal"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=1.0 / math.sqrt(self.dim))
            elif isinstance(module, nn.Embedding):
                nn.init.trunc_normal_(module.weight, std=0.02)

    def get_initial_states(self, batch_size: int) -> List[torch.Tensor]:
        """Get initial hidden states for all layers"""
        return [
            init_state.expand(batch_size, -1, -1)
            for init_state in self.init_states
        ]

    def forward_pass(
        self,
        x: torch.Tensor,
        states: Optional[List[torch.Tensor]] = None
    ) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Single forward pass of HRM (total_steps iterations)

        Args:
            x: Input tokens [batch, seq_len]
            states: List of previous states for each layer (or None for initial)

        Returns:
            states: Updated states for each layer
            y_hat: Output predictions [batch, seq_len, vocab_size]
            q_values: ACT confidence values [batch, 2]
        """
        batch_size = x.shape[0]
        x_embed = self.input_embedding(x)

        if states is None:
            states = self.get_initial_states(batch_size)

        # 1-step gradient approximation: run all but last step without gradients
        with torch.no_grad():
            for step in range(self.total_steps - 1):
                states = self._update_step(states, x_embed, step)

        # Final step with gradient
        states = self._update_step(states, x_embed, self.total_steps - 1)

        # Output from the slowest (last) layer
        slowest_state = states[-1]
        y_hat = self.output_head(slowest_state)
        q_values = torch.sigmoid(self.q_head(slowest_state.mean(dim=1)))

        return states, y_hat, q_values

    def _update_step(
        self,
        states: List[torch.Tensor],
        x_embed: torch.Tensor,
        step: int
    ) -> List[torch.Tensor]:
        """
        Perform one update step across all layers.

        Each layer is updated only if (step + 1) % update_freq == 0
        """
        new_states = []

        for i, (layer, config, state) in enumerate(zip(self.hrm_layers, self.layer_configs, states)):
            # Check if this layer should be updated at this step
            if (step + 1) % config.update_freq == 0:
                # Compute sum of other states + input embedding
                other_sum = x_embed.clone()
                for j, other_state in enumerate(states):
                    if j != i:
                        other_sum = other_sum + other_state

                new_state = layer(state, other_sum)
            else:
                new_state = state

            new_states.append(new_state)

        return new_states

    def forward(
        self,
        x: torch.Tensor,
        num_segments: int = 1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward with deep supervision (multiple segments)

        Args:
            x: Input tokens [batch, seq_len]
            num_segments: Number of times to run forward_pass

        Returns:
            y_hat: Final output predictions
            q_values: Final ACT confidence values
        """
        states = None

        for _ in range(num_segments):
            states, y_hat, q_values = self.forward_pass(x, states)
            # Detach states between segments for truncated BPTT
            states = [s.detach() for s in states]

        return y_hat, q_values

    # Legacy compatibility methods
    def get_initial_state(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Legacy method for backward compatibility (returns H, L states)"""
        states = self.get_initial_states(batch_size)
        if len(states) >= 2:
            return states[1], states[0]  # H, L order for legacy
        return states[0], states[0]


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Convenience function to create common configurations
def create_hrm(
    vocab_size: int,
    dim: int = 64,
    seq_len: int = 64,
    num_hierarchies: int = 2,
    base_update_freq: int = 2,
    num_layers: int = 1,
    num_heads: int = 4,
    total_steps: int = 8
) -> HRM:
    """
    Create HRM with automatically generated layer configs.

    Args:
        vocab_size: Vocabulary size
        dim: Hidden dimension
        seq_len: Sequence length
        num_hierarchies: Number of hierarchical layers
        base_update_freq: Multiplier for update frequency (layer i updates every base_update_freq^i steps)
        num_layers: Transformer layers per HRM layer
        num_heads: Attention heads
        total_steps: Total recurrent steps

    Returns:
        HRM model

    Example:
        # 3-layer HRM with update frequencies [1, 2, 4]
        model = create_hrm(vocab_size=100, num_hierarchies=3, base_update_freq=2)
    """
    layer_configs = [
        HRMLayerConfig(
            update_freq=base_update_freq ** i,
            num_layers=num_layers,
            num_heads=num_heads
        )
        for i in range(num_hierarchies)
    ]

    return HRM(
        vocab_size=vocab_size,
        dim=dim,
        seq_len=seq_len,
        layer_configs=layer_configs,
        total_steps=total_steps
    )
