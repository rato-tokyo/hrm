"""
Infini-HRM: Infini-Attention as input layer + standard HRM

Architecture:
  Embedding → Infini-Attention (with compressive memory) → HRM layers → output

The Infini-Attention layer handles long-range context,
while HRM layers handle deep reasoning on the enriched input.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Tuple, Optional

from .hrm import HRMLayerConfig, HRMLayer


class InfiniAttentionLayer(nn.Module):
    """
    Infini-Attention layer that combines local attention with compressive memory.

    For each position:
      output = gate * memory_retrieval + (1 - gate) * local_attention
    """

    def __init__(self, dim: int, num_heads: int = 4, max_seq_len: int = 1024):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Projections
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

        # Learnable gate (per head)
        self.gate = nn.Parameter(torch.zeros(1, num_heads, 1, 1))

        # Causal mask
        mask = torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).bool()
        self.register_buffer('causal_mask', mask)

    def forward(
        self,
        x: torch.Tensor,
        memory: Optional[torch.Tensor] = None,
        norm: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input [batch, seq_len, dim]
            memory: Compressive memory [batch, num_heads, head_dim, head_dim]
            norm: Memory normalization [batch, num_heads, head_dim]

        Returns:
            output, new_memory, new_norm
        """
        batch_size, seq_len, _ = x.shape
        device = x.device

        # Initialize memory if needed
        if memory is None or norm is None:
            memory = torch.zeros(batch_size, self.num_heads, self.head_dim, self.head_dim, device=device)
            norm = torch.zeros(batch_size, self.num_heads, self.head_dim, device=device)

        # Type assertion for mypy (memory and norm are now guaranteed non-None)
        assert memory is not None and norm is not None

        # Project Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # [batch, num_heads, seq_len, head_dim]

        # === Local Attention (standard causal attention) ===
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_scores = attn_scores.masked_fill(self.causal_mask[:seq_len, :seq_len], float('-inf'))
        attn_weights = F.softmax(attn_scores, dim=-1)
        local_out = torch.matmul(attn_weights, v)  # [batch, num_heads, seq_len, head_dim]

        # === Memory Retrieval ===
        # Use ELU+1 for non-negative queries (Linear Attention style)
        q_mem = F.elu(q) + 1

        # Retrieve: (Q × M) / (Q × z + eps)
        mem_out = torch.matmul(q_mem, memory)  # [batch, num_heads, seq_len, head_dim]

        # Normalize
        normalizer = torch.matmul(q_mem, norm.unsqueeze(-1)) + 1e-6  # [batch, num_heads, seq_len, 1]
        mem_out = mem_out / normalizer

        # === Combine with learnable gate ===
        gate = torch.sigmoid(self.gate)
        combined = gate * mem_out + (1 - gate) * local_out

        # === Update Memory with current K, V ===
        k_mem = F.elu(k) + 1
        # M_new = M_old + K^T × V
        new_memory = memory + torch.matmul(k_mem.transpose(-2, -1), v)
        # z_new = z_old + sum(K)
        new_norm = norm + k_mem.sum(dim=2)

        # Output projection
        output = combined.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        output = self.o_proj(output)

        return output, new_memory, new_norm


class InfiniHRM(nn.Module):
    """
    HRM with Infini-Attention as the input layer.

    Architecture:
        Input tokens
            ↓
        Embedding
            ↓
        Infini-Attention (with compressive memory for long-range context)
            ↓
        HRM Layers (standard hierarchical reasoning)
            ↓
        Output
    """

    def __init__(
        self,
        vocab_size: int,
        dim: int = 256,
        seq_len: int = 64,
        layer_configs: Optional[List[HRMLayerConfig]] = None,
        total_steps: int = 8,
        num_layers: int = 1,
        num_heads: int = 4,
        N: int = 2,
        T: int = 4
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.seq_len = seq_len
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # Default 2-layer HRM config
        if layer_configs is None:
            layer_configs = [
                HRMLayerConfig(update_freq=1, num_layers=num_layers, num_heads=num_heads),
                HRMLayerConfig(update_freq=T, num_layers=num_layers, num_heads=num_heads),
            ]
            total_steps = N * T

        self.layer_configs = layer_configs
        self.total_steps = total_steps
        self.num_hrm_layers = len(layer_configs)

        # Input embedding
        self.input_embedding = nn.Embedding(vocab_size, dim)

        # Infini-Attention layer (input processing with memory)
        self.infini_attention = InfiniAttentionLayer(dim, num_heads, seq_len)

        # Initial hidden states for HRM layers
        self.init_states = nn.ParameterList([
            nn.Parameter(torch.randn(1, seq_len, dim) * 0.02)
            for _ in range(self.num_hrm_layers)
        ])

        # Standard HRM layers
        self.hrm_layers = nn.ModuleList([
            HRMLayer(dim, config) for config in layer_configs
        ])

        # Output head
        self.output_head = nn.Linear(dim, vocab_size, bias=False)
        self.q_head = nn.Linear(dim, 2, bias=False)

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=1.0 / math.sqrt(self.dim))
            elif isinstance(module, nn.Embedding):
                nn.init.trunc_normal_(module.weight, std=0.02)

    def get_initial_states(self, batch_size: int) -> List[torch.Tensor]:
        return [
            init_state.expand(batch_size, -1, -1)
            for init_state in self.init_states
        ]

    def forward_pass(
        self,
        x: torch.Tensor,
        states: Optional[List[torch.Tensor]] = None,
        memory: Optional[torch.Tensor] = None,
        norm: Optional[torch.Tensor] = None
    ) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Single forward pass.

        Returns:
            states, memory, norm, y_hat, q_values
        """
        batch_size = x.shape[0]

        # Embedding
        x_embed = self.input_embedding(x)

        # Detach incoming memory to avoid backprop through previous segments
        if memory is not None and norm is not None:
            memory = memory.detach()
            norm = norm.detach()

        # Infini-Attention: enrich input with long-range context
        x_enriched, memory, norm = self.infini_attention(x_embed, memory, norm)

        # Initialize HRM states if needed
        if states is None:
            states = self.get_initial_states(batch_size)

        # HRM processing (standard)
        with torch.no_grad():
            for step in range(self.total_steps - 1):
                states = self._update_step(states, x_enriched, step)

        states = self._update_step(states, x_enriched, self.total_steps - 1)

        # Output
        slowest_state = states[-1]
        y_hat = self.output_head(slowest_state)
        q_values = torch.sigmoid(self.q_head(slowest_state.mean(dim=1)))

        return states, memory, norm, y_hat, q_values

    def _update_step(
        self,
        states: List[torch.Tensor],
        x_embed: torch.Tensor,
        step: int
    ) -> List[torch.Tensor]:
        """Standard HRM update step."""
        new_states = []

        for i, (layer, config, state) in enumerate(
            zip(self.hrm_layers, self.layer_configs, states)
        ):
            if (step + 1) % config.update_freq == 0:
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
        num_segments: int = 1,
        memory: Optional[torch.Tensor] = None,
        norm: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward with multiple segments, carrying memory across."""
        states = None

        for _ in range(num_segments):
            states, memory, norm, y_hat, q_values = self.forward_pass(
                x, states, memory, norm
            )
            states = [s.detach() for s in states]

        return y_hat, q_values


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
