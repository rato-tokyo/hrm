"""
InfiniSpanMemoryのテスト・可視化スクリプト

Local/Global分離メモリの動作を確認。
"""

import sys
import torch
import torch.nn as nn
from typing import List, Optional, Tuple
from dataclasses import dataclass, field

# 直接モジュールを定義（環境問題回避）
from torch import Tensor


@dataclass
class SpanInfo:
    """Span情報を保持するデータクラス。"""
    start_pos: int
    end_pos: int
    hidden_states: Tensor

    @property
    def length(self) -> int:
        return self.end_pos - self.start_pos + 1


@dataclass
class MemoryState:
    """メモリの状態を表すデータクラス。"""
    local_spans: List[SpanInfo] = field(default_factory=list)
    global_vectors: List[Tensor] = field(default_factory=list)
    global_positions: List[Tuple[int, int]] = field(default_factory=list)
    total_processed: int = 0


class BidirectionalSpanEncoder(nn.Module):
    def __init__(self, dim: int, use_projection: bool = True, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        if use_projection:
            self.projection: nn.Module = nn.Sequential(
                nn.Linear(dim, dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim, dim),
            )
        else:
            self.projection = nn.Identity()

    def forward(self, hidden_states: Tensor) -> Tensor:
        if hidden_states.dim() == 2:
            pooled = hidden_states.mean(dim=0)
        else:
            pooled = hidden_states.mean(dim=1)
        return self.projection(pooled)


class InfiniSpanMemory(nn.Module):
    def __init__(self, dim: int, span_size: int = 128, num_local_spans: int = 2,
                 max_global_spans: int = 256, use_projection: bool = True, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.span_size = span_size
        self.num_local_spans = num_local_spans
        self.max_global_spans = max_global_spans
        self.span_encoder = BidirectionalSpanEncoder(dim, use_projection, dropout)
        self._state = MemoryState()

    def reset(self):
        self._state = MemoryState()

    @property
    def state(self) -> MemoryState:
        return self._state

    def update(self, hidden_states: Tensor, force_span_boundary: bool = False) -> Tuple[Tensor, Optional[Tensor]]:
        seq_len = hidden_states.size(0)
        current_pos = self._state.total_processed

        new_span = SpanInfo(start_pos=current_pos, end_pos=current_pos + seq_len - 1,
                           hidden_states=hidden_states.detach())

        if self._state.local_spans:
            last_span = self._state.local_spans[-1]
            combined_hidden = torch.cat([last_span.hidden_states, hidden_states], dim=0)

            if combined_hidden.size(0) >= self.span_size or force_span_boundary:
                self._split_and_update_spans(combined_hidden, last_span.start_pos)
            else:
                self._state.local_spans[-1] = SpanInfo(
                    start_pos=last_span.start_pos, end_pos=current_pos + seq_len - 1,
                    hidden_states=combined_hidden.detach(),
                )
        else:
            self._state.local_spans.append(new_span)

        self._state.total_processed += seq_len
        return self._get_contexts()

    def _split_and_update_spans(self, hidden_states: Tensor, start_pos: int):
        total_len = hidden_states.size(0)
        current_pos = start_pos
        new_local_spans: List[SpanInfo] = []

        idx = 0
        while idx < total_len:
            end_idx = min(idx + self.span_size, total_len)
            span_hidden = hidden_states[idx:end_idx]
            span_info = SpanInfo(start_pos=current_pos, end_pos=current_pos + (end_idx - idx) - 1,
                                hidden_states=span_hidden.detach())
            new_local_spans.append(span_info)
            current_pos += (end_idx - idx)
            idx = end_idx

        all_spans = new_local_spans

        if len(all_spans) > self.num_local_spans:
            spans_to_compress = all_spans[:-self.num_local_spans]
            self._state.local_spans = all_spans[-self.num_local_spans:]
            for span in spans_to_compress:
                self._compress_to_global(span)
        else:
            self._state.local_spans = all_spans

    def _compress_to_global(self, span: SpanInfo):
        compressed = self.span_encoder(span.hidden_states)
        self._state.global_vectors.append(compressed.detach())
        self._state.global_positions.append((span.start_pos, span.end_pos))
        if len(self._state.global_vectors) > self.max_global_spans:
            self._state.global_vectors.pop(0)
            self._state.global_positions.pop(0)

    def _get_contexts(self) -> Tuple[Tensor, Optional[Tensor]]:
        if self._state.local_spans:
            local_hidden = torch.cat([span.hidden_states for span in self._state.local_spans], dim=0)
        else:
            local_hidden = torch.zeros(0, self.dim)
        if self._state.global_vectors:
            global_hidden = torch.stack(self._state.global_vectors, dim=0)
        else:
            global_hidden = None
        return local_hidden, global_hidden

    def get_full_context(self) -> Tensor:
        local_ctx, global_ctx = self._get_contexts()
        if global_ctx is not None:
            return torch.cat([global_ctx, local_ctx], dim=0)
        return local_ctx


class InfiniSpanAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1, gate_init: float = 0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.k_global_proj = nn.Linear(dim, dim)
        self.v_global_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.gate = nn.Parameter(torch.tensor(gate_init))
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(self, local_hidden: Tensor, global_hidden: Optional[Tensor] = None,
                local_mask: Optional[Tensor] = None) -> Tensor:
        batch_size, local_len, _ = local_hidden.shape
        q = self.q_proj(local_hidden).view(batch_size, local_len, self.num_heads, self.head_dim).transpose(1, 2)

        k_local = self.k_proj(local_hidden).view(batch_size, local_len, self.num_heads, self.head_dim).transpose(1, 2)
        v_local = self.v_proj(local_hidden).view(batch_size, local_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_local = torch.matmul(q, k_local.transpose(-2, -1)) * self.scale
        if local_mask is not None:
            attn_local = attn_local.masked_fill(local_mask == 0, float('-inf'))
        attn_local = torch.softmax(attn_local, dim=-1)
        attn_local = self.dropout(attn_local)
        out_local = torch.matmul(attn_local, v_local)

        if global_hidden is not None and global_hidden.size(1) > 0:
            num_global = global_hidden.size(1)
            k_global = self.k_global_proj(global_hidden).view(batch_size, num_global, self.num_heads, self.head_dim).transpose(1, 2)
            v_global = self.v_global_proj(global_hidden).view(batch_size, num_global, self.num_heads, self.head_dim).transpose(1, 2)
            attn_global = torch.matmul(q, k_global.transpose(-2, -1)) * self.scale
            attn_global = torch.softmax(attn_global, dim=-1)
            attn_global = self.dropout(attn_global)
            out_global = torch.matmul(attn_global, v_global)
            gate = torch.sigmoid(self.gate)
            out = gate * out_local + (1 - gate) * out_global
        else:
            out = out_local

        out = out.transpose(1, 2).contiguous().view(batch_size, local_len, self.dim)
        return self.out_proj(out)


def test_basic_memory():
    """基本的なメモリ動作をテスト。"""
    from cascade.infini_span_memory import (
        InfiniSpanMemory,
        BidirectionalSpanEncoder,
    )

    print("=" * 60)
    print("Test: Basic Memory Operation")
    print("=" * 60)

    dim = 64
    span_size = 4
    memory = InfiniSpanMemory(
        dim=dim,
        span_size=span_size,
        num_local_spans=2,
        max_global_spans=10,
    )

    # シミュレート: 連続したチャンクを投入
    total_tokens = 20
    chunk_size = 3

    print(f"\nConfig: dim={dim}, span_size={span_size}, num_local_spans=2")
    print(f"Input: {total_tokens} tokens in chunks of {chunk_size}")
    print("-" * 60)

    for i in range(0, total_tokens, chunk_size):
        chunk_len = min(chunk_size, total_tokens - i)
        hidden_chunk = torch.randn(chunk_len, dim)

        local_ctx, global_ctx = memory.update(hidden_chunk)

        global_info = f"shape={global_ctx.shape}" if global_ctx is not None else "None"

        print(f"\nChunk {i//chunk_size + 1}: tokens [{i}-{i+chunk_len-1}]")
        print(f"  Local context: shape={local_ctx.shape}")
        print(f"  Global context: {global_info}")
        print(f"  Total processed: {memory.state.total_processed}")
        print(f"  Local spans: {len(memory.state.local_spans)}")
        print(f"  Global vectors: {len(memory.state.global_vectors)}")

    print("\n" + "=" * 60)
    print("Final State")
    print("=" * 60)
    print(f"Local spans:")
    for i, span in enumerate(memory.state.local_spans):
        print(f"  Span {i}: pos=[{span.start_pos}-{span.end_pos}], len={span.length}")
    print(f"Global vectors: {len(memory.state.global_vectors)}")
    for i, (start, end) in enumerate(memory.state.global_positions):
        print(f"  Compressed span {i}: pos=[{start}-{end}]")


def test_attention_module():
    """InfiniSpanAttentionモジュールをテスト。"""
    from cascade.infini_span_memory import InfiniSpanAttention

    print("\n" + "=" * 60)
    print("Test: InfiniSpanAttention")
    print("=" * 60)

    dim = 64
    num_heads = 4
    batch_size = 2
    local_len = 8
    num_global = 3

    attention = InfiniSpanAttention(
        dim=dim,
        num_heads=num_heads,
        dropout=0.0,
    )

    # テスト入力
    local_hidden = torch.randn(batch_size, local_len, dim)
    global_hidden = torch.randn(batch_size, num_global, dim)

    # Causal mask
    local_mask = torch.tril(torch.ones(local_len, local_len))

    print(f"\nInput shapes:")
    print(f"  local_hidden: {local_hidden.shape}")
    print(f"  global_hidden: {global_hidden.shape}")
    print(f"  local_mask: {local_mask.shape}")

    # Forward
    output = attention(local_hidden, global_hidden, local_mask)

    print(f"\nOutput shape: {output.shape}")
    print(f"Gate value: {torch.sigmoid(attention.gate).item():.4f}")

    # Global無しの場合
    output_no_global = attention(local_hidden, None, local_mask)
    print(f"Output (no global) shape: {output_no_global.shape}")


def test_streaming_scenario():
    """ストリーミング処理のシナリオをテスト。"""
    from cascade.infini_span_memory import InfiniSpanMemory

    print("\n" + "=" * 60)
    print("Test: Streaming Scenario")
    print("=" * 60)

    dim = 32
    span_size = 8
    memory = InfiniSpanMemory(
        dim=dim,
        span_size=span_size,
        num_local_spans=2,
    )

    # シミュレート: 長いシーケンスをストリーミング処理
    print(f"\nSimulating streaming with span_size={span_size}")
    print("-" * 60)

    # 文章の区切りを想定した入力
    sentences = [
        ("Hello world", 4),
        ("How are you", 5),
        ("I am fine", 4),
        ("Thank you", 3),
        ("Goodbye", 2),
    ]

    for sentence, length in sentences:
        hidden = torch.randn(length, dim)
        local_ctx, global_ctx = memory.update(hidden)

        global_count = global_ctx.size(0) if global_ctx is not None else 0
        print(f"\n'{sentence}' ({length} tokens)")
        print(f"  Local: {local_ctx.size(0)} tokens")
        print(f"  Global: {global_count} compressed spans")

    print("\n" + "=" * 60)
    print("Memory Summary")
    print("=" * 60)

    full_context = memory.get_full_context()
    print(f"Full context shape: {full_context.shape}")
    print(f"  = {len(memory.state.global_vectors)} global + {sum(s.length for s in memory.state.local_spans)} local")


def visualize_memory_state():
    """メモリ状態を可視化。"""
    from cascade.infini_span_memory import InfiniSpanMemory

    print("\n" + "=" * 60)
    print("Visualization: Memory State Over Time")
    print("=" * 60)

    dim = 16
    span_size = 4
    memory = InfiniSpanMemory(dim=dim, span_size=span_size, num_local_spans=2)

    # 16トークンを1つずつ追加
    print(f"\nAdding tokens one by one (span_size={span_size})")
    print("Legend: [L] = Local span, [G] = Global (compressed)")
    print("-" * 60)

    for t in range(16):
        hidden = torch.randn(1, dim)
        local_ctx, global_ctx = memory.update(hidden)

        # 可視化
        visual = ""
        # Global
        for start, end in memory.state.global_positions:
            visual += f"[G:{start}-{end}] "
        # Local
        for span in memory.state.local_spans:
            visual += f"[L:{span.start_pos}-{span.end_pos}] "

        print(f"t={t:2d}: {visual}")


def main():
    print("InfiniSpanMemory Test Suite")
    print("=" * 60)

    test_basic_memory()
    test_attention_module()
    test_streaming_scenario()
    visualize_memory_state()

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
