"""
InfiniSpanMemoryのテスト・可視化スクリプト

Local/Global分離メモリと双方向エンコーダの動作を確認。

Note: このスクリプトは直接実行ではなく、
      python -c "import sys; sys.path.insert(0, '.'); exec(open('experiments/test_infini_span_memory.py').read())"
      または Colab で実行することを想定しています。
"""

import sys
import torch
import torch.nn as nn
from typing import List, Optional, Tuple
from dataclasses import dataclass, field
from torch import Tensor


# ========== 直接定義（依存関係回避） ==========

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
    """双方向エンコーダでspanを圧縮。"""

    def __init__(
        self,
        dim: int,
        mode: str = "bilstm",
        num_layers: int = 1,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dim = dim
        self.mode = mode
        self.num_layers = num_layers

        if mode == "bilstm":
            self.encoder = nn.LSTM(
                input_size=dim,
                hidden_size=dim // 2,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=True,
                dropout=dropout if num_layers > 1 else 0,
            )
            self.output_proj = nn.Sequential(
                nn.Linear(dim, dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dim, dim),
            )

        elif mode == "transformer":
            self.cls_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=num_heads,
                dim_feedforward=dim * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.output_proj = nn.Linear(dim, dim)

        else:  # pooling
            self.encoder = None
            self.output_proj = nn.Sequential(
                nn.Linear(dim, dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dim, dim),
            )

        self.layer_norm = nn.LayerNorm(dim)

    def forward(self, hidden_states: Tensor) -> Tensor:
        squeeze_batch = False
        if hidden_states.dim() == 2:
            hidden_states = hidden_states.unsqueeze(0)
            squeeze_batch = True

        batch_size, seq_len, _ = hidden_states.shape

        if self.mode == "bilstm":
            output, (h_n, _) = self.encoder(hidden_states)
            h_forward = h_n[-2]
            h_backward = h_n[-1]
            pooled = torch.cat([h_forward, h_backward], dim=-1)
            compressed = self.output_proj(pooled)

        elif self.mode == "transformer":
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat([cls_tokens, hidden_states], dim=1)
            encoded = self.encoder(x)
            cls_output = encoded[:, 0, :]
            compressed = self.output_proj(cls_output)

        else:  # pooling
            pooled = hidden_states.mean(dim=1)
            compressed = self.output_proj(pooled)

        compressed = self.layer_norm(compressed)

        if squeeze_batch:
            compressed = compressed.squeeze(0)

        return compressed

    def encode_spans(self, spans: list) -> Tensor:
        if not spans:
            return torch.zeros(0, self.dim)

        max_len = max(s.size(0) for s in spans)
        device = spans[0].device

        padded = torch.zeros(len(spans), max_len, self.dim, device=device)
        lengths = []
        for i, span in enumerate(spans):
            seq_len = span.size(0)
            padded[i, :seq_len] = span
            lengths.append(seq_len)

        if self.mode == "bilstm":
            from torch.nn.utils.rnn import pack_padded_sequence

            lengths_tensor = torch.tensor(lengths, device=device)
            sorted_lengths, sort_idx = lengths_tensor.sort(descending=True)
            sorted_padded = padded[sort_idx]

            packed = pack_padded_sequence(
                sorted_padded, sorted_lengths.cpu(), batch_first=True, enforce_sorted=True
            )
            _, (h_n, _) = self.encoder(packed)

            h_forward = h_n[-2]
            h_backward = h_n[-1]
            pooled = torch.cat([h_forward, h_backward], dim=-1)

            _, unsort_idx = sort_idx.sort()
            pooled = pooled[unsort_idx]

            compressed = self.output_proj(pooled)
        else:
            compressed = self.forward(padded)

        return self.layer_norm(compressed)


class InfiniSpanMemory(nn.Module):
    """Infini-Attention風のLocal/Global分離メモリ。"""

    def __init__(
        self,
        dim: int,
        span_size: int = 128,
        num_local_spans: int = 2,
        max_global_spans: int = 256,
        encoder_mode: str = "bilstm",
        encoder_layers: int = 1,
        encoder_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dim = dim
        self.span_size = span_size
        self.num_local_spans = num_local_spans
        self.max_global_spans = max_global_spans

        self.span_encoder = BidirectionalSpanEncoder(
            dim=dim,
            mode=encoder_mode,
            num_layers=encoder_layers,
            num_heads=encoder_heads,
            dropout=dropout,
        )

        self._state = MemoryState()

    def reset(self):
        self._state = MemoryState()

    @property
    def state(self) -> MemoryState:
        return self._state

    def update(
        self, hidden_states: Tensor, force_span_boundary: bool = False
    ) -> Tuple[Tensor, Optional[Tensor]]:
        seq_len = hidden_states.size(0)
        current_pos = self._state.total_processed

        new_span = SpanInfo(
            start_pos=current_pos,
            end_pos=current_pos + seq_len - 1,
            hidden_states=hidden_states.detach(),
        )

        if self._state.local_spans:
            last_span = self._state.local_spans[-1]
            combined_hidden = torch.cat([last_span.hidden_states, hidden_states], dim=0)

            if combined_hidden.size(0) >= self.span_size or force_span_boundary:
                self._split_and_update_spans(combined_hidden, last_span.start_pos)
            else:
                self._state.local_spans[-1] = SpanInfo(
                    start_pos=last_span.start_pos,
                    end_pos=current_pos + seq_len - 1,
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
            span_info = SpanInfo(
                start_pos=current_pos,
                end_pos=current_pos + (end_idx - idx) - 1,
                hidden_states=span_hidden.detach(),
            )
            new_local_spans.append(span_info)
            current_pos += (end_idx - idx)
            idx = end_idx

        all_spans = new_local_spans

        if len(all_spans) > self.num_local_spans:
            spans_to_compress = all_spans[: -self.num_local_spans]
            self._state.local_spans = all_spans[-self.num_local_spans :]
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
            local_hidden = torch.cat(
                [span.hidden_states for span in self._state.local_spans], dim=0
            )
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
    """Local AttentionとGlobal Attentionを統合するモジュール。"""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        gate_init: float = 0.0,
    ):
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

    def forward(
        self,
        local_hidden: Tensor,
        global_hidden: Optional[Tensor] = None,
        local_mask: Optional[Tensor] = None,
    ) -> Tensor:
        batch_size, local_len, _ = local_hidden.shape
        q = self.q_proj(local_hidden).view(
            batch_size, local_len, self.num_heads, self.head_dim
        ).transpose(1, 2)

        k_local = self.k_proj(local_hidden).view(
            batch_size, local_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        v_local = self.v_proj(local_hidden).view(
            batch_size, local_len, self.num_heads, self.head_dim
        ).transpose(1, 2)

        attn_local = torch.matmul(q, k_local.transpose(-2, -1)) * self.scale
        if local_mask is not None:
            attn_local = attn_local.masked_fill(local_mask == 0, float("-inf"))
        attn_local = torch.softmax(attn_local, dim=-1)
        attn_local = self.dropout(attn_local)
        out_local = torch.matmul(attn_local, v_local)

        if global_hidden is not None and global_hidden.size(1) > 0:
            num_global = global_hidden.size(1)
            k_global = self.k_global_proj(global_hidden).view(
                batch_size, num_global, self.num_heads, self.head_dim
            ).transpose(1, 2)
            v_global = self.v_global_proj(global_hidden).view(
                batch_size, num_global, self.num_heads, self.head_dim
            ).transpose(1, 2)
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


# ========== テスト関数 ==========


def test_basic_memory():
    """基本的なメモリ動作をテスト。"""
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
        encoder_mode="bilstm",
    )

    total_tokens = 20
    chunk_size = 3

    print(f"\nConfig: dim={dim}, span_size={span_size}, num_local_spans=2")
    print(f"Encoder: {memory.span_encoder.mode}")
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
    print("Local spans:")
    for i, span in enumerate(memory.state.local_spans):
        print(f"  Span {i}: pos=[{span.start_pos}-{span.end_pos}], len={span.length}")
    print(f"Global vectors: {len(memory.state.global_vectors)}")
    for i, (start, end) in enumerate(memory.state.global_positions):
        print(f"  Compressed span {i}: pos=[{start}-{end}]")


def test_bidirectional_encoder():
    """双方向エンコーダの各モードをテスト。"""
    print("\n" + "=" * 60)
    print("Test: BidirectionalSpanEncoder Modes")
    print("=" * 60)

    dim = 64
    seq_len = 8
    batch_size = 2

    modes = ["bilstm", "transformer", "pooling"]

    for mode in modes:
        print(f"\n--- Mode: {mode} ---")

        encoder = BidirectionalSpanEncoder(
            dim=dim,
            mode=mode,
            num_layers=1,
            num_heads=4,
            dropout=0.0,
        )

        # 単一シーケンス
        single_input = torch.randn(seq_len, dim)
        single_output = encoder(single_input)
        print(f"  Single input: {single_input.shape} -> {single_output.shape}")
        assert single_output.shape == (dim,), f"Expected ({dim},), got {single_output.shape}"

        # バッチ入力
        batch_input = torch.randn(batch_size, seq_len, dim)
        batch_output = encoder(batch_input)
        print(f"  Batch input: {batch_input.shape} -> {batch_output.shape}")
        assert batch_output.shape == (batch_size, dim), f"Expected ({batch_size}, {dim})"

        # パラメータ数
        num_params = sum(p.numel() for p in encoder.parameters())
        print(f"  Parameters: {num_params:,}")


def test_encoder_batch_spans():
    """複数spanの一括エンコードをテスト。"""
    print("\n" + "=" * 60)
    print("Test: Batch Span Encoding")
    print("=" * 60)

    dim = 64
    encoder = BidirectionalSpanEncoder(
        dim=dim,
        mode="bilstm",
        num_layers=1,
    )

    # 異なる長さのspan
    spans = [
        torch.randn(4, dim),
        torch.randn(8, dim),
        torch.randn(2, dim),
        torch.randn(6, dim),
    ]

    print(f"\nInput spans: {[s.size(0) for s in spans]} tokens each")

    compressed = encoder.encode_spans(spans)
    print(f"Compressed output: {compressed.shape}")
    assert compressed.shape == (len(spans), dim), f"Expected ({len(spans)}, {dim})"


def test_attention_module():
    """InfiniSpanAttentionモジュールをテスト。"""
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
    print("\n" + "=" * 60)
    print("Test: Streaming Scenario")
    print("=" * 60)

    dim = 32
    span_size = 8
    memory = InfiniSpanMemory(
        dim=dim,
        span_size=span_size,
        num_local_spans=2,
        encoder_mode="transformer",
        encoder_layers=1,
    )

    print(f"\nSimulating streaming with span_size={span_size}")
    print(f"Encoder: {memory.span_encoder.mode}")
    print("-" * 60)

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
    local_tokens = sum(s.length for s in memory.state.local_spans)
    print(f"  = {len(memory.state.global_vectors)} global + {local_tokens} local")


def visualize_memory_state():
    """メモリ状態を可視化。"""
    print("\n" + "=" * 60)
    print("Visualization: Memory State Over Time")
    print("=" * 60)

    dim = 16
    span_size = 4
    memory = InfiniSpanMemory(
        dim=dim,
        span_size=span_size,
        num_local_spans=2,
        encoder_mode="pooling",
    )

    print(f"\nAdding tokens one by one (span_size={span_size})")
    print(f"Encoder: {memory.span_encoder.mode}")
    print("Legend: [L] = Local span, [G] = Global (compressed)")
    print("-" * 60)

    for t in range(16):
        hidden = torch.randn(1, dim)
        local_ctx, global_ctx = memory.update(hidden)

        visual = ""
        for start, end in memory.state.global_positions:
            visual += f"[G:{start}-{end}] "
        for span in memory.state.local_spans:
            visual += f"[L:{span.start_pos}-{span.end_pos}] "

        print(f"t={t:2d}: {visual}")


def compare_encoder_modes():
    """エンコーダモードの比較。"""
    print("\n" + "=" * 60)
    print("Comparison: Encoder Modes")
    print("=" * 60)

    dim = 64
    span_size = 8
    total_tokens = 32

    modes = ["bilstm", "transformer", "pooling"]

    for mode in modes:
        memory = InfiniSpanMemory(
            dim=dim,
            span_size=span_size,
            num_local_spans=2,
            encoder_mode=mode,
        )

        for i in range(0, total_tokens, 4):
            hidden = torch.randn(4, dim)
            memory.update(hidden)

        num_params = sum(p.numel() for p in memory.span_encoder.parameters())
        local_tokens = sum(s.length for s in memory.state.local_spans)
        print(f"\n{mode}:")
        print(f"  Parameters: {num_params:,}")
        print(f"  Global vectors: {len(memory.state.global_vectors)}")
        print(f"  Local tokens: {local_tokens}")


def main():
    print("InfiniSpanMemory Test Suite")
    print("=" * 60)

    test_basic_memory()
    test_bidirectional_encoder()
    test_encoder_batch_spans()
    test_attention_module()
    test_streaming_scenario()
    visualize_memory_state()
    compare_encoder_modes()

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
