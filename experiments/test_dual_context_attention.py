"""
Dual-Context Attention (DCA) のテスト・実験スクリプト

L0 (Local Context) と L1 (Representative Context) の
2層コンテキストを使った推論実験。

実験内容:
1. 基本動作テスト
2. メモリ使用量の比較
3. 長シーケンスでの圧縮効果
4. Attention分布の可視化
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, NamedTuple
from dataclasses import dataclass, field


# ========== DCA実装（自己完結版） ==========

class DCAOutput(NamedTuple):
    """Dual-Context Attentionの出力。"""
    output: torch.Tensor
    l0_attention: Optional[torch.Tensor]
    l1_attention: Optional[torch.Tensor]


@dataclass
class DualContextState:
    """Dual-Contextの状態。"""
    l0_keys: Optional[torch.Tensor] = None
    l0_values: Optional[torch.Tensor] = None
    l1_representatives: List[torch.Tensor] = field(default_factory=list)
    l1_keys: Optional[torch.Tensor] = None
    l0_positions: List[Tuple[int, int]] = field(default_factory=list)
    l1_positions: List[Tuple[int, int]] = field(default_factory=list)
    total_processed: int = 0


class BidirectionalSpanEncoder(nn.Module):
    """双方向エンコーダ（テスト用簡易版）。"""

    def __init__(self, dim: int, mode: str = "bilstm", num_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.mode = mode

        if mode == "bilstm":
            self.encoder = nn.LSTM(
                input_size=dim, hidden_size=dim // 2,
                num_layers=num_layers, batch_first=True, bidirectional=True,
            )
            self.output_proj = nn.Sequential(
                nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim),
            )
        else:  # pooling
            self.output_proj = nn.Sequential(
                nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim),
            )

        self.layer_norm = nn.LayerNorm(dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        squeeze = False
        if hidden_states.dim() == 2:
            hidden_states = hidden_states.unsqueeze(0)
            squeeze = True

        if self.mode == "bilstm":
            _, (h_n, _) = self.encoder(hidden_states)
            pooled = torch.cat([h_n[-2], h_n[-1]], dim=-1)
            compressed = self.output_proj(pooled)
        else:
            pooled = hidden_states.mean(dim=1)
            compressed = self.output_proj(pooled)

        compressed = self.layer_norm(compressed)
        if squeeze:
            compressed = compressed.squeeze(0)
        return compressed


class DualContextMemory(nn.Module):
    """L0/L1の2層コンテキストを管理。"""

    def __init__(
        self, dim: int, window_size: int = 512, max_representatives: int = 256,
        encoder_mode: str = "bilstm",
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.max_representatives = max_representatives

        self.l0_key_proj = nn.Linear(dim, dim)
        self.l0_value_proj = nn.Linear(dim, dim)
        self.l1_key_proj = nn.Linear(dim, dim)

        self.span_encoder = BidirectionalSpanEncoder(dim=dim, mode=encoder_mode)

        self._state = DualContextState()
        self._l0_buffer: List[torch.Tensor] = []
        self._l0_buffer_positions: List[int] = []

    def reset(self):
        self._state = DualContextState()
        self._l0_buffer = []
        self._l0_buffer_positions = []

    @property
    def state(self) -> DualContextState:
        return self._state

    def update(self, hidden_states: torch.Tensor, force_compress: bool = False) -> DualContextState:
        if hidden_states.dim() == 2:
            hidden_states = hidden_states.unsqueeze(0)

        batch_size, seq_len, _ = hidden_states.shape

        for t in range(seq_len):
            self._l0_buffer.append(hidden_states[:, t:t+1, :])
            self._l0_buffer_positions.append(self._state.total_processed + t)

        self._state.total_processed += seq_len

        while len(self._l0_buffer) > self.window_size or force_compress:
            if len(self._l0_buffer) <= self.window_size // 2:
                break

            compress_size = len(self._l0_buffer) - self.window_size // 2
            compress_size = min(compress_size, self.window_size // 2)

            if compress_size < 1:
                break

            span_hidden = torch.cat(self._l0_buffer[:compress_size], dim=1)
            span_start = self._l0_buffer_positions[0]
            span_end = self._l0_buffer_positions[compress_size - 1]

            self._l0_buffer = self._l0_buffer[compress_size:]
            self._l0_buffer_positions = self._l0_buffer_positions[compress_size:]

            representative = self.span_encoder(span_hidden.squeeze(0))

            self._state.l1_representatives.append(representative.detach())
            self._state.l1_positions.append((span_start, span_end))

            if len(self._state.l1_representatives) > self.max_representatives:
                self._state.l1_representatives.pop(0)
                self._state.l1_positions.pop(0)

        if self._l0_buffer:
            l0_hidden = torch.cat(self._l0_buffer, dim=1)
            self._state.l0_keys = self.l0_key_proj(l0_hidden)
            self._state.l0_values = self.l0_value_proj(l0_hidden)
        else:
            self._state.l0_keys = None
            self._state.l0_values = None

        if self._state.l1_representatives:
            l1_reps = torch.stack(self._state.l1_representatives, dim=0).unsqueeze(0)
            l1_reps = l1_reps.expand(batch_size, -1, -1)
            self._state.l1_keys = self.l1_key_proj(l1_reps)
        else:
            self._state.l1_keys = None

        return self._state

    def get_l1_values(self, batch_size: int = 1) -> Optional[torch.Tensor]:
        """L1の代表ベクトル（Value用）を取得。"""
        if not self._state.l1_representatives:
            return None
        l1_reps = torch.stack(self._state.l1_representatives, dim=0).unsqueeze(0)
        return l1_reps.expand(batch_size, -1, -1)


class DualContextAttention(nn.Module):
    """L0とL1を統合するDual-Context Attention。"""

    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1, gate_init: float = 0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.gate = nn.Parameter(torch.tensor(gate_init))
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(
        self,
        query: torch.Tensor,
        l0_keys: Optional[torch.Tensor] = None,
        l0_values: Optional[torch.Tensor] = None,
        l1_keys: Optional[torch.Tensor] = None,
        l1_values: Optional[torch.Tensor] = None,
        l0_mask: Optional[torch.Tensor] = None,
    ) -> DCAOutput:
        batch_size, seq_len, _ = query.shape

        q = self.q_proj(query)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # L0 Attention
        if l0_keys is not None and l0_values is not None:
            l0_len = l0_keys.size(1)
            k0 = l0_keys.view(batch_size, l0_len, self.num_heads, self.head_dim).transpose(1, 2)
            v0 = l0_values.view(batch_size, l0_len, self.num_heads, self.head_dim).transpose(1, 2)

            attn_l0 = torch.matmul(q, k0.transpose(-2, -1)) * self.scale
            if l0_mask is not None:
                attn_l0 = attn_l0.masked_fill(l0_mask.unsqueeze(0).unsqueeze(0) == 0, float('-inf'))
            attn_l0_weights = F.softmax(attn_l0, dim=-1)
            attn_l0_weights = self.dropout(attn_l0_weights)
            out_l0 = torch.matmul(attn_l0_weights, v0)
            attn_l0_avg = attn_l0_weights.mean(dim=1)
        else:
            out_l0 = torch.zeros(batch_size, self.num_heads, seq_len, self.head_dim, device=query.device)
            attn_l0_avg = None

        # L1 Attention
        if l1_keys is not None and l1_values is not None:
            l1_len = l1_keys.size(1)
            k1 = l1_keys.view(batch_size, l1_len, self.num_heads, self.head_dim).transpose(1, 2)
            v1 = l1_values.view(batch_size, l1_len, self.num_heads, self.head_dim).transpose(1, 2)

            attn_l1 = torch.matmul(q, k1.transpose(-2, -1)) * self.scale
            attn_l1_weights = F.softmax(attn_l1, dim=-1)
            attn_l1_weights = self.dropout(attn_l1_weights)
            out_l1 = torch.matmul(attn_l1_weights, v1)
            attn_l1_avg = attn_l1_weights.mean(dim=1)
        else:
            out_l1 = torch.zeros(batch_size, self.num_heads, seq_len, self.head_dim, device=query.device)
            attn_l1_avg = None

        # Combine
        gate = torch.sigmoid(self.gate)
        if l1_keys is None:
            out = out_l0
        elif l0_keys is None:
            out = out_l1
        else:
            out = gate * out_l0 + (1 - gate) * out_l1

        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)
        out = self.out_proj(out)

        return DCAOutput(output=out, l0_attention=attn_l0_avg, l1_attention=attn_l1_avg)


# ========== テスト関数 ==========


def test_basic_dca():
    """基本的なDCA動作テスト。"""
    print("=" * 60)
    print("Test: Basic Dual-Context Attention")
    print("=" * 60)

    dim = 64
    num_heads = 4
    batch_size = 1
    seq_len = 8

    dca = DualContextAttention(dim=dim, num_heads=num_heads, dropout=0.0)

    # Query
    query = torch.randn(batch_size, seq_len, dim)

    # L0 context
    l0_len = 16
    l0_keys = torch.randn(batch_size, l0_len, dim)
    l0_values = torch.randn(batch_size, l0_len, dim)

    # L1 context (代表ベクトル)
    l1_len = 4
    l1_keys = torch.randn(batch_size, l1_len, dim)
    l1_values = torch.randn(batch_size, l1_len, dim)

    print("\nInput shapes:")
    print(f"  Query: {query.shape}")
    print(f"  L0 (Keys/Values): {l0_keys.shape}")
    print(f"  L1 (Keys/Values): {l1_keys.shape}")

    # Forward
    output = dca(query, l0_keys, l0_values, l1_keys, l1_values)

    print(f"\nOutput shape: {output.output.shape}")
    print(f"Gate value: {torch.sigmoid(dca.gate).item():.4f}")

    if output.l0_attention is not None:
        print(f"L0 attention shape: {output.l0_attention.shape}")
    if output.l1_attention is not None:
        print(f"L1 attention shape: {output.l1_attention.shape}")


def test_memory_update():
    """メモリ更新のテスト。"""
    print("\n" + "=" * 60)
    print("Test: Dual-Context Memory Update")
    print("=" * 60)

    dim = 64
    window_size = 16
    max_representatives = 10

    memory = DualContextMemory(
        dim=dim,
        window_size=window_size,
        max_representatives=max_representatives,
        encoder_mode="bilstm",
    )

    print(f"\nConfig: dim={dim}, window_size={window_size}, max_reps={max_representatives}")
    print("-" * 60)

    # ストリーミングシミュレーション
    total_tokens = 50
    chunk_size = 5

    for i in range(0, total_tokens, chunk_size):
        chunk_len = min(chunk_size, total_tokens - i)
        hidden_chunk = torch.randn(chunk_len, dim)

        state = memory.update(hidden_chunk)

        l0_len = state.l0_keys.size(1) if state.l0_keys is not None else 0
        l1_len = len(state.l1_representatives)

        print(f"\nTokens [{i}-{i+chunk_len-1}]:")
        print(f"  L0 (local): {l0_len} tokens")
        print(f"  L1 (representatives): {l1_len} vectors")
        print(f"  Total processed: {state.total_processed}")

    print("\n" + "-" * 60)
    print("Final Memory State:")
    print(f"  L0 positions: {memory.state.l0_positions}")
    print(f"  L1 positions: {memory.state.l1_positions}")


def test_long_sequence_compression():
    """長シーケンスでの圧縮効果テスト。"""
    print("\n" + "=" * 60)
    print("Test: Long Sequence Compression")
    print("=" * 60)

    dim = 64
    window_size = 32
    max_representatives = 16

    memory = DualContextMemory(
        dim=dim,
        window_size=window_size,
        max_representatives=max_representatives,
    )

    # 長いシーケンスを処理
    total_tokens = 256
    chunk_size = 16

    print(f"\nProcessing {total_tokens} tokens with window_size={window_size}")
    print("-" * 60)

    stats_history = []

    for i in range(0, total_tokens, chunk_size):
        chunk_len = min(chunk_size, total_tokens - i)
        hidden_chunk = torch.randn(chunk_len, dim)

        state = memory.update(hidden_chunk)

        l0_len = state.l0_keys.size(1) if state.l0_keys is not None else 0
        l1_len = len(state.l1_representatives)
        total = state.total_processed
        compression = 1.0 - (l0_len + l1_len) / total if total > 0 else 0

        stats_history.append({
            "processed": total,
            "l0_len": l0_len,
            "l1_len": l1_len,
            "compression": compression,
        })

    # 結果表示
    print(f"\n{'Processed':>10} | {'L0':>5} | {'L1':>5} | {'Compression':>12}")
    print("-" * 40)
    for stats in stats_history[::2]:  # 2つおきに表示
        print(f"{stats['processed']:>10} | {stats['l0_len']:>5} | {stats['l1_len']:>5} | {stats['compression']:>11.1%}")

    final = stats_history[-1]
    print(f"\nFinal: {final['l0_len']} L0 + {final['l1_len']} L1 = {final['l0_len'] + final['l1_len']} total")
    print(f"Original: {total_tokens} tokens")
    print(f"Compression ratio: {final['compression']:.1%}")


def test_attention_distribution():
    """Attention分布のテスト。"""
    print("\n" + "=" * 60)
    print("Test: Attention Distribution Analysis")
    print("=" * 60)

    dim = 64
    num_heads = 4
    batch_size = 1
    seq_len = 4

    dca = DualContextAttention(dim=dim, num_heads=num_heads, dropout=0.0)

    # 異なるL0/L1比率でテスト
    test_cases = [
        {"l0_len": 16, "l1_len": 0, "name": "L0 only"},
        {"l0_len": 16, "l1_len": 4, "name": "L0 + L1 (4:1)"},
        {"l0_len": 8, "l1_len": 8, "name": "L0 + L1 (1:1)"},
        {"l0_len": 0, "l1_len": 16, "name": "L1 only"},
    ]

    query = torch.randn(batch_size, seq_len, dim)

    print(f"\nQuery shape: {query.shape}")
    print("-" * 60)

    for case in test_cases:
        l0_keys = torch.randn(batch_size, case["l0_len"], dim) if case["l0_len"] > 0 else None
        l0_values = torch.randn(batch_size, case["l0_len"], dim) if case["l0_len"] > 0 else None
        l1_keys = torch.randn(batch_size, case["l1_len"], dim) if case["l1_len"] > 0 else None
        l1_values = torch.randn(batch_size, case["l1_len"], dim) if case["l1_len"] > 0 else None

        output = dca(query, l0_keys, l0_values, l1_keys, l1_values)

        print(f"\n{case['name']}:")
        print(f"  Output norm: {output.output.norm().item():.4f}")

        if output.l0_attention is not None:
            l0_attn_mean = output.l0_attention.mean().item()
            print(f"  L0 attention mean: {l0_attn_mean:.4f}")

        if output.l1_attention is not None:
            l1_attn_mean = output.l1_attention.mean().item()
            print(f"  L1 attention mean: {l1_attn_mean:.4f}")


def test_end_to_end():
    """エンドツーエンドのテスト。"""
    print("\n" + "=" * 60)
    print("Test: End-to-End DCA Pipeline")
    print("=" * 60)

    dim = 64
    num_heads = 4
    window_size = 16
    max_representatives = 8

    # メモリとAttentionを初期化
    memory = DualContextMemory(
        dim=dim,
        window_size=window_size,
        max_representatives=max_representatives,
    )
    dca = DualContextAttention(dim=dim, num_heads=num_heads, dropout=0.0)

    print(f"\nConfig: dim={dim}, window={window_size}, max_reps={max_representatives}")
    print("-" * 60)

    # ストリーミング処理
    total_tokens = 64
    query_size = 4

    for step in range(0, total_tokens, query_size):
        # 新しいトークンをメモリに追加
        new_hidden = torch.randn(query_size, dim)
        state = memory.update(new_hidden)

        # Query (最新のトークン)
        query = new_hidden.unsqueeze(0)

        # L1のValueを取得
        l1_values = memory.get_l1_values(batch_size=1)

        # DCA forward
        output = dca(
            query=query,
            l0_keys=state.l0_keys,
            l0_values=state.l0_values,
            l1_keys=state.l1_keys,
            l1_values=l1_values,
        )

        l0_len = state.l0_keys.size(1) if state.l0_keys is not None else 0
        l1_len = len(state.l1_representatives)

        print(f"\nStep {step // query_size + 1}: tokens [{step}-{step+query_size-1}]")
        print(f"  L0: {l0_len}, L1: {l1_len}")
        print(f"  Output shape: {output.output.shape}")
        print(f"  Gate: {torch.sigmoid(dca.gate).item():.4f}")


def compare_with_baseline():
    """通常のAttentionとの比較。"""
    print("\n" + "=" * 60)
    print("Test: Comparison with Full Attention (Baseline)")
    print("=" * 60)

    dim = 64

    # テストシーケンス長
    seq_lengths = [64, 128, 256, 512]

    print(f"\n{'Seq Len':>10} | {'Full Attn':>12} | {'DCA':>12} | {'Memory Ratio':>12}")
    print("-" * 55)

    for seq_len in seq_lengths:
        # Full Attention: O(n^2) メモリ
        full_attn_memory = seq_len * seq_len * dim  # 概算

        # DCA: window + representatives
        window_size = min(64, seq_len)
        num_reps = (seq_len - window_size) // (window_size // 2) if seq_len > window_size else 0
        dca_memory = (window_size + num_reps) * dim

        ratio = dca_memory / full_attn_memory if full_attn_memory > 0 else 1.0

        print(f"{seq_len:>10} | {full_attn_memory:>12,} | {dca_memory:>12,} | {ratio:>11.1%}")

    print("\nNote: Memory estimates are simplified (actual varies by implementation)")


def main():
    print("Dual-Context Attention (DCA) Test Suite")
    print("=" * 60)

    test_basic_dca()
    test_memory_update()
    test_long_sequence_compression()
    test_attention_distribution()
    test_end_to_end()
    compare_with_baseline()

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
