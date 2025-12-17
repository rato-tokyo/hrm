"""
Dual-Context Attention (DCA) - Colab実験スクリプト

Google Colabで即座に実行可能な自己完結型スクリプト。
依存関係: torch のみ

実験内容:
1. DCA基本動作テスト
2. 実LLM (GPT-2) での推論比較
3. 長シーケンスでの圧縮効果測定
4. Attention分布の可視化

使用方法:
    # Colabで実行
    !pip install torch transformers
    # このスクリプト全体をセルにコピー＆実行
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, NamedTuple
from dataclasses import dataclass, field
import time


# ========== DCA実装 ==========

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
    """双方向エンコーダでspanを圧縮。"""

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


# ========== 実験関数 ==========


def experiment_basic_dca(device):
    """実験1: 基本的なDCA動作テスト"""
    print("=" * 70)
    print("Experiment 1: Basic DCA Operation")
    print("=" * 70)

    dim = 768
    num_heads = 12
    batch_size = 1
    seq_len = 32

    dca = DualContextAttention(dim=dim, num_heads=num_heads, dropout=0.0).to(device)

    # Query
    query = torch.randn(batch_size, seq_len, dim, device=device)

    # L0 context (最新のKV)
    l0_len = 128
    l0_keys = torch.randn(batch_size, l0_len, dim, device=device)
    l0_values = torch.randn(batch_size, l0_len, dim, device=device)

    # L1 context (代表ベクトル)
    l1_len = 16
    l1_keys = torch.randn(batch_size, l1_len, dim, device=device)
    l1_values = torch.randn(batch_size, l1_len, dim, device=device)

    print("\nConfiguration:")
    print(f"  dim={dim}, num_heads={num_heads}")
    print(f"  Query: {query.shape}")
    print(f"  L0 (local): {l0_keys.shape}")
    print(f"  L1 (representatives): {l1_keys.shape}")

    # Forward
    output = dca(query, l0_keys, l0_values, l1_keys, l1_values)

    print("\nResults:")
    print(f"  Output shape: {output.output.shape}")
    print(f"  Gate value: {torch.sigmoid(dca.gate).item():.4f}")
    print(f"  L0 attention shape: {output.l0_attention.shape if output.l0_attention is not None else 'None'}")
    print(f"  L1 attention shape: {output.l1_attention.shape if output.l1_attention is not None else 'None'}")


def experiment_gpt2_comparison(device):
    """実験2: GPT-2の隠れ状態を使ったDCA実験"""
    print("\n" + "=" * 70)
    print("Experiment 2: DCA with GPT-2 Hidden States")
    print("=" * 70)

    try:
        from transformers import GPT2LMHeadModel, GPT2Tokenizer
    except ImportError:
        print("transformers not installed. Skipping GPT-2 experiment.")
        return

    print("\nLoading GPT-2...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    model.eval()

    dim = model.config.n_embd  # 768
    num_heads = model.config.n_head  # 12

    # テキスト
    text = """The quick brown fox jumps over the lazy dog.
    This is a longer text to test the dual-context attention mechanism.
    We want to see how the model handles context from both recent tokens (L0)
    and compressed historical representations (L1)."""

    inputs = tokenizer(text, return_tensors="pt").to(device)
    seq_len = inputs["input_ids"].size(1)

    print(f"\nInput: {seq_len} tokens")

    # GPT-2でhidden statesを取得
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # 最終層の隠れ状態を使用
    hidden_states = outputs.hidden_states[-1]  # (batch, seq_len, dim)

    # DCAメモリを設定
    window_size = 32
    memory = DualContextMemory(
        dim=dim,
        window_size=window_size,
        max_representatives=16,
        encoder_mode="bilstm",
    ).to(device)

    dca = DualContextAttention(dim=dim, num_heads=num_heads, dropout=0.0).to(device)

    # ストリーミングシミュレーション
    print(f"\nStreaming simulation (window_size={window_size}):")
    print("-" * 50)

    chunk_size = 16
    for i in range(0, seq_len, chunk_size):
        end_idx = min(i + chunk_size, seq_len)
        chunk = hidden_states[:, i:end_idx, :]

        state = memory.update(chunk.squeeze(0))

        l0_len = state.l0_keys.size(1) if state.l0_keys is not None else 0
        l1_len = len(state.l1_representatives)

        # Query: 最後のトークン
        query = chunk[:, -1:, :]

        # DCA forward
        l1_values = memory.get_l1_values(batch_size=1)
        if l1_values is not None:
            l1_values = l1_values.to(device)

        _ = dca(
            query=query,
            l0_keys=state.l0_keys,
            l0_values=state.l0_values,
            l1_keys=state.l1_keys,
            l1_values=l1_values,
        )

        print(f"  Tokens [{i:3d}-{end_idx-1:3d}]: L0={l0_len:3d}, L1={l1_len:2d}, "
              f"gate={torch.sigmoid(dca.gate).item():.3f}")

    # 最終統計
    total = state.total_processed
    context_size = l0_len + l1_len
    compression = 1.0 - context_size / total if total > 0 else 0

    print("\nFinal Statistics:")
    print(f"  Total tokens: {total}")
    print(f"  Context size: {context_size} (L0={l0_len}, L1={l1_len})")
    print(f"  Compression: {compression:.1%}")


def experiment_long_sequence(device):
    """実験3: 長シーケンスでの圧縮効果"""
    print("\n" + "=" * 70)
    print("Experiment 3: Long Sequence Compression")
    print("=" * 70)

    dim = 768
    window_sizes = [64, 128, 256]
    seq_lengths = [256, 512, 1024, 2048]

    print(f"\n{'Seq Len':>10} | {'Window':>8} | {'L0':>6} | {'L1':>6} | {'Total':>6} | {'Compression':>12}")
    print("-" * 65)

    for window_size in window_sizes:
        for seq_len in seq_lengths:
            memory = DualContextMemory(
                dim=dim,
                window_size=window_size,
                max_representatives=256,
                encoder_mode="pooling",  # 高速化のため
            ).to(device)

            # シーケンス処理
            chunk_size = 64
            for i in range(0, seq_len, chunk_size):
                chunk_len = min(chunk_size, seq_len - i)
                chunk = torch.randn(chunk_len, dim, device=device)
                memory.update(chunk)

            state = memory.state
            l0_len = len(memory._l0_buffer)
            l1_len = len(state.l1_representatives)
            total_ctx = l0_len + l1_len
            compression = 1.0 - total_ctx / seq_len

            print(f"{seq_len:>10} | {window_size:>8} | {l0_len:>6} | {l1_len:>6} | {total_ctx:>6} | {compression:>11.1%}")


def experiment_memory_benchmark(device):
    """実験4: メモリ使用量ベンチマーク"""
    print("\n" + "=" * 70)
    print("Experiment 4: Memory Usage Benchmark")
    print("=" * 70)

    num_heads = 12
    batch_size = 1
    seq_lengths = [512, 1024, 2048, 4096]

    print(f"\n{'Seq Len':>10} | {'Full Attn (MB)':>15} | {'DCA (MB)':>12} | {'Ratio':>10}")
    print("-" * 55)

    for seq_len in seq_lengths:
        # Full Attention: Q, K, V + attention matrix
        # Memory = batch * heads * seq * seq * 4 bytes (float32)
        full_attn_bytes = batch_size * num_heads * seq_len * seq_len * 4
        full_attn_mb = full_attn_bytes / (1024 * 1024)

        # DCA: window + representatives
        window_size = 256
        num_reps = max(0, (seq_len - window_size) // (window_size // 2))
        num_reps = min(num_reps, 256)  # max_representatives
        dca_context = window_size + num_reps

        # DCA Memory = batch * heads * query * context * 4 bytes
        query_len = 32  # typical query length
        dca_bytes = batch_size * num_heads * query_len * dca_context * 4
        dca_mb = dca_bytes / (1024 * 1024)

        ratio = dca_mb / full_attn_mb if full_attn_mb > 0 else 0

        print(f"{seq_len:>10} | {full_attn_mb:>15.2f} | {dca_mb:>12.2f} | {ratio:>9.2%}")


def experiment_attention_visualization(device):
    """実験5: Attention分布の可視化"""
    print("\n" + "=" * 70)
    print("Experiment 5: Attention Distribution Analysis")
    print("=" * 70)

    dim = 768
    num_heads = 12
    batch_size = 1
    query_len = 4

    dca = DualContextAttention(dim=dim, num_heads=num_heads, dropout=0.0).to(device)
    query = torch.randn(batch_size, query_len, dim, device=device)

    # 異なるL0/L1比率でテスト
    configs = [
        {"l0": 64, "l1": 0, "name": "L0 only (64)"},
        {"l0": 64, "l1": 16, "name": "L0(64) + L1(16)"},
        {"l0": 32, "l1": 32, "name": "L0(32) + L1(32)"},
        {"l0": 16, "l1": 48, "name": "L0(16) + L1(48)"},
    ]

    print(f"\nQuery shape: {query.shape}")
    print("-" * 60)

    for config in configs:
        l0_keys = torch.randn(batch_size, config["l0"], dim, device=device) if config["l0"] > 0 else None
        l0_values = torch.randn(batch_size, config["l0"], dim, device=device) if config["l0"] > 0 else None
        l1_keys = torch.randn(batch_size, config["l1"], dim, device=device) if config["l1"] > 0 else None
        l1_values = torch.randn(batch_size, config["l1"], dim, device=device) if config["l1"] > 0 else None

        output = dca(query, l0_keys, l0_values, l1_keys, l1_values)

        print(f"\n{config['name']}:")

        if output.l0_attention is not None:
            l0_attn = output.l0_attention[0]  # (query_len, l0_len)
            l0_max = l0_attn.max(dim=-1).values.mean().item()
            l0_entropy = -(l0_attn * torch.log(l0_attn + 1e-10)).sum(dim=-1).mean().item()
            print(f"  L0: max_attn={l0_max:.4f}, entropy={l0_entropy:.4f}")

        if output.l1_attention is not None:
            l1_attn = output.l1_attention[0]
            l1_max = l1_attn.max(dim=-1).values.mean().item()
            l1_entropy = -(l1_attn * torch.log(l1_attn + 1e-10)).sum(dim=-1).mean().item()
            print(f"  L1: max_attn={l1_max:.4f}, entropy={l1_entropy:.4f}")


def experiment_latency_comparison(device):
    """実験6: 推論レイテンシ比較"""
    print("\n" + "=" * 70)
    print("Experiment 6: Inference Latency Comparison")
    print("=" * 70)

    dim = 768
    num_heads = 12
    batch_size = 1
    query_len = 32
    num_iterations = 100

    # Standard Self-Attention
    class StandardAttention(nn.Module):
        def __init__(self, dim, num_heads):
            super().__init__()
            self.num_heads = num_heads
            self.head_dim = dim // num_heads
            self.q_proj = nn.Linear(dim, dim)
            self.k_proj = nn.Linear(dim, dim)
            self.v_proj = nn.Linear(dim, dim)
            self.out_proj = nn.Linear(dim, dim)
            self.scale = self.head_dim ** -0.5

        def forward(self, x, kv_cache=None):
            batch, seq, _ = x.shape
            q = self.q_proj(x).view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)

            if kv_cache is not None:
                k = self.k_proj(kv_cache).view(batch, -1, self.num_heads, self.head_dim).transpose(1, 2)
                v = self.v_proj(kv_cache).view(batch, -1, self.num_heads, self.head_dim).transpose(1, 2)
            else:
                k = self.k_proj(x).view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
                v = self.v_proj(x).view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)

            attn = F.softmax(torch.matmul(q, k.transpose(-2, -1)) * self.scale, dim=-1)
            out = torch.matmul(attn, v)
            return self.out_proj(out.transpose(1, 2).contiguous().view(batch, seq, -1))

    context_sizes = [256, 512, 1024, 2048]

    print(f"\nLatency comparison (ms, {num_iterations} iterations):")
    print(f"{'Context':>10} | {'Standard':>12} | {'DCA':>12} | {'Speedup':>10}")
    print("-" * 50)

    for ctx_size in context_sizes:
        # Standard Attention
        std_attn = StandardAttention(dim, num_heads).to(device)
        query = torch.randn(batch_size, query_len, dim, device=device)
        kv_cache = torch.randn(batch_size, ctx_size, dim, device=device)

        # Warmup
        for _ in range(10):
            _ = std_attn(query, kv_cache)

        torch.cuda.synchronize() if device.type == 'cuda' else None
        start = time.time()
        for _ in range(num_iterations):
            _ = std_attn(query, kv_cache)
        torch.cuda.synchronize() if device.type == 'cuda' else None
        std_time = (time.time() - start) / num_iterations * 1000

        # DCA
        dca = DualContextAttention(dim, num_heads, dropout=0.0).to(device)
        window_size = min(256, ctx_size)
        num_reps = max(0, (ctx_size - window_size) // (window_size // 2))
        num_reps = min(num_reps, 64)

        l0_keys = torch.randn(batch_size, window_size, dim, device=device)
        l0_values = torch.randn(batch_size, window_size, dim, device=device)
        l1_keys = torch.randn(batch_size, num_reps, dim, device=device) if num_reps > 0 else None
        l1_values = torch.randn(batch_size, num_reps, dim, device=device) if num_reps > 0 else None

        # Warmup
        for _ in range(10):
            _ = dca(query, l0_keys, l0_values, l1_keys, l1_values)

        torch.cuda.synchronize() if device.type == 'cuda' else None
        start = time.time()
        for _ in range(num_iterations):
            _ = dca(query, l0_keys, l0_values, l1_keys, l1_values)
        torch.cuda.synchronize() if device.type == 'cuda' else None
        dca_time = (time.time() - start) / num_iterations * 1000

        speedup = std_time / dca_time if dca_time > 0 else 0

        print(f"{ctx_size:>10} | {std_time:>11.3f} | {dca_time:>11.3f} | {speedup:>9.2f}x")


def main():
    """メイン実験関数"""
    print("=" * 70)
    print("Dual-Context Attention (DCA) Experiments")
    print("=" * 70)

    # デバイス設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # 実験実行
    experiment_basic_dca(device)
    experiment_gpt2_comparison(device)
    experiment_long_sequence(device)
    experiment_memory_benchmark(device)
    experiment_attention_visualization(device)
    experiment_latency_comparison(device)

    print("\n" + "=" * 70)
    print("All experiments completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
