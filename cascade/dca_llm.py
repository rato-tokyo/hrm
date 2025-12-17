"""
DCA-LLM: Dual-Context Attention統合言語モデル

ベースLLM（GPT-2等）にDual-Context Attentionを統合し、
長系列のメモリ効率的な処理を可能にする。

アーキテクチャ:
- Base LLM: 通常のTransformerブロック（frozen or trainable）
- DCA Layer: L0（ローカル）+ L1（圧縮済み過去コンテキスト）
- LM Head: ベースLLMのlm_headを使用

推論フロー:
1. Input tokens → Base LLM embedding
2. Hidden states → DCA Memory（L0/L1管理）
3. DCA Attention → 出力hidden states
4. LM Head → logits → next token
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple
from dataclasses import dataclass

from transformers import PreTrainedModel, AutoModelForCausalLM

from .dual_context_attention import (
    DualContextMemory,
    DualContextAttention,
    DualContextState,
)


@dataclass
class DCALLMOutput:
    """DCA-LLMの出力。"""
    logits: Tensor  # (batch, seq_len, vocab_size)
    loss: Optional[Tensor] = None
    hidden_states: Optional[Tensor] = None
    dca_state: Optional[DualContextState] = None


class DCALLM(nn.Module):
    """
    Dual-Context Attention統合言語モデル。

    ベースLLM（GPT-2等）の出力hidden statesにDCAを適用し、
    長系列を効率的に処理する。

    使用例:
        from transformers import AutoModelForCausalLM

        base_llm = AutoModelForCausalLM.from_pretrained('gpt2')
        dca_llm = DCALLM(base_llm, window_size=512)

        # 訓練
        outputs = dca_llm(input_ids, labels=labels)
        loss = outputs.loss

        # 推論（ストリーミング）
        dca_llm.reset_memory()
        for chunk in chunks:
            outputs = dca_llm(chunk)
            next_token = outputs.logits[:, -1, :].argmax(dim=-1)
    """

    def __init__(
        self,
        base_llm: PreTrainedModel,
        window_size: int = 512,
        max_representatives: int = 256,
        encoder_mode: str = "pooling",
        encoder_layers: int = 1,
        num_heads: Optional[int] = None,
        dropout: float = 0.1,
        freeze_base: bool = False,
        use_dca: bool = True,
    ):
        """
        Args:
            base_llm: ベースとなるHugging Face CausalLM
            window_size: L0のウィンドウサイズ
            max_representatives: L1の最大代表ベクトル数
            encoder_mode: span圧縮エンコーダのモード
            encoder_layers: エンコーダのレイヤー数
            num_heads: DCA Attentionのヘッド数（Noneならbase_llmから取得）
            dropout: ドロップアウト率
            freeze_base: ベースLLMをfreezeするか
            use_dca: DCAを使用するか（Falseならベースラインとして動作）
        """
        super().__init__()
        self.base_llm = base_llm
        self.use_dca = use_dca

        # モデル設定を取得
        self.dim = self._get_dim(base_llm.config)
        self.vocab_size = base_llm.config.vocab_size
        self._num_heads = num_heads or self._get_num_heads(base_llm.config)

        # ベースLLMのfreeze
        if freeze_base:
            for param in base_llm.parameters():
                param.requires_grad = False

        if use_dca:
            # DCA Memory
            self.dca_memory = DualContextMemory(
                dim=self.dim,
                window_size=window_size,
                max_representatives=max_representatives,
                encoder_mode=encoder_mode,
                encoder_layers=encoder_layers,
                encoder_heads=self._num_heads,
                dropout=dropout,
            )

            # DCA Attention Layer
            self.dca_attention = DualContextAttention(
                dim=self.dim,
                num_heads=self._num_heads,
                dropout=dropout,
                combine_mode="gate",
            )

            # DCA後の処理
            self.dca_ffn = nn.Sequential(
                nn.LayerNorm(self.dim),
                nn.Linear(self.dim, self.dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(self.dim * 4, self.dim),
                nn.Dropout(dropout),
            )
            self.dca_ln = nn.LayerNorm(self.dim)

    def _get_dim(self, config) -> int:
        """モデル次元を取得。"""
        if hasattr(config, 'hidden_size'):
            return config.hidden_size
        if hasattr(config, 'n_embd'):
            return config.n_embd
        if hasattr(config, 'd_model'):
            return config.d_model
        raise AttributeError("モデルの次元を特定できません")

    def _get_num_heads(self, config) -> int:
        """ヘッド数を取得。"""
        if hasattr(config, 'num_attention_heads'):
            return config.num_attention_heads
        if hasattr(config, 'n_head'):
            return config.n_head
        raise AttributeError("モデルのヘッド数を特定できません")

    def reset_memory(self):
        """DCAメモリをリセット。"""
        if self.use_dca:
            self.dca_memory.reset()

    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        return_hidden_states: bool = False,
    ) -> DCALLMOutput:
        """
        Forward pass。

        Args:
            input_ids: (batch, seq_len) 入力トークンID
            attention_mask: (batch, seq_len) アテンションマスク
            labels: (batch, seq_len) ラベル（訓練時）
            inputs_embeds: (batch, seq_len, dim) 入力埋め込み（input_idsの代わり）
            use_cache: KVキャッシュを使用するか
            return_hidden_states: hidden statesを返すか

        Returns:
            DCALLMOutput
        """
        # ベースLLMでhidden statesを取得
        base_outputs = self.base_llm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_hidden_states=True,
            return_dict=True,
        )

        # 最終層のhidden states
        hidden_states = base_outputs.hidden_states[-1]
        batch_size, seq_len, _ = hidden_states.shape

        if self.use_dca:
            # DCAを適用
            hidden_states = self._apply_dca(hidden_states)

        # LM Headでlogits計算
        logits = self.base_llm.lm_head(hidden_states)

        # Loss計算
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return DCALLMOutput(
            logits=logits,
            loss=loss,
            hidden_states=hidden_states if return_hidden_states else None,
            dca_state=self.dca_memory.state if self.use_dca else None,
        )

    def _apply_dca(self, hidden_states: Tensor) -> Tensor:
        """
        DCAを適用。

        訓練時: 各バッチは独立に処理（メモリはリセット済み）
        L0 = hidden_states自体をKey/Valueとして使用（self-attention的）
        L1 = 現在は空（ストリーミング推論時のみ使用）

        Args:
            hidden_states: (batch, seq_len, dim)

        Returns:
            output: (batch, seq_len, dim)
        """
        batch_size, seq_len, _ = hidden_states.shape

        # 訓練時はhidden_states自体をL0として使用（メモリのストリーミングは不要）
        # これにより各バッチが独立に処理される
        l0_keys = self.dca_memory.l0_key_proj(hidden_states)
        l0_values = self.dca_memory.l0_value_proj(hidden_states)

        # L1は訓練時は使用しない（ストリーミング推論時のみ）
        l1_keys = None
        l1_values = None

        # Causal mask: 下三角行列（位置iは位置i以前のみattend可能）
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=hidden_states.device))

        # DCA Attention
        dca_output = self.dca_attention(
            query=hidden_states,
            l0_keys=l0_keys,
            l0_values=l0_values,
            l1_keys=l1_keys,
            l1_values=l1_values,
            l0_mask=causal_mask,
        )

        # Residual + FFN
        hidden_states = hidden_states + dca_output.output
        hidden_states = hidden_states + self.dca_ffn(self.dca_ln(hidden_states))

        return hidden_states

    @torch.no_grad()
    def generate(
        self,
        input_ids: Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: bool = False,
        eos_token_id: Optional[int] = None,
    ) -> Tensor:
        """
        テキスト生成。

        Args:
            input_ids: (batch, seq_len) 入力トークンID
            max_new_tokens: 生成する最大トークン数
            temperature: サンプリング温度
            top_k: Top-k サンプリング
            top_p: Nucleus (top-p) サンプリング
            do_sample: サンプリングを使用するか
            eos_token_id: 終了トークンID

        Returns:
            generated_ids: (batch, seq_len + max_new_tokens)
        """
        self.eval()
        self.reset_memory()

        generated = input_ids.clone()

        for _ in range(max_new_tokens):
            # Forward
            outputs = self.forward(input_ids=generated)
            next_token_logits = outputs.logits[:, -1, :]

            # Temperature scaling
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature

            # Top-k filtering
            if top_k is not None:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')

            # Top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = float('-inf')

            # サンプリング or Greedy
            if do_sample:
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = next_token_logits.argmax(dim=-1, keepdim=True)

            generated = torch.cat([generated, next_token], dim=-1)

            # EOS check
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

        return generated


def create_dca_llm(
    model_name: str = "gpt2",
    window_size: int = 512,
    max_representatives: int = 256,
    encoder_mode: str = "pooling",
    freeze_base: bool = False,
    use_dca: bool = True,
    device: Optional[str] = None,
) -> DCALLM:
    """
    DCA-LLMを作成するファクトリ関数。

    Args:
        model_name: ベースモデル名
        window_size: L0のウィンドウサイズ
        max_representatives: L1の最大代表ベクトル数
        encoder_mode: span圧縮エンコーダのモード
        freeze_base: ベースLLMをfreezeするか
        use_dca: DCAを使用するか（Falseならベースライン）
        device: デバイス

    Returns:
        DCALLM インスタンス
    """
    base_llm = AutoModelForCausalLM.from_pretrained(model_name)

    dca_llm = DCALLM(
        base_llm=base_llm,
        window_size=window_size,
        max_representatives=max_representatives,
        encoder_mode=encoder_mode,
        freeze_base=freeze_base,
        use_dca=use_dca,
    )

    if device:
        dca_llm = dca_llm.to(device)

    return dca_llm


class IntegratedDCABlock(nn.Module):
    """
    DCAを内蔵したTransformerブロック（L0/L1 2層構造）。

    L0: ローカルコンテキスト（ウィンドウ内の詳細なattention）
    L1: 圧縮コンテキスト（ウィンドウ外の要約情報）

    訓練時は長いシーケンスを分割してL0/L1を使用。
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int = 256,
        compression_ratio: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.compression_ratio = compression_ratio

        # Q projection (shared)
        self.q_proj = nn.Linear(dim, dim)

        # L0: Local context (within window)
        self.l0_k_proj = nn.Linear(dim, dim)
        self.l0_v_proj = nn.Linear(dim, dim)

        # L1: Compressed context (outside window)
        self.l1_k_proj = nn.Linear(dim, dim)
        self.l1_v_proj = nn.Linear(dim, dim)

        # Compression layer for L1 (average pooling + linear projection)
        self.l1_compressor = nn.Linear(dim, dim)

        # Output projection
        self.out_proj = nn.Linear(dim, dim)

        # Gating mechanism to balance L0 and L1
        self.gate = nn.Linear(dim, 2)

        self.scale = self.head_dim ** -0.5
        self.dropout = nn.Dropout(dropout)

        # FFN
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

    def _compress_context(self, hidden: Tensor) -> Tensor:
        """
        過去のコンテキストを圧縮してL1表現を作成。

        Args:
            hidden: (batch, past_len, dim)

        Returns:
            compressed: (batch, past_len // compression_ratio, dim)
        """
        batch_size, seq_len, dim = hidden.shape

        if seq_len == 0:
            return hidden

        # Pad to make divisible by compression_ratio
        pad_len = (self.compression_ratio - seq_len % self.compression_ratio) % self.compression_ratio
        if pad_len > 0:
            hidden = F.pad(hidden, (0, 0, 0, pad_len))

        # Reshape and average pool
        new_len = hidden.size(1) // self.compression_ratio
        hidden = hidden.view(batch_size, new_len, self.compression_ratio, dim)
        compressed = hidden.mean(dim=2)  # (batch, new_len, dim)

        # Project
        compressed = self.l1_compressor(compressed)
        return compressed

    def forward(
        self,
        hidden_states: Tensor,
        causal_mask: Tensor,
        past_context: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward pass with dual-context attention.

        Args:
            hidden_states: (batch, seq_len, dim) - 現在のウィンドウ
            causal_mask: (seq_len, seq_len) - causal mask for L0
            past_context: (batch, past_len, dim) - 過去のコンテキスト（L1用）

        Returns:
            output: (batch, seq_len, dim)
            current_context: hidden_states（次のブロック用）
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Pre-norm
        normed = self.ln1(hidden_states)

        # Query projection
        q = self.q_proj(normed)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # === L0: Local Context Attention ===
        l0_k = self.l0_k_proj(normed)
        l0_v = self.l0_v_proj(normed)
        l0_k = l0_k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        l0_v = l0_v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # L0 attention with causal mask
        attn_l0 = torch.matmul(q, l0_k.transpose(-2, -1)) * self.scale
        attn_l0 = attn_l0.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0) == 0, float('-inf'))
        attn_l0 = F.softmax(attn_l0, dim=-1)
        attn_l0 = self.dropout(attn_l0)
        out_l0 = torch.matmul(attn_l0, l0_v)  # (batch, heads, seq, head_dim)

        # === L1: Compressed Context Attention ===
        if past_context is not None and past_context.size(1) > 0:
            # Compress past context
            compressed = self._compress_context(past_context)
            comp_len = compressed.size(1)

            # L1 K/V projections
            l1_k = self.l1_k_proj(compressed)
            l1_v = self.l1_v_proj(compressed)
            l1_k = l1_k.view(batch_size, comp_len, self.num_heads, self.head_dim).transpose(1, 2)
            l1_v = l1_v.view(batch_size, comp_len, self.num_heads, self.head_dim).transpose(1, 2)

            # L1 attention (no causal mask - all past is visible)
            attn_l1 = torch.matmul(q, l1_k.transpose(-2, -1)) * self.scale
            attn_l1 = F.softmax(attn_l1, dim=-1)
            attn_l1 = self.dropout(attn_l1)
            out_l1 = torch.matmul(attn_l1, l1_v)  # (batch, heads, seq, head_dim)

            # Gating: learn to balance L0 and L1
            gate_input = normed.mean(dim=1)  # (batch, dim)
            gate_weights = F.softmax(self.gate(gate_input), dim=-1)  # (batch, 2)
            gate_l0 = gate_weights[:, 0].view(batch_size, 1, 1, 1)
            gate_l1 = gate_weights[:, 1].view(batch_size, 1, 1, 1)

            # Combine L0 and L1
            out = gate_l0 * out_l0 + gate_l1 * out_l1
        else:
            # No past context, use L0 only
            out = out_l0

        # Reshape and project
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)
        out = self.out_proj(out)

        # Residual connection
        hidden_states = hidden_states + out

        # FFN with pre-norm and residual
        hidden_states = hidden_states + self.ffn(self.ln2(hidden_states))

        return hidden_states, normed  # Return normed as context for next window


class IntegratedDCALLM(nn.Module):
    """
    DCAを内部に統合した言語モデル（L0/L1 2層構造）。

    長いシーケンスをウィンドウに分割し、各ウィンドウで:
    - L0: 現在のウィンドウ内でcausal attention
    - L1: 過去のウィンドウを圧縮してattention

    使用例:
        model = IntegratedDCALLM(vocab_size=50257, dim=256, num_layers=4)
        outputs = model(input_ids, labels=labels)
    """

    def __init__(
        self,
        vocab_size: int,
        dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        max_seq_len: int = 1024,
        window_size: int = 256,
        compression_ratio: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.window_size = window_size
        self.compression_ratio = compression_ratio

        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.position_embedding = nn.Embedding(max_seq_len, dim)
        self.embed_dropout = nn.Dropout(dropout)

        # DCA Blocks
        self.blocks = nn.ModuleList([
            IntegratedDCABlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                compression_ratio=compression_ratio,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        # Output
        self.ln_f = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.token_embedding.weight

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights similar to GPT-2."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                torch.nn.init.ones_(module.weight)
                torch.nn.init.zeros_(module.bias)

    def reset_memory(self):
        """Compatibility method (no-op for training)."""
        pass

    def forward(
        self,
        input_ids: Tensor,
        labels: Optional[Tensor] = None,
    ) -> DCALLMOutput:
        """
        Forward pass with windowed DCA.

        長いシーケンスをwindow_sizeに分割し、各ウィンドウで:
        - L0: 現在のウィンドウ内のcausal attention
        - L1: 過去のウィンドウを圧縮したattention

        Args:
            input_ids: (batch, seq_len)
            labels: (batch, seq_len) for loss calculation

        Returns:
            DCALLMOutput
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Embeddings
        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        hidden_states = self.token_embedding(input_ids) + self.position_embedding(positions)
        hidden_states = self.embed_dropout(hidden_states)

        # Split into windows
        num_windows = (seq_len + self.window_size - 1) // self.window_size
        pad_len = num_windows * self.window_size - seq_len

        if pad_len > 0:
            hidden_states = F.pad(hidden_states, (0, 0, 0, pad_len))

        # Process windows
        all_outputs = []

        for window_idx in range(num_windows):
            start_idx = window_idx * self.window_size
            end_idx = start_idx + self.window_size
            window_hidden = hidden_states[:, start_idx:end_idx, :]

            # Causal mask for this window
            causal_mask = torch.tril(torch.ones(
                self.window_size, self.window_size, device=device
            ))

            # Collect past context from previous windows (all layers)
            if window_idx > 0:
                past_context = hidden_states[:, :start_idx, :]
            else:
                past_context = None

            # Forward through all DCA blocks
            layer_contexts = []
            for block in self.blocks:
                window_hidden, context = block(window_hidden, causal_mask, past_context)
                layer_contexts.append(context)

            all_outputs.append(window_hidden)

        # Concatenate all window outputs
        hidden_states = torch.cat(all_outputs, dim=1)

        # Remove padding
        if pad_len > 0:
            hidden_states = hidden_states[:, :seq_len, :]

        # Output
        hidden_states = self.ln_f(hidden_states)
        logits = self.lm_head(hidden_states)

        # Loss
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return DCALLMOutput(logits=logits, loss=loss)


def create_integrated_dca_llm(
    vocab_size: int,
    dim: int = 256,
    num_layers: int = 4,
    num_heads: int = 4,
    max_seq_len: int = 1024,
    window_size: int = 256,
    compression_ratio: int = 4,
    dropout: float = 0.1,
    device: Optional[str] = None,
) -> IntegratedDCALLM:
    """
    DCA統合言語モデルを作成するファクトリ関数。

    Args:
        vocab_size: 語彙サイズ
        dim: モデル次元
        num_layers: レイヤー数
        num_heads: ヘッド数
        max_seq_len: 最大シーケンス長
        window_size: L0のウィンドウサイズ
        compression_ratio: L1の圧縮率
        dropout: ドロップアウト率
        device: デバイス

    Returns:
        IntegratedDCALLM インスタンス
    """
    model = IntegratedDCALLM(
        vocab_size=vocab_size,
        dim=dim,
        num_layers=num_layers,
        num_heads=num_heads,
        max_seq_len=max_seq_len,
        window_size=window_size,
        compression_ratio=compression_ratio,
        dropout=dropout,
    )

    if device:
        model = model.to(device)

    return model


def create_dca_llm_from_scratch(
    vocab_size: int,
    dim: int = 768,
    num_layers: int = 6,
    num_heads: int = 12,
    max_seq_len: int = 1024,
    window_size: int = 512,
    max_representatives: int = 256,
    encoder_mode: str = "pooling",
    use_dca: bool = True,
    device: Optional[str] = None,
) -> DCALLM:
    """
    新規にDCA-LLMを作成するファクトリ関数（事前学習用）。

    Args:
        vocab_size: 語彙サイズ
        dim: モデル次元
        num_layers: Transformerレイヤー数
        num_heads: Attentionヘッド数
        max_seq_len: 最大シーケンス長
        window_size: L0のウィンドウサイズ
        max_representatives: L1の最大代表ベクトル数
        encoder_mode: span圧縮エンコーダのモード
        use_dca: DCAを使用するか
        device: デバイス

    Returns:
        DCALLM インスタンス
    """
    from transformers import GPT2Config, GPT2LMHeadModel

    config = GPT2Config(
        vocab_size=vocab_size,
        n_embd=dim,
        n_head=num_heads,
        n_layer=num_layers,
        n_positions=max_seq_len,
    )
    base_llm = GPT2LMHeadModel(config)

    dca_llm = DCALLM(
        base_llm=base_llm,
        window_size=window_size,
        max_representatives=max_representatives,
        encoder_mode=encoder_mode,
        freeze_base=False,
        use_dca=use_dca,
    )

    if device:
        dca_llm = dca_llm.to(device)

    return dca_llm
