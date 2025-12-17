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
from typing import Optional
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

        # DCA Attention
        dca_output = self.dca_attention(
            query=hidden_states,
            l0_keys=l0_keys,
            l0_values=l0_values,
            l1_keys=l1_keys,
            l1_values=l1_values,
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
