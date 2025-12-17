"""
CASCADEフレームワーク - LLMクラス

任意のHugging Face CausalLMをラップし、Early Exit機能を追加する汎用クラス。
既存の訓練済みLLMも、新規の未学習LLMも、同じLLMクラスでラップする。
"""

from __future__ import annotations

import warnings
import torch
import torch.nn as nn
from typing import Tuple, List, Optional

from transformers import PreTrainedModel

from .exit_fn import ExitFn, default_exit_fn


# 型エイリアス: 意味的な区別を明確化（ドキュメント用）
# TokenTensor: token_ids (整数テンソル、shape: batch_size, seq_len)
# HiddenTensor: hidden_states (浮動小数点テンソル、shape: batch_size, seq_len, dim)
TokenTensor = torch.Tensor
HiddenTensor = torch.Tensor


class LLM(nn.Module):
    """
    Hugging Face CausalLM + Early Exit機能を持つ汎用LLMラッパー。

    任意のHugging Face CausalLM（GPT2LMHeadModel, LlamaForCausalLM等）をラップし、
    Early Exit機能を追加する。LM Headは元モデルのものをそのまま使用。

    設計方針:
        各LLMは独立したモデルとして自己完結する。
        LM Headは元のCausalLMが持つものを使用（独自実装不要）。
        Early Exit機能のみを追加。

    使用例:
        from transformers import AutoModelForCausalLM

        # 既存の訓練済みLLMをラップ
        gpt2 = AutoModelForCausalLM.from_pretrained('gpt2')
        llm = LLM(gpt2)

        # token_idsを入力
        h_out, hidden_history = llm.forward_token_ids(token_ids)

        # hidden_statesを入力
        h_out, hidden_history = llm.forward_hidden_states(hidden_states)

    Args:
        base_llm: ラップするHugging Face CausalLM
        exit_fn: オプションのカスタムexit関数。Noneの場合default_exit_fn（CALM式）を使用
    """

    def __init__(
        self,
        base_llm: PreTrainedModel,
        exit_fn: Optional[ExitFn] = None,
    ):
        super().__init__()
        self.base_llm = base_llm
        self.exit_fn = exit_fn or default_exit_fn
        self.threshold = 0.0  # CascadeTrainerで設定

    @property
    def vocab_size(self) -> int:
        """語彙サイズ（base_llmのconfigから取得）。"""
        return self.base_llm.config.vocab_size

    @property
    def dim(self) -> int:
        """モデル次元（base_llmのconfigから取得）。"""
        config = self.base_llm.config
        # 様々なモデルに対応
        if hasattr(config, 'hidden_size'):
            return config.hidden_size
        if hasattr(config, 'n_embd'):
            return config.n_embd
        if hasattr(config, 'd_model'):
            return config.d_model
        raise AttributeError("モデルの次元を特定できません")

    @property
    def num_layers(self) -> int:
        """レイヤー数（base_llmのconfigから取得）。"""
        config = self.base_llm.config
        # 様々なモデルに対応
        if hasattr(config, 'num_hidden_layers'):
            return config.num_hidden_layers
        if hasattr(config, 'n_layer'):
            return config.n_layer
        if hasattr(config, 'num_layers'):
            return config.num_layers
        raise AttributeError("モデルのレイヤー数を特定できません")

    def forward_token_ids(
        self, token_ids: TokenTensor
    ) -> Tuple[HiddenTensor, List[HiddenTensor]]:
        """
        token_idsを入力として処理し、hidden statesを返す。

        Args:
            token_ids: 入力トークンID (batch_size, seq_len)、整数テンソル

        Returns:
            h_out: 出力hidden states (batch_size, seq_len, dim)
            hidden_history: 各レイヤーのhidden statesリスト（exit判定用）
        """
        outputs = self.base_llm(
            input_ids=token_ids,
            output_hidden_states=True,
            return_dict=True,
        )

        # hidden_statesは (入力埋め込み, layer1出力, layer2出力, ...) のタプル
        hidden_history = list(outputs.hidden_states)
        h_out = hidden_history[-1]

        return h_out, hidden_history

    def forward_hidden_states(
        self, hidden_states: HiddenTensor
    ) -> Tuple[HiddenTensor, List[HiddenTensor]]:
        """
        hidden_statesを入力として処理し、出力hidden statesを返す。

        Args:
            hidden_states: 入力hidden states (batch_size, seq_len, dim)、浮動小数点テンソル

        Returns:
            h_out: 出力hidden states (batch_size, seq_len, dim)
            hidden_history: 各レイヤーのhidden statesリスト（exit判定用）
        """
        # モデルのdtypeに自動変換（float16対応）
        model_dtype = next(self.base_llm.parameters()).dtype
        if hidden_states.dtype != model_dtype:
            hidden_states = hidden_states.to(dtype=model_dtype)

        outputs = self.base_llm(
            inputs_embeds=hidden_states,
            output_hidden_states=True,
            return_dict=True,
        )

        # hidden_statesは (入力埋め込み, layer1出力, layer2出力, ...) のタプル
        hidden_history = list(outputs.hidden_states)
        h_out = hidden_history[-1]

        return h_out, hidden_history

    def forward(
        self, x: torch.Tensor, input_type: str = "token_ids"
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        入力を処理し、hidden statesを返す。

        .. deprecated::
            forward()は非推奨です。代わりにforward_token_ids()または
            forward_hidden_states()を使用してください。

        Args:
            x: 入力（token_ids or hidden states）
            input_type: "token_ids" または "hidden_states"

        Returns:
            h_out: 出力hidden states (batch_size, seq_len, dim)
            hidden_history: 各レイヤーのhidden statesリスト（exit判定用）
        """
        warnings.warn(
            "forward()は非推奨です。forward_token_ids()または"
            "forward_hidden_states()を使用してください。",
            DeprecationWarning,
            stacklevel=2,
        )
        if input_type == "token_ids":
            return self.forward_token_ids(x)
        else:
            return self.forward_hidden_states(x)

    def get_logits(self, h: HiddenTensor) -> torch.Tensor:
        """
        hidden statesからlogitsを計算。

        Args:
            h: hidden states (batch_size, seq_len, dim)

        Returns:
            logits: (batch_size, seq_len, vocab_size)
        """
        # CausalLMのlm_headを使用
        return self.base_llm.lm_head(h)

    def should_exit(self, hidden_history: List[HiddenTensor]) -> torch.Tensor:
        """
        exit判定。

        Args:
            hidden_history: 各レイヤーのhidden statesリスト

        Returns:
            should_exit: Booleanマスク True=exitすべき (batch_size, seq_len)
        """
        return self.exit_fn(hidden_history, self.threshold)
