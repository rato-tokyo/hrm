"""
CASCADEフレームワーク - LLMクラス

任意のHugging Face CausalLMをラップし、Early Exit機能を追加する汎用クラス。
既存の訓練済みLLMも、新規の未学習LLMも、同じLLMクラスでラップする。
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Tuple, List, Optional, TYPE_CHECKING

from transformers import PreTrainedModel

from .exit_fn import ExitFn, default_exit_fn, compute_cos_sim

if TYPE_CHECKING:
    from .sequence_data import SequenceData


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

        # Ensembleで統合
        ensemble = Ensemble([llm_0, llm_1])

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
        self.threshold = 0.0  # collect_hard_tokensで設定

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

    def forward(
        self, x: torch.Tensor, input_type: str = "token_ids"
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        入力を処理し、hidden statesを返す。

        Args:
            x: 入力（token_ids or hidden states）
            input_type: "token_ids" または "hidden_states"

        Returns:
            h_out: 出力hidden states (batch_size, seq_len, dim)
            hidden_history: 各レイヤーのhidden statesリスト（exit判定用）
        """
        if input_type == "token_ids":
            outputs = self.base_llm(
                input_ids=x,
                output_hidden_states=True,
                return_dict=True,
            )
        else:  # hidden_states
            outputs = self.base_llm(
                inputs_embeds=x,
                output_hidden_states=True,
                return_dict=True,
            )

        # hidden_statesは (入力埋め込み, layer1出力, layer2出力, ...) のタプル
        hidden_history = list(outputs.hidden_states)
        h_out = hidden_history[-1]

        return h_out, hidden_history

    def get_logits(self, h: torch.Tensor) -> torch.Tensor:
        """
        hidden statesからlogitsを計算。

        Args:
            h: hidden states (batch_size, seq_len, dim)

        Returns:
            logits: (batch_size, seq_len, vocab_size)
        """
        # CausalLMのlm_headを使用
        return self.base_llm.lm_head(h)

    def should_exit(self, hidden_history: List[torch.Tensor]) -> torch.Tensor:
        """
        exit判定。

        Args:
            hidden_history: 各レイヤーのhidden statesリスト

        Returns:
            should_exit: Booleanマスク True=exitすべき (batch_size, seq_len)
        """
        return self.exit_fn(hidden_history, self.threshold)

    def collect_hard_tokens(
        self,
        data: "SequenceData",
        hard_ratio: float,
        batch_size: int,
    ) -> "SequenceData":
        """
        【訓練用】hard tokensを収集。

        訓練後に呼ばれ、次のLLM用のデータを準備。
        Hard tokensは低cos_sim（LLM通過で大きく変化した）トークン。
        推論時のexit判定用にself.thresholdも設定。

        Args:
            data: hard tokensを収集するSequenceData
            hard_ratio: hard tokenとして収集するトークンの割合 (0.0-1.0)
            batch_size: 処理用バッチサイズ

        Returns:
            hard tokensのみを含むSequenceData（出力hidden states）
        """
        from .sequence_data import SequenceData as SD

        device = next(self.parameters()).device
        self.eval()

        all_cos_sim: List[torch.Tensor] = []
        all_hidden_out: List[torch.Tensor] = []
        all_targets: List[torch.Tensor] = []

        with torch.no_grad():
            for h, y in data.to(str(device)).batches(batch_size, shuffle=False):
                h_out, hidden_history = self.forward(h, input_type="hidden_states")
                h_in = hidden_history[-2]
                cos_sim = compute_cos_sim(h_in, h_out)

                all_cos_sim.append(cos_sim.cpu())
                all_hidden_out.append(h_out.cpu())
                all_targets.append(y.cpu())

        cos_sim_all = torch.cat(all_cos_sim)
        hidden_out_all = torch.cat(all_hidden_out)
        targets_all = torch.cat(all_targets)

        # 閾値を計算
        all_cos_flat = cos_sim_all.view(-1)
        if hard_ratio >= 1.0:
            self.threshold = float('inf')
        elif hard_ratio <= 0.0:
            self.threshold = float('-inf')
        else:
            self.threshold = float(torch.quantile(all_cos_flat, hard_ratio).item())

        # トークン単位のhardマスク
        hard_token_mask = cos_sim_all < self.threshold
        hard_hidden = hidden_out_all[hard_token_mask]
        hard_targets = targets_all[hard_token_mask]

        num_hard_tokens = hard_hidden.shape[0]
        seq_len = data.seq_len
        dim = hidden_out_all.shape[-1]

        if num_hard_tokens == 0:
            return SD.empty(seq_len, dim, str(device))

        # シーケンスに再構成
        num_complete_sequences = num_hard_tokens // seq_len
        if num_complete_sequences == 0:
            return SD.empty(seq_len, dim, str(device))

        usable_tokens = num_complete_sequences * seq_len
        hard_hidden = hard_hidden[:usable_tokens].view(num_complete_sequences, seq_len, -1)
        hard_targets = hard_targets[:usable_tokens].view(num_complete_sequences, seq_len)

        return SD(hard_hidden, hard_targets)

    def transform_data(self, data: "SequenceData", batch_size: int) -> "SequenceData":
        """
        【訓練用】SequenceDataをこのLLMで変換。

        全データをLLMに通し、出力hidden statesを返す。
        次のLLM用の検証データ準備に使用。

        Args:
            data: 入力SequenceData
            batch_size: 処理用バッチサイズ

        Returns:
            変換されたhidden statesを持つSequenceData
        """
        from .sequence_data import SequenceData as SD

        device = next(self.parameters()).device
        self.eval()

        all_hidden: List[torch.Tensor] = []
        all_targets: List[torch.Tensor] = []

        with torch.no_grad():
            for h, y in data.to(str(device)).batches(batch_size, shuffle=False):
                h_out, _ = self.forward(h, input_type="hidden_states")
                all_hidden.append(h_out)
                all_targets.append(y)

        return SD(torch.cat(all_hidden), torch.cat(all_targets))
