"""
LEGOフレームワーク - EarlyExitLLM

任意のBaseLLMをラップし、Early Exit機能を追加するクラス。
各EarlyExitLLMは独立したLLMとして機能し、LEGOEnsembleで統合される。
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Tuple, List, Optional, TYPE_CHECKING

from .modules import TransformerBlock
from .exit_fn import ExitFn, default_exit_fn, compute_cos_sim

if TYPE_CHECKING:
    from .sequence_data import SequenceData


class EarlyExitLLM(nn.Module):
    """
    BaseLLM + Early Exit機能。

    任意のLLM（TransformerBlockなど）をラップし、以下を追加:
    - exit_fnによるexit判定（デフォルト: CALM式cos_sim）
    - logits計算用の共有output_head
    - 推論時のexit判定用threshold
    - 訓練後のhard token収集

    Args:
        base_llm: ラップするBaseLLM（TransformerBlockなど）
        exit_fn: オプションのカスタムexit関数。Noneの場合default_exit_fn（CALM式）を使用
    """

    def __init__(
        self,
        base_llm: TransformerBlock,
        exit_fn: Optional[ExitFn] = None,
    ):
        super().__init__()
        self.base_llm = base_llm
        self.exit_fn = exit_fn or default_exit_fn
        self.threshold = 0.0  # collect_hard_tokensで設定
        self.output_head: nn.Linear | None = None  # LEGOEnsembleが設定

    @property
    def dim(self) -> int:
        """モデル次元（base_llmに委譲）。"""
        return self.base_llm.dim

    @property
    def num_heads(self) -> int:
        """Attentionヘッド数（base_llmに委譲）。"""
        return self.base_llm.num_heads

    @property
    def num_layers(self) -> int:
        """レイヤー数（base_llmに委譲）。"""
        return self.base_llm.num_layers

    def train_forward(
        self, h: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """
        【訓練用】forward pass。

        exit判定は行わず、hidden statesとlogitsのみを計算。
        llm_trainer.pyから呼ばれる。

        Args:
            h: hidden states (batch_size, seq_len, dim)

        Returns:
            h_out: 出力hidden states (batch_size, seq_len, dim)
            logits: 出力logits (batch_size, seq_len, vocab_size)
            hidden_history: cos_sim計算用の全hidden statesリスト
        """
        if self.output_head is None:
            raise RuntimeError("output_headが未設定。先にset_output_head()を呼んでください。")

        h_out, hidden_history = self.base_llm(h)
        logits = self.output_head(h_out)

        return h_out, logits, hidden_history

    def evaluate(
        self, h: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        【評価用】forward pass（TRUE Early Exit）。

        exit判定を含む。LEGOEnsemble.evaluate()から呼ばれる。

        Args:
            h: hidden states (batch_size, seq_len, dim)

        Returns:
            h_out: 出力hidden states (batch_size, seq_len, dim)
            logits: 出力logits (batch_size, seq_len, vocab_size)
            should_exit: Booleanマスク True = exitすべき (batch_size, seq_len)
        """
        if self.output_head is None:
            raise RuntimeError("output_headが未設定。先にset_output_head()を呼んでください。")

        h_out, hidden_history = self.base_llm(h)
        logits = self.output_head(h_out)
        should_exit = self.exit_fn(hidden_history, self.threshold)

        return h_out, logits, should_exit

    def set_output_head(self, output_head: nn.Linear) -> None:
        """共有output headの参照を設定。"""
        self.output_head = output_head

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
                h_out, _, hidden_history = self.train_forward(h)
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
                h_out, _, _ = self.train_forward(h)
                all_hidden.append(h_out)
                all_targets.append(y)

        return SD(torch.cat(all_hidden), torch.cat(all_targets))
