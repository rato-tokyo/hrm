"""
DCA-LLM出力データクラス。
"""

from dataclasses import dataclass
from typing import Optional

from torch import Tensor


@dataclass
class DCALLMOutput:
    """DCA-LLMの出力。"""
    logits: Tensor  # (batch, seq_len, vocab_size)
    loss: Optional[Tensor] = None
