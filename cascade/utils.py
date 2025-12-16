"""
CASCADEフレームワーク - ユーティリティ関数
"""

import random
import numpy as np
import torch


def set_seed(seed: int) -> None:
    """再現性のためのランダムシード設定。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> str:
    """利用可能な計算デバイスを取得（CUDAが利用可能ならCUDA、それ以外はCPU）。"""
    return 'cuda' if torch.cuda.is_available() else 'cpu'
