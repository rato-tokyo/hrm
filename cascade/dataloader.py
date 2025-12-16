"""
CASCADEフレームワーク - データローディング

言語モデリング事前学習用のWikiText-2データローダー。
"""

import torch
from typing import Dict, List, Tuple


def create_wikitext_dataloaders(
    num_samples: int,
    batch_size: int,
    seq_len: int,
    seed: int
) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor]],
           List[Tuple[torch.Tensor, torch.Tensor]],
           int]:
    """
    言語モデリング用のWikiText-2データローダーを作成。

    Args:
        num_samples: 訓練サンプル数
        batch_size: バッチサイズ
        seq_len: シーケンス長
        seed: ランダムシード

    Returns:
        (train_batches, val_batches, vocab_size)のタプル
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("datasetsをインストールしてください: pip install datasets") from exc

    torch.manual_seed(seed)
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')

    def simple_tokenize(text: str) -> List[str]:
        return text.lower().split()

    # 語彙を構築
    vocab: Dict[str, int] = {'<unk>': 0, '<pad>': 1}
    for split in ['train', 'validation']:
        for item in dataset[split]:
            for token in simple_tokenize(item['text']):
                if token not in vocab:
                    vocab[token] = len(vocab)

    vocab_size = len(vocab)

    def tokenize_split(split_name: str) -> torch.Tensor:
        all_tokens: List[int] = []
        for item in dataset[split_name]:
            tokens = simple_tokenize(item['text'])
            token_ids = [vocab.get(t, vocab['<unk>']) for t in tokens]
            all_tokens.extend(token_ids)
        return torch.tensor(all_tokens, dtype=torch.long)

    train_data = tokenize_split('train')
    val_data = tokenize_split('validation')

    # サンプル数を制限
    max_tokens_train = num_samples * (seq_len + 1)
    max_tokens_val = int(num_samples * 0.2) * (seq_len + 1)
    train_data = train_data[:max_tokens_train]
    val_data = val_data[:max_tokens_val]

    def batchify(data: torch.Tensor) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        batches = []
        num_tokens = len(data)
        for i in range(0, num_tokens - seq_len - 1, batch_size * seq_len):
            batch_x, batch_y = [], []
            for j in range(batch_size):
                start_idx = i + j * seq_len
                if start_idx + seq_len + 1 <= num_tokens:
                    batch_x.append(data[start_idx:start_idx + seq_len])
                    batch_y.append(data[start_idx + 1:start_idx + seq_len + 1])
            if len(batch_x) == batch_size:
                batches.append((torch.stack(batch_x), torch.stack(batch_y)))
        return batches

    return batchify(train_data), batchify(val_data), vocab_size
