"""
CASCADEフレームワーク - データローディング

Hugging Face datasets + tokenizers を使用した言語モデリング用データローダー。
"""

import torch
from typing import List, Tuple
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, PreTrainedTokenizer


def create_wikitext_dataloaders(
    num_samples: int,
    batch_size: int,
    seq_len: int,
    seed: int,
    tokenizer_name: str = "gpt2",
) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor]],
           List[Tuple[torch.Tensor, torch.Tensor]],
           int]:
    """
    Hugging Face tokenizerを使用してWikiText-2データローダーを作成。

    Args:
        num_samples: 訓練サンプル数
        batch_size: バッチサイズ
        seq_len: シーケンス長
        seed: ランダムシード
        tokenizer_name: 使用するトークナイザ名（デフォルト: "gpt2"）

    Returns:
        (train_batches, val_batches, vocab_size)のタプル
    """
    torch.manual_seed(seed)

    # Hugging Face tokenizerをロード
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # WikiText-2データセットをロード
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')

    # テキストをトークナイズ
    train_tokens = _tokenize_split(dataset['train'], tokenizer)
    val_tokens = _tokenize_split(dataset['validation'], tokenizer)

    # サンプル数を制限
    max_tokens_train = num_samples * (seq_len + 1)
    max_tokens_val = int(num_samples * 0.2) * (seq_len + 1)
    train_tokens = train_tokens[:max_tokens_train]
    val_tokens = val_tokens[:max_tokens_val]

    # バッチ化
    train_batches = _batchify(train_tokens, batch_size, seq_len)
    val_batches = _batchify(val_tokens, batch_size, seq_len)

    return train_batches, val_batches, tokenizer.vocab_size


def _tokenize_split(split_dataset: Dataset, tokenizer: PreTrainedTokenizer) -> torch.Tensor:
    """
    データセット分割をトークナイズして単一のトークンテンソルに連結。

    Args:
        split_dataset: Hugging Face Dataset分割
        tokenizer: Hugging Face tokenizer

    Returns:
        全トークンを含むテンソル
    """
    all_tokens: List[int] = []
    for item in split_dataset:
        text = item['text']
        if text.strip():  # 空行をスキップ
            tokens = tokenizer.encode(text, add_special_tokens=False)
            all_tokens.extend(tokens)
    return torch.tensor(all_tokens, dtype=torch.long)


def _batchify(
    data: torch.Tensor,
    batch_size: int,
    seq_len: int,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    トークンデータを(input, target)バッチのリストに変換。

    Args:
        data: トークンテンソル
        batch_size: バッチサイズ
        seq_len: シーケンス長

    Returns:
        (x, y)バッチのリスト
    """
    batches: List[Tuple[torch.Tensor, torch.Tensor]] = []
    num_tokens = len(data)

    for i in range(0, num_tokens - seq_len - 1, batch_size * seq_len):
        batch_x: List[torch.Tensor] = []
        batch_y: List[torch.Tensor] = []
        for j in range(batch_size):
            start_idx = i + j * seq_len
            if start_idx + seq_len + 1 <= num_tokens:
                batch_x.append(data[start_idx:start_idx + seq_len])
                batch_y.append(data[start_idx + 1:start_idx + seq_len + 1])
        if len(batch_x) == batch_size:
            batches.append((torch.stack(batch_x), torch.stack(batch_y)))

    return batches


def create_dataset_from_tokenizer(
    texts: List[str],
    tokenizer: PreTrainedTokenizer,
    seq_len: int,
) -> Dataset:
    """
    テキストリストからHugging Face Datasetを作成。

    Args:
        texts: テキストのリスト
        tokenizer: Hugging Face tokenizer
        seq_len: シーケンス長

    Returns:
        Hugging Face Dataset
    """
    all_tokens: List[int] = []
    for text in texts:
        if text.strip():
            tokens = tokenizer.encode(text, add_special_tokens=False)
            all_tokens.extend(tokens)

    # シーケンスに分割
    input_ids: List[List[int]] = []
    labels: List[List[int]] = []

    for i in range(0, len(all_tokens) - seq_len - 1, seq_len):
        input_ids.append(all_tokens[i:i + seq_len])
        labels.append(all_tokens[i + 1:i + seq_len + 1])

    return Dataset.from_dict({
        'input_ids': input_ids,
        'labels': labels,
    })
