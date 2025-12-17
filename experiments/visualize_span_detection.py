"""
Span検出の可視化実験

実際のLLMでattention patternを取得し、
span分割がどのように行われるかを可視化する。
"""

import sys
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import List, Tuple
from dataclasses import dataclass


@dataclass
class Span:
    """検出されたspanを表すデータクラス。"""
    start: int
    end: int
    score: float

    @property
    def length(self) -> int:
        return self.end - self.start + 1


def detect_boundaries_by_row_change(
    attention_map: torch.Tensor,
    threshold: float = 0.3,
) -> List[int]:
    """行方向のattention分布の変化点を検出。"""
    seq_len = attention_map.size(0)

    if seq_len <= 2:
        return [0, seq_len - 1]

    boundaries = [0]

    for i in range(1, seq_len):
        if i < 2:
            continue

        row_prev = attention_map[i - 1, :i]
        row_curr = attention_map[i, :i]

        if row_prev.norm() < 1e-8 or row_curr.norm() < 1e-8:
            continue

        cos_sim = F.cosine_similarity(
            row_prev.unsqueeze(0),
            row_curr.unsqueeze(0),
            dim=1
        ).item()

        if cos_sim < threshold:
            boundaries.append(i)

    if boundaries[-1] != seq_len - 1:
        boundaries.append(seq_len - 1)

    return boundaries


def boundaries_to_spans(boundaries: List[int]) -> List[Span]:
    """境界位置リストをSpanリストに変換。"""
    if len(boundaries) < 2:
        return []

    spans = []
    for i in range(len(boundaries) - 1):
        spans.append(Span(
            start=boundaries[i],
            end=boundaries[i + 1],
            score=1.0,
        ))

    return spans


def aggregate_attention_maps(attention_maps: Tuple[torch.Tensor, ...]) -> torch.Tensor:
    """複数レイヤーのattention mapを集約。"""
    # Noneを除外
    valid_maps = [m for m in attention_maps if m is not None]
    if not valid_maps:
        raise ValueError("No valid attention maps")
    stacked = torch.stack(valid_maps, dim=0)
    # shape: (num_layers, batch, num_heads, seq, seq)
    aggregated = stacked.mean(dim=(0, 2))  # (batch, seq, seq)
    return aggregated


def set_seed(seed: int):
    """ランダムシードを設定。"""
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """利用可能なデバイスを取得。"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def visualize_attention_map(attention_map: torch.Tensor, tokens: List[str], title: str = ""):
    """Attention mapをテキストで可視化。"""
    seq_len = attention_map.size(0)

    print(f"\n{'='*60}")
    print(f"Attention Map: {title}")
    print(f"{'='*60}")

    # ヘッダー（トークン）
    header = "      "
    for i, tok in enumerate(tokens[:seq_len]):
        header += f"{tok[:4]:>5}"
    print(header)

    # Attention値
    for i in range(seq_len):
        row = f"{tokens[i][:4]:>5} "
        for j in range(seq_len):
            val = attention_map[i, j].item()
            if val > 0.3:
                row += "  ███"
            elif val > 0.1:
                row += "  ▓▓▓"
            elif val > 0.05:
                row += "  ░░░"
            else:
                row += "    ·"
        print(row)


def visualize_spans(tokens: List[str], spans: List[Span], boundaries: List[int]):
    """検出されたspanを可視化。"""
    print(f"\n{'='*60}")
    print("Detected Spans")
    print(f"{'='*60}")

    print(f"\nBoundaries: {boundaries}")
    print(f"Number of spans: {len(spans)}")

    for i, span in enumerate(spans):
        span_tokens = tokens[span.start:span.end+1]
        span_text = " ".join(span_tokens)
        print(f"\nSpan {i}: [{span.start}-{span.end}] (len={span.length})")
        print(f"  Tokens: {span_text[:80]}{'...' if len(span_text) > 80 else ''}")

    # テキスト表示（境界を★で表示）
    print(f"\n{'='*60}")
    print("Text with boundaries (★)")
    print(f"{'='*60}")

    result = ""
    for i, tok in enumerate(tokens):
        if i in boundaries:
            result += f"★{tok}"
        else:
            result += f" {tok}"
    print(result[:200] + "..." if len(result) > 200 else result)


def experiment_threshold_sensitivity(
    attention_map: torch.Tensor,
    tokens: List[str],
    thresholds: List[float] = [0.1, 0.3, 0.5, 0.7, 0.9],
):
    """閾値による分割の変化を確認。"""
    print(f"\n{'='*60}")
    print("Threshold Sensitivity Analysis")
    print(f"{'='*60}")

    for threshold in thresholds:
        boundaries = detect_boundaries_by_row_change(attention_map, threshold=threshold)
        spans = boundaries_to_spans(boundaries)
        compression = 1.0 - len(boundaries) / len(tokens) if len(tokens) > 0 else 0.0

        print(f"\nThreshold: {threshold:.1f}")
        print(f"  Boundaries: {len(boundaries)}")
        print(f"  Spans: {len(spans)}")
        print(f"  Compression: {compression:.1%}")
        print(f"  Boundary positions: {boundaries[:10]}{'...' if len(boundaries) > 10 else ''}")


def main():
    set_seed(42)
    device = get_device()
    print(f"Device: {device}")

    # モデルとトークナイザをロード
    model_name = "gpt2"  # 軽量モデルで実験
    print(f"\nLoading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # configでoutput_attentions=Trueに設定（デフォルトでattentionを返すように）
    config = AutoConfig.from_pretrained(model_name)
    config.output_attentions = True

    base_model = AutoModelForCausalLM.from_pretrained(model_name, config=config)
    base_model.to(device)
    base_model.eval()

    # テスト用テキスト（Alpaca形式のサンプル）
    test_texts = [
        # 短い文
        "The quick brown fox jumps over the lazy dog.",

        # Instruction形式
        """### Instruction:
Explain what machine learning is.

### Response:
Machine learning is a subset of artificial intelligence that enables computers to learn from data.""",

        # 日本語（GPT-2はsubword）
        "今日は天気がいいので散歩に行きました。",
    ]

    for text_idx, text in enumerate(test_texts):
        print(f"\n\n{'#'*70}")
        print(f"# Text {text_idx + 1}")
        print(f"{'#'*70}")
        print(f"Original: {text[:100]}{'...' if len(text) > 100 else ''}")

        # トークナイズ
        inputs = tokenizer(text, return_tensors="pt").to(device)
        token_ids = inputs["input_ids"]
        tokens = tokenizer.convert_ids_to_tokens(token_ids[0])

        print(f"Tokens ({len(tokens)}): {tokens[:15]}{'...' if len(tokens) > 15 else ''}")

        if len(tokens) < 3:
            print("Skipping: too short")
            continue

        # Attention mapを取得
        with torch.no_grad():
            outputs = base_model(
                input_ids=token_ids,
                output_hidden_states=True,
                output_attentions=True,
                return_dict=True,
            )

        attentions = outputs.attentions
        print(f"Attentions type: {type(attentions)}")
        if attentions is not None:
            print(f"Attentions length: {len(attentions)}")
            if len(attentions) > 0 and attentions[0] is not None:
                print(f"First attention shape: {attentions[0].shape}")

        if attentions is None or len(attentions) == 0:
            print("Warning: Could not get attention maps")
            continue

        # Attention mapを集約（全レイヤー、全ヘッドの平均）
        try:
            agg_attention = aggregate_attention_maps(attentions)
        except ValueError as e:
            print(f"Warning: {e}")
            continue
        attention_map = agg_attention[0]  # バッチの最初

        print(f"Attention map shape: {attention_map.shape}")

        # Attention map可視化（短いテキストのみ）
        if len(tokens) <= 20:
            visualize_attention_map(attention_map, tokens, f"Text {text_idx + 1}")

        # Span検出
        boundaries = detect_boundaries_by_row_change(attention_map, threshold=0.3)
        spans = boundaries_to_spans(boundaries)

        visualize_spans(tokens, spans, boundaries)

        # 閾値感度分析
        experiment_threshold_sensitivity(attention_map, tokens)

    print(f"\n\n{'='*70}")
    print("Experiment completed!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
