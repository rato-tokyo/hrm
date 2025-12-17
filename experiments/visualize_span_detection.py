"""
Span検出の可視化実験

実際のLLMでattention patternを取得し、
3つのspan検出方式を比較する：
1. TriangleScoreDetector (LTri-LLM論文ベース)
2. RowChangeDetector (独自実装)
3. FixedSpanDetector (ベースライン)
"""

import sys
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import List, Tuple, Optional, Union, Protocol
from dataclasses import dataclass


# ========== プロトコル ==========

class SpanDetector(Protocol):
    """Span検出器のプロトコル。"""
    def detect(self, attention_map: torch.Tensor) -> List["Span"]:
        ...


# ========== データクラス ==========

@dataclass
class Span:
    """検出されたspanを表すデータクラス。"""
    start: int
    end: int
    score: float

    @property
    def length(self) -> int:
        return self.end - self.start + 1

    def __repr__(self) -> str:
        return f"Span({self.start}-{self.end}, score={self.score:.3f})"


# ========== Detector実装 ==========

class TriangleScoreDetector:
    """LTri-LLM論文ベースの三角形スコアによるspan検出。"""

    def __init__(
        self,
        threshold: float = 0.0,
        iou_threshold: float = 0.1,
        min_span_length: int = 2,
        max_span_length: Optional[int] = None,
    ):
        self.threshold = threshold
        self.iou_threshold = iou_threshold
        self.min_span_length = min_span_length
        self.max_span_length = max_span_length

    def detect(self, attention_map: torch.Tensor) -> List[Span]:
        if attention_map.dim() == 3:
            attention_map = attention_map.sum(dim=0)

        seq_len = attention_map.size(0)
        triangle_scores = self._compute_triangle_scores(attention_map)
        candidates = self._generate_candidates(triangle_scores, seq_len)

        if not candidates:
            return [Span(start=0, end=seq_len - 1, score=0.0)]

        spans = self._apply_nms(candidates)
        spans = self._ensure_full_coverage(spans, seq_len)
        return spans

    def _compute_triangle_scores(self, attention_map: torch.Tensor) -> torch.Tensor:
        seq_len = attention_map.size(0)
        thresholded = attention_map - self.threshold
        cumsum = torch.zeros(seq_len + 1, seq_len + 1, device=attention_map.device)
        cumsum[1:, 1:] = torch.cumsum(torch.cumsum(thresholded, dim=0), dim=1)

        triangle_scores = torch.zeros(seq_len, seq_len, device=attention_map.device)
        for x in range(seq_len):
            for y in range(x, seq_len):
                score = (cumsum[y + 1, y + 1] - cumsum[x, y + 1]
                         - cumsum[y + 1, x] + cumsum[x, x])
                triangle_scores[x, y] = score
        return triangle_scores

    def _generate_candidates(self, triangle_scores: torch.Tensor, seq_len: int) -> List[Span]:
        candidates = []
        max_len = self.max_span_length or seq_len

        for x in range(seq_len):
            for y in range(x + self.min_span_length - 1, min(x + max_len, seq_len)):
                score = triangle_scores[x, y].item()
                if score > 0:
                    candidates.append(Span(start=x, end=y, score=score))

        candidates.sort(key=lambda s: s.score, reverse=True)
        return candidates

    def _apply_nms(self, candidates: List[Span]) -> List[Span]:
        if not candidates:
            return []

        selected: List[Span] = []
        for candidate in candidates:
            is_suppressed = False
            for selected_span in selected:
                iou = self._compute_iou(candidate, selected_span)
                if iou > self.iou_threshold:
                    is_suppressed = True
                    break
            if not is_suppressed:
                selected.append(candidate)
        return selected

    def _compute_iou(self, span1: Span, span2: Span) -> float:
        inter_start = max(span1.start, span2.start)
        inter_end = min(span1.end, span2.end)
        intersection = max(0, inter_end - inter_start + 1)
        union = span1.length + span2.length - intersection
        if union == 0:
            return 0.0
        return intersection / union

    def _ensure_full_coverage(self, spans: List[Span], seq_len: int) -> List[Span]:
        if not spans:
            return [Span(start=0, end=seq_len - 1, score=0.0)]

        spans = sorted(spans, key=lambda x: x.start)
        result: List[Span] = []
        covered_until = -1

        for span in spans:
            if span.start > covered_until + 1:
                result.append(Span(start=covered_until + 1, end=span.start - 1, score=0.0))
            if span.end > covered_until:
                result.append(span)
                covered_until = span.end

        if covered_until < seq_len - 1:
            result.append(Span(start=covered_until + 1, end=seq_len - 1, score=0.0))

        return result


class RowChangeDetector:
    """行変化検出によるspan検出（独自実装）。"""

    def __init__(self, threshold: float = 0.3, min_span_length: int = 1):
        self.threshold = threshold
        self.min_span_length = min_span_length

    def detect(self, attention_map: torch.Tensor) -> List[Span]:
        if attention_map.dim() == 3:
            attention_map = attention_map.mean(dim=0)

        seq_len = attention_map.size(0)
        boundaries = self._detect_boundaries(attention_map)
        spans = self._boundaries_to_spans(boundaries)
        spans = self._merge_short_spans(spans)
        spans = self._ensure_full_coverage(spans, seq_len)
        return spans

    def _detect_boundaries(self, attention_map: torch.Tensor) -> List[int]:
        seq_len = attention_map.size(0)
        if seq_len <= 2:
            return [0, seq_len - 1]

        boundaries = [0]
        for i in range(2, seq_len):
            row_prev = attention_map[i - 1, :i]
            row_curr = attention_map[i, :i]

            if row_prev.norm() < 1e-8 or row_curr.norm() < 1e-8:
                continue

            cos_sim = F.cosine_similarity(
                row_prev.unsqueeze(0), row_curr.unsqueeze(0), dim=1
            ).item()

            if cos_sim < self.threshold:
                boundaries.append(i)

        if boundaries[-1] != seq_len - 1:
            boundaries.append(seq_len - 1)

        return boundaries

    def _boundaries_to_spans(self, boundaries: List[int]) -> List[Span]:
        if len(boundaries) < 2:
            return []
        return [Span(start=boundaries[i], end=boundaries[i + 1], score=1.0)
                for i in range(len(boundaries) - 1)]

    def _merge_short_spans(self, spans: List[Span]) -> List[Span]:
        if not spans or self.min_span_length <= 1:
            return spans

        result: List[Span] = []
        for span in spans:
            if span.length < self.min_span_length and result:
                prev = result[-1]
                result[-1] = Span(start=prev.start, end=span.end, score=prev.score)
            else:
                result.append(span)
        return result

    def _ensure_full_coverage(self, spans: List[Span], seq_len: int) -> List[Span]:
        if not spans:
            return [Span(start=0, end=seq_len - 1, score=1.0)]

        spans = sorted(spans, key=lambda x: x.start)
        result: List[Span] = []
        covered_until = -1

        for span in spans:
            if span.start > covered_until + 1:
                result.append(Span(start=covered_until + 1, end=span.start - 1, score=0.0))
            if span.end > covered_until:
                result.append(span)
                covered_until = span.end

        if covered_until < seq_len - 1:
            result.append(Span(start=covered_until + 1, end=seq_len - 1, score=0.0))

        return result


class FixedSpanDetector:
    """固定長span検出（ベースライン）。"""

    def __init__(self, span_size: int = 4):
        self.span_size = span_size

    def detect(self, attention_map: torch.Tensor) -> List[Span]:
        if attention_map.dim() == 3:
            seq_len = attention_map.size(1)
        else:
            seq_len = attention_map.size(0)

        spans = []
        for start in range(0, seq_len, self.span_size):
            end = min(start + self.span_size - 1, seq_len - 1)
            spans.append(Span(start=start, end=end, score=1.0))
        return spans


# ========== ユーティリティ ==========

def aggregate_attention_maps(attention_maps: Tuple[torch.Tensor, ...]) -> torch.Tensor:
    """複数レイヤーのattention mapを集約。"""
    valid_maps = [m for m in attention_maps if m is not None]
    if not valid_maps:
        raise ValueError("No valid attention maps")
    stacked = torch.stack(valid_maps, dim=0)
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


def spans_to_boundaries(spans: List[Span]) -> List[int]:
    """SpanリストからユニークなboundaryリストをI取得。"""
    boundaries = set()
    for span in spans:
        boundaries.add(span.start)
        boundaries.add(span.end)
    return sorted(boundaries)


def compute_compression_stats(spans: List[Span], seq_len: int) -> dict:
    """圧縮統計を計算。"""
    boundaries = spans_to_boundaries(spans)
    n_boundaries = len(boundaries)
    compression_ratio = 1.0 - n_boundaries / seq_len if seq_len > 0 else 0.0
    return {
        "num_spans": len(spans),
        "num_boundaries": n_boundaries,
        "compression_ratio": compression_ratio,
        "boundaries": boundaries,
    }


# ========== 可視化 ==========

def visualize_attention_map(attention_map: torch.Tensor, tokens: List[str], title: str = ""):
    """Attention mapをテキストで可視化。"""
    seq_len = attention_map.size(0)

    print(f"\n{'='*60}")
    print(f"Attention Map: {title}")
    print(f"{'='*60}")

    # ヘッダー（トークン）
    header = "      "
    for tok in tokens[:seq_len]:
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


def visualize_spans(tokens: List[str], spans: List[Span], method_name: str):
    """検出されたspanを可視化。"""
    boundaries = spans_to_boundaries(spans)

    print(f"\n--- {method_name} ---")
    print(f"Spans: {len(spans)}, Boundaries: {len(boundaries)}")

    # 短い形式でspanを表示
    span_strs = [f"[{s.start}-{s.end}]" for s in spans[:8]]
    if len(spans) > 8:
        span_strs.append("...")
    print(f"Spans: {' '.join(span_strs)}")

    # テキスト表示（境界を|で表示）
    result = ""
    for i, tok in enumerate(tokens):
        if i in boundaries and i > 0:
            result += f"|{tok}"
        else:
            result += f" {tok}"
    print(f"Text: {result[:120]}{'...' if len(result) > 120 else ''}")


def compare_detectors(
    attention_map: torch.Tensor,
    tokens: List[str],
    title: str = "",
):
    """3つのDetectorを比較。"""
    seq_len = len(tokens)

    print(f"\n{'='*70}")
    print(f"Detector Comparison: {title}")
    print(f"Sequence length: {seq_len} tokens")
    print(f"{'='*70}")

    # 3つのDetectorを定義
    detectors: List[Tuple[str, SpanDetector]] = [
        ("TriangleScore (LTri-LLM)", TriangleScoreDetector(
            threshold=0.0, iou_threshold=0.3, min_span_length=2, max_span_length=8
        )),
        ("RowChange (Custom)", RowChangeDetector(threshold=0.5, min_span_length=2)),
        ("Fixed (Baseline)", FixedSpanDetector(span_size=4)),
    ]

    results = []

    for name, detector in detectors:
        spans = detector.detect(attention_map)
        stats = compute_compression_stats(spans, seq_len)

        results.append({
            "name": name,
            "spans": spans,
            "stats": stats,
        })

        visualize_spans(tokens, spans, name)

    # 比較表
    print(f"\n{'='*70}")
    print("Summary Comparison")
    print(f"{'='*70}")
    print(f"{'Method':<25} {'Spans':>8} {'Boundaries':>12} {'Compression':>12}")
    print("-" * 60)

    for r in results:
        stats = r["stats"]
        print(f"{r['name']:<25} {stats['num_spans']:>8} {stats['num_boundaries']:>12} {stats['compression_ratio']:>11.1%}")

    return results


def experiment_parameter_sensitivity(
    attention_map: torch.Tensor,
    tokens: List[str],
):
    """パラメータ感度分析。"""
    seq_len = len(tokens)

    print(f"\n{'='*70}")
    print("Parameter Sensitivity Analysis")
    print(f"{'='*70}")

    # RowChangeDetector: threshold感度
    print("\n--- RowChangeDetector: threshold sensitivity ---")
    print(f"{'Threshold':>10} {'Spans':>8} {'Boundaries':>12} {'Compression':>12}")
    print("-" * 45)

    for threshold in [0.1, 0.3, 0.5, 0.7, 0.9]:
        row_detector = RowChangeDetector(threshold=threshold, min_span_length=2)
        spans = row_detector.detect(attention_map)
        stats = compute_compression_stats(spans, seq_len)
        print(f"{threshold:>10.1f} {stats['num_spans']:>8} {stats['num_boundaries']:>12} {stats['compression_ratio']:>11.1%}")

    # TriangleScoreDetector: iou_threshold感度
    print("\n--- TriangleScoreDetector: iou_threshold sensitivity ---")
    print(f"{'IoU Thresh':>10} {'Spans':>8} {'Boundaries':>12} {'Compression':>12}")
    print("-" * 45)

    for iou in [0.1, 0.3, 0.5, 0.7, 0.9]:
        tri_detector = TriangleScoreDetector(
            threshold=0.0, iou_threshold=iou, min_span_length=2, max_span_length=8
        )
        spans = tri_detector.detect(attention_map)
        stats = compute_compression_stats(spans, seq_len)
        print(f"{iou:>10.1f} {stats['num_spans']:>8} {stats['num_boundaries']:>12} {stats['compression_ratio']:>11.1%}")

    # FixedSpanDetector: span_size感度
    print("\n--- FixedSpanDetector: span_size sensitivity ---")
    print(f"{'Span Size':>10} {'Spans':>8} {'Boundaries':>12} {'Compression':>12}")
    print("-" * 45)

    for size in [2, 4, 8, 16, 32]:
        fixed_detector = FixedSpanDetector(span_size=size)
        spans = fixed_detector.detect(attention_map)
        stats = compute_compression_stats(spans, seq_len)
        print(f"{size:>10} {stats['num_spans']:>8} {stats['num_boundaries']:>12} {stats['compression_ratio']:>11.1%}")


def main():
    set_seed(42)
    device = get_device()
    print(f"Device: {device}")

    # モデルとトークナイザをロード
    model_name = "gpt2"
    print(f"\nLoading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = AutoConfig.from_pretrained(model_name)
    config.output_attentions = True

    base_model = AutoModelForCausalLM.from_pretrained(model_name, config=config)
    base_model.to(device)
    base_model.eval()

    # テスト用テキスト
    test_texts = [
        # 短い文
        "The quick brown fox jumps over the lazy dog.",

        # Instruction形式
        """### Instruction:
Explain what machine learning is.

### Response:
Machine learning is a subset of artificial intelligence that enables computers to learn from data.""",

        # 日本語
        "今日は天気がいいので散歩に行きました。",

        # より長い文
        "In the field of computer science, algorithms are fundamental building blocks. They provide step-by-step procedures for solving problems efficiently.",
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

        if attentions is None or len(attentions) == 0:
            print("Warning: Could not get attention maps")
            continue

        # Attention mapを集約
        try:
            agg_attention = aggregate_attention_maps(attentions)
        except ValueError as e:
            print(f"Warning: {e}")
            continue
        attention_map = agg_attention[0]  # バッチの最初

        print(f"Attention map shape: {attention_map.shape}")

        # Attention map可視化（短いテキストのみ）
        if len(tokens) <= 15:
            visualize_attention_map(attention_map, tokens, f"Text {text_idx + 1}")

        # 3つのDetectorを比較
        compare_detectors(attention_map, tokens, f"Text {text_idx + 1}")

        # パラメータ感度分析（最初のテキストのみ）
        if text_idx == 0:
            experiment_parameter_sensitivity(attention_map, tokens)

    print(f"\n\n{'='*70}")
    print("Experiment completed!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
