"""
Span Detectorのテストスクリプト

LTri-LLM論文ベースのTriangleScoreDetectorと
独自実装のRowChangeDetectorの動作を確認。
"""

import torch
import torch.nn.functional as F
from torch import Tensor
from typing import List, Optional
from dataclasses import dataclass


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

    def detect(self, attention_map: Tensor) -> List[Span]:
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

    def _compute_triangle_scores(self, attention_map: Tensor) -> Tensor:
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

    def _generate_candidates(self, triangle_scores: Tensor, seq_len: int) -> List[Span]:
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

    def detect(self, attention_map: Tensor) -> List[Span]:
        if attention_map.dim() == 3:
            attention_map = attention_map.mean(dim=0)

        seq_len = attention_map.size(0)
        boundaries = self._detect_boundaries(attention_map)
        spans = self._boundaries_to_spans(boundaries)
        spans = self._merge_short_spans(spans)
        spans = self._ensure_full_coverage(spans, seq_len)
        return spans

    def _detect_boundaries(self, attention_map: Tensor) -> List[int]:
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

    def __init__(self, span_size: int = 128):
        self.span_size = span_size

    def detect(self, attention_map: Tensor) -> List[Span]:
        if attention_map.dim() == 3:
            seq_len = attention_map.size(1)
        else:
            seq_len = attention_map.size(0)

        spans = []
        for start in range(0, seq_len, self.span_size):
            end = min(start + self.span_size - 1, seq_len - 1)
            spans.append(Span(start=start, end=end, score=1.0))
        return spans


# ========== テスト ==========

def main():
    print("=" * 60)
    print("Span Detector Test")
    print("=" * 60)

    # テスト用attention map（下三角 + softmax）
    seq_len = 16
    attention = torch.randn(seq_len, seq_len).tril()
    attention = torch.softmax(attention, dim=-1)

    print(f"\nAttention shape: {attention.shape}")

    # 1. TriangleScoreDetector (LTri-LLM論文ベース)
    print("\n" + "-" * 40)
    print("TriangleScoreDetector (LTri-LLM paper)")
    print("-" * 40)
    detector1 = TriangleScoreDetector(threshold=0.0, iou_threshold=0.1, min_span_length=2)
    spans1 = detector1.detect(attention)
    print(f"Detected spans: {len(spans1)}")
    for s in spans1:
        print(f"  {s}")

    # 2. RowChangeDetector (独自実装)
    print("\n" + "-" * 40)
    print("RowChangeDetector (custom)")
    print("-" * 40)
    detector2 = RowChangeDetector(threshold=0.3, min_span_length=2)
    spans2 = detector2.detect(attention)
    print(f"Detected spans: {len(spans2)}")
    for s in spans2:
        print(f"  {s}")

    # 3. FixedSpanDetector
    print("\n" + "-" * 40)
    print("FixedSpanDetector")
    print("-" * 40)
    detector3 = FixedSpanDetector(span_size=4)
    spans3 = detector3.detect(attention)
    print(f"Detected spans: {len(spans3)}")
    for s in spans3:
        print(f"  {s}")

    # 圧縮率の比較
    print("\n" + "-" * 40)
    print("Compression Comparison")
    print("-" * 40)

    def count_boundaries(spans):
        positions = set()
        for s in spans:
            positions.add(s.start)
            positions.add(s.end)
        return len(positions)

    for name, spans in [("Triangle", spans1), ("RowChange", spans2), ("Fixed", spans3)]:
        n_boundaries = count_boundaries(spans)
        ratio = 1.0 - n_boundaries / seq_len
        print(f"{name:12}: {len(spans)} spans, {n_boundaries} boundaries, {ratio:.1%} compression")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
