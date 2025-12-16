"""
SmolLM2レイヤー別saturation分析

各レイヤーのcos_sim（入力と出力のコサイン類似度）分布を調査し、
hard token ratioの検討材料を提供する。

使用方法:
    # 少量データでクイック分析（推奨）
    python experiments/analyze_saturation.py --num-samples 500

    # 中程度のデータ量で分析
    python experiments/analyze_saturation.py --num-samples 2000

出力例:
    Layer  0: mean=0.912, std=0.045, p10=0.856, p50=0.921, p90=0.958
    Layer  1: mean=0.887, std=0.062, p10=0.803, p50=0.898, p90=0.943
    ...
    Layer 29: mean=0.823, std=0.089, p10=0.701, p50=0.841, p90=0.912

    === Hard Ratio推奨値 ===
    50% exit (threshold=0.841): Layer 29でexit
"""

import argparse
from typing import List, Dict
import torch

from cascade import (
    LLM,
    load_pretrained,
    set_seed,
    get_device,
    create_alpaca_dataloaders,
)
from cascade.exit_fn import compute_cos_sim


def parse_args() -> argparse.Namespace:
    """コマンドライン引数をパース。"""
    parser = argparse.ArgumentParser(
        description="SmolLM2レイヤー別saturation分析"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="smollm2-135m",
        help="分析するモデル (default: smollm2-135m)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=500,
        help="使用するサンプル数 (default: 500)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="バッチサイズ (default: 8)",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=128,
        help="シーケンス長 (default: 128)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="乱数シード (default: 42)",
    )
    parser.add_argument(
        "--target-ratios",
        type=float,
        nargs="+",
        default=[0.3, 0.5, 0.7],
        help="調査するhard ratio (default: 0.3 0.5 0.7)",
    )
    return parser.parse_args()


def analyze_layer_saturation(
    llm: LLM,
    batches: List[torch.Tensor],
    device: torch.device,
) -> Dict[int, torch.Tensor]:
    """
    各レイヤーのcos_simを収集。

    Returns:
        {layer_idx: cos_sim_tensor} のDict
    """
    layer_cos_sims: Dict[int, List[torch.Tensor]] = {}

    llm.eval()
    with torch.no_grad():
        for batch in batches:
            batch = batch.to(device)
            _, hidden_history = llm.forward_token_ids(batch)

            # 各レイヤー間のcos_simを計算
            for layer_idx in range(1, len(hidden_history)):
                h_in = hidden_history[layer_idx - 1]
                h_out = hidden_history[layer_idx]
                cos_sim = compute_cos_sim(h_in, h_out).cpu()

                if layer_idx not in layer_cos_sims:
                    layer_cos_sims[layer_idx] = []
                layer_cos_sims[layer_idx].append(cos_sim.flatten())

    # 結合
    return {
        layer_idx: torch.cat(cos_sims)
        for layer_idx, cos_sims in layer_cos_sims.items()
    }


def print_statistics(layer_cos_sims: Dict[int, torch.Tensor]) -> None:
    """統計情報を表示。"""
    print("\n" + "=" * 70)
    print("レイヤー別cos_sim統計")
    print("=" * 70)
    print(f"{'Layer':>6}  {'mean':>7}  {'std':>7}  {'p10':>7}  {'p50':>7}  {'p90':>7}")
    print("-" * 70)

    for layer_idx in sorted(layer_cos_sims.keys()):
        cos_sim = layer_cos_sims[layer_idx].float()
        mean = cos_sim.mean().item()
        std = cos_sim.std().item()
        p10 = torch.quantile(cos_sim, 0.1).item()
        p50 = torch.quantile(cos_sim, 0.5).item()
        p90 = torch.quantile(cos_sim, 0.9).item()

        print(f"{layer_idx:>6}  {mean:>7.4f}  {std:>7.4f}  {p10:>7.4f}  {p50:>7.4f}  {p90:>7.4f}")


def print_threshold_recommendations(
    layer_cos_sims: Dict[int, torch.Tensor],
    target_ratios: List[float],
) -> None:
    """hard ratio推奨値を表示。"""
    # 最終レイヤーのcos_simを使用
    last_layer = max(layer_cos_sims.keys())
    cos_sim = layer_cos_sims[last_layer].float()

    print("\n" + "=" * 70)
    print(f"Hard Ratio推奨値 (Layer {last_layer}基準)")
    print("=" * 70)

    for ratio in target_ratios:
        threshold = torch.quantile(cos_sim, ratio).item()
        # 実際のhard比率を確認
        actual_hard = (cos_sim < threshold).float().mean().item()
        print(f"  hard_ratio={ratio:.0%}: threshold={threshold:.4f} (実際: {actual_hard:.1%})")

    print("\n使用例:")
    for ratio in target_ratios:
        threshold = torch.quantile(cos_sim, ratio).item()
        print(f"  python experiments/smollm2_cascade.py --threshold {threshold:.4f} --hard-ratio {ratio}")


def print_layer_comparison(layer_cos_sims: Dict[int, torch.Tensor]) -> None:
    """レイヤー間の比較を表示。"""
    print("\n" + "=" * 70)
    print("レイヤー別saturation概要")
    print("=" * 70)

    sorted_layers = sorted(layer_cos_sims.keys())

    # 最もsaturationが高い（cos_simが高い）レイヤー
    means = {k: v.float().mean().item() for k, v in layer_cos_sims.items()}
    most_saturated = max(means, key=means.get)  # type: ignore
    least_saturated = min(means, key=means.get)  # type: ignore

    print(f"  最もsaturated (cos_sim高): Layer {most_saturated} (mean={means[most_saturated]:.4f})")
    print(f"  最もactive (cos_sim低):    Layer {least_saturated} (mean={means[least_saturated]:.4f})")

    # 後半レイヤーの傾向
    mid_point = len(sorted_layers) // 2
    first_half_mean = sum(means[layer] for layer in sorted_layers[:mid_point]) / mid_point
    second_half_mean = sum(means[layer] for layer in sorted_layers[mid_point:]) / (len(sorted_layers) - mid_point)

    print(f"\n  前半レイヤー平均: {first_half_mean:.4f}")
    print(f"  後半レイヤー平均: {second_half_mean:.4f}")

    if second_half_mean < first_half_mean:
        print("  -> 後半レイヤーがよりactive（変化が大きい）")
    else:
        print("  -> 前半レイヤーがよりactive（変化が大きい）")


def main() -> None:
    """メイン実行関数。"""
    args = parse_args()

    device = get_device()
    print("=" * 70)
    print("SmolLM2 Saturation分析")
    print("=" * 70)
    print(f"デバイス: {device}")
    print(f"モデル: {args.base_model}")
    print(f"サンプル数: {args.num_samples}")

    set_seed(args.seed)

    # モデルロード
    print("\nモデルをロード中...")
    is_cuda = device.type == "cuda"
    base_model, tokenizer = load_pretrained(
        args.base_model,
        device="auto" if is_cuda else None,
        torch_dtype=torch.float16 if is_cuda else None,
    )

    llm = LLM(base_model)
    if is_cuda and not hasattr(base_model, "hf_device_map"):
        llm = llm.to(device)

    print(f"レイヤー数: {llm.num_layers}")

    # データロード
    print("\nデータをロード中...")
    train_batches, _, _ = create_alpaca_dataloaders(
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        seed=args.seed,
        tokenizer_name=tokenizer.name_or_path,
        val_ratio=0.0,
    )
    batches = [x for x, _ in train_batches]
    print(f"バッチ数: {len(batches)}")

    # 分析実行
    print("\ncos_simを計算中...")
    layer_cos_sims = analyze_layer_saturation(llm, batches, device)

    # 結果表示
    print_statistics(layer_cos_sims)
    print_layer_comparison(layer_cos_sims)
    print_threshold_recommendations(layer_cos_sims, args.target_ratios)

    print("\n完了！")


if __name__ == "__main__":
    main()
