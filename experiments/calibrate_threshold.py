"""
CASCADE閾値キャリブレーション

ファインチューニング用データに対し、各レイヤーのcos_simを測定し、
指定したhard_ratioになるようにthresholdを決定する。

使用方法:
    # デフォルト設定（hard_ratio=0.5）
    python experiments/calibrate_threshold.py

    # カスタム設定
    python experiments/calibrate_threshold.py --hard-ratio 0.3 --data-file data.txt

    # 出力例:
    # Layer 28 (入力→Layer 29): threshold=0.847, hard_ratio=50.0%
    # Layer 29 (入力→Layer 30): threshold=0.823, hard_ratio=50.0%
"""

import argparse
from typing import List, Tuple, Optional
import torch
import numpy as np
from transformers import AutoTokenizer

from cascade import (
    LLM,
    load_pretrained,
    set_seed,
    get_device,
)
from cascade.exit_fn import compute_cos_sim


def parse_args() -> argparse.Namespace:
    """コマンドライン引数をパース。"""
    parser = argparse.ArgumentParser(
        description="CASCADE閾値キャリブレーション"
    )

    parser.add_argument(
        "--base-model",
        type=str,
        default="smollm2-135m",
        help="ベースモデル名 (default: smollm2-135m)",
    )
    parser.add_argument(
        "--hard-ratio",
        type=float,
        default=0.5,
        help="hard tokenの目標比率 (default: 0.5)",
    )
    parser.add_argument(
        "--data-file",
        type=str,
        default=None,
        help="キャリブレーション用テキストファイル（省略時はサンプルテキスト使用）",
    )
    parser.add_argument(
        "--target-layers",
        type=int,
        nargs="+",
        default=None,
        help="測定対象のレイヤー（省略時は最後の2層）",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=128,
        help="シーケンス長 (default: 128)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="バッチサイズ (default: 8)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="乱数シード (default: 42)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="結果を保存するJSONファイル（省略時は標準出力のみ）",
    )

    return parser.parse_args()


def load_calibration_data(
    data_file: Optional[str],
    tokenizer: AutoTokenizer,
    seq_len: int,
    batch_size: int,
) -> List[torch.Tensor]:
    """
    キャリブレーション用データをロード。

    Args:
        data_file: テキストファイルパス（Noneの場合はサンプルテキスト）
        tokenizer: トークナイザ
        seq_len: シーケンス長
        batch_size: バッチサイズ

    Returns:
        トークンIDのバッチリスト
    """
    if data_file is not None:
        with open(data_file, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        # サンプルテキスト（WikiTextスタイル）
        text = """
        The quick brown fox jumps over the lazy dog. This is a sample text
        for calibration purposes. Machine learning models require careful
        calibration to achieve optimal performance. Natural language processing
        has made significant advances in recent years, with transformer models
        leading the way. These models can understand and generate human-like
        text with remarkable accuracy. The field continues to evolve rapidly,
        with new architectures and training methods being developed regularly.
        """
        # 繰り返して十分な量にする
        text = text * 100

    # トークナイズ
    tokens = tokenizer.encode(text, add_special_tokens=False)
    tokens = torch.tensor(tokens, dtype=torch.long)

    # シーケンスに分割
    num_seqs = len(tokens) // seq_len
    tokens = tokens[: num_seqs * seq_len]
    tokens = tokens.view(num_seqs, seq_len)

    # バッチに分割
    batches = []
    for i in range(0, len(tokens), batch_size):
        batch = tokens[i : i + batch_size]
        if len(batch) > 0:
            batches.append(batch)

    return batches


def compute_layer_cos_sim(
    llm: LLM,
    batches: List[torch.Tensor],
    device: str,
) -> List[Tuple[int, torch.Tensor]]:
    """
    各レイヤーのcos_simを計算。

    Args:
        llm: LLMモデル
        batches: トークンIDのバッチリスト
        device: デバイス

    Returns:
        (レイヤー番号, cos_simテンソル)のリスト
    """
    all_cos_sims = []

    llm.eval()
    with torch.no_grad():
        for batch in batches:
            batch = batch.to(device)
            _, hidden_history = llm.forward(batch, input_type="token_ids")

            # 各レイヤー間のcos_simを計算
            # hidden_history: [embedding, layer0_out, layer1_out, ..., layerN_out]
            for layer_idx in range(1, len(hidden_history)):
                h_in = hidden_history[layer_idx - 1]
                h_out = hidden_history[layer_idx]
                cos_sim = compute_cos_sim(h_in, h_out)  # (batch, seq_len)

                # 結果を保存
                if layer_idx > len(all_cos_sims):
                    all_cos_sims.append([])
                if len(all_cos_sims) < layer_idx:
                    for _ in range(layer_idx - len(all_cos_sims)):
                        all_cos_sims.append([])
                if layer_idx - 1 < len(all_cos_sims):
                    all_cos_sims[layer_idx - 1].append(cos_sim.cpu())

    # 結合
    results = []
    for layer_idx, cos_sims in enumerate(all_cos_sims):
        if cos_sims:
            combined = torch.cat([cs.flatten() for cs in cos_sims])
            results.append((layer_idx, combined))

    return results


def compute_threshold_for_ratio(
    cos_sims: torch.Tensor,
    hard_ratio: float,
) -> float:
    """
    指定したhard_ratioになるthresholdを計算。

    hard tokenはcos_sim < thresholdで定義される。
    hard_ratio=0.5なら、50%のトークンがhard tokenになるthresholdを返す。

    Args:
        cos_sims: cos_sim値のテンソル
        hard_ratio: hard tokenの目標比率

    Returns:
        threshold値
    """
    # hard_ratio分位点を計算
    # hard_ratio=0.5 → 下位50%がhard → 50パーセンタイル
    percentile = hard_ratio * 100
    threshold = float(np.percentile(cos_sims.numpy(), percentile))
    return threshold


def main() -> None:
    """メイン実行関数。"""
    args = parse_args()

    device = get_device()
    print("=" * 60)
    print("CASCADE閾値キャリブレーション")
    print("=" * 60)
    print(f"デバイス: {device}")
    print(f"ベースモデル: {args.base_model}")
    print(f"目標hard_ratio: {args.hard_ratio}")

    set_seed(args.seed)

    # モデルロード
    print(f"\nモデルをロード中: {args.base_model}")
    base_model, tokenizer = load_pretrained(
        args.base_model,
        device="auto" if device == "cuda" else None,
        torch_dtype=torch.float16 if device == "cuda" else None,
    )

    llm = LLM(base_model)
    if device == "cuda" and not hasattr(base_model, "hf_device_map"):
        llm = llm.to(device)

    num_layers = llm.num_layers
    print(f"レイヤー数: {num_layers}")

    # 対象レイヤー決定
    if args.target_layers is not None:
        target_layers = args.target_layers
    else:
        # デフォルト: 最後の2層
        target_layers = [num_layers - 2, num_layers - 1]

    print(f"測定対象レイヤー: {target_layers}")

    # データロード
    print("\nキャリブレーションデータをロード中...")
    batches = load_calibration_data(
        args.data_file,
        tokenizer,
        args.seq_len,
        args.batch_size,
    )
    print(f"バッチ数: {len(batches)}")

    # cos_sim計算
    print("\ncos_simを計算中...")
    layer_cos_sims = compute_layer_cos_sim(llm, batches, device)

    # 結果を整理
    results = {}
    print("\n" + "=" * 60)
    print("キャリブレーション結果")
    print("=" * 60)

    for layer_idx, cos_sims in layer_cos_sims:
        if layer_idx in target_layers:
            threshold = compute_threshold_for_ratio(cos_sims, args.hard_ratio)
            actual_hard = float((cos_sims < threshold).float().mean())

            results[f"layer_{layer_idx}"] = {
                "threshold": threshold,
                "actual_hard_ratio": actual_hard,
                "cos_sim_mean": float(cos_sims.mean()),
                "cos_sim_std": float(cos_sims.std()),
                "cos_sim_min": float(cos_sims.min()),
                "cos_sim_max": float(cos_sims.max()),
            }

            print(
                f"Layer {layer_idx} (入力→Layer {layer_idx + 1}): "
                f"threshold={threshold:.4f}, hard_ratio={actual_hard*100:.1f}%"
            )
            print(
                f"  cos_sim: mean={cos_sims.mean():.4f}, "
                f"std={cos_sims.std():.4f}, "
                f"range=[{cos_sims.min():.4f}, {cos_sims.max():.4f}]"
            )

    # 推奨設定を表示
    print("\n" + "=" * 60)
    print("推奨設定（experiments/smollm2_cascade.py用）")
    print("=" * 60)

    if target_layers:
        # 最後のレイヤーのthresholdを使用
        last_layer = max(target_layers)
        if f"layer_{last_layer}" in results:
            rec_threshold = results[f"layer_{last_layer}"]["threshold"]
            print(f"--threshold {rec_threshold:.4f}")

    # JSON出力
    if args.output:
        import json

        output_data = {
            "model": args.base_model,
            "target_hard_ratio": args.hard_ratio,
            "num_layers": num_layers,
            "target_layers": target_layers,
            "results": results,
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"\n結果を保存しました: {args.output}")

    print("\n完了！")


if __name__ == "__main__":
    main()
