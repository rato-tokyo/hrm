"""
SmolLM2レイヤー別cos_simデータ収集

各レイヤーのcos_sim（入力と出力のコサイン類似度）分布を収集し、
npzファイルに保存する。ダウンロードしてローカルで分析可能。

使用方法:
    # Colabで実行
    !PYTHONPATH=/content/hrm python experiments/collect_cos_sim_data.py

    # サンプル数を変更
    !PYTHONPATH=/content/hrm python experiments/collect_cos_sim_data.py --num-samples 1000

出力:
    cos_sim_data.npz - 以下のキーを含む:
        - layer_{i}: 各レイヤーのcos_sim値 (shape: num_tokens,)
        - metadata: サンプル数、バッチサイズ、シーケンス長等の情報
"""

import argparse
from typing import Dict, List
import numpy as np
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
        description="SmolLM2 cos_simデータ収集"
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
        "--output",
        type=str,
        default="cos_sim_data.npz",
        help="出力ファイル名 (default: cos_sim_data.npz)",
    )
    return parser.parse_args()


def collect_layer_cos_sims(
    llm: LLM,
    batches: List[torch.Tensor],
    device: torch.device,
) -> Dict[int, np.ndarray]:
    """
    各レイヤーのcos_simを収集。

    Returns:
        {layer_idx: cos_sim_array} のDict
    """
    layer_cos_sims: Dict[int, List[np.ndarray]] = {}

    llm.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(batches):
            if batch_idx % 10 == 0:
                print(f"  バッチ {batch_idx + 1}/{len(batches)}")

            batch = batch.to(device)
            _, hidden_history = llm.forward_token_ids(batch)

            # 各レイヤー間のcos_simを計算
            for layer_idx in range(1, len(hidden_history)):
                h_in = hidden_history[layer_idx - 1]
                h_out = hidden_history[layer_idx]
                cos_sim = compute_cos_sim(h_in, h_out).cpu().numpy()

                if layer_idx not in layer_cos_sims:
                    layer_cos_sims[layer_idx] = []
                layer_cos_sims[layer_idx].append(cos_sim.flatten())

    # 結合してnumpy配列に
    return {
        layer_idx: np.concatenate(cos_sims)
        for layer_idx, cos_sims in layer_cos_sims.items()
    }


def main() -> None:
    """メイン実行関数。"""
    args = parse_args()

    device = get_device()
    print("=" * 70)
    print("SmolLM2 cos_simデータ収集")
    print("=" * 70)
    print(f"デバイス: {device}")
    print(f"モデル: {args.base_model}")
    print(f"サンプル数: {args.num_samples}")
    print(f"出力ファイル: {args.output}")

    set_seed(args.seed)

    # モデルロード
    print("\nモデルをロード中...")
    is_cuda = device.type == "cuda"
    base_model, tokenizer = load_pretrained(
        args.base_model,
        device="auto" if is_cuda else None,
        dtype=torch.float16 if is_cuda else None,
    )

    llm = LLM(base_model)
    if is_cuda and not hasattr(base_model, "hf_device_map"):
        llm = llm.to(device)

    num_layers = llm.num_layers
    print(f"レイヤー数: {num_layers}")

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

    # cos_sim収集
    print("\ncos_simを計算中...")
    layer_cos_sims = collect_layer_cos_sims(llm, batches, device)

    # npz形式で保存
    print(f"\n{args.output}に保存中...")

    save_dict = {}
    for layer_idx, cos_sim in layer_cos_sims.items():
        save_dict[f"layer_{layer_idx}"] = cos_sim
        print(f"  layer_{layer_idx}: shape={cos_sim.shape}, mean={cos_sim.mean():.4f}")

    # メタデータを追加
    save_dict["metadata"] = np.array([
        args.num_samples,
        args.batch_size,
        args.seq_len,
        num_layers,
        args.seed,
    ])

    np.savez_compressed(args.output, **save_dict)

    print(f"\n保存完了: {args.output}")
    print("\nColabでダウンロード:")
    print("  from google.colab import files")
    print(f"  files.download('{args.output}')")


if __name__ == "__main__":
    main()
