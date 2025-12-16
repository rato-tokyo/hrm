"""
Hard Token Attention分析スクリプト

Hard tokenを出力する際に、モデルがどのトークンに注目しているか（Attention score）を分析する。
仮説「重要トークン = Hard + Attentionスコア上位k個」を検証するためのスクリプト。

出力形式:
- JSON: 詳細な分析データ
- TXT: 人間が読みやすい形式

使用方法:
    python experiments/analyze_hard_token_attention.py --threshold 0.934 --num-samples 10 --top-k 3,5,10
"""

import argparse
import json
from typing import Dict, List, Any, Tuple, Optional

import torch

from cascade import (
    LLM,
    set_seed,
    get_device,
    load_pretrained,
    create_alpaca_dataloaders,
)
from cascade.exit_fn import compute_cos_sim


def parse_args() -> argparse.Namespace:
    """コマンドライン引数をパース。"""
    parser = argparse.ArgumentParser(
        description="Hard Token Attention分析: Hardトークン出力時のAttention分布を分析"
    )

    parser.add_argument(
        "--base-model",
        type=str,
        default="smollm2-135m",
        help="ベースモデル名 (default: smollm2-135m)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.934,
        help="hard token判定の閾値 (default: 0.934)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="分析するサンプル数 (default: 10)",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=128,
        help="シーケンス長 (default: 128)",
    )
    parser.add_argument(
        "--top-k",
        type=str,
        default="3,5,10",
        help="Attention上位k個のトークンを分析 (default: 3,5,10)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="hard_token_attention",
        help="出力ファイル名（拡張子なし）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="乱数シード (default: 42)",
    )

    return parser.parse_args()


def get_attention_weights(
    model: torch.nn.Module,
    token_ids: torch.Tensor,
    device: torch.device,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """
    Attention weightsを取得。

    Args:
        model: HuggingFace CausalLM
        token_ids: 入力トークンID (batch_size, seq_len)
        device: デバイス

    Returns:
        hidden_states: 最終hidden state
        attentions: 各レイヤーのAttention weights [(batch, heads, seq, seq), ...]
    """
    token_ids = token_ids.to(device)

    with torch.no_grad():
        outputs = model(
            input_ids=token_ids,
            output_attentions=True,
            output_hidden_states=True,
            return_dict=True,
        )

    return outputs.hidden_states, outputs.attentions


def compute_token_cos_sim(
    hidden_history: List[torch.Tensor],
) -> torch.Tensor:
    """
    トークンごとのcos_simを計算。

    Args:
        hidden_history: hidden statesのリスト

    Returns:
        cos_sim: トークンごとのcos_sim (batch_size, seq_len)
    """
    num_states = len(hidden_history)

    if num_states >= 5:
        h_prev2 = hidden_history[-4]
        h_prev1 = hidden_history[-3]
        h_last = hidden_history[-2]
        cos_sim_1 = compute_cos_sim(h_prev2, h_prev1)
        cos_sim_2 = compute_cos_sim(h_prev1, h_last)
        cos_sim = (cos_sim_1 + cos_sim_2) / 2.0
    elif num_states >= 3:
        h_prev = hidden_history[-3]
        h_last = hidden_history[-2]
        cos_sim = compute_cos_sim(h_prev, h_last)
    else:
        h_in = hidden_history[0]
        h_out = hidden_history[-2] if num_states > 1 else hidden_history[-1]
        cos_sim = compute_cos_sim(h_in, h_out)

    return cos_sim


def analyze_attention_for_hard_tokens(
    base_llm: torch.nn.Module,
    tokenizer: Any,
    batches: List[Tuple[torch.Tensor, torch.Tensor]],
    threshold: float,
    device: torch.device,
    num_samples: int,
    top_k_list: List[int],
) -> List[Dict[str, Any]]:
    """
    Hard tokenに対するAttention分析。

    Args:
        base_llm: HuggingFace CausalLM
        tokenizer: トークナイザ
        batches: データバッチ
        threshold: Hard判定閾値
        device: デバイス
        num_samples: サンプル数
        top_k_list: 分析するtop-kのリスト

    Returns:
        分析結果のリスト
    """
    results = []
    sample_count = 0
    max_k = max(top_k_list)

    base_llm.eval()

    for x_batch, y_batch in batches:
        if sample_count >= num_samples:
            break

        x_batch = x_batch.to(device)
        batch_size = x_batch.shape[0]

        # Attentionとhidden statesを取得
        hidden_states, attentions = get_attention_weights(base_llm, x_batch, device)

        # cos_simを計算
        cos_sim = compute_token_cos_sim(list(hidden_states))
        is_hard = (cos_sim < threshold).cpu()

        # 最終レイヤーのAttentionを使用（全ヘッドの平均）
        # attentions[-1]: (batch, heads, seq, seq)
        last_layer_attn = attentions[-1].mean(dim=1)  # (batch, seq, seq)

        for i in range(batch_size):
            if sample_count >= num_samples:
                break

            tokens = x_batch[i].cpu().tolist()
            cos_sims = cos_sim[i].cpu().tolist()
            hard_mask = is_hard[i].tolist()
            attn_matrix = last_layer_attn[i].cpu()  # (seq, seq)

            token_strs = [tokenizer.decode([t]) for t in tokens]

            sample_result = {
                "sample_id": sample_count,
                "text": tokenizer.decode(tokens),
                "threshold": threshold,
                "num_tokens": len(tokens),
                "num_hard": sum(hard_mask),
                "hard_ratio": sum(hard_mask) / len(tokens),
                "hard_tokens": [],
                "top_k_coverage": {k: {"total": 0, "includes_prev": 0} for k in top_k_list},
            }

            # 各Hardトークンを分析
            for pos, (token_id, token_str, cs, is_h) in enumerate(
                zip(tokens, token_strs, cos_sims, hard_mask)
            ):
                if not is_h:
                    continue

                # このトークン位置のAttention分布（自分より前のトークンのみ）
                if pos == 0:
                    # 最初のトークンはスキップ
                    continue

                attn_scores = attn_matrix[pos, :pos].tolist()  # 自分より前のみ

                # 上位k個のインデックスを取得
                attn_with_idx = [(score, idx) for idx, score in enumerate(attn_scores)]
                attn_with_idx.sort(reverse=True)

                top_k_results = {}
                for k in top_k_list:
                    top_indices = [idx for _, idx in attn_with_idx[:k]]
                    top_tokens = [
                        {
                            "position": idx,
                            "token": token_strs[idx],
                            "attention_score": round(attn_scores[idx], 4),
                        }
                        for idx in top_indices
                    ]
                    top_k_results[k] = top_tokens

                    # 直前トークンが含まれるか
                    sample_result["top_k_coverage"][k]["total"] += 1
                    if pos - 1 in top_indices:
                        sample_result["top_k_coverage"][k]["includes_prev"] += 1

                hard_token_info = {
                    "position": pos,
                    "token": token_str,
                    "token_id": token_id,
                    "cos_sim": round(cs, 4),
                    "prev_token": token_strs[pos - 1] if pos > 0 else None,
                    "top_k_attention": top_k_results,
                }

                sample_result["hard_tokens"].append(hard_token_info)

            results.append(sample_result)
            sample_count += 1

    return results


def compute_statistics(results: List[Dict[str, Any]], top_k_list: List[int]) -> Dict[str, Any]:
    """統計を計算。"""
    stats = {
        "total_samples": len(results),
        "total_hard_tokens": sum(r["num_hard"] for r in results),
        "avg_hard_ratio": sum(r["hard_ratio"] for r in results) / len(results) if results else 0,
        "top_k_prev_coverage": {},
    }

    # 各kについて、直前トークンがtop-kに含まれる割合
    for k in top_k_list:
        total = sum(r["top_k_coverage"][k]["total"] for r in results)
        includes_prev = sum(r["top_k_coverage"][k]["includes_prev"] for r in results)
        coverage = includes_prev / total if total > 0 else 0
        stats["top_k_prev_coverage"][k] = {
            "total_hard_tokens": total,
            "includes_prev": includes_prev,
            "coverage_rate": round(coverage * 100, 1),
        }

    return stats


def save_json(results: List[Dict[str, Any]], stats: Dict[str, Any], output_path: str) -> None:
    """JSON形式で保存。"""
    data = {
        "statistics": stats,
        "samples": results,
    }
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"JSON保存: {output_path}")


def save_readable(
    results: List[Dict[str, Any]],
    stats: Dict[str, Any],
    top_k_list: List[int],
    output_path: str,
) -> None:
    """人間が読みやすい形式で保存。"""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("Hard Token Attention分析レポート\n")
        f.write("=" * 80 + "\n\n")

        # 統計
        f.write("## 全体統計\n\n")
        f.write(f"サンプル数: {stats['total_samples']}\n")
        f.write(f"総Hardトークン数: {stats['total_hard_tokens']}\n")
        f.write(f"平均Hard比率: {stats['avg_hard_ratio']*100:.1f}%\n\n")

        f.write("## 直前トークンがAttention上位k個に含まれる割合\n\n")
        for k in top_k_list:
            cov = stats["top_k_prev_coverage"][k]
            f.write(f"Top-{k}: {cov['coverage_rate']}% ({cov['includes_prev']}/{cov['total_hard_tokens']})\n")
        f.write("\n")

        f.write("=" * 80 + "\n")
        f.write("各サンプルの詳細\n")
        f.write("=" * 80 + "\n\n")

        for sample in results[:10]:  # 最初の10サンプル
            f.write(f"--- サンプル {sample['sample_id']} ---\n")
            f.write(f"Hard比率: {sample['hard_ratio']*100:.1f}% ({sample['num_hard']}/{sample['num_tokens']})\n\n")

            for hard_token in sample["hard_tokens"][:5]:  # 最初の5個
                f.write(f"位置 {hard_token['position']}: \"{hard_token['token']}\" (cos_sim={hard_token['cos_sim']})\n")
                f.write(f"  直前: \"{hard_token['prev_token']}\"\n")

                for k in top_k_list[:2]:  # Top-3とTop-5のみ表示
                    if k in hard_token["top_k_attention"]:
                        top_tokens = hard_token["top_k_attention"][k]
                        top_str = ", ".join([f"\"{t['token']}\"({t['attention_score']:.3f})" for t in top_tokens])
                        f.write(f"  Top-{k}: {top_str}\n")
                f.write("\n")

            f.write("\n")

    print(f"テキスト保存: {output_path}")


def main() -> None:
    """メイン実行関数。"""
    args = parse_args()

    device = get_device()
    print("=" * 60)
    print("Hard Token Attention分析")
    print("=" * 60)
    print(f"デバイス: {device}")

    set_seed(args.seed)

    # top-kリストをパース
    top_k_list = [int(k.strip()) for k in args.top_k.split(",")]
    print(f"分析するTop-k: {top_k_list}")

    # モデルロード
    print(f"\nモデルをロード中: {args.base_model}")
    is_cuda = device.type == "cuda"
    base_llm, tokenizer = load_pretrained(
        args.base_model,
        device="auto" if is_cuda else None,
        torch_dtype=torch.float16 if is_cuda else None,
    )

    print(f"レイヤー数: {base_llm.config.num_hidden_layers}")
    print(f"閾値: {args.threshold:.4f}")

    # データロード
    print(f"\nデータをロード中... (num_samples={args.num_samples})")
    train_batches, _, _ = create_alpaca_dataloaders(
        num_samples=args.num_samples * 2,  # 余裕を持って
        batch_size=1,  # Attention分析は1サンプルずつ
        seq_len=args.seq_len,
        seed=args.seed,
        tokenizer_name=tokenizer.name_or_path,
        val_ratio=0.0,
    )

    # 分析実行
    print("\n分析中...")
    results = analyze_attention_for_hard_tokens(
        base_llm,
        tokenizer,
        train_batches,
        args.threshold,
        device,
        args.num_samples,
        top_k_list,
    )

    # 統計計算
    stats = compute_statistics(results, top_k_list)

    # 結果表示
    print("\n分析完了:")
    print(f"  サンプル数: {stats['total_samples']}")
    print(f"  総Hardトークン数: {stats['total_hard_tokens']}")
    print(f"  平均Hard比率: {stats['avg_hard_ratio']*100:.1f}%")
    print("\n直前トークンがAttention上位に含まれる割合:")
    for k in top_k_list:
        cov = stats["top_k_prev_coverage"][k]
        print(f"  Top-{k}: {cov['coverage_rate']}%")

    # ファイル保存
    print("\nファイル保存中...")
    save_json(results, stats, f"{args.output}.json")
    save_readable(results, stats, top_k_list, f"{args.output}.txt")

    print("\nColabでダウンロード:")
    print("  from google.colab import files")
    print(f"  files.download('{args.output}.json')")
    print(f"  files.download('{args.output}.txt')")


if __name__ == "__main__":
    main()
