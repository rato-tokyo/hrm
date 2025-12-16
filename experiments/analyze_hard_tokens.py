"""
Hard Token分析スクリプト

訓練済みCASCADEモデルを使用して、テキストのどの部分がhard tokenとして
判定されるかを分析し、結果をファイルに出力する。

出力形式:
- JSON: 詳細な分析データ（プログラムで処理しやすい）
- TSV: 人間が読みやすい形式（スプレッドシートで開ける）
- TXT: カラー表示用のテキスト形式

使用方法:
    # 訓練済みモデルをロードして分析
    python experiments/analyze_hard_tokens.py --model-dir ./cascade_smollm2_output/trained_model

    # モデルなしでベースモデルのみで分析（閾値を指定）
    python experiments/analyze_hard_tokens.py --threshold 0.934 --num-samples 100

    # 出力ファイル名を指定
    python experiments/analyze_hard_tokens.py --output hard_token_analysis
"""

import argparse
import json
import os
from typing import Dict, List, Any, Tuple

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
        description="Hard Token分析: どのトークンがhardと判定されるか分析"
    )

    # モデル設定
    parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="訓練済みモデルのディレクトリ（cascade_config.json含む）",
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
        default=None,
        help="hard token判定の閾値（model-dir指定時は自動ロード）",
    )

    # データ設定
    parser.add_argument(
        "--num-samples",
        type=int,
        default=50,
        help="分析するサンプル数 (default: 50)",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=128,
        help="シーケンス長 (default: 128)",
    )

    # 出力設定
    parser.add_argument(
        "--output",
        type=str,
        default="hard_token_analysis",
        help="出力ファイル名（拡張子なし）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="乱数シード (default: 42)",
    )

    return parser.parse_args()


def compute_token_cos_sim(
    llm: LLM,
    token_ids: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    トークンごとのcos_simを計算。

    Args:
        llm: LLMモデル
        token_ids: 入力トークンID (batch_size, seq_len)
        device: デバイス

    Returns:
        cos_sim: トークンごとのcos_sim (batch_size, seq_len)
    """
    token_ids = token_ids.to(device)

    with torch.no_grad():
        _, hidden_history = llm.forward_token_ids(token_ids)

        # レイヤー数に応じたcos_sim計算（exit_fn.pyと同じロジック）
        num_states = len(hidden_history)

        if num_states >= 5:
            # 4層以上: 最後から2-3番目のレイヤーの平均
            h_prev2 = hidden_history[-4]
            h_prev1 = hidden_history[-3]
            h_last = hidden_history[-2]
            cos_sim_1 = compute_cos_sim(h_prev2, h_prev1)
            cos_sim_2 = compute_cos_sim(h_prev1, h_last)
            cos_sim = (cos_sim_1 + cos_sim_2) / 2.0
        elif num_states >= 3:
            # 2-3層: 最後から1-2番目のcos_sim
            h_prev = hidden_history[-3]
            h_last = hidden_history[-2]
            cos_sim = compute_cos_sim(h_prev, h_last)
        else:
            # 1層: 入力と出力のcos_sim
            h_in = hidden_history[0]
            h_out = hidden_history[-2] if num_states > 1 else hidden_history[-1]
            cos_sim = compute_cos_sim(h_in, h_out)

    return cos_sim.cpu()


def analyze_samples(
    llm: LLM,
    tokenizer: Any,
    batches: List[Tuple[torch.Tensor, torch.Tensor]],
    threshold: float,
    device: torch.device,
    num_samples: int,
) -> List[Dict[str, Any]]:
    """
    サンプルを分析してhard token情報を収集。

    Args:
        llm: LLMモデル
        tokenizer: トークナイザ
        batches: (x, y)バッチのリスト
        threshold: hard判定の閾値
        device: デバイス
        num_samples: 分析するサンプル数

    Returns:
        分析結果のリスト
    """
    results = []
    sample_count = 0

    llm.eval()

    for x_batch, y_batch in batches:
        if sample_count >= num_samples:
            break

        # cos_simを計算
        cos_sim = compute_token_cos_sim(llm, x_batch, device)

        batch_size = x_batch.shape[0]
        for i in range(batch_size):
            if sample_count >= num_samples:
                break

            tokens = x_batch[i].tolist()
            cos_sims = cos_sim[i].tolist()
            is_hard = [cs < threshold for cs in cos_sims]

            # トークンをデコード
            token_strs = [tokenizer.decode([t]) for t in tokens]

            # 結果を構築
            sample_result = {
                "sample_id": sample_count,
                "text": tokenizer.decode(tokens),
                "threshold": threshold,
                "num_tokens": len(tokens),
                "num_hard": sum(is_hard),
                "hard_ratio": sum(is_hard) / len(tokens),
                "tokens": []
            }

            for j, (token_id, token_str, cs, hard) in enumerate(
                zip(tokens, token_strs, cos_sims, is_hard)
            ):
                sample_result["tokens"].append({
                    "position": j,
                    "token_id": token_id,
                    "token": token_str,
                    "cos_sim": round(cs, 4),
                    "is_hard": hard,
                })

            results.append(sample_result)
            sample_count += 1

    return results


def save_json(results: List[Dict[str, Any]], output_path: str) -> None:
    """JSON形式で保存。"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"JSON保存: {output_path}")


def save_tsv(results: List[Dict[str, Any]], output_path: str) -> None:
    """TSV形式で保存（スプレッドシート用）。"""
    with open(output_path, 'w', encoding='utf-8') as f:
        # ヘッダー
        f.write("sample_id\tposition\ttoken\ttoken_id\tcos_sim\tis_hard\n")

        for sample in results:
            for token in sample["tokens"]:
                # タブと改行をエスケープ
                token_str = token["token"].replace('\t', '\\t').replace('\n', '\\n')
                f.write(f"{sample['sample_id']}\t{token['position']}\t{token_str}\t"
                       f"{token['token_id']}\t{token['cos_sim']}\t{token['is_hard']}\n")

    print(f"TSV保存: {output_path}")


def save_readable(results: List[Dict[str, Any]], output_path: str, threshold: float) -> None:
    """人間が読みやすい形式で保存。"""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("Hard Token分析レポート\n")
        f.write(f"閾値: {threshold:.4f}\n")
        f.write(f"サンプル数: {len(results)}\n")
        f.write("=" * 80 + "\n\n")

        # 全体統計
        total_tokens = sum(r["num_tokens"] for r in results)
        total_hard = sum(r["num_hard"] for r in results)
        f.write("全体統計:\n")
        f.write(f"  総トークン数: {total_tokens}\n")
        f.write(f"  Hard トークン数: {total_hard}\n")
        f.write(f"  Hard 比率: {total_hard/total_tokens*100:.1f}%\n\n")

        f.write("=" * 80 + "\n")
        f.write("各サンプルの詳細\n")
        f.write("（[H]はHard token、cos_simの値も表示）\n")
        f.write("=" * 80 + "\n\n")

        for sample in results:
            f.write(f"--- サンプル {sample['sample_id']} ---\n")
            f.write(f"Hard比率: {sample['hard_ratio']*100:.1f}% ({sample['num_hard']}/{sample['num_tokens']})\n\n")

            # トークンを表示（hardは[H]マーク付き）
            line = ""
            for token in sample["tokens"]:
                token_str = token["token"]
                if token["is_hard"]:
                    # Hard tokenは強調表示
                    display = f"[H:{token['cos_sim']:.2f}]{token_str}"
                else:
                    display = token_str

                # 改行処理
                if '\n' in display:
                    parts = display.split('\n')
                    line += parts[0]
                    f.write(line + "\n")
                    for part in parts[1:-1]:
                        f.write(part + "\n")
                    line = parts[-1]
                else:
                    line += display

            if line:
                f.write(line + "\n")
            f.write("\n")

    print(f"テキスト保存: {output_path}")


def save_markdown(results: List[Dict[str, Any]], output_path: str, threshold: float) -> None:
    """Markdown形式で保存（HTMLカラー表示用）。"""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# Hard Token分析レポート\n\n")
        f.write(f"- **閾値**: {threshold:.4f}\n")
        f.write(f"- **サンプル数**: {len(results)}\n\n")

        # 全体統計
        total_tokens = sum(r["num_tokens"] for r in results)
        total_hard = sum(r["num_hard"] for r in results)
        f.write("## 全体統計\n\n")
        f.write("| 指標 | 値 |\n")
        f.write("|------|----|\n")
        f.write(f"| 総トークン数 | {total_tokens} |\n")
        f.write(f"| Hard トークン数 | {total_hard} |\n")
        f.write(f"| Hard 比率 | {total_hard/total_tokens*100:.1f}% |\n\n")

        f.write("## サンプル詳細\n\n")
        f.write("凡例: <span style='background-color:#ffcccc'>赤背景</span> = Hard token\n\n")

        for sample in results[:20]:  # 最初の20サンプルのみ
            f.write(f"### サンプル {sample['sample_id']}\n\n")
            f.write(f"Hard比率: {sample['hard_ratio']*100:.1f}%\n\n")

            # HTMLでカラー表示
            html_parts = []
            for token in sample["tokens"]:
                token_str = token["token"].replace('<', '&lt;').replace('>', '&gt;')
                token_str = token_str.replace('\n', '↵\n')
                if token["is_hard"]:
                    cos_sim_val = token["cos_sim"]
                    html_parts.append(
                        f"<span style='background-color:#ffcccc' title='cos_sim={cos_sim_val}'>"
                        f"{token_str}</span>"
                    )
                else:
                    html_parts.append(token_str)

            f.write("<pre>\n")
            f.write("".join(html_parts))
            f.write("\n</pre>\n\n")

    print(f"Markdown保存: {output_path}")


def main() -> None:
    """メイン実行関数。"""
    args = parse_args()

    device = get_device()
    print("=" * 60)
    print("Hard Token分析")
    print("=" * 60)
    print(f"デバイス: {device}")

    set_seed(args.seed)

    # 閾値をロード
    threshold = args.threshold
    base_model = args.base_model

    if args.model_dir:
        config_path = os.path.join(args.model_dir, "cascade_config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            threshold = config.get("threshold_llm0", threshold)
            base_model = config.get("base_model", base_model)
            print(f"CASCADE設定をロード: {config_path}")
            print(f"  ベースモデル: {base_model}")
            print(f"  閾値: {threshold}")
        else:
            print(f"警告: {config_path} が見つかりません")

    if threshold is None:
        print("エラー: 閾値が指定されていません。--threshold または --model-dir を指定してください。")
        return

    # モデルロード
    print(f"\nモデルをロード中: {base_model}")
    is_cuda = device.type == "cuda"
    base_llm, tokenizer = load_pretrained(
        base_model,
        device="auto" if is_cuda else None,
        dtype=torch.float16 if is_cuda else None,
    )

    llm = LLM(base_llm)
    llm.threshold = threshold

    if is_cuda and not hasattr(base_llm, "hf_device_map"):
        llm = llm.to(device)

    print(f"レイヤー数: {llm.num_layers}")
    print(f"閾値: {threshold:.4f}")

    # データロード
    print(f"\nデータをロード中... (num_samples={args.num_samples})")
    train_batches, _, _ = create_alpaca_dataloaders(
        num_samples=args.num_samples,
        batch_size=16,
        seq_len=args.seq_len,
        seed=args.seed,
        tokenizer_name=tokenizer.name_or_path,
        val_ratio=0.0,
    )

    # 分析実行
    print("\n分析中...")
    results = analyze_samples(
        llm, tokenizer, train_batches, threshold, device, args.num_samples
    )

    # 統計表示
    total_tokens = sum(r["num_tokens"] for r in results)
    total_hard = sum(r["num_hard"] for r in results)
    print("\n分析完了:")
    print(f"  サンプル数: {len(results)}")
    print(f"  総トークン数: {total_tokens}")
    print(f"  Hard トークン数: {total_hard}")
    print(f"  Hard 比率: {total_hard/total_tokens*100:.1f}%")

    # ファイル保存
    print("\nファイル保存中...")
    save_json(results, f"{args.output}.json")
    save_tsv(results, f"{args.output}.tsv")
    save_readable(results, f"{args.output}.txt", threshold)
    save_markdown(results, f"{args.output}.md", threshold)

    print("\nColabでダウンロード:")
    print("  from google.colab import files")
    print(f"  files.download('{args.output}.json')")
    print(f"  files.download('{args.output}.tsv')")
    print(f"  files.download('{args.output}.md')")


if __name__ == "__main__":
    main()
