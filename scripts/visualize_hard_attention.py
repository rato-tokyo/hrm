"""
Hard Token + Attention Top-k 可視化スクリプト

元の文に対して:
- Hardトークンを★マーク
- 各HardのAttention Top-kトークンを表示
- 抽出されるトークン集合を可視化
"""

import json
import sys
from typing import Dict, List, Any, Set


def load_data(attention_file: str, analysis_file: str) -> tuple:
    """データファイルを読み込み"""
    with open(attention_file, 'r') as f:
        attention_data = json.load(f)

    with open(analysis_file, 'r') as f:
        analysis_data = json.load(f)

    return attention_data, analysis_data


def visualize_sample(
    sample_attention: Dict[str, Any],
    sample_analysis: Dict[str, Any],
    top_k: int = 5,
) -> str:
    """サンプルを可視化"""
    output = []

    # 全トークン情報（analysis_dataから）
    all_tokens = {t['position']: t for t in sample_analysis['tokens']}

    # Hardトークン情報（attention_dataから）
    hard_tokens = {ht['position']: ht for ht in sample_attention['hard_tokens']}

    # Attention Top-kで参照されるトークンの位置を収集
    attention_refs: Dict[int, Set[int]] = {}  # hard_pos -> set of attended positions
    for ht in sample_attention['hard_tokens']:
        hard_pos = ht['position']
        attention_refs[hard_pos] = set()
        if str(top_k) in ht['top_k_attention']:
            for attn in ht['top_k_attention'][str(top_k)]:
                attention_refs[hard_pos].add(attn['position'])

    # ヘッダー
    output.append("=" * 100)
    output.append(f"【サンプル {sample_attention['sample_id']}】")
    output.append(f"Hard数: {sample_attention['num_hard']} / {sample_attention['num_tokens']} "
                  f"({sample_attention['hard_ratio']*100:.1f}%)")
    output.append("=" * 100)

    # 元の文（Hardを★マーク）
    output.append("\n■ 元の文（★=Hard）:")
    output.append("-" * 100)

    text_parts = []
    for pos in sorted(all_tokens.keys()):
        token = all_tokens[pos]['token']
        if pos in hard_tokens:
            text_parts.append(f"[★{token}]")
        else:
            text_parts.append(token)

    # 改行を保持しつつ結合
    full_text = "".join(text_parts)
    output.append(full_text)
    output.append("-" * 100)

    # 各Hardトークンの詳細
    output.append(f"\n■ Hardトークン詳細（Attention Top-{top_k}）:")
    output.append("-" * 100)

    for ht in sample_attention['hard_tokens']:
        pos = ht['position']
        token = ht['token']
        cos_sim = ht['cos_sim']
        prev_token = ht['prev_token']

        output.append(f"\n位置{pos}: '{token}' (cos_sim={cos_sim:.4f})")
        output.append(f"  直前: '{prev_token}'")

        if str(top_k) in ht['top_k_attention']:
            attn_list = ht['top_k_attention'][str(top_k)]
            attn_str = ", ".join([
                f"'{a['token']}'(pos{a['position']}, {a['attention_score']:.3f})"
                for a in attn_list
            ])
            output.append(f"  Attention Top-{top_k}: {attn_str}")

    # Hard + Attention Top-k で抽出されるトークン
    output.append(f"\n■ 抽出トークン集合（Hard + Attention Top-{top_k}）:")
    output.append("-" * 100)

    # 重要な位置を収集
    important_positions: Set[int] = set()
    for hard_pos in hard_tokens.keys():
        important_positions.add(hard_pos)
        if hard_pos in attention_refs:
            important_positions.update(attention_refs[hard_pos])

    # 位置順にトークンを並べる
    extracted_tokens = []
    for pos in sorted(important_positions):
        if pos in all_tokens:
            token = all_tokens[pos]['token']
            if pos in hard_tokens:
                extracted_tokens.append(f"[★{token}]")
            else:
                extracted_tokens.append(f"[{token}]")

    extracted_text = "".join(extracted_tokens)
    output.append(extracted_text)
    output.append(f"\n抽出トークン数: {len(important_positions)} / {len(all_tokens)} "
                  f"({len(important_positions)/len(all_tokens)*100:.1f}%)")

    # マークなし版（意味確認用）
    output.append("\n■ 抽出トークン（マークなし、意味確認用）:")
    output.append("-" * 100)
    plain_tokens = []
    for pos in sorted(important_positions):
        if pos in all_tokens:
            plain_tokens.append(all_tokens[pos]['token'])
    output.append("".join(plain_tokens))

    return "\n".join(output)


def main():
    # ファイルパス
    attention_file = "hard_token_attention.json"
    analysis_file = "hard_token_analysis.json"

    # top_k の値（引数から取得、デフォルト5）
    top_k = int(sys.argv[1]) if len(sys.argv) > 1 else 5

    # サンプルID（引数から取得、デフォルトは全サンプル）
    sample_id = int(sys.argv[2]) if len(sys.argv) > 2 else None

    # データ読み込み
    attention_data, analysis_data = load_data(attention_file, analysis_file)

    print(f"Attention分析ファイル: {len(attention_data['samples'])}サンプル")
    print(f"トークン分析ファイル: {len(analysis_data)}サンプル")
    print(f"Top-k: {top_k}")
    print()

    # 可視化
    if sample_id is not None:
        # 特定サンプルのみ
        sample_attn = attention_data['samples'][sample_id]
        sample_anal = analysis_data[sample_id]
        print(visualize_sample(sample_attn, sample_anal, top_k))
    else:
        # 全サンプル（attention_dataにあるもののみ）
        for sample_attn in attention_data['samples']:
            sid = sample_attn['sample_id']
            if sid < len(analysis_data):
                sample_anal = analysis_data[sid]
                print(visualize_sample(sample_attn, sample_anal, top_k))
                print("\n" + "=" * 100 + "\n")


if __name__ == "__main__":
    main()
