"""
SmolLM2-135M互換LLM作成テスト

SmolLM2-135Mをベースに、後続LLMとして使用する
互換モデルを正しく作成できるかを検証する。

使用方法:
    python scripts/test_smollm2_compatible_llm.py
"""

import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaConfig


def create_llm_from_base(base_model, num_layers: int):
    """ベースモデルと同じアーキテクチャで新規LLMを作成"""
    config = base_model.config

    new_config = LlamaConfig(
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        num_hidden_layers=num_layers,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=getattr(
            config, "num_key_value_heads", config.num_attention_heads
        ),
        max_position_embeddings=config.max_position_embeddings,
        rms_norm_eps=config.rms_norm_eps,
        rope_theta=getattr(config, "rope_theta", 10000),
        hidden_act=config.hidden_act,
        tie_word_embeddings=config.tie_word_embeddings,
    )

    return AutoModelForCausalLM.from_config(new_config)


def test_create_compatible_llm():
    """SmolLM2-135M互換LLMの作成テスト"""
    print("=" * 70)
    print("SmolLM2-135M互換LLM作成テスト")
    print("=" * 70)

    # ベースモデルをロード
    print("\n1. SmolLM2-135Mをロード中...")
    model_name = "HuggingFaceTB/SmolLM2-135M-Instruct"
    base_model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # ベースモデルの仕様を表示
    config = base_model.config
    print("\n【ベースモデル (SmolLM2-135M) の仕様】")
    print(f"  アーキテクチャ: {config.architectures}")
    print(f"  hidden_size: {config.hidden_size}")
    print(f"  num_hidden_layers: {config.num_hidden_layers}")
    print(f"  num_attention_heads: {config.num_attention_heads}")
    print(f"  num_key_value_heads: {config.num_key_value_heads}")
    print(f"  intermediate_size: {config.intermediate_size}")
    print(f"  vocab_size: {config.vocab_size}")
    print(f"  max_position_embeddings: {config.max_position_embeddings}")
    print(f"  rope_theta: {getattr(config, 'rope_theta', 'N/A')}")
    print(f"  hidden_act: {config.hidden_act}")
    print(f"  rms_norm_eps: {config.rms_norm_eps}")
    print(f"  tie_word_embeddings: {config.tie_word_embeddings}")

    # パラメータ数を計算
    base_params = sum(p.numel() for p in base_model.parameters())
    print(f"  総パラメータ数: {base_params:,} ({base_params/1e6:.1f}M)")

    # 後続LLMを作成（レイヤー数を変えて複数テスト）
    test_layers = [2, 4, 6]

    for num_layers in test_layers:
        print(f"\n2. 後続LLM作成テスト (num_layers={num_layers})")
        print("-" * 50)

        # create_llm_from_baseで作成
        additional_llm = create_llm_from_base(base_model, num_layers=num_layers)

        # 作成されたモデルの仕様を確認
        new_config = additional_llm.config
        print(f"  アーキテクチャ: {new_config.architectures}")
        print(f"  hidden_size: {new_config.hidden_size}")
        print(f"  num_hidden_layers: {new_config.num_hidden_layers}")
        print(f"  num_attention_heads: {new_config.num_attention_heads}")
        print(f"  num_key_value_heads: {new_config.num_key_value_heads}")
        print(f"  intermediate_size: {new_config.intermediate_size}")
        print(f"  vocab_size: {new_config.vocab_size}")

        # パラメータ数を計算
        new_params = sum(p.numel() for p in additional_llm.parameters())
        print(f"  総パラメータ数: {new_params:,} ({new_params/1e6:.1f}M)")

        # hidden_sizeの一致を確認
        assert new_config.hidden_size == config.hidden_size, "hidden_sizeが一致しません"
        assert new_config.vocab_size == config.vocab_size, "vocab_sizeが一致しません"
        print("  ✓ hidden_size, vocab_size一致確認OK")

    # 実際のforward passテスト
    print("\n3. Forward Passテスト")
    print("-" * 50)

    # テスト用の入力を作成
    test_text = "Hello, this is a test."
    inputs = tokenizer(test_text, return_tensors="pt")
    input_ids = inputs["input_ids"]
    print(f"  入力テキスト: '{test_text}'")
    print(f"  トークン数: {input_ids.shape[1]}")

    # ベースモデルでhidden statesを取得
    base_model.eval()
    with torch.no_grad():
        base_outputs = base_model(
            input_ids,
            output_hidden_states=True,
            return_dict=True,
        )
        base_hidden = base_outputs.hidden_states[-1]  # 最終層の出力
        print(f"  ベースモデル出力hidden shape: {base_hidden.shape}")

    # 後続LLMにhidden statesを入力（2層版）
    additional_llm = create_llm_from_base(base_model, num_layers=2)
    additional_llm.eval()

    with torch.no_grad():
        # hidden statesを直接入力（embeddingをバイパス）
        # LlamaForCausalLMはinputs_embedsを受け取れる
        additional_outputs = additional_llm(
            inputs_embeds=base_hidden,
            output_hidden_states=True,
            return_dict=True,
        )
        additional_hidden = additional_outputs.hidden_states[-1]
        additional_logits = additional_outputs.logits
        print(f"  後続LLM出力hidden shape: {additional_hidden.shape}")
        print(f"  後続LLM出力logits shape: {additional_logits.shape}")

    print("\n" + "=" * 70)
    print("全テスト完了!")
    print("=" * 70)

    # サマリー
    print("\n【CASCADE実験用モデル構成の提案】")
    print("-" * 50)

    additional_2layer = create_llm_from_base(base_model, num_layers=2)
    additional_4layer = create_llm_from_base(base_model, num_layers=4)
    additional_6layer = create_llm_from_base(base_model, num_layers=6)

    params_2 = sum(p.numel() for p in additional_2layer.parameters())
    params_4 = sum(p.numel() for p in additional_4layer.parameters())
    params_6 = sum(p.numel() for p in additional_6layer.parameters())

    print(f"ベースモデル: SmolLM2-135M ({base_params/1e6:.1f}M)")
    print(f"")
    print(f"後続LLM候補:")
    print(f"  - 2層: {params_2/1e6:.1f}M (合計: {(base_params + params_2)/1e6:.1f}M)")
    print(f"  - 4層: {params_4/1e6:.1f}M (合計: {(base_params + params_4)/1e6:.1f}M)")
    print(f"  - 6層: {params_6/1e6:.1f}M (合計: {(base_params + params_6)/1e6:.1f}M)")
    print(f"")
    print(f"比較対象: SmolLM2-360M ({360}M)")


if __name__ == "__main__":
    test_create_compatible_llm()
