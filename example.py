"""
CASCADEフレームワーク実行例 - LLM統合

任意のLLMをLLMクラスでラップし、Ensembleで統合する例。
TRUE Early Exitにより、簡単なトークンは前段LLMで処理完了。
"""

from cascade import (
    Ensemble,
    LLM,
    TransformerBlock,
    ExperimentConfig,
    TrainerConfig,
    set_seed,
    get_device,
    create_wikitext_dataloaders,
    train_ensemble,
    evaluate_ensemble,
    create_sequence_data,
)


def main() -> None:
    """CASCADE統合訓練の実行例。"""
    config = ExperimentConfig(
        dim=64,
        num_heads=4,
        ffn_dim=256,
        max_seq_len=1024,
        causal=True,
        eps=1e-6,
        seq_len=32,
        num_samples=10000,
        llm_layers=(2, 2),  # LLM 0: 2層, LLM 1: 2層
    )

    trainer_config = TrainerConfig(
        batch_size=64,
        max_epochs=50,
        patience=3,
        grad_clip=1.0,
        hard_ratio=0.5,
        lr=1e-3,
        verbose=True,
    )

    device = get_device()

    print("=" * 60)
    print("CASCADEフレームワーク - LLM統合訓練例")
    print("=" * 60)
    print(f"デバイス: {device}")
    print(f"モデル: dim={config.dim}, heads={config.num_heads}")
    print(f"LLM構成: {config.llm_layers}")

    # セットアップ
    set_seed(42)
    train_batches, val_batches, vocab_size = create_wikitext_dataloaders(
        config.num_samples, trainer_config.batch_size, config.seq_len, seed=42
    )

    # LLMを作成
    # 任意のLLMをLLMクラスでラップ
    llms = [
        LLM(
            TransformerBlock(
                config.dim, config.num_heads, num_layers,
                config.ffn_dim, config.max_seq_len, config.causal, config.eps
            )
        )
        for num_layers in config.llm_layers
    ]
    ensemble = Ensemble(vocab_size, config.dim, llms).to(device)

    print(f"LLMあたりのレイヤー数: {[llm.num_layers for llm in ensemble.llms]}")

    # バッチからシーケンスデータを作成
    train_data = create_sequence_data(ensemble, train_batches)
    val_data = create_sequence_data(ensemble, val_batches)
    print(f"訓練データ: {len(train_data)}シーケンス ({train_data.num_tokens}トークン)")
    print(f"検証データ: {len(val_data)}シーケンス ({val_data.num_tokens}トークン)")

    # 全LLMを訓練
    # 各LLMは順番に訓練され、hard tokensを次に渡す
    train_stats = train_ensemble(
        ensemble=ensemble,
        train_data=train_data,
        val_data=val_data,
        config=trainer_config,
        lr_decay=0.1,
    )

    # 評価
    print(f"\n{'=' * 60}")
    print("評価: TRUE Early Exit")
    print("=" * 60)

    eval_stats = evaluate_ensemble(ensemble, val_batches)

    print("\n最終結果:")
    print(f"  Accuracy: {eval_stats['accuracy']*100:.2f}%")
    print(f"  PPL: {eval_stats['ppl']:.2f}")
    print(f"  Shallow ratio: {eval_stats['shallow_ratio']*100:.1f}%")
    print(f"  Compute cost: {eval_stats['compute_cost']*100:.1f}%")
    print(f"  Compute savings: {eval_stats['compute_savings']*100:.1f}%")

    # 健全性チェック: 最終PPLは最悪LLMのval_pplを超えてはならない
    llm_stats = train_stats['llm_stats']
    worst_llm_ppl = max(s['best_val_ppl'] for s in llm_stats)
    if eval_stats['ppl'] > worst_llm_ppl:
        print("\n" + "!" * 60)
        print("バグ検出: 最終PPLが最悪LLMのval_pplを超えています！")
        print(f"  最終PPL: {eval_stats['ppl']:.2f}")
        print(f"  最悪LLM val_ppl: {worst_llm_ppl:.2f}")
        print("  これは起こるべきではありません - 調査してください。")
        print("!" * 60)

    print("\n" + "=" * 60)
    print("実験完了！")
    print("=" * 60)


if __name__ == "__main__":
    main()
