"""
CASCADEフレームワーク実行例 - LLM統合

任意のHugging Face CausalLMをLLMクラスでラップし、Ensembleで統合する例。
TRUE Early Exitにより、簡単なトークンは前段LLMで処理完了。

Hugging Face Transformersとの統合:
- AutoTokenizer（GPT-2 BPEトークナイザ）を使用
- AutoModelForCausalLMでモデル作成
- TrainingArgumentsを直接使用
- datasets.Datasetを直接使用
- HF Trainerで訓練
"""

from transformers import AutoModelForCausalLM, TrainingArguments

from cascade import (
    Ensemble,
    LLM,
    ExperimentConfig,
    CascadeConfig,
    set_seed,
    get_device,
    create_wikitext_dataloaders,
    train_ensemble,
    create_initial_dataset,
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
        tokenizer_name="gpt2",  # Hugging Face GPT-2トークナイザを使用
    )

    # Hugging Face TrainingArgumentsを直接使用
    training_args = TrainingArguments(
        output_dir="./cascade_output",
        num_train_epochs=50,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        learning_rate=1e-3,
        max_grad_norm=1.0,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=1,
        report_to="none",
        remove_unused_columns=False,
        disable_tqdm=False,
    )

    # CASCADE固有の設定
    cascade_config = CascadeConfig(
        patience=3,
        hard_ratio=0.5,
        lr_decay=0.1,
    )

    device = get_device()

    print("=" * 60)
    print("CASCADEフレームワーク - LLM統合訓練例")
    print("=" * 60)
    print(f"デバイス: {device}")
    print(f"モデル: dim={config.dim}, heads={config.num_heads}")
    print(f"LLM構成: {config.llm_layers}")
    print(f"トークナイザ: {config.tokenizer_name}")

    # セットアップ
    set_seed(42)

    # Hugging Face tokenizerを使用してデータをロード
    train_batches, val_batches, vocab_size = create_wikitext_dataloaders(
        num_samples=config.num_samples,
        batch_size=training_args.per_device_train_batch_size,
        seq_len=config.seq_len,
        seed=42,
        tokenizer_name=config.tokenizer_name,
    )

    print(f"語彙サイズ: {vocab_size}")

    # Hugging Face AutoModelForCausalLMを使用してLLMを作成
    llms = []
    for num_layers in config.llm_layers:
        # ExperimentConfig.to_gpt2_config()を使用
        gpt2_config = config.to_gpt2_config(vocab_size)
        gpt2_config.n_layer = num_layers

        # AutoModelForCausalLM.from_config()で新規モデル作成
        # または from_pretrained("gpt2") で訓練済みモデルを使用
        causal_lm = AutoModelForCausalLM.from_config(gpt2_config)
        llm = LLM(causal_lm)
        llms.append(llm)

    ensemble = Ensemble(llms).to(device)

    print(f"LLMあたりのレイヤー数: {[llm.num_layers for llm in ensemble.llms]}")

    # バッチからHF Datasetを作成
    train_dataset = create_initial_dataset(ensemble, train_batches)
    val_dataset = create_initial_dataset(ensemble, val_batches)
    print(f"訓練データ: {len(train_dataset)}シーケンス")
    print(f"検証データ: {len(val_dataset)}シーケンス")

    # 全LLMを訓練（CascadeTrainer使用）
    train_stats = train_ensemble(
        ensemble=ensemble,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        training_args=training_args,
        cascade_config=cascade_config,
    )

    # 評価
    print(f"\n{'=' * 60}")
    print("評価: TRUE Early Exit")
    print("=" * 60)

    eval_stats = ensemble.evaluate(val_batches, training_args.per_device_eval_batch_size)

    print("\n最終結果:")
    print(f"  Accuracy: {eval_stats['accuracy']*100:.2f}%")
    print(f"  PPL: {eval_stats['ppl']:.2f}")

    # 各LLMの統計
    print("\n各LLMの統計:")
    for i, s in enumerate(eval_stats['llm_stats']):
        print(f"  LLM {i}: input={s['input_tokens']}, exit={s['exit_tokens']}, layers_computed={s['layers_computed']}")

    # 計算コスト
    compute_cost = eval_stats['total_layers_computed'] / eval_stats['max_layers_computed']
    print(f"\n計算コスト: {compute_cost*100:.1f}% (savings: {(1-compute_cost)*100:.1f}%)")

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
