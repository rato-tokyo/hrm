"""
CASCADE実験: SmolLM2-135M + 後付けLLM (2層)

既存の訓練済みLLM（SmolLM2-135M-Instruct）に、2層の新規LLMを後付けして訓練する実験。

実験概要:
1. SmolLM2-135M-Instructをフリーズ（訓練しない）
2. SmolLM2のhard tokensを収集（事前にcalibrate_threshold.pyで閾値決定）
3. 2層の新規LLM（同じアーキテクチャ）をhard tokensで訓練
4. TRUE Early Exitで評価

使用方法:
    # デフォルト設定で実行（2層追加）
    python experiments/smollm2_cascade.py

    # 閾値を指定して実行（calibrate_threshold.pyで事前計算）
    python experiments/smollm2_cascade.py --threshold 0.85

    # モデルを切り替えて実行
    python experiments/smollm2_cascade.py --base-model smollm2-360m

    # パラメータをカスタマイズ
    python experiments/smollm2_cascade.py --additional-layers 4 --hard-ratio 0.3
"""

import argparse
import torch
from transformers import TrainingArguments

from cascade import (
    Ensemble,
    LLM,
    CascadeConfig,
    set_seed,
    get_device,
    load_pretrained,
    list_available_models,
    create_llm_from_base,
    create_wikitext_dataloaders,
    train_ensemble,
    create_initial_dataset,
)


def parse_args() -> argparse.Namespace:
    """コマンドライン引数をパース。"""
    parser = argparse.ArgumentParser(
        description="CASCADE実験: 事前学習済みLLM + 後付けLLM"
    )

    # モデル設定
    parser.add_argument(
        "--base-model",
        type=str,
        default="smollm2-135m",
        help="ベースモデル名 (default: smollm2-135m)",
    )
    parser.add_argument(
        "--additional-layers",
        type=int,
        default=2,
        help="後付けLLMのレイヤー数 (default: 2)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="hard token判定の閾値（省略時はhard_ratioから自動計算）",
    )

    # データ設定
    parser.add_argument(
        "--seq-len",
        type=int,
        default=128,
        help="シーケンス長 (default: 128)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5000,
        help="訓練サンプル数 (default: 5000)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="バッチサイズ (default: 16)",
    )

    # 訓練設定
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="エポック数 (default: 10)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="学習率 (default: 1e-4)",
    )
    parser.add_argument(
        "--hard-ratio",
        type=float,
        default=0.3,
        help="hard token比率 (default: 0.3)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=3,
        help="Early stoppingのpatience (default: 3)",
    )

    # その他
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="乱数シード (default: 42)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./cascade_smollm2_output",
        help="出力ディレクトリ (default: ./cascade_smollm2_output)",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="利用可能なモデル一覧を表示して終了",
    )

    return parser.parse_args()


def main() -> None:
    """メイン実行関数。"""
    args = parse_args()

    # モデル一覧表示
    if args.list_models:
        print("利用可能なモデル:")
        for name, spec in list_available_models().items():
            print(f"  {name}: {spec.description}")
        return

    # デバイス設定
    device = get_device()
    print("=" * 60)
    print("CASCADE実験: 事前学習済みLLM + 後付けLLM (2層)")
    print("=" * 60)
    print(f"デバイス: {device}")
    print(f"ベースモデル: {args.base_model}")
    print(f"後付けLLMレイヤー数: {args.additional_layers}")
    print(f"hard ratio: {args.hard_ratio}")
    if args.threshold is not None:
        print(f"threshold: {args.threshold} (固定値)")

    set_seed(args.seed)

    # ベースモデルのロード
    print(f"\nベースモデルをロード中: {args.base_model}")
    base_model, tokenizer = load_pretrained(
        args.base_model,
        device="auto" if device == "cuda" else None,
        torch_dtype=torch.float16 if device == "cuda" else None,
    )

    print(f"  次元: {base_model.config.hidden_size}")
    print(f"  レイヤー数: {base_model.config.num_hidden_layers}")
    print(f"  語彙サイズ: {base_model.config.vocab_size}")

    # LLMクラスでラップ
    llm_base = LLM(base_model)

    # ベースモデルをフリーズ
    for param in llm_base.parameters():
        param.requires_grad = False

    print(f"ベースLLM: dim={llm_base.dim}, layers={llm_base.num_layers}, frozen=True")

    # 後付けLLMの作成（ベースモデルと同じアーキテクチャ）
    additional_model = create_llm_from_base(base_model, num_layers=args.additional_layers)

    if device == "cuda":
        additional_model = additional_model.to(device).half()

    llm_additional = LLM(additional_model)

    arch = base_model.config.architectures[0] if base_model.config.architectures else "Unknown"
    print(
        f"後付けLLM: dim={llm_additional.dim}, "
        f"layers={llm_additional.num_layers}, arch={arch}, trainable=True"
    )

    # Ensembleの構築
    ensemble = Ensemble([llm_base, llm_additional])
    if device == "cuda":
        ensemble = ensemble.to(device)

    print("\nEnsemble構成:")
    for i, llm in enumerate(ensemble.llms):
        trainable = sum(p.numel() for p in llm.parameters() if p.requires_grad)
        total = sum(p.numel() for p in llm.parameters())
        print(f"  LLM {i}: layers={llm.num_layers}, trainable={trainable:,}/{total:,}")

    # データ準備
    print("\nデータをロード中...")
    train_batches, val_batches, vocab_size = create_wikitext_dataloaders(
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        seed=args.seed,
        tokenizer_name=tokenizer.name_or_path,
    )

    print(f"訓練バッチ数: {len(train_batches)}")
    print(f"検証バッチ数: {len(val_batches)}")
    print(f"語彙サイズ: {vocab_size}")

    # HF Datasetを作成
    train_dataset = create_initial_dataset(ensemble, train_batches)
    val_dataset = create_initial_dataset(ensemble, val_batches)

    print(f"訓練Dataset: {len(train_dataset)} samples")
    print(f"検証Dataset: {len(val_dataset)} samples")

    # 訓練設定
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        max_grad_norm=1.0,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=10,
        report_to="none",
        remove_unused_columns=False,
        fp16=device == "cuda",
    )

    cascade_config = CascadeConfig(
        patience=args.patience,
        hard_ratio=args.hard_ratio,
        lr_decay=0.5,
    )

    # 訓練実行
    print("\n訓練開始...")
    print("注意: LLM 0（SmolLM2）はフリーズ済み、LLM 1（後付け）のみ訓練")

    train_stats = train_ensemble(
        ensemble=ensemble,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        training_args=training_args,
        cascade_config=cascade_config,
    )

    # 評価
    print("\n" + "=" * 60)
    print("評価: TRUE Early Exit")
    print("=" * 60)

    eval_stats = ensemble.evaluate(val_batches, args.batch_size)

    print("\n最終結果:")
    print(f"  Accuracy: {eval_stats['accuracy']*100:.2f}%")
    print(f"  PPL: {eval_stats['ppl']:.2f}")

    print("\n各LLMの統計:")
    for i, s in enumerate(eval_stats["llm_stats"]):
        print(
            f"  LLM {i}: input={s['input_tokens']}, "
            f"exit={s['exit_tokens']}, layers={s['layers_computed']}"
        )

    compute_cost = (
        eval_stats["total_layers_computed"] / eval_stats["max_layers_computed"]
    )
    print(f"\n計算コスト: {compute_cost*100:.1f}% (節約: {(1-compute_cost)*100:.1f}%)")

    # 訓練統計の表示
    print("\n訓練統計:")
    for i, llm_stat in enumerate(train_stats["llm_stats"]):
        print(
            f"  LLM {i}: best_val_ppl={llm_stat['best_val_ppl']:.2f}, "
            f"threshold={llm_stat.get('threshold', 'N/A')}"
        )

    print("\n" + "=" * 60)
    print("実験完了！")
    print("=" * 60)


if __name__ == "__main__":
    main()
