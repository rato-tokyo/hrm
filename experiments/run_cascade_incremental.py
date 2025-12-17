"""
CASCADE段階的訓練実験

SmolLM2-135Mをベースに、後続LLMを段階的に追加・訓練する実験。
フレームワーク（cascade/）のIncrementalTrainerを使用。

使用方法:
    python experiments/run_cascade_incremental.py

設定を変更する場合:
    下記の CONFIG セクションを編集してください。
"""

import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from datasets import load_dataset  # noqa: E402

from cascade import (  # noqa: E402
    IncrementalTrainer,
    IncrementalConfig,
    load_pretrained,
)

# ============================================================
# CONFIG: 実験パラメータ（ここを編集して実験設定を変更）
# ============================================================

# ベースモデル
BASE_MODEL = "smollm2-135m"  # cascade.model_registryの識別子

# 段階的追加の設定
LAYERS_PER_STAGE = 8      # 各段階で追加するレイヤー数
HARD_RATIO = 0.6          # Hard token比率（0.6 = cos_sim下位60%）
NUM_STAGES = 5            # 段階数

# 訓練設定
EPOCHS = 10               # 各段階の最大エポック数
BATCH_SIZE = 32           # バッチサイズ
LEARNING_RATE = 1e-4      # 学習率
SEQ_LEN = 128             # シーケンス長
PATIENCE = 1              # Early stopping patience（必ず1）

# データ設定
NUM_TRAIN_SAMPLES = 1000  # 訓練サンプル数
NUM_VAL_SAMPLES = 100     # 検証サンプル数

# その他
SEED = 42                 # 乱数シード
OUTPUT_DIR = None         # 出力ディレクトリ（Noneで自動生成）

# ============================================================
# 以下は実行コード（通常は編集不要）
# ============================================================


def load_alpaca_data(tokenizer, num_train: int, num_val: int, seq_len: int):
    """Alpacaデータセットをロードしてテキストのリストとして返す"""
    dataset = load_dataset("tatsu-lab/alpaca", split="train")

    all_texts = []
    for i in range(len(dataset)):
        text = f"{dataset[i]['instruction']}\n{dataset[i]['input']}\n{dataset[i]['output']}"
        all_texts.append(text)

    # テキストをシーケンスにチャンク化（トークナイザを使用）
    train_texts = all_texts[:num_train * 10]  # 余裕を持って取得
    val_texts = all_texts[num_train * 10:num_train * 10 + num_val * 10]

    return train_texts, val_texts


def main():
    print("=" * 80)
    print("CASCADE段階的訓練実験（フレームワーク使用）")
    print("=" * 80)

    # モデルとトークナイザをロード
    print(f"\nベースモデルをロード中: {BASE_MODEL}")
    base_model, tokenizer = load_pretrained(BASE_MODEL)

    # データをロード
    print("\nデータをロード中...")
    train_texts, val_texts = load_alpaca_data(
        tokenizer,
        NUM_TRAIN_SAMPLES,
        NUM_VAL_SAMPLES,
        SEQ_LEN,
    )
    print(f"  訓練テキスト数: {len(train_texts)}")
    print(f"  検証テキスト数: {len(val_texts)}")

    # 設定を作成
    config = IncrementalConfig(
        layers_per_stage=LAYERS_PER_STAGE,
        hard_ratio=HARD_RATIO,
        num_stages=NUM_STAGES,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        patience=PATIENCE,
        seq_len=SEQ_LEN,
        seed=SEED,
    )

    # トレーナーを作成して訓練
    trainer = IncrementalTrainer(config, tokenizer)
    results = trainer.train(
        base_model,
        train_texts=train_texts,
        val_texts=val_texts,
        output_dir=OUTPUT_DIR,
    )

    print("\n訓練完了!")
    print(f"結果: {len(results)}段階")

    return results


if __name__ == "__main__":
    main()
