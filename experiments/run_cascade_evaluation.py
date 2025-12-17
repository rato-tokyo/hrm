"""
CASCADE Ensemble評価実験

訓練済みCASCADEモデルの各Early Exitパターンでのval_pplを測定。
フレームワーク（cascade/）のIncrementalEvaluatorを使用。

使用方法:
    python experiments/run_cascade_evaluation.py outputs/cascade_YYYYMMDD_HHMMSS

    引数:
        output_dir: 訓練出力ディレクトリ（config.json, results.json, stage_*を含む）
"""

import json
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from datasets import load_dataset  # noqa: E402

from cascade import (  # noqa: E402
    IncrementalEvaluator,
    load_pretrained,
)

# ============================================================
# CONFIG: 評価パラメータ
# ============================================================

# ベースモデル
BASE_MODEL = "smollm2-135m"  # cascade.model_registryの識別子

# データ設定
NUM_VAL_SAMPLES = 100     # 検証サンプル数
SEQ_LEN = 128             # シーケンス長
BATCH_SIZE = 32           # バッチサイズ

# ============================================================
# 以下は実行コード（通常は編集不要）
# ============================================================


def load_alpaca_val_texts(num_val: int):
    """Alpacaデータセットから検証テキストをロード"""
    dataset = load_dataset("tatsu-lab/alpaca", split="train")

    all_texts = []
    for i in range(len(dataset)):
        text = f"{dataset[i]['instruction']}\n{dataset[i]['input']}\n{dataset[i]['output']}"
        all_texts.append(text)

    # 訓練に使用しない後半部分を検証に使用
    val_texts = all_texts[-num_val * 10:]
    return val_texts


def main():
    if len(sys.argv) < 2:
        print("使用方法: python run_cascade_evaluation.py <output_dir>")
        print("例: python run_cascade_evaluation.py outputs/cascade_20251217_094348")
        sys.exit(1)

    output_dir = Path(sys.argv[1])

    if not output_dir.exists():
        print(f"エラー: ディレクトリが存在しません: {output_dir}")
        sys.exit(1)

    print("=" * 80)
    print("CASCADE Ensemble評価")
    print("=" * 80)
    print(f"\n評価対象: {output_dir}")

    # 訓練結果を読み込み
    results_path = output_dir / "results.json"
    if not results_path.exists():
        print(f"エラー: results.jsonが見つかりません: {results_path}")
        sys.exit(1)

    with open(results_path) as f:
        training_results = json.load(f)

    print(f"訓練段階数: {len(training_results)}")

    # 段階モデルのパスと閾値を取得
    stage_model_paths = []
    thresholds = []

    for result in training_results:
        stage_model_paths.append(result["model_path"])
        thresholds.append(result["threshold"])

    print(f"段階モデル: {stage_model_paths}")
    print(f"閾値: {thresholds}")

    # ベースモデルとトークナイザをロード
    print(f"\nベースモデルをロード中: {BASE_MODEL}")
    base_model, tokenizer = load_pretrained(BASE_MODEL)

    # 検証データをロード
    print("\n検証データをロード中...")
    val_texts = load_alpaca_val_texts(NUM_VAL_SAMPLES)
    print(f"  検証テキスト数: {len(val_texts)}")

    # 評価を実行
    evaluator = IncrementalEvaluator(
        tokenizer,
        seq_len=SEQ_LEN,
        batch_size=BATCH_SIZE,
    )

    results = evaluator.evaluate(
        base_model,
        stage_model_paths=stage_model_paths,
        thresholds=thresholds,
        val_texts=val_texts,
        output_path=str(output_dir / "evaluation_results.json"),
    )

    # 結果を表示
    evaluator.print_results(results)

    return results


if __name__ == "__main__":
    main()
