"""
CASCADE Ensemble評価スクリプト

訓練済みのCASCADE Ensembleを評価し、各パターン（Early Exitポイント）での
val_pplを測定する。

使用方法:
    # 実験ディレクトリを指定して評価
    python experiments/evaluate_cascade_ensemble.py outputs/cascade_20251217_XXXXXX

    # 最新の実験ディレクトリを自動検出
    python experiments/evaluate_cascade_ensemble.py

出力:
    - 各パターン（Base only, Base+Stage1, ...）のval_ppl
    - Early Exit率
    - 推奨停止ポイント
"""

import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaConfig


# ============================================================
# 評価パラメータ
# ============================================================

BATCH_SIZE = 32           # 評価時のバッチサイズ
NUM_VAL_SAMPLES = 100     # 検証サンプル数（訓練時と同じ）
SEQ_LEN = 128             # シーケンス長
SEED = 42                 # 乱数シード

# ============================================================
# 以下は実装コード
# ============================================================


@dataclass
class EvaluationResult:
    """評価結果"""
    pattern_name: str
    num_stages: int
    total_params: int
    val_ppl: float
    val_loss: float
    exit_rates: List[float]  # 各段階でのexit率


def set_seed(seed: int):
    """乱数シードを設定"""
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """利用可能なデバイスを取得"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def create_val_dataloader(
    tokenizer,
    num_val: int,
    seq_len: int,
    batch_size: int,
) -> DataLoader:
    """Alpacaデータセットから検証用データローダーを作成"""
    from datasets import load_dataset

    dataset = load_dataset("tatsu-lab/alpaca", split="train")

    def tokenize_and_chunk(max_samples: int) -> List[List[int]]:
        all_input_ids: List[List[int]] = []
        # 訓練データをスキップして検証データを取得
        skip_count = 1000  # NUM_TRAIN_SAMPLES相当

        sample_idx = 0
        for i in range(len(dataset)):
            if len(all_input_ids) >= max_samples:
                break

            full_text = f"{dataset[i]['instruction']}\n{dataset[i]['input']}\n{dataset[i]['output']}"
            tokens = tokenizer.encode(full_text, add_special_tokens=True)

            for j in range(0, len(tokens) - seq_len, seq_len):
                if sample_idx < skip_count:
                    sample_idx += 1
                    continue
                if len(all_input_ids) >= max_samples:
                    break
                chunk = tokens[j:j + seq_len + 1]
                if len(chunk) == seq_len + 1:
                    all_input_ids.append(chunk)
                sample_idx += 1

        return all_input_ids

    print("  Alpacaデータセットをトークナイズ中...")
    val_data_list = tokenize_and_chunk(num_val)
    print(f"  検証サンプル数: {len(val_data_list)}")

    val_data = torch.tensor(val_data_list, dtype=torch.long)
    val_x, val_y = val_data[:, :-1], val_data[:, 1:]

    val_loader = DataLoader(
        TensorDataset(val_x, val_y),
        batch_size=batch_size,
        shuffle=False,
    )

    return val_loader


def compute_cos_sim(h_in: torch.Tensor, h_out: torch.Tensor) -> torch.Tensor:
    """コサイン類似度を計算"""
    h_in_norm = h_in / (h_in.norm(dim=-1, keepdim=True) + 1e-8)
    h_out_norm = h_out / (h_out.norm(dim=-1, keepdim=True) + 1e-8)
    return (h_in_norm * h_out_norm).sum(dim=-1)


def load_stage_model(model_path: Path, base_model) -> nn.Module:
    """保存されたステージモデルをロード"""
    config = LlamaConfig.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_config(config)
    state_dict = torch.load(model_path / "pytorch_model.bin", map_location="cpu")
    model.load_state_dict(state_dict)
    return model


def evaluate_base_only(
    base_model,
    val_loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    """ベースモデルのみで評価"""
    base_model.eval()
    base_model.to(device)

    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_count = 0

    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            outputs = base_model(x_batch, return_dict=True)
            logits = outputs.logits

            loss = criterion(
                logits.view(-1, logits.size(-1)),
                y_batch.view(-1)
            )
            total_loss += loss.item() * y_batch.numel()
            total_count += y_batch.numel()

    val_loss = total_loss / total_count
    val_ppl = torch.exp(torch.tensor(val_loss)).item()

    base_model.cpu()
    return val_loss, val_ppl


def evaluate_ensemble(
    base_model,
    stage_models: List[nn.Module],
    thresholds: List[float],
    val_loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float, List[float]]:
    """Ensemble（Early Exit適用）で評価

    Args:
        base_model: ベースモデル
        stage_models: 各段階のモデルリスト
        thresholds: 各段階の閾値リスト
        val_loader: 検証データローダー
        device: デバイス

    Returns:
        val_loss: 検証損失
        val_ppl: 検証Perplexity
        exit_rates: 各段階でのexit率
    """
    base_model.eval()
    base_model.to(device)
    for model in stage_models:
        model.eval()
        model.to(device)

    criterion = nn.CrossEntropyLoss(reduction='none')

    # 全トークンの損失を蓄積
    all_losses = []
    exit_counts = [0] * (len(stage_models) + 1)  # Base + 各Stage
    total_tokens = 0

    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            batch_size, seq_len = x_batch.shape

            # ベースモデルで処理
            outputs = base_model(
                x_batch,
                output_hidden_states=True,
                return_dict=True,
            )
            base_logits = outputs.logits  # (batch, seq, vocab)
            h_in = outputs.hidden_states[0]
            h_out = outputs.hidden_states[-1]
            cos_sim = compute_cos_sim(h_in, h_out)  # (batch, seq)

            # 最終的なlogitsを初期化（ベースモデルの出力で初期化）
            final_logits = base_logits.clone()

            # 処理済みマスク（Trueなら既にexitした）
            exited_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)

            # 各段階でEarly Exit判定
            current_hidden = h_out
            for stage_idx, (stage_model, threshold) in enumerate(zip(stage_models, thresholds)):
                # このステージで処理すべきトークン（まだexitしていない）
                should_process = ~exited_mask

                if not should_process.any():
                    break

                # exit判定: cos_sim > threshold ならexit
                exit_mask = (cos_sim > threshold) & should_process

                # exitするトークン数をカウント
                exit_counts[stage_idx] += exit_mask.sum().item()

                # exitしたトークンは処理済みとしてマーク
                exited_mask = exited_mask | exit_mask

                # まだexitしていないトークンを次の段階へ
                continue_mask = should_process & ~exit_mask

                if not continue_mask.any():
                    continue

                # 次の段階のモデルで処理
                # continue_maskに該当するトークンのhidden statesを取得
                continue_indices = continue_mask.nonzero(as_tuple=False)

                if len(continue_indices) == 0:
                    continue

                # バッチ処理のため、continue_maskがTrueのトークンを抽出
                flat_hidden = current_hidden.view(-1, current_hidden.size(-1))
                flat_mask = continue_mask.view(-1)
                selected_hidden = flat_hidden[flat_mask]  # (num_continue, dim)

                # seq_len=1として処理
                selected_hidden = selected_hidden.unsqueeze(1)  # (num_continue, 1, dim)

                stage_outputs = stage_model(
                    inputs_embeds=selected_hidden,
                    output_hidden_states=True,
                    return_dict=True,
                )
                stage_logits = stage_outputs.logits.squeeze(1)  # (num_continue, vocab)
                stage_h_out = stage_outputs.hidden_states[-1].squeeze(1)  # (num_continue, dim)

                # cos_simを更新
                stage_h_in = selected_hidden.squeeze(1)
                stage_cos_sim = compute_cos_sim(stage_h_in, stage_h_out)

                # final_logitsを更新
                flat_logits = final_logits.view(-1, final_logits.size(-1))
                flat_logits[flat_mask] = stage_logits
                final_logits = flat_logits.view(batch_size, seq_len, -1)

                # current_hiddenとcos_simを更新（次の段階用）
                flat_current = current_hidden.view(-1, current_hidden.size(-1))
                flat_current[flat_mask] = stage_h_out
                current_hidden = flat_current.view(batch_size, seq_len, -1)

                flat_cos_sim = cos_sim.view(-1)
                flat_cos_sim[flat_mask] = stage_cos_sim
                cos_sim = flat_cos_sim.view(batch_size, seq_len)

            # 最終段階まで残ったトークン
            final_remaining = (~exited_mask).sum().item()
            exit_counts[-1] += final_remaining

            # 損失計算
            losses = criterion(
                final_logits.view(-1, final_logits.size(-1)),
                y_batch.view(-1)
            )
            all_losses.append(losses)
            total_tokens += y_batch.numel()

    # 全体の損失とPPL
    all_losses = torch.cat(all_losses)
    val_loss = all_losses.mean().item()
    val_ppl = torch.exp(torch.tensor(val_loss)).item()

    # exit率を計算
    exit_rates = [count / total_tokens for count in exit_counts]

    # GPUメモリ解放
    base_model.cpu()
    for model in stage_models:
        model.cpu()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return val_loss, val_ppl, exit_rates


def find_latest_experiment_dir() -> Optional[Path]:
    """最新の実験ディレクトリを検索"""
    outputs_dir = Path("outputs")
    if not outputs_dir.exists():
        return None

    cascade_dirs = list(outputs_dir.glob("cascade_*"))
    if not cascade_dirs:
        return None

    # 最新のディレクトリを返す
    return max(cascade_dirs, key=lambda p: p.name)


def run_evaluation(experiment_dir: Path):
    """評価を実行"""
    print("=" * 80)
    print("CASCADE Ensemble評価")
    print("=" * 80)
    print(f"\n実験ディレクトリ: {experiment_dir}")

    # 設定を読み込み
    config_path = experiment_dir / "config.json"
    if not config_path.exists():
        print(f"エラー: {config_path} が見つかりません")
        return

    with open(config_path) as f:
        config = json.load(f)

    print("\n実験設定:")
    print(f"  ベースモデル: {config['base_model']}")
    print(f"  Hard token比率: {config['hard_ratio'] * 100:.1f}%")
    print(f"  段階数: {config['num_stages']}")

    # 結果を読み込み
    results_path = experiment_dir / "results.json"
    if not results_path.exists():
        print(f"エラー: {results_path} が見つかりません")
        return

    with open(results_path) as f:
        results = json.load(f)

    print(f"  完了した段階数: {len(results)}")

    device = get_device()
    print(f"\nデバイス: {device}")
    set_seed(SEED)

    # ベースモデルをロード
    print(f"\nベースモデルをロード中: {config['base_model']}")
    base_model = AutoModelForCausalLM.from_pretrained(config['base_model'])
    tokenizer = AutoTokenizer.from_pretrained(config['base_model'])
    base_params = sum(p.numel() for p in base_model.parameters())
    print(f"  パラメータ数: {base_params:,} ({base_params/1e6:.1f}M)")

    # 各段階のモデルをロード
    stage_models = []
    thresholds = []
    stage_params_list = []

    for result in results:
        model_path = Path(result['model_path'])
        if not model_path.exists():
            print(f"警告: {model_path} が見つかりません。スキップします。")
            continue

        print(f"  Stage {result['stage']} モデルをロード中...")
        stage_model = load_stage_model(model_path, base_model)
        stage_models.append(stage_model)
        thresholds.append(result['threshold'])
        stage_params = sum(p.numel() for p in stage_model.parameters())
        stage_params_list.append(stage_params)

    print(f"\n  ロードしたステージ数: {len(stage_models)}")

    # 検証データをロード
    print("\n検証データをロード中...")
    val_loader = create_val_dataloader(
        tokenizer,
        NUM_VAL_SAMPLES,
        SEQ_LEN,
        BATCH_SIZE,
    )

    # 評価結果を格納
    evaluation_results: List[EvaluationResult] = []

    # パターン0: ベースモデルのみ
    print("\n" + "=" * 60)
    print("パターン0: Base LLMのみ")
    print("=" * 60)
    val_loss, val_ppl = evaluate_base_only(base_model, val_loader, device)
    print(f"  val_loss: {val_loss:.4f}")
    print(f"  val_ppl: {val_ppl:.2f}")

    evaluation_results.append(EvaluationResult(
        pattern_name="Base only",
        num_stages=0,
        total_params=base_params,
        val_ppl=val_ppl,
        val_loss=val_loss,
        exit_rates=[1.0],  # 全トークンがBaseでexit
    ))

    # パターン1〜N: Base + Stage1, Base + Stage1-2, ...
    for num_stages in range(1, len(stage_models) + 1):
        print(f"\n{'=' * 60}")
        print(f"パターン{num_stages}: Base + Stage1-{num_stages}")
        print("=" * 60)

        models_to_use = stage_models[:num_stages]
        thresholds_to_use = thresholds[:num_stages]
        total_params = base_params + sum(stage_params_list[:num_stages])

        print(f"  使用モデル数: {num_stages + 1} (Base + {num_stages} stages)")
        print(f"  合計パラメータ: {total_params:,} ({total_params/1e6:.1f}M)")
        print(f"  閾値: {thresholds_to_use}")

        val_loss, val_ppl, exit_rates = evaluate_ensemble(
            base_model,
            models_to_use,
            thresholds_to_use,
            val_loader,
            device,
        )

        print(f"  val_loss: {val_loss:.4f}")
        print(f"  val_ppl: {val_ppl:.2f}")
        print("  Exit率:")
        for i, rate in enumerate(exit_rates[:-1]):
            stage_name = "Base" if i == 0 else f"Stage{i}"
            print(f"    {stage_name}: {rate*100:.1f}%")
        print(f"    最終段階まで残存: {exit_rates[-1]*100:.1f}%")

        evaluation_results.append(EvaluationResult(
            pattern_name=f"Base + Stage1-{num_stages}",
            num_stages=num_stages,
            total_params=total_params,
            val_ppl=val_ppl,
            val_loss=val_loss,
            exit_rates=exit_rates,
        ))

    # 結果サマリー
    print("\n" + "=" * 80)
    print("評価結果サマリー")
    print("=" * 80)

    print(f"\n{'パターン':<25} {'段階数':<8} {'パラメータ':<15} {'Val PPL':<12}")
    print("-" * 70)
    for result in evaluation_results:
        print(f"{result.pattern_name:<25} {result.num_stages:<8} "
              f"{result.total_params/1e6:<15.1f}M {result.val_ppl:<12.2f}")

    # 最適パターンを特定
    best_result = min(evaluation_results, key=lambda r: r.val_ppl)
    print(f"\n【推奨】最良パターン: {best_result.pattern_name}")
    print(f"  Val PPL: {best_result.val_ppl:.2f}")
    print(f"  パラメータ数: {best_result.total_params/1e6:.1f}M")

    # 結果を保存
    eval_results_path = experiment_dir / "evaluation_results.json"
    eval_results_dict = [asdict(r) for r in evaluation_results]
    with open(eval_results_path, "w") as f:
        json.dump(eval_results_dict, f, indent=2, ensure_ascii=False)
    print(f"\n評価結果保存先: {eval_results_path}")

    return evaluation_results


def main():
    """メイン関数"""
    if len(sys.argv) > 1:
        experiment_dir = Path(sys.argv[1])
    else:
        experiment_dir = find_latest_experiment_dir()
        if experiment_dir is None:
            print("エラー: 実験ディレクトリが見つかりません")
            print("使用方法: python experiments/evaluate_cascade_ensemble.py <experiment_dir>")
            sys.exit(1)

    if not experiment_dir.exists():
        print(f"エラー: ディレクトリが存在しません: {experiment_dir}")
        sys.exit(1)

    run_evaluation(experiment_dir)


if __name__ == "__main__":
    main()
