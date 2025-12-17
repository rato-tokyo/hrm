"""
CASCADE vs Baseline 比較実験

SmolLM2-135M + Stage LLMs (CASCADE) vs SmolLM2-360M (Baseline) の比較。

CASCADEの仕様:
- 各LLMは独自のlm_head（output head）を持つ
- Early Exit時: そのLLMのlm_headでlogits計算 → exit
- Hard token時: hidden statesのまま後段に渡す（lm_headは使わない）

使用方法:
    python experiments/run_comparison.py
"""

import json
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import numpy as np  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402
from datasets import load_dataset  # noqa: E402
from transformers import PreTrainedModel, PreTrainedTokenizer  # noqa: E402

from cascade import (  # noqa: E402
    load_pretrained,
    compute_cos_sim_from_history,
)
from cascade.model_registry import create_llm_from_base  # noqa: E402


# ============================================================
# CONFIG: 実験パラメータ
# ============================================================

# 共通設定
NUM_TRAIN_SAMPLES = 1000
NUM_VAL_SAMPLES = 100
SEQ_LEN = 128
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 1e-5
PATIENCE = 3
SEED = 42

# CASCADE設定
CASCADE_BASE_MODEL = "smollm2-135m"
CASCADE_LAYERS_PER_STAGE = 8
CASCADE_HARD_RATIO = 0.5
CASCADE_NUM_STAGES = 3

# Baseline設定
BASELINE_MODEL = "smollm2-360m"

# ============================================================


@dataclass
class ExperimentResult:
    """実験結果"""
    method: str
    model_name: str
    total_params: int
    num_layers: int
    train_tokens: int
    val_tokens: int
    epochs_trained: int
    training_time: float
    best_val_ppl: float
    final_val_ppl: float
    exit_distribution: Optional[Dict[str, float]] = None
    compute_ratio: Optional[float] = None


def set_seed(seed: int):
    """乱数シードを設定"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """デバイスを取得"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_alpaca_data(
    tokenizer: PreTrainedTokenizer,
    num_train: int,
    num_val: int,
    seq_len: int,
    batch_size: int,
) -> Tuple[DataLoader, DataLoader]:
    """Alpacaデータをロードしてトークナイズ"""
    print("Alpacaデータセットをロード中...")
    dataset = load_dataset("tatsu-lab/alpaca", split="train")

    all_texts = []
    for item in dataset:
        text = f"{item['instruction']}\n{item['input']}\n{item['output']}"
        all_texts.append(text)

    # 訓練と検証に分割
    train_texts = all_texts[:num_train * 10]
    val_texts = all_texts[-(num_val * 10):]

    def tokenize_and_chunk(texts: List[str], max_samples: int) -> List[List[int]]:
        all_input_ids = []
        for text in texts:
            if len(all_input_ids) >= max_samples:
                break
            tokens = tokenizer.encode(text, add_special_tokens=True)
            for j in range(0, len(tokens) - seq_len, seq_len):
                if len(all_input_ids) >= max_samples:
                    break
                chunk = tokens[j:j + seq_len + 1]
                if len(chunk) == seq_len + 1:
                    all_input_ids.append(chunk)
        return all_input_ids

    train_data = torch.tensor(tokenize_and_chunk(train_texts, num_train), dtype=torch.long)
    val_data = torch.tensor(tokenize_and_chunk(val_texts, num_val), dtype=torch.long)

    train_x, train_y = train_data[:, :-1], train_data[:, 1:]
    val_x, val_y = val_data[:, :-1], val_data[:, 1:]

    train_loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(val_x, val_y),
        batch_size=batch_size,
        shuffle=False,
    )

    print(f"  訓練サンプル数: {len(train_x)}")
    print(f"  検証サンプル数: {len(val_x)}")
    print(f"  訓練トークン数: {len(train_x) * seq_len:,}")
    print(f"  検証トークン数: {len(val_x) * seq_len:,}")

    return train_loader, val_loader


def train_baseline(
    model: PreTrainedModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    patience: int,
) -> Dict:
    """Baselineモデルを訓練"""
    model.train()
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    best_model_state = None
    patience_counter = 0
    epochs_trained = 0

    start_time = time.time()

    for epoch in range(epochs):
        epoch_start = time.time()
        epochs_trained = epoch + 1

        # 訓練
        model.train()
        train_loss_sum = 0.0
        train_count = 0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            outputs = model(x_batch, return_dict=True)
            logits = outputs.logits

            loss = criterion(logits.view(-1, logits.size(-1)), y_batch.view(-1))
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * y_batch.numel()
            train_count += y_batch.numel()

        train_loss = train_loss_sum / train_count

        # 検証
        model.eval()
        val_loss_sum = 0.0
        val_count = 0

        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                outputs = model(x_batch, return_dict=True)
                logits = outputs.logits

                loss = criterion(logits.view(-1, logits.size(-1)), y_batch.view(-1))
                val_loss_sum += loss.item() * y_batch.numel()
                val_count += y_batch.numel()

        val_loss = val_loss_sum / val_count
        val_ppl = float(np.exp(val_loss))
        epoch_time = time.time() - epoch_start

        print(f"    Epoch {epoch + 1}/{epochs}: train_loss={train_loss:.4f}, "
              f"val_loss={val_loss:.4f}, val_ppl={val_ppl:.2f}, time={epoch_time:.1f}s")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"    Early stopping at epoch {epoch + 1}")
                break

    training_time = time.time() - start_time

    # 最良モデルを復元
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return {
        "best_val_loss": best_val_loss,
        "best_val_ppl": float(np.exp(best_val_loss)),
        "final_val_ppl": val_ppl,
        "epochs_trained": epochs_trained,
        "training_time": training_time,
    }


def train_cascade(
    base_model: PreTrainedModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    layers_per_stage: int,
    hard_ratio: float,
    num_stages: int,
    epochs: int,
    learning_rate: float,
    patience: int,
) -> Tuple[List[PreTrainedModel], List[float], Dict]:
    """
    CASCADE方式で訓練。

    仕様:
    - 各Stage LLMは独自のlm_headを持つ
    - 訓練時: hidden_statesを入力とし、lm_headで予測
    - Hard tokens: hidden_statesのまま次のstageに渡す
    """
    print("\nベースモデルからhidden statesを抽出中...")
    base_model.eval()
    base_model.to(device)

    # 訓練データからhidden statesを抽出
    train_hidden, train_labels, train_cos_sim = extract_hidden_states(
        base_model, train_loader, device
    )
    val_hidden, val_labels, val_cos_sim = extract_hidden_states(
        base_model, val_loader, device
    )

    print(f"  訓練トークン数: {len(train_labels):,}")
    print(f"  検証トークン数: {len(val_labels):,}")

    # メモリ解放
    base_model.cpu()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Stage LLMsを訓練
    stage_models: List[PreTrainedModel] = []
    thresholds: List[float] = []

    current_train = (train_hidden, train_labels, train_cos_sim)
    current_val = (val_hidden, val_labels, val_cos_sim)

    total_training_time = 0.0
    total_stage_params = 0

    for stage in range(1, num_stages + 1):
        print(f"\n{'=' * 60}")
        print(f"Stage {stage}/{num_stages}")
        print(f"{'=' * 60}")

        # Hard tokensをフィルタリング
        threshold = float(torch.quantile(current_train[2].float(), hard_ratio).item())
        thresholds.append(threshold)

        hard_mask_train = current_train[2] <= threshold
        hard_mask_val = current_val[2] <= threshold

        hard_train = (current_train[0][hard_mask_train], current_train[1][hard_mask_train])
        hard_val = (current_val[0][hard_mask_val], current_val[1][hard_mask_val])

        print(f"  閾値: {threshold:.4f}")
        print(f"  Hard tokens (train): {hard_train[0].shape[0]:,} / {current_train[0].shape[0]:,} "
              f"({hard_train[0].shape[0] / current_train[0].shape[0] * 100:.1f}%)")
        print(f"  Hard tokens (val): {hard_val[0].shape[0]:,}")

        # Stage LLMを作成（lm_headを含む完全なCausalLM）
        # NOTE: base_modelを一時的にGPUに戻してconfigを取得
        base_model.to(device)
        stage_model = create_llm_from_base(base_model, layers_per_stage)
        base_model.cpu()

        stage_params = sum(p.numel() for p in stage_model.parameters())
        total_stage_params += stage_params
        print(f"  Stage LLMパラメータ数: {stage_params:,} ({stage_params / 1e6:.1f}M)")

        # Stage LLMを訓練
        print("  訓練開始...")
        start_time = time.time()

        stage_model, train_result = train_stage_llm(
            stage_model,
            hard_train[0], hard_train[1],
            hard_val[0], hard_val[1],
            device, epochs, learning_rate, patience,
        )

        stage_training_time = time.time() - start_time
        total_training_time += stage_training_time

        print(f"  訓練完了: {stage_training_time:.1f}秒")
        print(f"  Best val_ppl: {train_result['best_val_ppl']:.2f}")

        stage_models.append(stage_model)

        # 次のstage用にhidden statesを更新
        if stage < num_stages:
            print("  次のstage用にhidden statesを更新中...")
            stage_model.eval()

            new_train = transform_hidden_states(
                stage_model, hard_train[0], hard_train[1], device
            )
            new_val = transform_hidden_states(
                stage_model, hard_val[0], hard_val[1], device
            )

            current_train = new_train
            current_val = new_val

        # メモリ解放
        stage_model.cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    base_params = sum(p.numel() for p in base_model.parameters())

    return stage_models, thresholds, {
        "total_params": base_params + total_stage_params,
        "base_params": base_params,
        "stage_params": total_stage_params,
        "training_time": total_training_time,
    }


def extract_hidden_states(
    model: PreTrainedModel,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """モデルからhidden statesを抽出"""
    model.eval()

    all_hidden = []
    all_labels = []
    all_cos_sim = []

    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            outputs = model(
                x_batch,
                output_hidden_states=True,
                return_dict=True,
            )

            hidden_history = outputs.hidden_states
            cos_sim = compute_cos_sim_from_history(hidden_history)
            h_out = hidden_history[-1]

            batch_size, seq_len, dim = h_out.shape
            all_hidden.append(h_out.view(-1, dim).cpu())
            all_labels.append(y_batch.view(-1).cpu())
            all_cos_sim.append(cos_sim.view(-1).cpu())

    return (
        torch.cat(all_hidden, dim=0),
        torch.cat(all_labels, dim=0),
        torch.cat(all_cos_sim, dim=0),
    )


def transform_hidden_states(
    model: PreTrainedModel,
    hidden: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
    batch_size: int = 32,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """hidden statesをモデルで変換"""
    model.eval()
    model.to(device)

    all_hidden = []
    all_cos_sim = []

    with torch.no_grad():
        for i in range(0, hidden.shape[0], batch_size):
            h_batch = hidden[i:i + batch_size].to(device)
            h_batch = h_batch.unsqueeze(1)  # シーケンス長1

            outputs = model(
                inputs_embeds=h_batch,
                output_hidden_states=True,
                return_dict=True,
            )

            hidden_history = outputs.hidden_states
            h_out = hidden_history[-1].squeeze(1)
            cos_sim = compute_cos_sim_from_history(hidden_history)
            cos_sim = cos_sim.squeeze(1)

            all_hidden.append(h_out.cpu())
            all_cos_sim.append(cos_sim.cpu())

    return (
        torch.cat(all_hidden, dim=0),
        labels,
        torch.cat(all_cos_sim, dim=0),
    )


def train_stage_llm(
    model: PreTrainedModel,
    train_hidden: torch.Tensor,
    train_labels: torch.Tensor,
    val_hidden: torch.Tensor,
    val_labels: torch.Tensor,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    patience: int,
) -> Tuple[PreTrainedModel, Dict]:
    """Stage LLMを訓練"""
    model.train()
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    train_dataset = TensorDataset(train_hidden, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    val_dataset = TensorDataset(val_hidden, val_labels)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    best_val_loss = float("inf")
    best_model_state = None
    patience_counter = 0

    for epoch in range(epochs):
        epoch_start = time.time()

        # 訓練
        model.train()
        train_loss_sum = 0.0
        train_count = 0

        for h_batch, y_batch in train_loader:
            h_batch = h_batch.to(device).float()
            y_batch = y_batch.to(device)
            h_batch = h_batch.unsqueeze(1)

            optimizer.zero_grad()

            # Stage LLMのlm_headを使って予測
            outputs = model(inputs_embeds=h_batch, return_dict=True)
            logits = outputs.logits.squeeze(1)

            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * y_batch.size(0)
            train_count += y_batch.size(0)

        train_loss = train_loss_sum / train_count

        # 検証
        model.eval()
        val_loss_sum = 0.0
        val_count = 0

        with torch.no_grad():
            for h_batch, y_batch in val_loader:
                h_batch = h_batch.to(device).float()
                y_batch = y_batch.to(device)
                h_batch = h_batch.unsqueeze(1)

                outputs = model(inputs_embeds=h_batch, return_dict=True)
                logits = outputs.logits.squeeze(1)

                loss = criterion(logits, y_batch)
                val_loss_sum += loss.item() * y_batch.size(0)
                val_count += y_batch.size(0)

        val_loss = val_loss_sum / val_count
        val_ppl = float(np.exp(val_loss))
        epoch_time = time.time() - epoch_start

        print(f"      Epoch {epoch + 1}/{epochs}: train_loss={train_loss:.4f}, "
              f"val_loss={val_loss:.4f}, val_ppl={val_ppl:.2f}, time={epoch_time:.1f}s")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"      Early stopping at epoch {epoch + 1}")
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, {
        "best_val_loss": best_val_loss,
        "best_val_ppl": float(np.exp(best_val_loss)),
    }


def evaluate_cascade(
    base_model: PreTrainedModel,
    stage_models: List[PreTrainedModel],
    thresholds: List[float],
    val_loader: DataLoader,
    device: torch.device,
) -> Dict:
    """
    CASCADE Ensembleを評価。

    仕様:
    - Exit時: そのLLMのlm_headでlogits計算
    - Hard token時: hidden statesのまま次のstageに渡す
    """
    base_model.eval()
    base_model.to(device)

    for sm in stage_models:
        sm.eval()
        sm.to(device)

    total_loss = 0.0
    total_correct = 0
    total_tokens = 0

    exit_counts = {"base": 0}
    for i in range(len(stage_models)):
        exit_counts[f"stage_{i + 1}"] = 0

    base_layers = base_model.config.num_hidden_layers
    total_layers_computed = 0

    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            batch_size, seq_len = x_batch.shape

            # ベースモデルを通す
            outputs = base_model(
                x_batch,
                output_hidden_states=True,
                return_dict=True,
            )

            hidden_history = outputs.hidden_states
            h_out = hidden_history[-1]
            # ベースモデルのlm_headでlogits計算
            logits = outputs.logits

            # Exit判定
            if len(stage_models) == 0:
                exit_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)
            else:
                cos_sim = compute_cos_sim_from_history(hidden_history)
                exit_mask = cos_sim > thresholds[0]

            # ベースでexitしたトークンのloss計算（ベースのlm_headを使用）
            exit_logits = logits[exit_mask]
            exit_targets = y_batch[exit_mask]
            if exit_logits.shape[0] > 0:
                total_loss += F.cross_entropy(
                    exit_logits.view(-1, logits.shape[-1]),
                    exit_targets.view(-1),
                    reduction='sum'
                ).item()
                total_correct += int((exit_logits.argmax(dim=-1) == exit_targets).sum().item())

            exit_counts["base"] += int(exit_mask.sum().item())
            total_layers_computed += int(exit_mask.sum().item()) * base_layers

            # Hard tokensを後段に渡す（hidden statesのまま）
            current_hidden = h_out
            current_mask = ~exit_mask
            current_targets = y_batch

            for stage_idx, stage_model in enumerate(stage_models):
                if not current_mask.any():
                    break

                # 継続トークンのhidden statesを抽出
                continue_hidden = current_hidden[current_mask]
                continue_targets = current_targets[current_mask]

                # シーケンス長1として処理（hidden statesを入力）
                continue_hidden = continue_hidden.unsqueeze(1)

                outputs = stage_model(
                    inputs_embeds=continue_hidden,
                    output_hidden_states=True,
                    return_dict=True,
                )

                hidden_history = outputs.hidden_states
                stage_h_out = hidden_history[-1].squeeze(1)
                # Stage LLMのlm_headでlogits計算
                stage_logits = outputs.logits.squeeze(1)

                is_last_stage = (stage_idx == len(stage_models) - 1)

                if is_last_stage:
                    stage_exit_mask = torch.ones(
                        continue_hidden.shape[0], dtype=torch.bool, device=device
                    )
                else:
                    cos_sim = compute_cos_sim_from_history(hidden_history)
                    cos_sim = cos_sim.squeeze(1)
                    stage_exit_mask = cos_sim > thresholds[stage_idx + 1]

                # Exitトークンのloss計算（Stage LLMのlm_headを使用）
                exit_logits = stage_logits[stage_exit_mask]
                exit_targets = continue_targets[stage_exit_mask]
                if exit_logits.shape[0] > 0:
                    total_loss += F.cross_entropy(
                        exit_logits,
                        exit_targets,
                        reduction='sum'
                    ).item()
                    total_correct += int((exit_logits.argmax(dim=-1) == exit_targets).sum().item())

                exit_counts[f"stage_{stage_idx + 1}"] += int(stage_exit_mask.sum().item())
                total_layers_computed += int(stage_exit_mask.sum().item()) * stage_model.config.num_hidden_layers

                # 次のstage用に更新（hidden statesのまま渡す）
                current_hidden = stage_h_out
                current_mask = ~stage_exit_mask
                current_targets = continue_targets

            total_tokens += batch_size * seq_len

    # 統計計算
    val_ppl = float(np.exp(total_loss / total_tokens)) if total_tokens > 0 else float('inf')
    accuracy = total_correct / total_tokens if total_tokens > 0 else 0.0

    exit_distribution = {k: v / total_tokens for k, v in exit_counts.items()}

    total_stage_layers = sum(sm.config.num_hidden_layers for sm in stage_models)
    max_layers = total_tokens * (base_layers + total_stage_layers)
    compute_ratio = total_layers_computed / max_layers if max_layers > 0 else 0.0

    # メモリ解放
    base_model.cpu()
    for sm in stage_models:
        sm.cpu()

    return {
        "val_ppl": val_ppl,
        "accuracy": accuracy,
        "exit_distribution": exit_distribution,
        "compute_ratio": compute_ratio,
    }


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"outputs/comparison_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(SEED)
    device = get_device()

    print("=" * 80)
    print("CASCADE vs Baseline 比較実験")
    print("=" * 80)
    print(f"\nデバイス: {device}")
    print(f"出力先: {output_dir}")

    # ============================================================
    # Baseline: SmolLM2-360M
    # ============================================================
    print("\n" + "=" * 80)
    print("Baseline: SmolLM2-360M")
    print("=" * 80)

    baseline_model, baseline_tokenizer = load_pretrained(BASELINE_MODEL)
    baseline_params = sum(p.numel() for p in baseline_model.parameters())
    baseline_layers = baseline_model.config.num_hidden_layers

    print(f"  パラメータ数: {baseline_params:,} ({baseline_params / 1e6:.1f}M)")
    print(f"  レイヤー数: {baseline_layers}")

    # データロード（Baselineのtokenizerを使用）
    train_loader, val_loader = load_alpaca_data(
        baseline_tokenizer, NUM_TRAIN_SAMPLES, NUM_VAL_SAMPLES, SEQ_LEN, BATCH_SIZE
    )

    print("\n訓練開始...")
    baseline_result = train_baseline(
        baseline_model, train_loader, val_loader, device,
        EPOCHS, LEARNING_RATE, PATIENCE
    )

    baseline_exp_result = ExperimentResult(
        method="Baseline",
        model_name=BASELINE_MODEL,
        total_params=baseline_params,
        num_layers=baseline_layers,
        train_tokens=NUM_TRAIN_SAMPLES * SEQ_LEN,
        val_tokens=NUM_VAL_SAMPLES * SEQ_LEN,
        epochs_trained=baseline_result["epochs_trained"],
        training_time=baseline_result["training_time"],
        best_val_ppl=baseline_result["best_val_ppl"],
        final_val_ppl=baseline_result["final_val_ppl"],
    )

    print("\nBaseline結果:")
    print(f"  Best val_ppl: {baseline_result['best_val_ppl']:.2f}")

    # メモリ解放
    del baseline_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ============================================================
    # CASCADE: SmolLM2-135M + Stage LLMs
    # ============================================================
    print("\n" + "=" * 80)
    print(f"CASCADE: {CASCADE_BASE_MODEL} + Stage LLMs")
    print("=" * 80)

    cascade_base_model, cascade_tokenizer = load_pretrained(CASCADE_BASE_MODEL)
    cascade_base_params = sum(p.numel() for p in cascade_base_model.parameters())
    cascade_base_layers = cascade_base_model.config.num_hidden_layers

    print(f"  ベースパラメータ数: {cascade_base_params:,} ({cascade_base_params / 1e6:.1f}M)")
    print(f"  ベースレイヤー数: {cascade_base_layers}")
    print(f"  Stage数: {CASCADE_NUM_STAGES}")
    print(f"  Stage毎のレイヤー数: {CASCADE_LAYERS_PER_STAGE}")
    print(f"  Hard比率: {CASCADE_HARD_RATIO * 100:.0f}%")

    # データロード（CASCADEのtokenizerを使用）
    train_loader, val_loader = load_alpaca_data(
        cascade_tokenizer, NUM_TRAIN_SAMPLES, NUM_VAL_SAMPLES, SEQ_LEN, BATCH_SIZE
    )

    # CASCADE訓練
    stage_models, thresholds, cascade_train_result = train_cascade(
        cascade_base_model, train_loader, val_loader, device,
        CASCADE_LAYERS_PER_STAGE, CASCADE_HARD_RATIO, CASCADE_NUM_STAGES,
        EPOCHS, LEARNING_RATE, PATIENCE
    )

    # CASCADE評価
    print("\n" + "=" * 60)
    print("CASCADE Ensemble評価")
    print("=" * 60)

    cascade_eval_result = evaluate_cascade(
        cascade_base_model, stage_models, thresholds, val_loader, device
    )

    total_cascade_layers = cascade_base_layers + CASCADE_NUM_STAGES * CASCADE_LAYERS_PER_STAGE

    cascade_exp_result = ExperimentResult(
        method="CASCADE",
        model_name=f"{CASCADE_BASE_MODEL} + {CASCADE_NUM_STAGES} stages",
        total_params=cascade_train_result["total_params"],
        num_layers=total_cascade_layers,
        train_tokens=NUM_TRAIN_SAMPLES * SEQ_LEN,
        val_tokens=NUM_VAL_SAMPLES * SEQ_LEN,
        epochs_trained=EPOCHS,
        training_time=cascade_train_result["training_time"],
        best_val_ppl=cascade_eval_result["val_ppl"],
        final_val_ppl=cascade_eval_result["val_ppl"],
        exit_distribution=cascade_eval_result["exit_distribution"],
        compute_ratio=cascade_eval_result["compute_ratio"],
    )

    print("\nCASCADE結果:")
    print(f"  val_ppl: {cascade_eval_result['val_ppl']:.2f}")
    print(f"  Exit分布: {cascade_eval_result['exit_distribution']}")
    print(f"  計算量比: {cascade_eval_result['compute_ratio']:.2%}")

    # ============================================================
    # 結果比較
    # ============================================================
    print("\n" + "=" * 80)
    print("結果比較")
    print("=" * 80)

    results = [baseline_exp_result, cascade_exp_result]

    print(f"\n{'Method':<15} {'Params':<12} {'Layers':<10} {'Val PPL':<12} {'Compute':<12} {'Time':<10}")
    print("-" * 75)

    for r in results:
        params_str = f"{r.total_params / 1e6:.1f}M"
        compute_str = f"{r.compute_ratio * 100:.1f}%" if r.compute_ratio else "100%"
        print(f"{r.method:<15} {params_str:<12} {r.num_layers:<10} "
              f"{r.best_val_ppl:<12.2f} {compute_str:<12} {r.training_time:.1f}s")

    # 結果を保存
    results_dict = [asdict(r) for r in results]
    with open(output_dir / "results.json", "w") as f:
        json.dump(results_dict, f, indent=2, ensure_ascii=False)

    # 設定を保存
    config = {
        "num_train_samples": NUM_TRAIN_SAMPLES,
        "num_val_samples": NUM_VAL_SAMPLES,
        "seq_len": SEQ_LEN,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "patience": PATIENCE,
        "seed": SEED,
        "cascade_base_model": CASCADE_BASE_MODEL,
        "cascade_layers_per_stage": CASCADE_LAYERS_PER_STAGE,
        "cascade_hard_ratio": CASCADE_HARD_RATIO,
        "cascade_num_stages": CASCADE_NUM_STAGES,
        "baseline_model": BASELINE_MODEL,
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n結果保存先: {output_dir}")

    # 勝者を判定
    winner = min(results, key=lambda r: r.best_val_ppl)
    print(f"\n勝者: {winner.method} (val_ppl={winner.best_val_ppl:.2f})")

    return results


if __name__ == "__main__":
    main()
