"""
CASCADE段階的訓練実験

SmolLM2-135Mをベースに、後続LLMを段階的に追加・訓練する実験。
各段階で:
1. 新しいLLM（x層）を追加
2. Hard tokens（上位y%）のみで訓練
3. 訓練済みモデルを保存
4. 次の段階へ

Colabでの実行方法:
    !python experiments/cascade_incremental_training.py
"""

import json
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaConfig

# ============================================================
# 実験パラメータ（ここを編集して実験設定を変更）
# ============================================================

# ベースモデル
BASE_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"

# 段階的追加の設定
LAYERS_PER_STAGE = 8      # 各段階で追加するレイヤー数
HARD_RATIO = 0.6          # Hard token比率（0.6 = cos_sim下位60%）
NUM_STAGES = 5            # 段階数（5回繰り返し）

# 訓練設定
EPOCHS = 10               # 各段階の最大エポック数
BATCH_SIZE = 32           # バッチサイズ
LEARNING_RATE = 1e-4      # 学習率
SEQ_LEN = 128             # シーケンス長
PATIENCE = 1              # Early stoppingのpatience（1=最良に近いモデルを保存）

# データ設定
NUM_TRAIN_SAMPLES = 1000  # 訓練サンプル数
NUM_VAL_SAMPLES = 100     # 検証サンプル数（精度測定の最小限）

# その他
SEED = 42                 # 乱数シード
OUTPUT_DIR = None         # 出力ディレクトリ（Noneで自動生成）

# ============================================================
# 以下は実装コード（通常は編集不要）
# ============================================================


@dataclass
class StageResult:
    """各段階の結果"""
    stage: int
    num_layers: int
    total_layers: int
    train_tokens: int
    val_tokens: int
    hard_ratio: float
    threshold: float
    final_train_loss: float
    final_val_loss: float
    val_ppl: float
    training_time: float
    model_path: str
    total_params: int


@dataclass
class ExperimentConfig:
    """実験設定"""
    base_model: str
    layers_per_stage: int
    hard_ratio: float
    num_stages: int
    epochs: int
    batch_size: int
    learning_rate: float
    seq_len: int
    patience: int
    num_train_samples: int
    num_val_samples: int
    seed: int
    output_dir: str


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


def create_dataloaders(
    tokenizer,
    num_train: int,
    num_val: int,
    seq_len: int,
    batch_size: int,
) -> Tuple[DataLoader, DataLoader]:
    """Alpacaデータセットからデータローダーを作成"""
    from datasets import load_dataset

    dataset = load_dataset("tatsu-lab/alpaca", split="train")

    def tokenize_and_chunk(max_samples: int) -> List[List[int]]:
        all_input_ids: List[List[int]] = []

        for i in range(len(dataset)):
            if len(all_input_ids) >= max_samples:
                break

            full_text = f"{dataset[i]['instruction']}\n{dataset[i]['input']}\n{dataset[i]['output']}"
            tokens = tokenizer.encode(full_text, add_special_tokens=True)

            for j in range(0, len(tokens) - seq_len, seq_len):
                if len(all_input_ids) >= max_samples:
                    break
                chunk = tokens[j:j + seq_len + 1]
                if len(chunk) == seq_len + 1:
                    all_input_ids.append(chunk)

        return all_input_ids

    print("  Alpacaデータセットをトークナイズ中...")
    all_data = tokenize_and_chunk(num_train + num_val)
    print(f"  取得サンプル数: {len(all_data)}")

    train_data = torch.tensor(all_data[:num_train], dtype=torch.long)
    val_data = torch.tensor(all_data[num_train:num_train + num_val], dtype=torch.long)

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

    return train_loader, val_loader


def compute_cos_sim(h_in: torch.Tensor, h_out: torch.Tensor) -> torch.Tensor:
    """コサイン類似度を計算"""
    h_in_norm = h_in / (h_in.norm(dim=-1, keepdim=True) + 1e-8)
    h_out_norm = h_out / (h_out.norm(dim=-1, keepdim=True) + 1e-8)
    return (h_in_norm * h_out_norm).sum(dim=-1)


def extract_hidden_states(
    model,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """モデルからhidden statesを抽出し、cos_simを計算"""
    model.eval()

    all_hidden_list = []
    all_labels_list = []
    all_cos_sim_list = []

    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            outputs = model(
                x_batch,
                output_hidden_states=True,
                return_dict=True,
            )

            h_in = outputs.hidden_states[0]
            h_out = outputs.hidden_states[-1]

            cos_sim = compute_cos_sim(h_in, h_out)

            batch_size, seq_len, dim = h_out.shape
            all_hidden_list.append(h_out.view(-1, dim).cpu())
            all_labels_list.append(y_batch.view(-1).cpu())
            all_cos_sim_list.append(cos_sim.view(-1).cpu())

    all_hidden = torch.cat(all_hidden_list, dim=0)
    all_labels = torch.cat(all_labels_list, dim=0)
    all_cos_sim = torch.cat(all_cos_sim_list, dim=0)

    return all_hidden, all_labels, all_cos_sim


def extract_hidden_from_hidden(
    model,
    input_hidden: torch.Tensor,
    input_labels: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """hidden statesを入力としてモデルを通し、新しいhidden statesを抽出"""
    model.eval()

    all_hidden_list = []
    all_cos_sim_list = []

    num_tokens = input_hidden.shape[0]

    with torch.no_grad():
        for i in range(0, num_tokens, batch_size):
            h_batch = input_hidden[i:i + batch_size].to(device)
            h_batch = h_batch.unsqueeze(1)

            outputs = model(
                inputs_embeds=h_batch,
                output_hidden_states=True,
                return_dict=True,
            )

            h_in = h_batch.squeeze(1)
            h_out = outputs.hidden_states[-1].squeeze(1)

            cos_sim = compute_cos_sim(h_in, h_out)

            all_hidden_list.append(h_out.cpu())
            all_cos_sim_list.append(cos_sim.cpu())

    all_hidden = torch.cat(all_hidden_list, dim=0)
    all_cos_sim = torch.cat(all_cos_sim_list, dim=0)

    return all_hidden, input_labels, all_cos_sim


def filter_hard_tokens(
    hidden: torch.Tensor,
    labels: torch.Tensor,
    cos_sim: torch.Tensor,
    hard_ratio: float,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """Hard tokens（cos_sim下位hard_ratio%）をフィルタリング"""
    threshold = torch.quantile(cos_sim, hard_ratio).item()
    hard_mask = cos_sim <= threshold

    hard_hidden = hidden[hard_mask]
    hard_labels = labels[hard_mask]

    return hard_hidden, hard_labels, threshold


def train_stage(
    model,
    train_hidden: torch.Tensor,
    train_labels: torch.Tensor,
    val_hidden: torch.Tensor,
    val_labels: torch.Tensor,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    patience: int,
    device: torch.device,
) -> Tuple[float, float, float]:
    """1段階の訓練を実行"""
    model.train()
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    train_dataset = TensorDataset(train_hidden, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = TensorDataset(val_hidden, val_labels)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        epoch_start = time.time()

        model.train()
        train_loss_sum = 0.0
        train_count = 0

        for h_batch, y_batch in train_loader:
            h_batch = h_batch.to(device).float()
            y_batch = y_batch.to(device)
            h_batch = h_batch.unsqueeze(1)

            optimizer.zero_grad()

            outputs = model(inputs_embeds=h_batch, return_dict=True)
            logits = outputs.logits.squeeze(1)

            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * y_batch.size(0)
            train_count += y_batch.size(0)

        train_loss = train_loss_sum / train_count

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
        val_ppl = torch.exp(torch.tensor(val_loss)).item()
        epoch_time = time.time() - epoch_start

        print(f"    Epoch {epoch + 1}/{epochs}: "
              f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
              f"val_ppl={val_ppl:.2f}, time={epoch_time:.1f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"    Early stopping at epoch {epoch + 1} (patience={patience})")
                break

    return train_loss, val_loss, val_ppl


def save_model(model, output_dir: Path, stage: int, num_layers: int):
    """モデルを保存"""
    model_dir = output_dir / f"stage_{stage}_layers_{num_layers}"
    model_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), model_dir / "pytorch_model.bin")
    model.config.save_pretrained(model_dir)

    return str(model_dir)


def run_experiment():
    """実験を実行"""
    # 出力ディレクトリを設定
    output_dir_str = OUTPUT_DIR
    if output_dir_str is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir_str = f"outputs/cascade_{timestamp}"

    config = ExperimentConfig(
        base_model=BASE_MODEL,
        layers_per_stage=LAYERS_PER_STAGE,
        hard_ratio=HARD_RATIO,
        num_stages=NUM_STAGES,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        seq_len=SEQ_LEN,
        patience=PATIENCE,
        num_train_samples=NUM_TRAIN_SAMPLES,
        num_val_samples=NUM_VAL_SAMPLES,
        seed=SEED,
        output_dir=output_dir_str,
    )

    print("=" * 80)
    print("CASCADE段階的訓練実験")
    print("=" * 80)
    print("\n設定:")
    print(f"  ベースモデル: {config.base_model}")
    print(f"  段階あたりのレイヤー数: {config.layers_per_stage}")
    print(f"  Hard token比率: {config.hard_ratio * 100:.1f}%")
    print(f"  段階数: {config.num_stages}")
    print(f"  エポック数: {config.epochs}")
    print(f"  バッチサイズ: {config.batch_size}")
    print(f"  学習率: {config.learning_rate}")
    print(f"  Early stopping patience: {config.patience}")
    print()

    device = get_device()
    print(f"デバイス: {device}")
    set_seed(config.seed)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "config.json", "w") as f:
        json.dump(asdict(config), f, indent=2, ensure_ascii=False)

    print(f"\nベースモデルをロード中: {config.base_model}")
    base_model = AutoModelForCausalLM.from_pretrained(config.base_model)
    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    base_model.eval()
    base_model.to(device)

    base_params = sum(p.numel() for p in base_model.parameters())
    base_layers = base_model.config.num_hidden_layers
    print(f"  パラメータ数: {base_params:,} ({base_params/1e6:.1f}M)")
    print(f"  レイヤー数: {base_layers}")

    print("\nデータをロード中...")
    train_loader, val_loader = create_dataloaders(
        tokenizer,
        config.num_train_samples,
        config.num_val_samples,
        config.seq_len,
        config.batch_size,
    )
    print(f"  訓練サンプル数: {len(train_loader.dataset)}")
    print(f"  検証サンプル数: {len(val_loader.dataset)}")

    print("\nベースモデルからhidden statesを抽出中...")
    train_hidden, train_labels, train_cos_sim = extract_hidden_states(
        base_model, train_loader, device
    )
    val_hidden, val_labels, val_cos_sim = extract_hidden_states(
        base_model, val_loader, device
    )
    print(f"  訓練トークン数: {len(train_labels):,}")
    print(f"  検証トークン数: {len(val_labels):,}")

    base_model.cpu()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    results: List[StageResult] = []
    current_train_hidden = train_hidden
    current_train_labels = train_labels
    current_train_cos_sim = train_cos_sim
    current_val_hidden = val_hidden
    current_val_labels = val_labels
    current_val_cos_sim = val_cos_sim

    total_additional_layers = 0

    for stage in range(1, config.num_stages + 1):
        print(f"\n{'=' * 80}")
        print(f"段階 {stage}/{config.num_stages}")
        print(f"{'=' * 80}")

        print("\nHard tokensをフィルタリング中...")
        hard_train_hidden, hard_train_labels, threshold = filter_hard_tokens(
            current_train_hidden,
            current_train_labels,
            current_train_cos_sim,
            config.hard_ratio,
        )
        hard_val_hidden, hard_val_labels, _ = filter_hard_tokens(
            current_val_hidden,
            current_val_labels,
            current_val_cos_sim,
            config.hard_ratio,
        )

        print(f"  閾値: {threshold:.4f}")
        print(f"  訓練Hard tokens: {len(hard_train_labels):,} / {len(current_train_labels):,} "
              f"({len(hard_train_labels)/len(current_train_labels)*100:.1f}%)")
        print(f"  検証Hard tokens: {len(hard_val_labels):,}")

        num_layers = config.layers_per_stage
        total_additional_layers += num_layers
        print(f"\n新しいLLM（{num_layers}層）を作成中...")

        stage_model = create_llm_from_base(base_model, num_layers)
        stage_params = sum(p.numel() for p in stage_model.parameters())
        print(f"  パラメータ数: {stage_params:,} ({stage_params/1e6:.1f}M)")

        print("\n訓練開始...")
        start_time = time.time()

        train_loss, val_loss, val_ppl = train_stage(
            stage_model,
            hard_train_hidden,
            hard_train_labels,
            hard_val_hidden,
            hard_val_labels,
            config.epochs,
            config.batch_size,
            config.learning_rate,
            config.patience,
            device,
        )

        training_time = time.time() - start_time
        print(f"\n訓練完了: {training_time:.1f}秒")
        print(f"  最終val_loss: {val_loss:.4f}")
        print(f"  最終val_ppl: {val_ppl:.2f}")

        model_path = save_model(stage_model, output_dir, stage, num_layers)
        print(f"  モデル保存先: {model_path}")

        total_params = base_params + stage_params * stage

        result = StageResult(
            stage=stage,
            num_layers=num_layers,
            total_layers=base_layers + total_additional_layers,
            train_tokens=len(hard_train_labels),
            val_tokens=len(hard_val_labels),
            hard_ratio=config.hard_ratio,
            threshold=threshold,
            final_train_loss=train_loss,
            final_val_loss=val_loss,
            val_ppl=val_ppl,
            training_time=training_time,
            model_path=model_path,
            total_params=total_params,
        )
        results.append(result)

        print("\n次の段階用にhidden statesを更新中...")
        stage_model.eval()

        new_train_hidden, new_train_labels, new_train_cos_sim = extract_hidden_from_hidden(
            stage_model,
            hard_train_hidden,
            hard_train_labels,
            config.batch_size,
            device,
        )
        new_val_hidden, new_val_labels, new_val_cos_sim = extract_hidden_from_hidden(
            stage_model,
            hard_val_hidden,
            hard_val_labels,
            config.batch_size,
            device,
        )

        current_train_hidden = new_train_hidden
        current_train_labels = new_train_labels
        current_train_cos_sim = new_train_cos_sim
        current_val_hidden = new_val_hidden
        current_val_labels = new_val_labels
        current_val_cos_sim = new_val_cos_sim

        stage_model.cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\n{'=' * 80}")
    print("実験完了")
    print(f"{'=' * 80}")

    results_dict = [asdict(r) for r in results]
    with open(output_dir / "results.json", "w") as f:
        json.dump(results_dict, f, indent=2, ensure_ascii=False)

    print("\n【結果サマリー】")
    print(f"{'段階':<6} {'追加層':<8} {'合計層':<8} {'訓練tokens':<12} {'Val PPL':<12} {'合計パラメータ':<16}")
    print("-" * 70)
    for r in results:
        print(f"{r.stage:<6} {r.num_layers:<8} {r.total_layers:<8} "
              f"{r.train_tokens:<12,} {r.val_ppl:<12.2f} {r.total_params/1e6:<16.1f}M")

    print(f"\n結果保存先: {output_dir}")

    return results


if __name__ == "__main__":
    run_experiment()
