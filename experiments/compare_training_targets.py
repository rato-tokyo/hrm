"""
Hard vs All Tokens 訓練比較実験

仮説: Hard tokensだけで後段LLMは十分に学習できる

実験設計:
- 条件A: Hard tokensのみで訓練 → Hard tokens(val)で評価
- 条件B: All tokensで訓練 → Hard tokens(val)で評価

両者のval PPLを比較することで、Hard tokensに十分な情報が
含まれているかを検証する。

使用方法:
    python experiments/compare_training_targets.py --num-samples 1000 --epochs 10
"""

import argparse
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
import time
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from cascade import (
    set_seed,
    get_device,
    load_pretrained,
    create_alpaca_dataloaders,
)
from cascade.exit_fn import compute_cos_sim


@dataclass
class ExperimentResult:
    """実験結果を格納"""
    condition: str
    train_tokens: int
    val_tokens: int
    final_train_loss: float
    final_val_loss: float
    val_ppl: float
    training_time: float


def parse_args() -> argparse.Namespace:
    """コマンドライン引数をパース"""
    parser = argparse.ArgumentParser(
        description="Hard vs All Tokens 訓練比較実験"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="smollm2-135m",
        help="ベースモデル名 (default: smollm2-135m)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.934,
        help="Hard token判定の閾値 (default: 0.934)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1000,
        help="訓練サンプル数 (default: 1000)",
    )
    parser.add_argument(
        "--val-samples",
        type=int,
        default=200,
        help="検証サンプル数 (default: 200)",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=128,
        help="シーケンス長 (default: 128)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="バッチサイズ (default: 32)",
    )
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
        "--seed",
        type=int,
        default=42,
        help="乱数シード (default: 42)",
    )
    return parser.parse_args()


def collect_hidden_states(
    base_llm: nn.Module,
    batches: List[Tuple[torch.Tensor, torch.Tensor]],
    threshold: float,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    ベースLLMを通してhidden statesとhard/all tokensを収集

    Returns:
        all_hidden: 全トークンのhidden states
        all_labels: 全トークンのラベル
        hard_hidden: Hard tokensのhidden states
        hard_labels: Hard tokensのラベル
        hard_mask: Hard tokenのマスク（統計用）
        cos_sims: 各トークンのcos_sim値
    """
    all_hidden_list = []
    all_labels_list = []
    hard_hidden_list = []
    hard_labels_list = []
    cos_sim_list = []

    base_llm.eval()

    with torch.no_grad():
        for x_batch, y_batch in batches:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            # Forward pass
            outputs = base_llm(
                input_ids=x_batch,
                output_hidden_states=True,
                return_dict=True,
            )

            hidden_states = outputs.hidden_states

            # cos_simを計算（最後の2層のhidden statesを使用）
            if len(hidden_states) >= 3:
                h_prev = hidden_states[-3]
                h_last = hidden_states[-2]
            else:
                h_prev = hidden_states[0]
                h_last = hidden_states[-2] if len(hidden_states) > 1 else hidden_states[-1]

            cos_sim = compute_cos_sim(h_prev, h_last)  # (batch, seq_len)

            # 最終層のhidden states
            h_out = hidden_states[-1]  # (batch, seq_len, dim)

            # Hard tokenマスク
            hard_mask = cos_sim < threshold  # (batch, seq_len)

            # 全トークン収集
            batch_size, seq_len, dim = h_out.shape
            all_hidden_list.append(h_out.view(-1, dim).cpu())
            all_labels_list.append(y_batch.view(-1).cpu())

            # Hard tokens収集
            for i in range(batch_size):
                mask = hard_mask[i]  # (seq_len,)
                if mask.any():
                    hard_h = h_out[i, mask]  # (num_hard, dim)
                    hard_y = y_batch[i, mask]  # (num_hard,)
                    hard_hidden_list.append(hard_h.cpu())
                    hard_labels_list.append(hard_y.cpu())

            cos_sim_list.append(cos_sim.view(-1).cpu())

    # 結合
    all_hidden = torch.cat(all_hidden_list, dim=0).float()  # float32に変換
    all_labels = torch.cat(all_labels_list, dim=0)

    if hard_hidden_list:
        hard_hidden = torch.cat(hard_hidden_list, dim=0).float()  # float32に変換
        hard_labels = torch.cat(hard_labels_list, dim=0)
    else:
        hard_hidden = torch.empty(0, all_hidden.shape[1])
        hard_labels = torch.empty(0, dtype=torch.long)

    cos_sims = torch.cat(cos_sim_list, dim=0)
    hard_mask = cos_sims < threshold

    return all_hidden, all_labels, hard_hidden, hard_labels, hard_mask, cos_sims


class SimpleHead(nn.Module):
    """
    Hidden states → Logits の簡易ヘッド
    ベースLLMのlm_headと同じ構造
    """
    def __init__(self, hidden_dim: int, vocab_size: int):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, vocab_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.linear(hidden_states)


def train_head(
    head: nn.Module,
    train_hidden: torch.Tensor,
    train_labels: torch.Tensor,
    val_hidden: torch.Tensor,
    val_labels: torch.Tensor,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
) -> Tuple[float, float, float]:
    """
    ヘッドを訓練してval lossとPPLを返す

    Returns:
        final_train_loss: 最終訓練loss
        final_val_loss: 最終検証loss
        val_ppl: 検証PPL
    """
    head = head.to(device)
    head.train()

    # DataLoader作成
    train_dataset = TensorDataset(train_hidden, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = TensorDataset(val_hidden, val_labels)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(head.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float('inf')

    for epoch in range(epochs):
        # Training
        head.train()
        train_loss = 0.0
        train_count = 0

        for hidden, labels in train_loader:
            hidden = hidden.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = head(hidden)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * hidden.shape[0]
            train_count += hidden.shape[0]

        avg_train_loss = train_loss / train_count

        # Validation
        head.eval()
        val_loss = 0.0
        val_count = 0

        with torch.no_grad():
            for hidden, labels in val_loader:
                hidden = hidden.to(device)
                labels = labels.to(device)

                logits = head(hidden)
                loss = criterion(logits, labels)

                val_loss += loss.item() * hidden.shape[0]
                val_count += hidden.shape[0]

        avg_val_loss = val_loss / val_count
        val_ppl = torch.exp(torch.tensor(avg_val_loss)).item()

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss

        print(f"  Epoch {epoch+1}/{epochs}: "
              f"train_loss={avg_train_loss:.4f}, "
              f"val_loss={avg_val_loss:.4f}, "
              f"val_ppl={val_ppl:.2f}")

    final_val_ppl = torch.exp(torch.tensor(best_val_loss)).item()
    return avg_train_loss, best_val_loss, final_val_ppl


def main() -> None:
    """メイン実行関数"""
    args = parse_args()

    device = get_device()
    print("=" * 70)
    print("Hard vs All Tokens 訓練比較実験")
    print("=" * 70)
    print(f"デバイス: {device}")
    print(f"ベースモデル: {args.base_model}")
    print(f"閾値: {args.threshold}")
    print(f"訓練サンプル数: {args.num_samples}")
    print(f"検証サンプル数: {args.val_samples}")
    print(f"エポック数: {args.epochs}")
    print(f"学習率: {args.lr}")

    set_seed(args.seed)

    # モデルロード
    print(f"\nモデルをロード中: {args.base_model}")
    is_cuda = device.type == "cuda"
    base_llm, tokenizer = load_pretrained(
        args.base_model,
        device="auto" if is_cuda else None,
        torch_dtype=torch.float16 if is_cuda else None,
    )
    base_llm.eval()

    hidden_dim = base_llm.config.hidden_size
    vocab_size = base_llm.config.vocab_size
    print(f"Hidden dim: {hidden_dim}, Vocab size: {vocab_size}")

    # データロード
    print(f"\nデータをロード中...")
    total_samples = args.num_samples + args.val_samples
    train_batches, val_batches, _ = create_alpaca_dataloaders(
        num_samples=total_samples,
        batch_size=1,  # 1サンプルずつ処理
        seq_len=args.seq_len,
        seed=args.seed,
        tokenizer_name=tokenizer.name_or_path,
        val_ratio=args.val_samples / total_samples,
    )

    print(f"訓練バッチ数: {len(train_batches)}")
    print(f"検証バッチ数: {len(val_batches)}")

    # Hidden states収集（訓練データ）
    print(f"\n訓練データからhidden statesを収集中...")
    (
        train_all_hidden, train_all_labels,
        train_hard_hidden, train_hard_labels,
        train_hard_mask, train_cos_sims
    ) = collect_hidden_states(base_llm, train_batches, args.threshold, device)

    train_hard_ratio = train_hard_mask.float().mean().item()
    print(f"  全トークン数: {len(train_all_labels)}")
    print(f"  Hardトークン数: {len(train_hard_labels)} ({train_hard_ratio*100:.1f}%)")

    # Hidden states収集（検証データ）
    print(f"\n検証データからhidden statesを収集中...")
    (
        val_all_hidden, val_all_labels,
        val_hard_hidden, val_hard_labels,
        val_hard_mask, val_cos_sims
    ) = collect_hidden_states(base_llm, val_batches, args.threshold, device)

    val_hard_ratio = val_hard_mask.float().mean().item()
    print(f"  全トークン数: {len(val_all_labels)}")
    print(f"  Hardトークン数: {len(val_hard_labels)} ({val_hard_ratio*100:.1f}%)")

    if len(val_hard_labels) == 0:
        print("\nエラー: 検証データにHard tokensがありません。閾値を調整してください。")
        return

    results: List[ExperimentResult] = []

    # ========================================
    # 条件A: Hard tokensのみで訓練
    # ========================================
    print("\n" + "=" * 70)
    print("条件A: Hard tokensのみで訓練")
    print("=" * 70)
    print(f"訓練トークン数: {len(train_hard_labels)}")
    print(f"検証トークン数: {len(val_hard_labels)} (Hard tokens)")

    head_a = SimpleHead(hidden_dim, vocab_size)
    start_time = time.time()

    train_loss_a, val_loss_a, val_ppl_a = train_head(
        head_a,
        train_hard_hidden, train_hard_labels,
        val_hard_hidden, val_hard_labels,
        args.epochs, args.batch_size, args.lr, device
    )

    training_time_a = time.time() - start_time

    results.append(ExperimentResult(
        condition="Hard tokens",
        train_tokens=len(train_hard_labels),
        val_tokens=len(val_hard_labels),
        final_train_loss=train_loss_a,
        final_val_loss=val_loss_a,
        val_ppl=val_ppl_a,
        training_time=training_time_a,
    ))

    # ========================================
    # 条件B: All tokensで訓練
    # ========================================
    print("\n" + "=" * 70)
    print("条件B: All tokensで訓練")
    print("=" * 70)
    print(f"訓練トークン数: {len(train_all_labels)}")
    print(f"検証トークン数: {len(val_hard_labels)} (Hard tokens)")

    head_b = SimpleHead(hidden_dim, vocab_size)
    start_time = time.time()

    train_loss_b, val_loss_b, val_ppl_b = train_head(
        head_b,
        train_all_hidden, train_all_labels,
        val_hard_hidden, val_hard_labels,
        args.epochs, args.batch_size, args.lr, device
    )

    training_time_b = time.time() - start_time

    results.append(ExperimentResult(
        condition="All tokens",
        train_tokens=len(train_all_labels),
        val_tokens=len(val_hard_labels),
        final_train_loss=train_loss_b,
        final_val_loss=val_loss_b,
        val_ppl=val_ppl_b,
        training_time=training_time_b,
    ))

    # ========================================
    # 結果サマリー
    # ========================================
    print("\n" + "=" * 70)
    print("実験結果サマリー")
    print("=" * 70)
    print(f"\n{'条件':<15} {'訓練トークン':>12} {'Val Loss':>10} {'Val PPL':>10} {'時間(s)':>10}")
    print("-" * 60)

    for r in results:
        print(f"{r.condition:<15} {r.train_tokens:>12,} {r.final_val_loss:>10.4f} "
              f"{r.val_ppl:>10.2f} {r.training_time:>10.1f}")

    # 比較分析
    print("\n" + "-" * 60)
    ppl_diff = val_ppl_a - val_ppl_b
    ppl_ratio = val_ppl_a / val_ppl_b if val_ppl_b > 0 else float('inf')

    print(f"\nPPL差 (Hard - All): {ppl_diff:+.2f}")
    print(f"PPL比 (Hard / All): {ppl_ratio:.3f}")

    if abs(ppl_diff) < 1.0:
        print("\n【結論】Hard tokensのみで訓練しても、ほぼ同等の性能")
        print("→ 仮説支持: Hard tokensに十分な情報が含まれている")
    elif ppl_diff > 0:
        print(f"\n【結論】All tokensで訓練した方がPPLが{abs(ppl_diff):.2f}低い")
        print("→ Easy tokensにも後段LLMに有用な情報が含まれる可能性")
    else:
        print(f"\n【結論】Hard tokensのみで訓練した方がPPLが{abs(ppl_diff):.2f}低い")
        print("→ Easy tokensはノイズであり、Hardのみの方が効率的")

    # 効率性の比較
    train_ratio = len(train_hard_labels) / len(train_all_labels)
    print(f"\n訓練データ削減率: {(1-train_ratio)*100:.1f}%")
    print(f"（Hard: {len(train_hard_labels):,} / All: {len(train_all_labels):,}）")


if __name__ == "__main__":
    main()
