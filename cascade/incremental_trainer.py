"""
CASCADEフレームワーク - IncrementalTrainer

事前学習済みベースモデルに対して、段階的にLLMを追加・訓練するトレーナー。
各段階でhard tokensのみを後段に渡し、専門的なモデルを訓練。

使用例:
    from cascade import IncrementalTrainer, IncrementalConfig
    from cascade import load_pretrained, LLM

    # ベースモデルをロード
    base_model, tokenizer = load_pretrained("smollm2-135m")

    # 設定
    config = IncrementalConfig(
        layers_per_stage=8,
        hard_ratio=0.6,
        num_stages=5,
        epochs=10,
        batch_size=32,
        learning_rate=1e-4,
        patience=1,
    )

    # 訓練
    trainer = IncrementalTrainer(config, tokenizer, device="cuda")
    results = trainer.train(
        base_model,
        train_texts=["text1", "text2", ...],
        val_texts=["val1", "val2", ...],
        output_dir="outputs/cascade_exp",
    )
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, Union

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from transformers import PreTrainedModel, PreTrainedTokenizer

from .exit_fn import compute_cos_sim_from_history
from .model_registry import create_llm_from_base
from .utils import set_seed


@dataclass
class IncrementalConfig:
    """
    段階的訓練の設定。

    Attributes:
        layers_per_stage: 各段階で追加するレイヤー数
        hard_ratio: Hard token比率（0.0-1.0、cos_sim下位N%）
        num_stages: 段階数
        epochs: 各段階の最大エポック数
        batch_size: バッチサイズ
        learning_rate: 学習率
        patience: Early stoppingのpatience（1=最良に近いモデルを保存）
        seq_len: シーケンス長
        seed: 乱数シード
    """
    layers_per_stage: int = 8
    hard_ratio: float = 0.6
    num_stages: int = 5
    epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 1e-4
    patience: int = 1
    seq_len: int = 128
    seed: int = 42


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
    best_val_loss: float
    val_ppl: float
    training_time: float
    model_path: str
    total_params: int


class IncrementalTrainer:
    """
    段階的CASCADE訓練トレーナー。

    事前学習済みベースモデルに対して、段階的にLLMを追加・訓練。
    各段階でhard tokensのみを後段に渡す。
    """

    def __init__(
        self,
        config: IncrementalConfig,
        tokenizer: PreTrainedTokenizer,
        device: Optional[Union[str, torch.device]] = None,
        verbose: bool = True,
    ):
        """
        Args:
            config: 訓練設定
            tokenizer: トークナイザ（データ前処理用）
            device: 訓練デバイス（None=自動検出）
            verbose: 詳細出力
        """
        self.config = config
        self.tokenizer = tokenizer
        self.verbose = verbose

        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device) if isinstance(device, str) else device

        set_seed(config.seed)

    def _log(self, message: str):
        """メッセージを出力"""
        if self.verbose:
            print(message)

    def train(
        self,
        base_model: PreTrainedModel,
        train_texts: Optional[List[str]] = None,
        val_texts: Optional[List[str]] = None,
        train_loader: Optional[DataLoader] = None,
        val_loader: Optional[DataLoader] = None,
        output_dir: Optional[str] = None,
    ) -> List[StageResult]:
        """
        段階的訓練を実行。

        データは以下のいずれかで渡す:
        - train_texts, val_texts: 生テキストのリスト（トークナイズはtrainer内部で実行）
        - train_loader, val_loader: 事前に準備したDataLoader

        Args:
            base_model: ベースとなる事前学習済みモデル
            train_texts: 訓練テキストのリスト
            val_texts: 検証テキストのリスト
            train_loader: 訓練データローダー（token_idsとlabels）
            val_loader: 検証データローダー
            output_dir: 出力ディレクトリ（Noneで自動生成）

        Returns:
            各段階の結果リスト
        """
        # 出力ディレクトリの設定
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"outputs/cascade_{timestamp}"
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 設定を保存
        with open(output_path / "config.json", "w") as f:
            json.dump(asdict(self.config), f, indent=2, ensure_ascii=False)

        self._log("=" * 80)
        self._log("CASCADE段階的訓練")
        self._log("=" * 80)
        self._log("\n設定:")
        self._log(f"  段階あたりのレイヤー数: {self.config.layers_per_stage}")
        self._log(f"  Hard token比率: {self.config.hard_ratio * 100:.1f}%")
        self._log(f"  段階数: {self.config.num_stages}")
        self._log(f"  エポック数: {self.config.epochs}")
        self._log(f"  バッチサイズ: {self.config.batch_size}")
        self._log(f"  学習率: {self.config.learning_rate}")
        self._log(f"  Early stopping patience: {self.config.patience}")
        self._log(f"  デバイス: {self.device}")
        self._log("")

        # データローダーの準備
        if train_loader is None:
            if train_texts is None:
                raise ValueError("train_textsまたはtrain_loaderのどちらかを指定してください")
            train_loader, val_loader = self._create_dataloaders(train_texts, val_texts or [])

        # val_loaderが確実に存在することを保証
        if val_loader is None:
            raise ValueError("val_textsまたはval_loaderのどちらかを指定してください")

        # ベースモデル情報
        base_model.eval()
        base_model.to(self.device)  # type: ignore[arg-type]
        base_params = sum(p.numel() for p in base_model.parameters())
        base_layers = base_model.config.num_hidden_layers

        self._log("ベースモデル情報:")
        self._log(f"  パラメータ数: {base_params:,} ({base_params/1e6:.1f}M)")
        self._log(f"  レイヤー数: {base_layers}")

        # 初期hidden statesを抽出
        self._log("\nベースモデルからhidden statesを抽出中...")
        train_hidden, train_labels, train_cos_sim = self._extract_hidden_states(
            base_model, train_loader
        )
        val_hidden, val_labels, val_cos_sim = self._extract_hidden_states(
            base_model, val_loader
        )
        self._log(f"  訓練トークン数: {len(train_labels):,}")
        self._log(f"  検証トークン数: {len(val_labels):,}")

        # メモリ解放
        base_model.cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 段階的訓練
        results: List[StageResult] = []
        current_train = (train_hidden, train_labels, train_cos_sim)
        current_val = (val_hidden, val_labels, val_cos_sim)
        total_additional_layers = 0

        for stage in range(1, self.config.num_stages + 1):
            self._log(f"\n{'=' * 80}")
            self._log(f"段階 {stage}/{self.config.num_stages}")
            self._log(f"{'=' * 80}")

            # Hard tokensをフィルタリング
            hard_train, threshold = self._filter_hard_tokens(
                current_train[0], current_train[1], current_train[2]
            )
            hard_val, _ = self._filter_hard_tokens(
                current_val[0], current_val[1], current_val[2]
            )

            self._log("\nHard tokens:")
            self._log(f"  閾値: {threshold:.4f}")
            self._log(f"  訓練: {hard_train[0].shape[0]:,} / {current_train[0].shape[0]:,} "
                     f"({hard_train[0].shape[0]/current_train[0].shape[0]*100:.1f}%)")
            self._log(f"  検証: {hard_val[0].shape[0]:,}")

            # 新しいLLMを作成
            num_layers = self.config.layers_per_stage
            total_additional_layers += num_layers

            self._log(f"\n新しいLLM（{num_layers}層）を作成中...")
            stage_model = create_llm_from_base(base_model, num_layers)
            stage_params = sum(p.numel() for p in stage_model.parameters())
            self._log(f"  パラメータ数: {stage_params:,} ({stage_params/1e6:.1f}M)")

            # 訓練
            self._log("\n訓練開始...")
            start_time = time.time()

            train_result = self._train_single_stage(
                stage_model,
                hard_train[0], hard_train[1],
                hard_val[0], hard_val[1],
            )

            training_time = time.time() - start_time
            self._log(f"\n訓練完了: {training_time:.1f}秒")
            self._log(f"  最良val_loss: {train_result['best_val_loss']:.4f}")
            self._log(f"  最良val_ppl: {train_result['best_val_ppl']:.2f}")

            # モデル保存
            model_path = self._save_model(stage_model, output_path, stage, num_layers)
            self._log(f"  モデル保存先: {model_path}")

            # 結果を記録
            total_params = base_params + stage_params * stage
            result = StageResult(
                stage=stage,
                num_layers=num_layers,
                total_layers=base_layers + total_additional_layers,
                train_tokens=hard_train[0].shape[0],
                val_tokens=hard_val[0].shape[0],
                hard_ratio=self.config.hard_ratio,
                threshold=threshold,
                final_train_loss=train_result['final_train_loss'],
                final_val_loss=train_result['final_val_loss'],
                best_val_loss=train_result['best_val_loss'],
                val_ppl=train_result['best_val_ppl'],
                training_time=training_time,
                model_path=model_path,
                total_params=total_params,
            )
            results.append(result)

            # 次の段階用にhidden statesを更新
            self._log("\n次の段階用にhidden statesを更新中...")
            stage_model.eval()

            new_train = self._transform_hidden_states(
                stage_model, hard_train[0], hard_train[1]
            )
            new_val = self._transform_hidden_states(
                stage_model, hard_val[0], hard_val[1]
            )

            current_train = new_train
            current_val = new_val

            # メモリ解放
            stage_model.cpu()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # 結果を保存
        self._log(f"\n{'=' * 80}")
        self._log("実験完了")
        self._log(f"{'=' * 80}")

        results_dict = [asdict(r) for r in results]
        with open(output_path / "results.json", "w") as f:
            json.dump(results_dict, f, indent=2, ensure_ascii=False)

        self._print_summary(results)
        self._log(f"\n結果保存先: {output_path}")

        return results

    def _create_dataloaders(
        self,
        train_texts: List[str],
        val_texts: List[str],
    ) -> Tuple[DataLoader, DataLoader]:
        """テキストからDataLoaderを作成"""
        def tokenize_and_chunk(texts: List[str], max_samples: int) -> List[List[int]]:
            all_input_ids: List[List[int]] = []
            for text in texts:
                if len(all_input_ids) >= max_samples:
                    break
                tokens = self.tokenizer.encode(text, add_special_tokens=True)
                for j in range(0, len(tokens) - self.config.seq_len, self.config.seq_len):
                    if len(all_input_ids) >= max_samples:
                        break
                    chunk = tokens[j:j + self.config.seq_len + 1]
                    if len(chunk) == self.config.seq_len + 1:
                        all_input_ids.append(chunk)
            return all_input_ids

        train_data = torch.tensor(tokenize_and_chunk(train_texts, len(train_texts)), dtype=torch.long)
        val_data = torch.tensor(tokenize_and_chunk(val_texts, len(val_texts)), dtype=torch.long)

        train_x, train_y = train_data[:, :-1], train_data[:, 1:]
        val_x, val_y = val_data[:, :-1], val_data[:, 1:]

        train_loader = DataLoader(
            TensorDataset(train_x, train_y),
            batch_size=self.config.batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            TensorDataset(val_x, val_y),
            batch_size=self.config.batch_size,
            shuffle=False,
        )

        return train_loader, val_loader

    def _extract_hidden_states(
        self,
        model: PreTrainedModel,
        dataloader: DataLoader,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """モデルからhidden statesを抽出し、cos_simを計算"""
        model.eval()

        all_hidden: List[torch.Tensor] = []
        all_labels: List[torch.Tensor] = []
        all_cos_sim: List[torch.Tensor] = []

        with torch.no_grad():
            for x_batch, y_batch in dataloader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                outputs = model(
                    x_batch,
                    output_hidden_states=True,
                    return_dict=True,
                )

                hidden_history = outputs.hidden_states
                cos_sim = compute_cos_sim_from_history(hidden_history)

                h_out = hidden_history[-1]  # 最終hidden states

                batch_size, seq_len, dim = h_out.shape
                all_hidden.append(h_out.view(-1, dim).cpu())
                all_labels.append(y_batch.view(-1).cpu())
                all_cos_sim.append(cos_sim.view(-1).cpu())

        return (
            torch.cat(all_hidden, dim=0),
            torch.cat(all_labels, dim=0),
            torch.cat(all_cos_sim, dim=0),
        )

    def _transform_hidden_states(
        self,
        model: PreTrainedModel,
        hidden: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """hidden statesをモデルで変換し、新しいcos_simを計算"""
        model.eval()
        model.to(self.device)  # type: ignore[arg-type]

        all_hidden: List[torch.Tensor] = []
        all_cos_sim: List[torch.Tensor] = []

        with torch.no_grad():
            for i in range(0, hidden.shape[0], self.config.batch_size):
                h_batch = hidden[i:i + self.config.batch_size].to(self.device)
                # シーケンス長1として入力
                h_batch = h_batch.unsqueeze(1)

                outputs = model(
                    inputs_embeds=h_batch,
                    output_hidden_states=True,
                    return_dict=True,
                )

                hidden_history = outputs.hidden_states
                h_out = hidden_history[-1].squeeze(1)

                # cos_simをexit_fn.pyの共通関数で計算
                cos_sim = compute_cos_sim_from_history(hidden_history)
                cos_sim = cos_sim.squeeze(1)  # シーケンス長1を削除

                all_hidden.append(h_out.cpu())
                all_cos_sim.append(cos_sim.cpu())

        return (
            torch.cat(all_hidden, dim=0),
            labels,  # labelsは変更なし
            torch.cat(all_cos_sim, dim=0),
        )

    def _filter_hard_tokens(
        self,
        hidden: torch.Tensor,
        labels: torch.Tensor,
        cos_sim: torch.Tensor,
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], float]:
        """Hard tokens（cos_sim下位hard_ratio%）をフィルタリング"""
        # hard_ratio位置の閾値を計算
        threshold = float(torch.quantile(cos_sim.float(), self.config.hard_ratio).item())

        # 閾値以下がhard（cos_simが低い = 大きく変化した = 難しい）
        hard_mask = cos_sim <= threshold

        hard_hidden = hidden[hard_mask]
        hard_labels = labels[hard_mask]

        return (hard_hidden, hard_labels), threshold

    def _train_single_stage(
        self,
        model: PreTrainedModel,
        train_hidden: torch.Tensor,
        train_labels: torch.Tensor,
        val_hidden: torch.Tensor,
        val_labels: torch.Tensor,
    ) -> Dict[str, Any]:
        """1段階の訓練を実行"""
        model.train()
        model.to(self.device)  # type: ignore[arg-type]

        optimizer = torch.optim.AdamW(model.parameters(), lr=self.config.learning_rate)
        criterion = nn.CrossEntropyLoss()

        train_dataset = TensorDataset(train_hidden, train_labels)
        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)

        val_dataset = TensorDataset(val_hidden, val_labels)
        val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False)

        best_val_loss = float("inf")
        best_model_state = None
        patience_counter = 0

        final_train_loss = 0.0
        final_val_loss = 0.0

        for epoch in range(self.config.epochs):
            epoch_start = time.time()

            # 訓練
            model.train()
            train_loss_sum = 0.0
            train_count = 0

            for h_batch, y_batch in train_loader:
                h_batch = h_batch.to(self.device).float()
                y_batch = y_batch.to(self.device)
                h_batch = h_batch.unsqueeze(1)  # シーケンス長1

                optimizer.zero_grad()

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
                    h_batch = h_batch.to(self.device).float()
                    y_batch = y_batch.to(self.device)
                    h_batch = h_batch.unsqueeze(1)

                    outputs = model(inputs_embeds=h_batch, return_dict=True)
                    logits = outputs.logits.squeeze(1)

                    loss = criterion(logits, y_batch)
                    val_loss_sum += loss.item() * y_batch.size(0)
                    val_count += y_batch.size(0)

            val_loss = val_loss_sum / val_count
            val_ppl = float(np.exp(val_loss))
            epoch_time = time.time() - epoch_start

            self._log(f"    Epoch {epoch + 1}/{self.config.epochs}: "
                     f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
                     f"val_ppl={val_ppl:.2f}, time={epoch_time:.1f}s")

            final_train_loss = train_loss
            final_val_loss = val_loss

            # Best model保存とearly stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.config.patience:
                    self._log(f"    Early stopping at epoch {epoch + 1} (patience={self.config.patience})")
                    break

        # 最良モデルを復元
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        return {
            'final_train_loss': final_train_loss,
            'final_val_loss': final_val_loss,
            'best_val_loss': best_val_loss,
            'best_val_ppl': float(np.exp(best_val_loss)),
        }

    def _save_model(
        self,
        model: PreTrainedModel,
        output_dir: Path,
        stage: int,
        num_layers: int,
    ) -> str:
        """モデルを保存"""
        model_dir = output_dir / f"stage_{stage}_layers_{num_layers}"
        model_dir.mkdir(parents=True, exist_ok=True)

        torch.save(model.state_dict(), model_dir / "pytorch_model.bin")
        model.config.save_pretrained(model_dir)

        return str(model_dir)

    def _print_summary(self, results: List[StageResult]):
        """結果サマリーを出力"""
        self._log("\n【結果サマリー】")
        self._log(f"{'段階':<6} {'追加層':<8} {'合計層':<8} {'訓練tokens':<12} {'Val PPL':<12} {'合計パラメータ':<16}")
        self._log("-" * 70)
        for r in results:
            self._log(f"{r.stage:<6} {r.num_layers:<8} {r.total_layers:<8} "
                     f"{r.train_tokens:<12,} {r.val_ppl:<12.2f} {r.total_params/1e6:<16.1f}M")
