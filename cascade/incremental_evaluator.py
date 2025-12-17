"""
CASCADEフレームワーク - IncrementalEvaluator

段階的訓練されたモデルのEnsemble評価。
各Early Exitパターンでのval_ppl、Exit率、計算量を測定。

使用例:
    from cascade import IncrementalEvaluator
    from cascade import load_pretrained

    # ベースモデルをロード
    base_model, tokenizer = load_pretrained("smollm2-135m")

    # 評価
    evaluator = IncrementalEvaluator(tokenizer, device="cuda")
    results = evaluator.evaluate(
        base_model,
        stage_model_paths=["outputs/stage_1", "outputs/stage_2", ...],
        val_texts=["text1", "text2", ...],
    )

    # 結果を表示
    evaluator.print_results(results)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Optional, Union

import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from transformers import PreTrainedModel, PreTrainedTokenizer, AutoModelForCausalLM

from .exit_fn import compute_cos_sim_from_history


@dataclass
class EvaluationResult:
    """評価結果"""
    pattern_name: str
    num_stages: int
    total_params: int
    val_ppl: float
    accuracy: float
    total_tokens: int
    exit_distribution: Dict[str, float]  # {"base": 0.4, "stage_1": 0.3, ...}
    layers_computed: int
    max_layers_possible: int
    compute_ratio: float  # 実際の計算量 / 最大計算量


@dataclass
class StageModel:
    """段階モデル情報"""
    model: PreTrainedModel
    threshold: float
    num_layers: int


class IncrementalEvaluator:
    """
    段階的CASCADE訓練モデルの評価。

    各Early Exitパターン（Base only, Base+Stage1, ...）での
    性能を測定し、比較可能な形式で結果を出力。
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        device: Optional[Union[str, torch.device]] = None,
        seq_len: int = 128,
        batch_size: int = 32,
        verbose: bool = True,
    ):
        """
        Args:
            tokenizer: トークナイザ
            device: 評価デバイス
            seq_len: シーケンス長
            batch_size: バッチサイズ
            verbose: 詳細出力
        """
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.batch_size = batch_size
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

    def _log(self, message: str):
        if self.verbose:
            print(message)

    def evaluate(
        self,
        base_model: PreTrainedModel,
        stage_model_paths: List[str],
        thresholds: List[float],
        val_texts: Optional[List[str]] = None,
        val_loader: Optional[DataLoader] = None,
        output_path: Optional[str] = None,
    ) -> List[EvaluationResult]:
        """
        各Early Exitパターンで評価。

        Args:
            base_model: ベースモデル
            stage_model_paths: 各段階モデルのパス
            thresholds: 各段階の閾値
            val_texts: 検証テキスト
            val_loader: 検証データローダー
            output_path: 結果保存先

        Returns:
            各パターンの評価結果
        """
        if val_loader is None:
            if val_texts is None:
                raise ValueError("val_textsまたはval_loaderを指定してください")
            val_loader = self._create_dataloader(val_texts)

        # 段階モデルをロード
        stage_models = self._load_stage_models(stage_model_paths, thresholds)

        # 各パターンで評価
        results: List[EvaluationResult] = []

        # Base only
        self._log("\n評価: Base only")
        result = self._evaluate_pattern(base_model, [], val_loader, "Base only")
        results.append(result)
        self._log(f"  val_ppl={result.val_ppl:.2f}, accuracy={result.accuracy:.2%}")

        # Base + Stage 1, 2, ...
        for i in range(len(stage_models)):
            pattern_name = f"Base + Stage1-{i+1}" if i > 0 else "Base + Stage1"
            self._log(f"\n評価: {pattern_name}")

            result = self._evaluate_pattern(
                base_model,
                stage_models[:i+1],
                val_loader,
                pattern_name,
            )
            results.append(result)
            self._log(f"  val_ppl={result.val_ppl:.2f}, accuracy={result.accuracy:.2%}")
            self._log(f"  Exit分布: {result.exit_distribution}")

        # 結果を保存
        if output_path:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, "w") as f:
                json.dump([asdict(r) for r in results], f, indent=2, ensure_ascii=False)
            self._log(f"\n結果保存先: {output_file}")

        return results

    def _load_stage_models(
        self,
        model_paths: List[str],
        thresholds: List[float],
    ) -> List[StageModel]:
        """段階モデルをロード"""
        stage_models: List[StageModel] = []

        for i, (path, threshold) in enumerate(zip(model_paths, thresholds)):
            self._log(f"Stage {i+1} モデルをロード: {path}")

            # configからモデルを作成し、重みをロード
            model = AutoModelForCausalLM.from_pretrained(path)
            model.eval()
            model.to(self.device)  # type: ignore[arg-type]

            num_layers = model.config.num_hidden_layers

            stage_models.append(StageModel(
                model=model,
                threshold=threshold,
                num_layers=num_layers,
            ))

        return stage_models

    def _create_dataloader(self, texts: List[str]) -> DataLoader:
        """テキストからDataLoaderを作成"""
        all_input_ids: List[List[int]] = []

        for text in texts:
            tokens = self.tokenizer.encode(text, add_special_tokens=True)
            for j in range(0, len(tokens) - self.seq_len, self.seq_len):
                chunk = tokens[j:j + self.seq_len + 1]
                if len(chunk) == self.seq_len + 1:
                    all_input_ids.append(chunk)

        data = torch.tensor(all_input_ids, dtype=torch.long)
        x, y = data[:, :-1], data[:, 1:]

        return DataLoader(
            TensorDataset(x, y),
            batch_size=self.batch_size,
            shuffle=False,
        )

    def _evaluate_pattern(
        self,
        base_model: PreTrainedModel,
        stage_models: List[StageModel],
        val_loader: DataLoader,
        pattern_name: str,
    ) -> EvaluationResult:
        """特定のパターンで評価"""
        base_model.eval()
        base_model.to(self.device)  # type: ignore[arg-type]

        total_loss = 0.0
        total_correct = 0
        total_tokens = 0

        # Exit分布のカウント
        exit_counts = {"base": 0}
        for i in range(len(stage_models)):
            exit_counts[f"stage_{i+1}"] = 0

        total_layers_computed = 0
        base_layers = base_model.config.num_hidden_layers

        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                batch_size, seq_len = x_batch.shape

                # ベースモデルを通す
                outputs = base_model(
                    x_batch,
                    output_hidden_states=True,
                    return_dict=True,
                )

                hidden_history = outputs.hidden_states
                h_out = hidden_history[-1]
                logits = outputs.logits

                # Exit判定
                if len(stage_models) == 0:
                    # Base onlyの場合、全トークンがここでexit
                    exit_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=self.device)
                else:
                    # cos_simでexit判定（exit_fn.pyの共通関数を使用）
                    cos_sim = compute_cos_sim_from_history(hidden_history)
                    exit_mask = cos_sim > stage_models[0].threshold

                # ベースでexitしたトークンのloss計算
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

                # 継続トークンを後段に渡す
                current_hidden = h_out
                current_mask = ~exit_mask
                current_targets = y_batch

                for stage_idx, stage_model in enumerate(stage_models):
                    if not current_mask.any():
                        break

                    # 継続トークンのhidden statesを抽出
                    continue_hidden = current_hidden[current_mask]
                    continue_targets = current_targets[current_mask]

                    # シーケンス長1として処理
                    continue_hidden = continue_hidden.unsqueeze(1)

                    outputs = stage_model.model(
                        inputs_embeds=continue_hidden,
                        output_hidden_states=True,
                        return_dict=True,
                    )

                    hidden_history = outputs.hidden_states
                    stage_h_out = hidden_history[-1].squeeze(1)
                    stage_logits = outputs.logits.squeeze(1)

                    is_last_stage = (stage_idx == len(stage_models) - 1)

                    if is_last_stage:
                        # 最終段階: 全トークンがexit
                        stage_exit_mask = torch.ones(continue_hidden.shape[0], dtype=torch.bool, device=self.device)
                    else:
                        # cos_simでexit判定（exit_fn.pyの共通関数を使用）
                        cos_sim = compute_cos_sim_from_history(hidden_history)
                        cos_sim = cos_sim.squeeze(1)  # シーケンス長1を削除
                        stage_exit_mask = cos_sim > stage_models[stage_idx + 1].threshold

                    # Exitトークンのloss計算
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
                    total_layers_computed += int(stage_exit_mask.sum().item()) * stage_model.num_layers

                    # 次のステージ用に更新
                    current_hidden = stage_h_out
                    current_mask = ~stage_exit_mask
                    current_targets = continue_targets

                total_tokens += batch_size * seq_len

        # 統計を計算
        val_ppl = float(np.exp(total_loss / total_tokens)) if total_tokens > 0 else float('inf')
        accuracy = total_correct / total_tokens if total_tokens > 0 else 0.0

        # Exit分布を比率に変換
        exit_distribution = {
            k: v / total_tokens for k, v in exit_counts.items()
        }

        # 合計パラメータ数
        total_params = sum(p.numel() for p in base_model.parameters())
        for sm in stage_models:
            total_params += sum(p.numel() for p in sm.model.parameters())

        # 最大計算量（全トークンが全レイヤーを通過した場合）
        total_stage_layers = sum(sm.num_layers for sm in stage_models)
        max_layers = total_tokens * (base_layers + total_stage_layers)

        return EvaluationResult(
            pattern_name=pattern_name,
            num_stages=len(stage_models),
            total_params=total_params,
            val_ppl=val_ppl,
            accuracy=accuracy,
            total_tokens=total_tokens,
            exit_distribution=exit_distribution,
            layers_computed=total_layers_computed,
            max_layers_possible=max_layers,
            compute_ratio=total_layers_computed / max_layers if max_layers > 0 else 0.0,
        )

    def print_results(self, results: List[EvaluationResult]):
        """結果をテーブル形式で出力"""
        print("\n" + "=" * 100)
        print("評価結果サマリー")
        print("=" * 100)

        print(f"\n{'パターン':<20} {'段階数':<8} {'パラメータ':<12} {'Val PPL':<12} {'Accuracy':<12} {'計算量比':<10}")
        print("-" * 80)

        for r in results:
            params_str = f"{r.total_params/1e6:.1f}M"
            print(f"{r.pattern_name:<20} {r.num_stages:<8} {params_str:<12} "
                  f"{r.val_ppl:<12.2f} {r.accuracy:<12.2%} {r.compute_ratio:<10.2%}")

        # 最良パターンを表示
        best = min(results, key=lambda r: r.val_ppl)
        print(f"\n推奨パターン: {best.pattern_name} (val_ppl={best.val_ppl:.2f})")
