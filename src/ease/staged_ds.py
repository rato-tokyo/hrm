"""
Staged Deep Supervision (SDS)

統一的な訓練フレームワーク。すべての訓練戦略をStageの組み合わせとして表現。

Core Concept:
- Stage: 訓練の1つのフェーズ（どの層を、どのデータで、どう訓練するか）
- Deep Supervision = 1 stage, all layers, all data
- ASHEM = 2 stages, progressive layers, filtered data
- SDS = N stages, flexible configuration

Key Innovation:
Deep SupervisionとASHEMは同じ概念の時間的変種として統一的に理解できる。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Callable, Any
from dataclasses import dataclass, field


@dataclass
class StageConfig:
    """
    単一の訓練ステージの設定。

    各ステージは以下を定義：
    - どの層を訓練するか（layer_range）
    - 各層の損失重み（layer_weights）
    - どのデータを使用するか（data_filter）
    - 訓練ハイパーパラメータ（learning_rate, max_epochs等）
    - 前のステージの層を凍結するか（freeze_prev_layers）
    """
    name: str
    layer_range: Tuple[int, int]  # (start, end) inclusive, 1-indexed
    layer_weights: Dict[int, float]  # {layer_idx: weight}
    data_filter: Optional[Callable] = None  # Filter function for data
    max_epochs: int = 50
    learning_rate: float = 1e-3
    patience: int = 1
    freeze_prev_layers: bool = False

    # Early Exit設定（推論時）
    routing_threshold: Optional[float] = None
    exit_layer: Optional[int] = None


@dataclass
class StagedDSConfig:
    """
    Staged Deep Supervisionの完全な設定。

    複数のStageを順次実行することで、柔軟な訓練戦略を実現。
    """
    stages: List[StageConfig]
    total_layers: int
    vocab_size: int

    def validate(self) -> None:
        """設定の妥当性を検証"""
        if not self.stages:
            raise ValueError("At least one stage is required")

        for stage in self.stages:
            start, end = stage.layer_range
            if start < 1 or end > self.total_layers:
                raise ValueError(f"Invalid layer range {stage.layer_range} for {self.total_layers} layers")
            if start > end:
                raise ValueError(f"Invalid layer range {stage.layer_range}: start > end")


# ==============================================================================
# Hard Example Mining用のユーティリティ関数
# ==============================================================================

def compute_confidence(model: nn.Module, hidden_state: torch.Tensor) -> torch.Tensor:
    """
    Per-token confidence計算。

    Args:
        model: output_headを持つモデル
        hidden_state: (batch_size, seq_len, dim)

    Returns:
        Confidence values: (batch_size, seq_len)
    """
    logits = model.output_head(hidden_state)
    probs = F.softmax(logits, dim=-1)
    return probs.max(dim=-1).values


def compute_confidence_threshold(
    model: nn.Module,
    val_batches: List[Tuple[torch.Tensor, torch.Tensor]],
    target_ratio: float,
    device: str,
    num_layers: int
) -> float:
    """
    Per-token quantile計算でthresholdを決定。

    Args:
        model: 評価対象モデル
        val_batches: 検証データ
        target_ratio: Hard exampleの目標比率（e.g., 0.5 = 50%）
        device: デバイス
        num_layers: Forward通過する層数

    Returns:
        Confidence threshold
    """
    model.eval()
    all_confidences = []

    with torch.no_grad():
        for x, _ in val_batches:
            x = x.to(device)

            # Forward through specified layers
            h = model.embedding(x)
            for i in range(num_layers):
                h = model.layers[i](h)

            # Compute per-token confidence
            confidence = compute_confidence(model, h)
            all_confidences.append(confidence.view(-1))  # Per-token flatten

    all_confidences_tensor = torch.cat(all_confidences)
    threshold = torch.quantile(all_confidences_tensor, target_ratio).item()

    return threshold


def collect_hard_examples(
    model: nn.Module,
    val_batches: List[Tuple[torch.Tensor, torch.Tensor]],
    threshold: float,
    device: str,
    num_layers: int
) -> Dict[str, torch.Tensor]:
    """
    Per-token filteringでHard examplesを収集。

    Args:
        model: 訓練済みモデル
        val_batches: 検証データ
        threshold: Confidence threshold
        device: デバイス
        num_layers: Forward通過する層数

    Returns:
        Dictionary with 'inputs', 'hidden_states', 'targets', 'confidences'
    """
    model.eval()

    hard_inputs = []
    hard_hidden_states = []
    hard_targets = []
    hard_confidences = []

    with torch.no_grad():
        for x, y in val_batches:
            x, y = x.to(device), y.to(device)

            # Forward through specified layers
            h = model.embedding(x)
            for i in range(num_layers):
                h = model.layers[i](h)

            # Compute per-token confidence
            confidence = compute_confidence(model, h)

            # Per-token filtering
            mask = confidence < threshold  # (batch_size, seq_len)

            # Flatten and filter
            x_flat = x.view(-1)
            h_flat = h.view(-1, h.shape[-1])
            y_flat = y.view(-1)
            confidence_flat = confidence.view(-1)
            mask_flat = mask.view(-1)

            if mask_flat.any():
                hard_inputs.append(x_flat[mask_flat])
                hard_hidden_states.append(h_flat[mask_flat])
                hard_targets.append(y_flat[mask_flat])
                hard_confidences.append(confidence_flat[mask_flat])

    if not hard_inputs:
        raise ValueError("No hard examples found. Try increasing threshold.")

    return {
        'inputs': torch.cat(hard_inputs),
        'hidden_states': torch.cat(hard_hidden_states),
        'targets': torch.cat(hard_targets),
        'confidences': torch.cat(hard_confidences)
    }


def create_hard_example_loader(
    hard_examples: Dict[str, torch.Tensor],
    batch_size: int
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Hard examplesからバッチローダーを作成。

    Args:
        hard_examples: collect_hard_examplesの出力
        batch_size: バッチサイズ

    Returns:
        List of (hidden_state, target) batches
    """
    hidden_states = hard_examples['hidden_states']
    targets = hard_examples['targets']

    num_samples = len(targets)
    indices = torch.randperm(num_samples)

    batches = []
    for i in range(0, num_samples, batch_size):
        batch_indices = indices[i:i + batch_size]
        # Add seq_len dimension: (batch_size, dim) -> (batch_size, 1, dim)
        h_batch = hidden_states[batch_indices].unsqueeze(1)
        t_batch = targets[batch_indices]
        batches.append((h_batch, t_batch))

    return batches


# ==============================================================================
# Staged Trainer
# ==============================================================================

class StagedTrainer:
    """
    Stageベースの訓練フレームワーク。

    各Stageを順次実行し、Deep Supervision、ASHEM等を統一的に実現。
    """

    def __init__(self, config: StagedDSConfig, device: str = 'cuda'):
        self.config = config
        self.device = device
        config.validate()

    def train_stage(
        self,
        stage: StageConfig,
        model: nn.Module,
        train_batches: List[Tuple[torch.Tensor, torch.Tensor]],
        val_batches: List[Tuple[torch.Tensor, torch.Tensor]],
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        単一Stageの訓練を実行。

        Args:
            stage: Stage設定
            model: 訓練対象モデル
            train_batches: 訓練データ
            val_batches: 検証データ
            verbose: ログ出力

        Returns:
            訓練結果（loss履歴等）
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"Stage: {stage.name}")
            print(f"{'='*60}")
            print(f"  Layers: {stage.layer_range[0]}-{stage.layer_range[1]}")
            print(f"  Learning rate: {stage.learning_rate:.1e}")
            print(f"  Max epochs: {stage.max_epochs}")
            print(f"  Patience: {stage.patience}")

        # Freeze previous layers if needed
        if stage.freeze_prev_layers:
            self._freeze_layers(model, stage.layer_range[0] - 1)

        # Create optimizer
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=stage.learning_rate
        )

        # Training loop with early stopping
        best_val_loss = float('inf')
        best_model_state = None
        patience_counter = 0

        train_losses = []
        val_losses = []

        for epoch in range(stage.max_epochs):
            # Train
            train_loss = self._train_epoch(model, train_batches, optimizer, stage)

            # Validate
            val_loss = self._validate(model, val_batches, stage)

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            if verbose:
                print(f"Epoch {epoch+1}/{stage.max_epochs} - "
                      f"Train Loss: {train_loss:.4f} | "
                      f"Val Loss: {val_loss:.4f}")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
                if verbose:
                    print(f"  → New best (val_loss: {val_loss:.4f})")
            else:
                patience_counter += 1
                if verbose:
                    print(f"  → No improvement ({patience_counter}/{stage.patience})")

            if patience_counter >= stage.patience:
                if verbose:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                break

        # Restore best model
        if best_model_state is not None:
            model.load_state_dict({k: v.to(self.device) for k, v in best_model_state.items()})
            if verbose:
                print(f"\nRestored best model (val_loss: {best_val_loss:.4f})")

        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_val_loss': best_val_loss
        }

    def _train_epoch(
        self,
        model: nn.Module,
        train_batches: List[Tuple[torch.Tensor, torch.Tensor]],
        optimizer: torch.optim.Optimizer,
        stage: StageConfig
    ) -> float:
        """1エポックの訓練"""
        model.train()
        total_loss = 0.0

        for x, y in train_batches:
            x, y = x.to(self.device), y.to(self.device)
            optimizer.zero_grad()

            # Forward through stage layers
            loss = self._compute_stage_loss(model, x, y, stage)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_batches)

    def _validate(
        self,
        model: nn.Module,
        val_batches: List[Tuple[torch.Tensor, torch.Tensor]],
        stage: StageConfig
    ) -> float:
        """検証"""
        model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for x, y in val_batches:
                x, y = x.to(self.device), y.to(self.device)
                loss = self._compute_stage_loss(model, x, y, stage)
                total_loss += loss.item()

        return total_loss / len(val_batches)

    def _compute_stage_loss(
        self,
        model: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        stage: StageConfig
    ) -> torch.Tensor:
        """
        Stageに基づいて損失を計算。

        指定されたlayer_rangeとlayer_weightsに基づいて、
        各層の出力に対する損失を計算し、重み付け和を返す。
        """
        # Forward through all layers up to stage end
        h = model.embedding(x)

        total_loss = 0.0
        start_layer, end_layer = stage.layer_range

        for i in range(end_layer):
            h = model.layers[i](h)

            # Compute loss if this layer is in the stage and has weight
            layer_idx = i + 1  # 1-indexed
            if start_layer <= layer_idx <= end_layer and layer_idx in stage.layer_weights:
                weight = stage.layer_weights[layer_idx]
                if weight > 0:
                    logits = model.output_head(h)
                    loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), y.view(-1))
                    total_loss = total_loss + weight * loss

        return total_loss

    def _freeze_layers(self, model: nn.Module, num_layers: int) -> None:
        """指定された層数まで凍結"""
        model.embedding.requires_grad_(False)
        for i in range(num_layers):
            model.layers[i].requires_grad_(False)


# ==============================================================================
# プリセット設定
# ==============================================================================

def create_standard_config(
    total_layers: int,
    vocab_size: int,
    learning_rate: float = 1e-3,
    max_epochs: int = 50
) -> StagedDSConfig:
    """
    Standard Transformer設定（最終層のみ）。

    従来のLLM訓練と同等。
    """
    return StagedDSConfig(
        stages=[
            StageConfig(
                name="standard",
                layer_range=(1, total_layers),
                layer_weights={total_layers: 1.0},
                max_epochs=max_epochs,
                learning_rate=learning_rate
            )
        ],
        total_layers=total_layers,
        vocab_size=vocab_size
    )


def create_deep_supervision_config(
    total_layers: int,
    vocab_size: int,
    learning_rate: float = 1e-3,
    max_epochs: int = 50
) -> StagedDSConfig:
    """
    Deep Supervision設定（全層均等）。
    """
    weight_per_layer = 1.0 / total_layers
    layer_weights = {i: weight_per_layer for i in range(1, total_layers + 1)}

    return StagedDSConfig(
        stages=[
            StageConfig(
                name="deep_supervision",
                layer_range=(1, total_layers),
                layer_weights=layer_weights,
                max_epochs=max_epochs,
                learning_rate=learning_rate
            )
        ],
        total_layers=total_layers,
        vocab_size=vocab_size
    )


def create_ashem_config(
    phase1_layers: int = 2,
    phase2_layers: int = 4,
    vocab_size: int = 69830,
    phase1_lr: float = 1e-3,
    phase2_lr: float = 1e-4,
    phase1_epochs: int = 50,
    phase2_epochs: int = 50,
    hard_example_ratio: float = 0.5
) -> StagedDSConfig:
    """
    ASHEM設定（2-stage, Hard example mining）。

    Phase 1: 浅層モデルで全データ訓練
    Phase 2: 深層モデルでHard examplesのみ訓練
    """
    return StagedDSConfig(
        stages=[
            StageConfig(
                name="phase1_shallow",
                layer_range=(1, phase1_layers),
                layer_weights={phase1_layers: 1.0},
                max_epochs=phase1_epochs,
                learning_rate=phase1_lr,
                patience=1
            ),
            StageConfig(
                name="phase2_hard",
                layer_range=(phase1_layers + 1, phase2_layers),
                layer_weights={phase2_layers: 1.0},
                max_epochs=phase2_epochs,
                learning_rate=phase2_lr,
                patience=3,
                freeze_prev_layers=True
            )
        ],
        total_layers=phase2_layers,
        vocab_size=vocab_size
    )
