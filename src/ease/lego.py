"""
LEGO Framework - Cascading Multi-Phase Training

LEGO: Layered Ensemble with Gradual Optimization

A multi-phase training strategy that focuses computational resources on hard examples:
1. Phase 1: Train Block 1 (shallow layers) on all data
2. Phase 2: Train Block 2 (deeper layers) on hard examples from Phase 1
3. Phase N: Train Block N on progressively harder examples
4. Inference: Multi-exit routing using confidence thresholds

Key Benefits:
- ~20% improvement on hard examples through cascading training
- Compute cost reduction using adaptive routing
- Flexible multi-phase configuration

Usage:
    from ease import LEGOConfig, PhaseConfig, LEGOTrainer

    # Define multi-phase configuration
    config = LEGOConfig(
        phases=[
            PhaseConfig(layers=(1, 2), lr=1e-3, patience=1),
            PhaseConfig(layers=(3, 4), lr=1e-4, patience=3),
        ],
        hard_example_ratio=0.5,
    )

    # Train with cascading phases
    trainer = LEGOTrainer(config, vocab_size=10000, device='cuda')
    result = trainer.train(model, train_loader, val_loader)

    # Access thresholds for inference
    thresholds = result['thresholds']
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
from dataclasses import dataclass, field

from .types import DataBatch, HardBatch, HardExamples, LEGOResult, PhaseHistory


# ==============================================================================
# Configuration
# ==============================================================================

@dataclass
class PhaseConfig:
    """
    Configuration for a single training phase.

    Each phase trains a block of layers. In cascading LEGO:
    - Phase 1 trains on all data
    - Phase 2+ trains on hard examples from the previous phase

    Args:
        layers: Tuple of (start_layer, end_layer) inclusive, 1-indexed.
                e.g., (1, 2) trains layers 1 and 2
        lr: Learning rate for this phase
        patience: Early stopping patience (epochs without improvement)
        freeze_lower: Whether to freeze layers below this block (default: True)
        max_epochs: Maximum epochs for this phase (default: 50)
    """
    layers: Tuple[int, int]
    lr: float = 1e-3
    patience: int = 3
    freeze_lower: bool = True
    max_epochs: int = 50


@dataclass
class LEGOConfig:
    """
    Multi-phase LEGO training configuration.

    LEGO (Layered Ensemble with Gradual Optimization) uses cascading phases:
    - Each phase trains a block of layers
    - Hard examples are passed from one phase to the next
    - Inference uses multi-exit routing with learned thresholds

    Args:
        phases: List of PhaseConfig defining each training phase
        hard_example_ratio: Target ratio of hard examples (0.0-1.0).
                           e.g., 0.5 means 50% of examples are classified as hard

    Example:
        # 4-layer model with 2 phases
        config = LEGOConfig(
            phases=[
                PhaseConfig(layers=(1, 2), lr=1e-3, patience=1),
                PhaseConfig(layers=(3, 4), lr=1e-4, patience=3),
            ],
            hard_example_ratio=0.5,
        )

        # 6-layer model with 3 phases
        config = LEGOConfig(
            phases=[
                PhaseConfig(layers=(1, 2), lr=1e-3, patience=1),
                PhaseConfig(layers=(3, 4), lr=1e-4, patience=2),
                PhaseConfig(layers=(5, 6), lr=1e-5, patience=3),
            ],
            hard_example_ratio=0.5,
        )
    """
    phases: List[PhaseConfig] = field(default_factory=list)
    hard_example_ratio: float = 0.5

    @property
    def num_phases(self) -> int:
        """Number of training phases."""
        return len(self.phases)

    @property
    def total_layers(self) -> int:
        """Total number of layers across all phases."""
        if not self.phases:
            return 0
        return max(phase.layers[1] for phase in self.phases)

    def describe(self) -> str:
        """Human-readable description."""
        phases_str = ", ".join([
            f"Phase{i+1}:L{p.layers[0]}-{p.layers[1]}(lr={p.lr})"
            for i, p in enumerate(self.phases)
        ])
        return f"LEGO[{phases_str}] hard_ratio={self.hard_example_ratio}"


# ==============================================================================
# Utility Functions
# ==============================================================================

def compute_confidence_threshold(
    model: nn.Module,
    val_batches: List[DataBatch],
    target_ratio: float,
    device: str,
    exit_layer: Optional[int] = None
) -> float:
    """
    Compute confidence threshold to achieve target hard example ratio.

    The threshold is set such that approximately target_ratio of examples
    fall below this confidence value (i.e., classified as hard examples).

    Args:
        model: Trained model (must have forward_to_layer method or forward_to_hidden)
        val_batches: Validation data batches
        target_ratio: Desired ratio of hard examples (e.g., 0.5 for 50%)
        device: Device to run computation on
        exit_layer: Layer to compute confidence at (None = final layer)

    Returns:
        Confidence threshold value
    """
    model.eval()
    all_confidences: List[torch.Tensor] = []

    with torch.no_grad():
        for x, _ in val_batches:
            x = x.to(device)

            if exit_layer is not None and hasattr(model, 'forward_to_layer'):
                h = model.forward_to_layer(x, exit_layer)
            else:
                h = model.forward_to_hidden(x)

            confidence = model.compute_confidence(h)
            all_confidences.append(confidence.view(-1))

    all_confidences_tensor = torch.cat(all_confidences)
    threshold = torch.quantile(all_confidences_tensor, target_ratio).item()

    return threshold


def collect_hard_examples(
    model: nn.Module,
    val_batches: List[DataBatch],
    threshold: float,
    device: str,
    exit_layer: Optional[int] = None
) -> HardExamples:
    """
    Collect hard examples (low confidence tokens) from validation set.

    Hard examples are defined as tokens where the model's prediction
    confidence falls below the specified threshold.

    Args:
        model: Trained model
        val_batches: Validation data batches
        threshold: Confidence threshold for identifying hard examples
        device: Device to run computation on
        exit_layer: Layer to compute confidence at (None = final layer)

    Returns:
        HardExamples dictionary containing inputs, hidden_states, targets, confidences
    """
    model.eval()

    hard_inputs = []
    hard_hidden_states = []
    hard_targets = []
    hard_confidences = []

    with torch.no_grad():
        for x, y in val_batches:
            x, y = x.to(device), y.to(device)

            if exit_layer is not None and hasattr(model, 'forward_to_layer'):
                h = model.forward_to_layer(x, exit_layer)
            else:
                h = model.forward_to_hidden(x)

            confidence = model.compute_confidence(h)

            # Identify low-confidence tokens
            mask = confidence < threshold

            if mask.any():
                # Flatten batch and sequence dimensions
                x_flat = x.view(-1)
                h_flat = h.view(-1, h.shape[-1])
                y_flat = y.view(-1)
                confidence_flat = confidence.view(-1)
                mask_flat = mask.view(-1)

                # Collect hard examples (token-level)
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
    hard_examples: HardExamples,
    batch_size: int
) -> List[HardBatch]:
    """
    Create batched dataloader from collected hard examples.

    Args:
        hard_examples: Dictionary with 'hidden_states' and 'targets'
        batch_size: Number of examples per batch

    Returns:
        List of batches (hidden_state, target) tuples
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


def train_upper_layers(
    model: nn.Module,
    hard_batches: List[HardBatch],
    optimizer: torch.optim.Optimizer,
    vocab_size: int,
    device: str,
    num_lower_layers: int = 2
) -> float:
    """
    Train upper layers on hard examples only.

    The lower layers are frozen (already trained in previous phase).
    Only the newly added upper layers are trained on hard examples.

    Args:
        model: Extended model with upper layers
        hard_batches: Batches of (hidden_state, target) pairs
        optimizer: Optimizer for trainable parameters
        vocab_size: Vocabulary size for loss computation
        device: Device to run training on
        num_lower_layers: Number of frozen lower layers

    Returns:
        Average training loss for this epoch
    """
    model.train()
    total_loss = 0.0

    for h, y in hard_batches:
        h, y = h.to(device), y.to(device)
        optimizer.zero_grad()

        # Process through upper layers only
        for i in range(num_lower_layers, model.num_layers):
            h = model.layers[i](h)

        # Compute classification loss
        logits = model.output_head(h).squeeze(1)
        loss = F.cross_entropy(logits, y)

        loss.backward()  # type: ignore[no-untyped-call]
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(hard_batches)


def evaluate_on_hard_examples(
    model: nn.Module,
    hard_examples: HardExamples,
    vocab_size: int,
    device: str,
    batch_size: int = 64,
    num_lower_layers: int = 2
) -> float:
    """
    Evaluate model performance on hard examples only.

    Args:
        model: Model to evaluate
        hard_examples: Dictionary with 'hidden_states' and 'targets'
        vocab_size: Vocabulary size for loss computation
        device: Device to run evaluation on
        batch_size: Batch size for evaluation
        num_lower_layers: Number of lower layers (for deep model evaluation)

    Returns:
        Perplexity on hard examples
    """
    model.eval()
    total_loss = 0.0
    total_samples = 0

    hidden_states = hard_examples['hidden_states']
    targets = hard_examples['targets']
    num_samples = len(targets)

    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            h_batch = hidden_states[i:i + batch_size].unsqueeze(1).to(device)
            y_batch = targets[i:i + batch_size].to(device)

            # If deep model, process through upper layers
            if hasattr(model, 'num_layers') and model.num_layers > num_lower_layers:
                for layer_idx in range(num_lower_layers, model.num_layers):
                    h_batch = model.layers[layer_idx](h_batch)

            logits = model.output_head(h_batch).squeeze(1)
            loss = F.cross_entropy(logits, y_batch, reduction='sum')

            total_loss += loss.item()
            total_samples += len(y_batch)

    avg_loss = total_loss / total_samples
    ppl = torch.exp(torch.tensor(avg_loss)).item()
    return ppl


# ==============================================================================
# LEGOTrainer
# ==============================================================================

class LEGOTrainer:
    """
    Trainer for multi-phase LEGO training with cascading hard examples.

    Implements the cascading LEGO training strategy:
    1. Phase 1: Train Block 1 on all data
    2. Compute threshold and collect hard examples
    3. Phase 2: Train Block 2 on hard examples only
    4. Repeat for additional phases

    Usage:
        config = LEGOConfig(
            phases=[
                PhaseConfig(layers=(1, 2), lr=1e-3, patience=1),
                PhaseConfig(layers=(3, 4), lr=1e-4, patience=3),
            ],
            hard_example_ratio=0.5,
        )
        trainer = LEGOTrainer(config, vocab_size=10000, device='cuda')
        result = trainer.train(model, train_loader, val_loader)
    """

    def __init__(
        self,
        config: LEGOConfig,
        vocab_size: int,
        device: str = 'cpu',
        verbose: bool = True
    ):
        """
        Initialize LEGOTrainer.

        Args:
            config: LEGOConfig with phase definitions
            vocab_size: Vocabulary size
            device: Device to train on
            verbose: Print progress
        """
        self.config = config
        self.vocab_size = vocab_size
        self.device = device
        self.verbose = verbose

    def _log(self, msg: str) -> None:
        """Print message if verbose."""
        if self.verbose:
            print(msg)

    def _train_phase_standard(
        self,
        model: nn.Module,
        train_batches: List[DataBatch],
        val_batches: List[DataBatch],
        phase_config: PhaseConfig,
    ) -> PhaseHistory:
        """
        Train a phase using standard data batches (Phase 1).

        Args:
            model: Model to train
            train_batches: Training data
            val_batches: Validation data
            phase_config: Phase configuration

        Returns:
            PhaseHistory with training statistics
        """
        # Freeze lower layers if configured
        if phase_config.freeze_lower:
            start_layer = phase_config.layers[0]
            for param in model.embedding.parameters():
                param.requires_grad = start_layer == 1
            for i in range(start_layer - 1):
                for param in model.layers[i].parameters():
                    param.requires_grad = False

        # Create optimizer for trainable parameters
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=phase_config.lr)

        best_val_loss = float('inf')
        patience_counter = 0
        best_state = None

        train_losses = []
        val_ppls = []

        for epoch in range(phase_config.max_epochs):
            # Training
            model.train()
            total_loss = 0.0

            for x, y in train_batches:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()

                output = model(x)
                loss = F.cross_entropy(
                    output.view(-1, self.vocab_size),
                    y.view(-1)
                )

                loss.backward()  # type: ignore[no-untyped-call]
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item()

            avg_train_loss = total_loss / len(train_batches)
            train_losses.append(avg_train_loss)

            # Validation
            model.eval()
            total_val_loss = 0.0
            total_tokens = 0

            with torch.no_grad():
                for x, y in val_batches:
                    x, y = x.to(self.device), y.to(self.device)
                    output = model(x)
                    loss = F.cross_entropy(
                        output.view(-1, self.vocab_size),
                        y.view(-1),
                        reduction='sum'
                    )
                    total_val_loss += loss.item()
                    total_tokens += y.numel()

            avg_val_loss = total_val_loss / total_tokens
            val_ppl = float(torch.exp(torch.tensor(avg_val_loss)).item())
            val_ppls.append(val_ppl)

            self._log(f"  Epoch {epoch+1}: Train Loss={avg_train_loss:.4f} | Val PPL={val_ppl:.2f}")

            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= phase_config.patience:
                    self._log(f"  Early stopping at epoch {epoch+1}")
                    break

        # Restore best model
        if best_state is not None:
            model.load_state_dict(best_state)

        return {
            'train_losses': train_losses,
            'val_ppls': val_ppls,
            'best_epoch': len(train_losses) - patience_counter - 1,
            'total_epochs': len(train_losses),
        }

    def _train_phase_hard(
        self,
        model: nn.Module,
        hard_examples: HardExamples,
        phase_config: PhaseConfig,
        num_lower_layers: int,
        batch_size: int = 64
    ) -> PhaseHistory:
        """
        Train a phase using hard examples (Phase 2+).

        Args:
            model: Model to train
            hard_examples: Hard examples from previous phase
            phase_config: Phase configuration
            num_lower_layers: Number of frozen lower layers
            batch_size: Batch size for hard example training

        Returns:
            PhaseHistory with training statistics
        """
        # Freeze lower layers
        for param in model.embedding.parameters():
            param.requires_grad = False
        for i in range(num_lower_layers):
            for param in model.layers[i].parameters():
                param.requires_grad = False

        # Create optimizer for trainable parameters
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=phase_config.lr)

        hard_batches = create_hard_example_loader(hard_examples, batch_size)

        best_ppl = float('inf')
        patience_counter = 0
        best_state = None

        train_losses = []
        val_ppls = []

        for epoch in range(phase_config.max_epochs):
            # Training on hard examples
            train_loss = train_upper_layers(
                model, hard_batches, optimizer,
                self.vocab_size, self.device, num_lower_layers
            )
            train_losses.append(train_loss)

            # Evaluation on hard examples
            hard_ppl = evaluate_on_hard_examples(
                model, hard_examples, self.vocab_size,
                self.device, batch_size, num_lower_layers
            )
            val_ppls.append(hard_ppl)

            self._log(f"  Epoch {epoch+1}: Train Loss={train_loss:.4f} | Hard PPL={hard_ppl:.2f}")

            # Early stopping
            if hard_ppl < best_ppl:
                best_ppl = hard_ppl
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= phase_config.patience:
                    self._log(f"  Early stopping at epoch {epoch+1}")
                    break

        # Restore best model
        if best_state is not None:
            model.load_state_dict(best_state)

        return {
            'train_losses': train_losses,
            'val_ppls': val_ppls,
            'best_epoch': len(train_losses) - patience_counter - 1,
            'total_epochs': len(train_losses),
        }

    def train(
        self,
        model: nn.Module,
        train_batches: List[DataBatch],
        val_batches: List[DataBatch],
        batch_size: int = 64
    ) -> LEGOResult:
        """
        Execute multi-phase LEGO training.

        Implements cascading training:
        1. Phase 1: Train on all data
        2. Compute threshold and collect hard examples
        3. Phase 2+: Train on hard examples from previous phase

        Args:
            model: Model to train (should have enough layers for all phases)
            train_batches: Training data
            val_batches: Validation data
            batch_size: Batch size for hard example training

        Returns:
            LEGOResult containing thresholds, histories, and hard_examples
        """
        thresholds: List[float] = []
        phase_histories: List[PhaseHistory] = []
        current_hard_examples: Optional[HardExamples] = None
        num_lower_layers = 0

        for phase_idx, phase_config in enumerate(self.config.phases):
            phase_num = phase_idx + 1
            self._log(f"\n{'='*50}")
            self._log(f"Phase {phase_num}: Layers {phase_config.layers[0]}-{phase_config.layers[1]}")
            self._log(f"{'='*50}")

            if phase_idx == 0:
                # Phase 1: Train on all data
                history = self._train_phase_standard(
                    model, train_batches, val_batches, phase_config
                )
            else:
                # Phase 2+: Train on hard examples
                assert current_hard_examples is not None
                history = self._train_phase_hard(
                    model, current_hard_examples, phase_config,
                    num_lower_layers, batch_size
                )

            phase_histories.append(history)

            # Collect hard examples for next phase (except for last phase)
            if phase_idx < len(self.config.phases) - 1:
                exit_layer = phase_config.layers[1]
                num_lower_layers = exit_layer

                self._log(f"\nComputing threshold at layer {exit_layer}...")
                threshold = compute_confidence_threshold(
                    model, val_batches, self.config.hard_example_ratio,
                    self.device, exit_layer
                )
                thresholds.append(threshold)
                self._log(f"Threshold: {threshold:.4f}")

                self._log("Collecting hard examples...")
                current_hard_examples = collect_hard_examples(
                    model, val_batches, threshold, self.device, exit_layer
                )
                num_hard = len(current_hard_examples['targets'])
                total_tokens = sum(x.numel() for x, _ in val_batches)
                self._log(f"Hard tokens: {num_hard}/{total_tokens} ({num_hard/total_tokens*100:.1f}%)")

        return {
            'thresholds': thresholds,
            'phase_histories': phase_histories,
            'hard_examples': current_hard_examples,
        }
