# EASE: Efficient Asymmetric Supervision for Early-Exit Transformers

A simple training framework with two base models and three options.

## Base Models

- **StandardTransformer**: Final layer loss only
- **DeepSupervisionTransformer**: Loss at all layers with early exit support

## Options

- **layer_weights**: Layer-wise loss weights
- **layer_lr_scales**: Layer-wise learning rates (Discriminative Fine-Tuning)
- **routing_threshold**: Early exit at inference

## Quick Start

```python
import sys
sys.path.insert(0, 'src')

from ease import DeepSupervisionTransformer, Trainer, TrainingConfig

# Create model
model = DeepSupervisionTransformer(vocab_size=1000, dim=64, num_layers=3)

# Configure training
config = TrainingConfig(
    layer_weights={1: 0.7, 2: 0, 3: 0.3},  # Asymmetric loss
    layer_lr_scales={1: 1.0, 2: 0.5, 3: 0.1},  # Discriminative Fine-Tuning
    routing_threshold=0.95,  # Early exit
)

# Create trainer
trainer = Trainer(config, vocab_size=1000)
optimizer = trainer.create_optimizer(model, base_lr=1e-3)

# Train
loss = trainer.train_epoch(model, train_batches, optimizer)

# Evaluate
stats = trainer.evaluate(model, val_batches)
```

## Helper Functions

```python
from ease import create_standard_config, create_deep_supervision_config

# Standard LLM: final layer loss only
config = create_standard_config(num_layers=3)

# Deep Supervision: equal loss on all layers
config = create_deep_supervision_config(num_layers=3)
```

## Run Experiments

```bash
python run_experiments.py
```

## References

- Lee et al. (2015) - Deep Supervision
- Howard & Ruder (2018) - Discriminative Fine-Tuning
- Teerapittayanon et al. (2016) - Early Exit

## License

MIT
