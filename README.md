# EASE: Efficient Asymmetric Supervision for Early-Exit Transformers

A unified training framework for Early-Exit Transformers.

## Key Findings

- **L2 Loss = 0**: Intermediate layers should NOT have prediction loss (+63% degradation with L2)
- **Asymmetric Loss (α=0.7)**: Better than standard α=0.5 auxiliary loss
- **Discriminative Fine-Tuning**: Layer-wise LR achieves best results (46.9% improvement)

## Quick Start

```python
import sys
sys.path.insert(0, 'src')

from ease import DEEDTransformer, UniversalTrainer, PRESETS

# Use preset configuration
config = PRESETS['asymmetric']  # α=0.7, L2=0, routing_threshold=0.95

# Create model and trainer
model = DEEDTransformer(vocab_size=1000, dim=64, num_layers=3)
trainer = UniversalTrainer(config, vocab_size=1000)
optimizer = trainer.create_optimizer(model, base_lr=1e-3)

# Train
loss, weights = trainer.train_epoch(model, train_batches, optimizer)

# Evaluate
stats = trainer.evaluate(model, val_batches)
```

## Available Presets

| Preset | Description |
|--------|-------------|
| `standard_llm` | Final layer loss only |
| `deep_supervision` | Equal loss on all layers (Lee et al., 2015) |
| `deed` | Deep Supervision + Early Exit (Tang et al., 2023) |
| `auxiliary_loss` | α=0.5 on L1 and L3 (Elbayad et al., 2020) |
| `asymmetric` | α=0.7, L2=0 (Ours) |

## Results

| Model | PPL | vs Standard |
|-------|-----|-------------|
| Layer-wise LR (Decreasing) | 18.52 | **46.9% better** |
| Asymmetric (α=0.7) | 22.95 | 34.2% better |
| Standard LLM | 34.86 | baseline |

## Run Experiments

```bash
python run_experiments.py
```

## References

- Lee et al. (2015) - Deep Supervision
- Tang et al. (2023) - DEED: Deep Supervision + Dynamic Early Exit
- Elbayad et al. (2020) - Depth-Adaptive Transformer
- Howard & Ruder (2018) - Discriminative Fine-Tuning
- Teerapittayanon et al. (2016) - BranchyNet

## License

MIT
