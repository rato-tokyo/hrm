"""
EASE Framework - Experiment Utilities

Efficient Asymmetric Supervision for Early-Exit Transformers.
"""

from .utils import (
    set_seed,
    count_params,
    prepare_wikitext_data,
    ExperimentConfig,
)
from .models import (
    StandardTransformer,
    ConfidenceRoutedTransformer,
)
from .universal_trainer import (
    UniversalConfig,
    UniversalTrainer,
    AlphaSchedule,
    PRESETS,
)

__all__ = [
    # Utils
    'set_seed',
    'count_params',
    'prepare_wikitext_data',
    'ExperimentConfig',
    # Models
    'StandardTransformer',
    'ConfidenceRoutedTransformer',
    # Universal Trainer
    'UniversalConfig',
    'UniversalTrainer',
    'AlphaSchedule',
    'PRESETS',
]
