"""Synthetic data generation for ControlNet fine-tuning.

Modules:
  - pair_generator: Generate training pairs from face images
  - augmentation: Clinical degradation augmentations
  - tps_warp: TPS warping with rigid region preservation
"""

from landmarkdiff.synthetic.augmentation import apply_clinical_augmentation
from landmarkdiff.synthetic.pair_generator import (
    TrainingPair,
    generate_pair,
    generate_pairs_from_directory,
)
from landmarkdiff.synthetic.tps_warp import warp_image_tps

__all__ = [
    "generate_pair",
    "generate_pairs_from_directory",
    "TrainingPair",
    "apply_clinical_augmentation",
    "warp_image_tps",
]
