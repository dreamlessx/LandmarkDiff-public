"""Curriculum learning support for progressive training difficulty.

Implements a schedule that controls which training samples are used
at different stages of training, starting with easy examples (small
displacements) and gradually introducing harder ones.

Usage in training loop::

    curriculum = TrainingCurriculum(
        total_steps=100000,
        warmup_fraction=0.1,   # first 10% easy only
        full_difficulty_at=0.5, # full dataset by 50%
    )

    # In training loop:
    difficulty = curriculum.get_difficulty(global_step)
    # Use difficulty to filter/weight samples

Or as a dataset wrapper::

    dataset = CurriculumDataset(
        base_dataset=SyntheticPairDataset(data_dir),
        metadata_path=Path(data_dir) / "metadata.json",
        total_steps=100000,
    )
    # Call dataset.set_step(global_step) each iteration
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


class TrainingCurriculum:
    """Schedule that maps training step to difficulty level [0, 1].

    Difficulty 0 = easiest (smallest displacements, lowest intensity).
    Difficulty 1 = full dataset (all difficulties).

    The schedule uses a cosine ramp:
    - During warmup: difficulty = 0 (easy only)
    - warmup → full_difficulty: cosine ramp from 0 → 1
    - After full_difficulty: difficulty = 1 (full dataset)
    """

    def __init__(
        self,
        total_steps: int,
        warmup_fraction: float = 0.1,
        full_difficulty_at: float = 0.5,
    ):
        self.total_steps = total_steps
        self.warmup_steps = int(total_steps * warmup_fraction)
        self.full_steps = int(total_steps * full_difficulty_at)

    def get_difficulty(self, step: int) -> float:
        """Get difficulty level [0, 1] for the given training step."""
        if step < self.warmup_steps:
            return 0.0
        if step >= self.full_steps:
            return 1.0
        progress = (step - self.warmup_steps) / max(1, self.full_steps - self.warmup_steps)
        return 0.5 * (1 - math.cos(math.pi * progress))

    def should_include(
        self,
        step: int,
        sample_difficulty: float,
        rng: np.random.Generator | None = None,
    ) -> bool:
        """Whether to include a sample of the given difficulty at this step.

        Uses probabilistic inclusion so harder samples gradually appear.

        Args:
            step: Current training step.
            sample_difficulty: Difficulty of the sample [0, 1].
            rng: Random number generator for stochastic inclusion.

        Returns:
            True if sample should be used.
        """
        curr_difficulty = self.get_difficulty(step)
        if sample_difficulty <= curr_difficulty:
            return True
        # Stochastic inclusion for samples slightly above threshold
        if rng is None:
            rng = np.random.default_rng()
        overshoot = sample_difficulty - curr_difficulty
        include_prob = max(0, 1.0 - overshoot * 5)  # drops off quickly
        return rng.random() < include_prob


class ProcedureCurriculum:
    """Procedure-aware curriculum that adjusts per-procedure weights.

    Some procedures are inherently harder (e.g., orthognathic with large
    deformations). This curriculum increases their weight over training.
    """

    # Difficulty ranking (0=easiest, 1=hardest)
    DEFAULT_PROCEDURE_DIFFICULTY = {
        "blepharoplasty": 0.3,  # small, localized changes
        "rhinoplasty": 0.5,  # moderate, central face
        "rhytidectomy": 0.7,  # large, affects face shape
        "orthognathic": 0.9,  # largest deformations
    }

    def __init__(
        self,
        total_steps: int,
        procedure_difficulty: dict[str, float] | None = None,
        warmup_fraction: float = 0.1,
    ):
        self.curriculum = TrainingCurriculum(total_steps, warmup_fraction)
        self.proc_difficulty = procedure_difficulty or self.DEFAULT_PROCEDURE_DIFFICULTY

    def get_weight(self, step: int, procedure: str) -> float:
        """Get sampling weight for a procedure at the given step.

        Returns a value in [0.1, 1.0] — never fully excludes any procedure.
        """
        difficulty = self.get_difficulty(step)
        proc_diff = self.proc_difficulty.get(procedure, 0.5)

        if proc_diff <= difficulty:
            return 1.0
        # Reduce weight for too-hard procedures
        return max(0.1, 1.0 - (proc_diff - difficulty) * 2)

    def get_difficulty(self, step: int) -> float:
        return self.curriculum.get_difficulty(step)

    def get_procedure_weights(self, step: int) -> dict[str, float]:
        """Get all procedure weights at the given step."""
        return {proc: self.get_weight(step, proc) for proc in self.proc_difficulty}


def compute_sample_difficulty(
    metadata_path: str | Path,
    displacement_model_path: str | Path | None = None,
) -> dict[str, float]:
    """Compute difficulty scores for each sample in the dataset.

    Difficulty is based on:
    1. Displacement intensity (from metadata)
    2. Procedure difficulty
    3. Source type (real > synthetic)

    Returns:
        Dict mapping sample prefix to difficulty score [0, 1].
    """
    with open(metadata_path) as f:
        meta = json.load(f)

    pairs = meta.get("pairs", {})
    difficulties = {}

    proc_base = {
        "blepharoplasty": 0.2,
        "rhinoplasty": 0.4,
        "rhytidectomy": 0.6,
        "orthognathic": 0.8,
        "unknown": 0.5,
    }

    source_bonus = {
        "synthetic": 0.0,
        "synthetic_v3": 0.1,  # realistic displacements slightly harder
        "real": 0.2,  # real data hardest
        "augmented": 0.0,
    }

    for prefix, info in pairs.items():
        proc = info.get("procedure", "unknown")
        source = info.get("source", "synthetic")
        intensity = info.get("intensity", 1.0)

        # Combine factors
        base = proc_base.get(proc, 0.5)
        src = source_bonus.get(source, 0.0)
        # Intensity scaling (higher intensity = harder)
        int_factor = min(1.0, intensity / 1.5) * 0.2

        difficulties[prefix] = min(1.0, base + src + int_factor)

    return difficulties
