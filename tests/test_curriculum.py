"""Tests for curriculum learning support."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from landmarkdiff.curriculum import (
    ProcedureCurriculum,
    TrainingCurriculum,
    compute_sample_difficulty,
)


class TestTrainingCurriculum:
    """Tests for TrainingCurriculum class."""

    def test_warmup_easy(self):
        curriculum = TrainingCurriculum(total_steps=10000, warmup_fraction=0.1)
        # During warmup, difficulty should be 0
        assert curriculum.get_difficulty(0) == 0.0
        assert curriculum.get_difficulty(500) == 0.0
        assert curriculum.get_difficulty(999) == 0.0

    def test_after_full_difficulty(self):
        curriculum = TrainingCurriculum(total_steps=10000, full_difficulty_at=0.5)
        # After full_difficulty_at, should be 1.0
        assert curriculum.get_difficulty(5000) == 1.0
        assert curriculum.get_difficulty(10000) == 1.0

    def test_cosine_ramp(self):
        curriculum = TrainingCurriculum(
            total_steps=10000, warmup_fraction=0.1, full_difficulty_at=0.5
        )
        # Middle of ramp should be around 0.5
        mid_step = int(10000 * 0.3)  # halfway between warmup end (0.1) and full (0.5)
        diff = curriculum.get_difficulty(mid_step)
        assert 0.3 < diff < 0.7  # roughly 0.5 for cosine

    def test_monotonically_increasing(self):
        curriculum = TrainingCurriculum(total_steps=10000)
        prev = -1.0
        for step in range(0, 10001, 100):
            d = curriculum.get_difficulty(step)
            assert d >= prev - 1e-6
            prev = d

    def test_should_include_easy_sample(self):
        curriculum = TrainingCurriculum(total_steps=10000)
        # Easy sample (difficulty=0) should always be included
        for step in range(0, 10001, 1000):
            assert curriculum.should_include(step, 0.0)

    def test_should_include_hard_sample_late(self):
        curriculum = TrainingCurriculum(total_steps=10000, full_difficulty_at=0.5)
        # Hard sample should be included after full difficulty
        assert curriculum.should_include(6000, 1.0)

    def test_should_exclude_hard_sample_early(self):
        curriculum = TrainingCurriculum(total_steps=10000, warmup_fraction=0.1)
        rng = np.random.default_rng(42)
        # Very hard sample during warmup should be excluded most of the time
        included = sum(curriculum.should_include(0, 1.0, rng=rng) for _ in range(100))
        assert included < 20  # should be excluded most times


class TestProcedureCurriculum:
    """Tests for ProcedureCurriculum class."""

    def test_easy_procedures_early(self):
        curriculum = ProcedureCurriculum(total_steps=10000, warmup_fraction=0.1)
        weights = curriculum.get_procedure_weights(0)
        # Blepharoplasty (easiest) should have higher weight early
        assert weights.get("blepharoplasty", 0) >= weights.get("orthognathic", 0)

    def test_all_procedures_late(self):
        curriculum = ProcedureCurriculum(total_steps=10000)
        weights = curriculum.get_procedure_weights(10000)
        # All procedures should have weight 1.0 at full difficulty
        for _proc, w in weights.items():
            assert w == 1.0

    def test_weight_range(self):
        curriculum = ProcedureCurriculum(total_steps=10000)
        for step in range(0, 10001, 500):
            weights = curriculum.get_procedure_weights(step)
            for _proc, w in weights.items():
                assert 0.1 <= w <= 1.0

    def test_custom_difficulty(self):
        custom = {"easy_proc": 0.1, "hard_proc": 0.9}
        curriculum = ProcedureCurriculum(
            total_steps=10000,
            procedure_difficulty=custom,
        )
        weights = curriculum.get_procedure_weights(0)
        assert weights["easy_proc"] >= weights["hard_proc"]

    def test_get_difficulty_delegates(self):
        curriculum = ProcedureCurriculum(total_steps=10000, warmup_fraction=0.2)
        assert curriculum.get_difficulty(0) == 0.0
        assert curriculum.get_difficulty(10000) == 1.0


class TestComputeSampleDifficulty:
    """Tests for compute_sample_difficulty function."""

    def test_basic(self, tmp_path):
        meta = {
            "pairs": {
                "rhinoplasty_000001": {
                    "procedure": "rhinoplasty",
                    "source": "synthetic",
                    "intensity": 0.5,
                },
                "orthognathic_000001": {
                    "procedure": "orthognathic",
                    "source": "real",
                    "intensity": 1.2,
                },
            }
        }
        meta_path = tmp_path / "metadata.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f)

        difficulties = compute_sample_difficulty(meta_path)
        assert "rhinoplasty_000001" in difficulties
        assert "orthognathic_000001" in difficulties
        # Orthognathic + real should be harder than rhinoplasty + synthetic
        assert difficulties["orthognathic_000001"] > difficulties["rhinoplasty_000001"]

    def test_range(self, tmp_path):
        meta = {
            "pairs": {
                f"sample_{i}": {
                    "procedure": "rhinoplasty",
                    "source": "synthetic",
                    "intensity": float(i) / 10,
                }
                for i in range(10)
            }
        }
        meta_path = tmp_path / "metadata.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f)

        difficulties = compute_sample_difficulty(meta_path)
        for d in difficulties.values():
            assert 0.0 <= d <= 1.0

    def test_empty_pairs(self, tmp_path):
        meta = {"pairs": {}}
        meta_path = tmp_path / "metadata.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f)

        difficulties = compute_sample_difficulty(meta_path)
        assert len(difficulties) == 0
