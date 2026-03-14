"""Tests for landmarkdiff.hyperparam."""

from __future__ import annotations

import json

import numpy as np
import pytest

from landmarkdiff.hyperparam import (
    HyperparamSearch,
    ParamSpec,
    SearchSpace,
    Trial,
)

# ---------------------------------------------------------------------------
# ParamSpec
# ---------------------------------------------------------------------------


class TestParamSpec:
    def test_float_sample(self):
        spec = ParamSpec(name="lr", param_type="float", low=1e-6, high=1e-4)
        rng = np.random.default_rng(42)
        val = spec.sample(rng)
        assert 1e-6 <= val <= 1e-4

    def test_float_log_sample(self):
        spec = ParamSpec(name="lr", param_type="float", low=1e-6, high=1e-3, log_scale=True)
        rng = np.random.default_rng(42)
        vals = [spec.sample(rng) for _ in range(100)]
        assert all(1e-6 <= v <= 1e-3 for v in vals)

    def test_int_sample(self):
        spec = ParamSpec(name="bs", param_type="int", low=2, high=16)
        rng = np.random.default_rng(42)
        val = spec.sample(rng)
        assert 2 <= val <= 16
        assert isinstance(val, int)

    def test_int_step_sample(self):
        spec = ParamSpec(name="bs", param_type="int", low=2, high=8, step=2)
        rng = np.random.default_rng(42)
        vals = {spec.sample(rng) for _ in range(100)}
        assert vals.issubset({2, 4, 6, 8})

    def test_choice_sample(self):
        spec = ParamSpec(name="opt", param_type="choice", choices=["adamw", "sgd"])
        rng = np.random.default_rng(42)
        val = spec.sample(rng)
        assert val in ["adamw", "sgd"]

    def test_grid_values_float(self):
        spec = ParamSpec(name="lr", param_type="float", low=0.0, high=1.0)
        vals = spec.grid_values(n_points=5)
        assert len(vals) == 5
        assert vals[0] == pytest.approx(0.0)
        assert vals[-1] == pytest.approx(1.0)

    def test_grid_values_log(self):
        spec = ParamSpec(name="lr", param_type="float", low=1e-4, high=1e-2, log_scale=True)
        vals = spec.grid_values(n_points=3)
        assert len(vals) == 3
        assert vals[0] == pytest.approx(1e-4, rel=1e-3)
        assert vals[-1] == pytest.approx(1e-2, rel=1e-3)

    def test_grid_values_int_step(self):
        spec = ParamSpec(name="bs", param_type="int", low=2, high=8, step=2)
        vals = spec.grid_values()
        assert vals == [2, 4, 6, 8]

    def test_grid_values_choice(self):
        spec = ParamSpec(name="opt", param_type="choice", choices=["a", "b", "c"])
        vals = spec.grid_values()
        assert vals == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# SearchSpace
# ---------------------------------------------------------------------------


class TestSearchSpace:
    def test_add_float(self):
        space = SearchSpace()
        space.add_float("lr", 1e-6, 1e-4, log_scale=True)
        assert "lr" in space
        assert len(space) == 1

    def test_add_int(self):
        space = SearchSpace()
        space.add_int("batch_size", 2, 16, step=2)
        assert "batch_size" in space

    def test_add_choice(self):
        space = SearchSpace()
        space.add_choice("optimizer", ["adamw", "sgd"])
        assert "optimizer" in space

    def test_chaining(self):
        space = (
            SearchSpace()
            .add_float("lr", 1e-6, 1e-4)
            .add_int("bs", 2, 8)
            .add_choice("opt", ["adamw"])
        )
        assert len(space) == 3


# ---------------------------------------------------------------------------
# Trial
# ---------------------------------------------------------------------------


class TestTrial:
    def test_config_hash_deterministic(self):
        t = Trial(trial_id="t1", config={"lr": 0.001, "bs": 4})
        h1 = t.config_hash
        h2 = t.config_hash
        assert h1 == h2
        assert len(h1) == 8

    def test_different_configs_different_hash(self):
        t1 = Trial(trial_id="t1", config={"lr": 0.001})
        t2 = Trial(trial_id="t2", config={"lr": 0.002})
        assert t1.config_hash != t2.config_hash

    def test_default_status(self):
        t = Trial(trial_id="t1", config={})
        assert t.status == "pending"
        assert t.result == {}


# ---------------------------------------------------------------------------
# HyperparamSearch — random
# ---------------------------------------------------------------------------


class TestRandomSearch:
    def _make_space(self):
        return (
            SearchSpace()
            .add_float("learning_rate", 1e-6, 1e-4, log_scale=True)
            .add_choice("optimizer", ["adamw", "adam8bit"])
            .add_int("batch_size", 2, 8, step=2)
        )

    def test_generates_n_trials(self):
        search = HyperparamSearch(self._make_space())
        trials = search.generate_trials(strategy="random", n_trials=10)
        assert len(trials) == 10

    def test_trials_are_unique(self):
        search = HyperparamSearch(self._make_space())
        trials = search.generate_trials(strategy="random", n_trials=10)
        hashes = {t.config_hash for t in trials}
        assert len(hashes) == 10

    def test_values_in_range(self):
        search = HyperparamSearch(self._make_space())
        trials = search.generate_trials(strategy="random", n_trials=50)
        for t in trials:
            assert 1e-6 <= t.config["learning_rate"] <= 1e-4
            assert t.config["optimizer"] in ["adamw", "adam8bit"]
            assert t.config["batch_size"] in [2, 4, 6, 8]

    def test_deterministic_with_seed(self):
        s1 = HyperparamSearch(self._make_space(), seed=123)
        s2 = HyperparamSearch(self._make_space(), seed=123)
        t1 = s1.generate_trials(strategy="random", n_trials=5)
        t2 = s2.generate_trials(strategy="random", n_trials=5)
        for a, b in zip(t1, t2, strict=False):
            assert a.config == b.config


# ---------------------------------------------------------------------------
# HyperparamSearch — grid
# ---------------------------------------------------------------------------


class TestGridSearch:
    def test_grid_count(self):
        space = SearchSpace().add_choice("a", [1, 2]).add_choice("b", ["x", "y", "z"])
        search = HyperparamSearch(space)
        trials = search.generate_trials(strategy="grid")
        assert len(trials) == 6  # 2 * 3

    def test_grid_coverage(self):
        space = SearchSpace().add_choice("opt", ["adamw", "sgd"])
        search = HyperparamSearch(space)
        trials = search.generate_trials(strategy="grid")
        opts = {t.config["opt"] for t in trials}
        assert opts == {"adamw", "sgd"}


# ---------------------------------------------------------------------------
# Result tracking
# ---------------------------------------------------------------------------


class TestResults:
    def _make_search(self):
        space = SearchSpace().add_float("lr", 1e-5, 1e-3)
        search = HyperparamSearch(space)
        search.generate_trials(strategy="random", n_trials=3)
        return search

    def test_record_result(self):
        search = self._make_search()
        tid = search.trials[0].trial_id
        search.record_result(tid, {"loss": 0.5, "ssim": 0.85})
        assert search.trials[0].status == "completed"
        assert search.trials[0].result["loss"] == 0.5

    def test_record_unknown_trial_raises(self):
        search = self._make_search()
        with pytest.raises(KeyError):
            search.record_result("nonexistent", {"loss": 0.1})

    def test_best_trial(self):
        search = self._make_search()
        search.record_result(search.trials[0].trial_id, {"loss": 0.5})
        search.record_result(search.trials[1].trial_id, {"loss": 0.3})
        search.record_result(search.trials[2].trial_id, {"loss": 0.7})
        best = search.best_trial(metric="loss", lower_is_better=True)
        assert best.result["loss"] == 0.3

    def test_best_trial_higher_better(self):
        search = self._make_search()
        search.record_result(search.trials[0].trial_id, {"ssim": 0.8})
        search.record_result(search.trials[1].trial_id, {"ssim": 0.9})
        best = search.best_trial(metric="ssim", lower_is_better=False)
        assert best.result["ssim"] == 0.9

    def test_best_trial_no_completed(self):
        search = self._make_search()
        assert search.best_trial() is None


# ---------------------------------------------------------------------------
# Save and table
# ---------------------------------------------------------------------------


class TestSaveAndTable:
    def test_save_configs(self, tmp_path):
        space = SearchSpace().add_float("lr", 1e-5, 1e-3)
        search = HyperparamSearch(space, output_dir=tmp_path / "hp")
        search.generate_trials(strategy="random", n_trials=3)
        out = search.save_configs()
        assert out.exists()
        assert (out / "search_index.json").exists()
        assert len(list(out.glob("trial_*.yaml"))) == 3

    def test_save_index_content(self, tmp_path):
        space = SearchSpace().add_float("lr", 1e-5, 1e-3).add_choice("opt", ["a"])
        search = HyperparamSearch(space, output_dir=tmp_path / "hp", seed=99)
        search.generate_trials(strategy="random", n_trials=1)
        search.save_configs()
        with open(tmp_path / "hp" / "search_index.json") as f:
            idx = json.load(f)
        assert idx["seed"] == 99
        assert "lr" in idx["params"]

    def test_results_table_no_completed(self):
        space = SearchSpace().add_float("lr", 1e-5, 1e-3)
        search = HyperparamSearch(space)
        search.generate_trials(strategy="random", n_trials=2)
        assert "No completed" in search.results_table()

    def test_results_table_with_data(self):
        space = SearchSpace().add_float("lr", 1e-5, 1e-3)
        search = HyperparamSearch(space)
        search.generate_trials(strategy="random", n_trials=2)
        search.record_result(search.trials[0].trial_id, {"loss": 0.42})
        table = search.results_table()
        assert "loss" in table
        assert "0.42" in table

    def test_invalid_strategy(self):
        space = SearchSpace().add_float("lr", 1e-5, 1e-3)
        search = HyperparamSearch(space)
        with pytest.raises(ValueError, match="Unknown strategy"):
            search.generate_trials(strategy="bayesian")
