"""Hyperparameter search utilities for systematic ControlNet tuning.

Supports grid search, random search, and Bayesian-inspired adaptive search
over training hyperparameters. Generates YAML configs for each trial and
tracks results for comparison.

Usage:
    from landmarkdiff.hyperparam import HyperparamSearch, SearchSpace

    space = SearchSpace()
    space.add_float("learning_rate", 1e-6, 1e-4, log_scale=True)
    space.add_choice("optimizer", ["adamw", "adam8bit"])
    space.add_int("batch_size", 2, 8, step=2)

    search = HyperparamSearch(space, output_dir="hp_search")
    for trial in search.generate_trials(strategy="random", n_trials=20):
        print(trial.config)
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _to_native(val: Any) -> Any:
    """Convert numpy/non-standard types to native Python for YAML serialization."""
    if hasattr(val, "item"):  # numpy scalar
        return val.item()
    return val


@dataclass
class ParamSpec:
    """Specification for a single hyperparameter."""

    name: str
    param_type: str  # "float", "int", "choice"
    low: float | None = None
    high: float | None = None
    step: float | None = None
    log_scale: bool = False
    choices: list[Any] | None = None

    def sample(self, rng) -> Any:
        """Sample a value from this parameter spec."""
        if self.param_type == "choice":
            return rng.choice(self.choices)
        elif self.param_type == "float":
            if self.log_scale:
                log_low = math.log(self.low)
                log_high = math.log(self.high)
                return float(math.exp(rng.uniform(log_low, log_high)))
            return float(rng.uniform(self.low, self.high))
        elif self.param_type == "int":
            if self.step and self.step > 1:
                n_steps = int((self.high - self.low) / self.step) + 1
                idx = rng.integers(0, n_steps)
                return int(self.low + idx * self.step)
            return int(rng.integers(int(self.low), int(self.high) + 1))
        raise ValueError(f"Unknown param type: {self.param_type}")

    def grid_values(self, n_points: int = 5) -> list[Any]:
        """Generate grid values for this parameter."""
        if self.param_type == "choice":
            return list(self.choices)
        elif self.param_type == "int":
            if self.step and self.step > 1:
                vals = []
                v = self.low
                while v <= self.high:
                    vals.append(int(v))
                    v += self.step
                return vals
            return list(range(int(self.low), int(self.high) + 1))
        elif self.param_type == "float":
            if self.log_scale:
                log_low = math.log(self.low)
                log_high = math.log(self.high)
                return [
                    float(math.exp(log_low + i * (log_high - log_low) / (n_points - 1)))
                    for i in range(n_points)
                ]
            return [
                float(self.low + i * (self.high - self.low) / (n_points - 1))
                for i in range(n_points)
            ]
        return []


class SearchSpace:
    """Define the hyperparameter search space."""

    def __init__(self) -> None:
        self.params: dict[str, ParamSpec] = {}

    def add_float(
        self,
        name: str,
        low: float,
        high: float,
        log_scale: bool = False,
    ) -> SearchSpace:
        """Add a continuous float parameter."""
        self.params[name] = ParamSpec(
            name=name,
            param_type="float",
            low=low,
            high=high,
            log_scale=log_scale,
        )
        return self

    def add_int(
        self,
        name: str,
        low: int,
        high: int,
        step: int = 1,
    ) -> SearchSpace:
        """Add an integer parameter."""
        self.params[name] = ParamSpec(
            name=name,
            param_type="int",
            low=low,
            high=high,
            step=step,
        )
        return self

    def add_choice(self, name: str, choices: list[Any]) -> SearchSpace:
        """Add a categorical parameter."""
        self.params[name] = ParamSpec(
            name=name,
            param_type="choice",
            choices=choices,
        )
        return self

    def __len__(self) -> int:
        return len(self.params)

    def __contains__(self, name: str) -> bool:
        return name in self.params


@dataclass
class Trial:
    """A single hyperparameter trial."""

    trial_id: str
    config: dict[str, Any]
    result: dict[str, float] = field(default_factory=dict)
    status: str = "pending"  # pending, running, completed, failed

    @property
    def config_hash(self) -> str:
        """Short hash of the config for deduplication."""
        s = json.dumps(self.config, sort_keys=True, default=str)
        return hashlib.md5(s.encode()).hexdigest()[:8]


class HyperparamSearch:
    """Hyperparameter search engine.

    Args:
        space: Search space definition.
        output_dir: Directory to save trial configs and results.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        space: SearchSpace,
        output_dir: str | Path = "hp_search",
        seed: int = 42,
    ) -> None:
        self.space = space
        self.output_dir = Path(output_dir)
        self.seed = seed
        self.trials: list[Trial] = []

    def generate_trials(
        self,
        strategy: str = "random",
        n_trials: int = 20,
        grid_points: int = 5,
    ) -> list[Trial]:
        """Generate trial configurations.

        Args:
            strategy: "random" or "grid".
            n_trials: Number of trials for random search.
            grid_points: Points per continuous dimension for grid search.

        Returns:
            List of Trial objects with configs.
        """
        if strategy == "grid":
            trials = self._grid_search(grid_points)
        elif strategy == "random":
            trials = self._random_search(n_trials)
        else:
            raise ValueError(f"Unknown strategy: {strategy}. Use 'random' or 'grid'.")

        self.trials.extend(trials)
        return trials

    def _random_search(self, n_trials: int) -> list[Trial]:
        """Generate random trial configs."""
        import numpy as np

        rng = np.random.default_rng(self.seed)
        seen_hashes: set[str] = set()
        trials: list[Trial] = []

        max_attempts = n_trials * 10
        attempts = 0
        while len(trials) < n_trials and attempts < max_attempts:
            attempts += 1
            config = {name: spec.sample(rng) for name, spec in self.space.params.items()}
            trial = Trial(
                trial_id=f"trial_{len(trials):04d}",
                config=config,
            )
            if trial.config_hash not in seen_hashes:
                seen_hashes.add(trial.config_hash)
                trials.append(trial)

        return trials

    def _grid_search(self, grid_points: int) -> list[Trial]:
        """Generate grid search configs."""
        import itertools

        param_names = list(self.space.params.keys())
        param_values = [self.space.params[name].grid_values(grid_points) for name in param_names]

        trials = []
        for combo in itertools.product(*param_values):
            config = dict(zip(param_names, combo, strict=False))
            trial = Trial(
                trial_id=f"trial_{len(trials):04d}",
                config=config,
            )
            trials.append(trial)

        return trials

    def record_result(
        self,
        trial_id: str,
        metrics: dict[str, float],
    ) -> None:
        """Record results for a trial."""
        for trial in self.trials:
            if trial.trial_id == trial_id:
                trial.result = metrics
                trial.status = "completed"
                return
        raise KeyError(f"Trial {trial_id} not found")

    def best_trial(
        self,
        metric: str = "loss",
        lower_is_better: bool = True,
    ) -> Trial | None:
        """Get the best completed trial by a metric."""
        completed = [t for t in self.trials if t.status == "completed" and metric in t.result]
        if not completed:
            return None
        return (min if lower_is_better else max)(completed, key=lambda t: t.result[metric])

    def save_configs(self) -> Path:
        """Save all trial configs as YAML files.

        Returns:
            Output directory path.
        """
        import yaml

        self.output_dir.mkdir(parents=True, exist_ok=True)
        for trial in self.trials:
            cfg_path = self.output_dir / f"{trial.trial_id}.yaml"
            # Convert numpy types to native Python for YAML serialization
            native_config = {k: _to_native(v) for k, v in trial.config.items()}
            with open(cfg_path, "w") as f:
                yaml.safe_dump(
                    {"trial_id": trial.trial_id, **native_config},
                    f,
                    default_flow_style=False,
                )

        # Save summary index
        index = {
            "seed": self.seed,
            "n_trials": len(self.trials),
            "params": {
                name: {
                    "type": spec.param_type,
                    "low": spec.low,
                    "high": spec.high,
                    "choices": spec.choices,
                    "log_scale": spec.log_scale,
                }
                for name, spec in self.space.params.items()
            },
        }
        with open(self.output_dir / "search_index.json", "w") as f:
            json.dump(index, f, indent=2, default=str)

        return self.output_dir

    def results_table(self) -> str:
        """Format results as a text table."""
        completed = [t for t in self.trials if t.status == "completed"]
        if not completed:
            return "No completed trials."

        # Collect all metric names
        metric_names = sorted(set().union(*(t.result.keys() for t in completed)))
        param_names = sorted(self.space.params.keys())

        # Header
        cols = ["Trial"] + param_names + metric_names
        lines = [" | ".join(f"{c:>12s}" for c in cols)]
        lines.append("-" * len(lines[0]))

        # Rows
        for trial in completed:
            parts = [f"{trial.trial_id:>12s}"]
            for p in param_names:
                val = trial.config.get(p, "")
                if isinstance(val, float):
                    parts.append(f"{val:>12.6f}")
                else:
                    parts.append(f"{str(val):>12s}")
            for m in metric_names:
                val = trial.result.get(m, float("nan"))
                parts.append(f"{val:>12.4f}")
            lines.append(" | ".join(parts))

        return "\n".join(lines)
