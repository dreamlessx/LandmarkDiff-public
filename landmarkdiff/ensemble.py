"""Ensemble inference for improved output quality.

Generates multiple outputs with different random seeds and combines them
to reduce per-sample variance. Supports multiple aggregation strategies:
- Pixel-space averaging (fast, slight blur)
- Feature-space averaging (better quality, requires VAE encode)
- Best-of-N selection (picks output with highest identity similarity)

Usage:
    from landmarkdiff.ensemble import EnsembleInference

    ensemble = EnsembleInference(
        mode="controlnet",
        controlnet_checkpoint="checkpoints/final/controlnet_ema",
        n_samples=5,
        strategy="best_of_n",
    )
    ensemble.load()
    result = ensemble.generate(image, procedure="rhinoplasty", intensity=65)
"""

from __future__ import annotations

import cv2
import numpy as np


class EnsembleInference:
    """Multi-sample ensemble inference for LandmarkDiff.

    Generates N outputs with different seeds and combines them using
    the specified aggregation strategy.
    """

    def __init__(
        self,
        mode: str = "controlnet",
        controlnet_checkpoint: str | None = None,
        displacement_model_path: str | None = None,
        n_samples: int = 5,
        strategy: str = "best_of_n",
        base_seed: int = 42,
        **pipeline_kwargs,
    ):
        """Initialize ensemble inference.

        Args:
            mode: Pipeline mode (controlnet, img2img, tps).
            controlnet_checkpoint: Path to fine-tuned ControlNet.
            displacement_model_path: Path to displacement model.
            n_samples: Number of ensemble members.
            strategy: Aggregation strategy:
                - "pixel_average": Average in pixel space.
                - "weighted_average": Weighted by quality metrics.
                - "best_of_n": Select best by identity similarity.
                - "median": Pixel-wise median (robust to outliers).
            base_seed: Base random seed (each sample uses base_seed + i).
            **pipeline_kwargs: Additional kwargs for LandmarkDiffPipeline.
        """
        self.mode = mode
        self.controlnet_checkpoint = controlnet_checkpoint
        self.displacement_model_path = displacement_model_path
        self.n_samples = n_samples
        self.strategy = strategy
        self.base_seed = base_seed
        self.pipeline_kwargs = pipeline_kwargs
        self._pipeline = None

    def load(self) -> None:
        """Load the inference pipeline."""
        from landmarkdiff.inference import LandmarkDiffPipeline

        self._pipeline = LandmarkDiffPipeline(
            mode=self.mode,
            controlnet_checkpoint=self.controlnet_checkpoint,
            displacement_model_path=self.displacement_model_path,
            **self.pipeline_kwargs,
        )
        self._pipeline.load()

    @property
    def is_loaded(self) -> bool:
        return self._pipeline is not None and self._pipeline.is_loaded

    def generate(
        self,
        image: np.ndarray,
        procedure: str = "rhinoplasty",
        intensity: float = 50.0,
        num_inference_steps: int = 30,
        guidance_scale: float = 9.0,
        controlnet_conditioning_scale: float = 0.9,
        strength: float = 0.5,
        seed: int | None = None,
        **kwargs,
    ) -> dict:
        """Generate ensemble output.

        Returns:
            Dict with keys:
                - output: Final ensembled image (np.ndarray, BGR, uint8)
                - outputs: List of all individual outputs
                - scores: Quality scores for each sample
                - selected_idx: Index of selected sample (for best_of_n)
                - strategy: Aggregation strategy used
                - n_samples: Number of ensemble members
        """
        if not self.is_loaded:
            raise RuntimeError("Pipeline not loaded. Call load() first.")

        base = seed if seed is not None else self.base_seed
        outputs = []
        results = []

        # Generate N samples
        for i in range(self.n_samples):
            sample_seed = base + i
            result = self._pipeline.generate(
                image,
                procedure=procedure,
                intensity=intensity,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                controlnet_conditioning_scale=controlnet_conditioning_scale,
                strength=strength,
                seed=sample_seed,
                **kwargs,
            )
            outputs.append(result["output"])
            results.append(result)

        # Aggregate
        if self.strategy == "pixel_average":
            final = self._pixel_average(outputs)
            scores = [1.0 / self.n_samples] * self.n_samples
            selected_idx = -1

        elif self.strategy == "weighted_average":
            final, scores = self._weighted_average(outputs, image)
            selected_idx = -1

        elif self.strategy == "best_of_n":
            final, scores, selected_idx = self._best_of_n(outputs, image)

        elif self.strategy == "median":
            final = self._pixel_median(outputs)
            scores = [1.0 / self.n_samples] * self.n_samples
            selected_idx = -1

        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        # Copy metadata from best result
        best_idx = selected_idx if selected_idx >= 0 else 0
        ensemble_result = dict(results[best_idx])
        ensemble_result.update(
            {
                "output": final,
                "outputs": outputs,
                "scores": scores,
                "selected_idx": selected_idx,
                "strategy": self.strategy,
                "n_samples": self.n_samples,
            }
        )

        return ensemble_result

    def _pixel_average(self, outputs: list[np.ndarray]) -> np.ndarray:
        """Simple pixel-space averaging."""
        stacked = np.stack(outputs, axis=0).astype(np.float32)
        return np.clip(stacked.mean(axis=0), 0, 255).astype(np.uint8)

    def _pixel_median(self, outputs: list[np.ndarray]) -> np.ndarray:
        """Pixel-wise median (robust to outliers)."""
        stacked = np.stack(outputs, axis=0)
        return np.median(stacked, axis=0).astype(np.uint8)

    def _weighted_average(
        self,
        outputs: list[np.ndarray],
        reference: np.ndarray,
    ) -> tuple[np.ndarray, list[float]]:
        """Quality-weighted averaging using SSIM as weight."""
        from landmarkdiff.evaluation import compute_ssim

        # Compute SSIM of each output to reference
        scores = []
        for output in outputs:
            ssim = compute_ssim(output, reference)
            scores.append(float(ssim))

        # Normalize to weights (higher SSIM = higher weight)
        total = sum(scores) or 1.0
        weights = [s / total for s in scores]

        # Weighted average
        result = np.zeros_like(outputs[0], dtype=np.float32)
        for output, weight in zip(outputs, weights, strict=False):
            result += output.astype(np.float32) * weight

        return np.clip(result, 0, 255).astype(np.uint8), scores

    def _best_of_n(
        self,
        outputs: list[np.ndarray],
        reference: np.ndarray,
    ) -> tuple[np.ndarray, list[float], int]:
        """Select the output with highest identity similarity to reference."""
        from landmarkdiff.evaluation import compute_identity_similarity

        scores = []
        for output in outputs:
            sim = compute_identity_similarity(output, reference)
            scores.append(float(sim))

        best_idx = int(np.argmax(scores))
        return outputs[best_idx], scores, best_idx


def ensemble_inference(
    image_path: str,
    procedure: str = "rhinoplasty",
    intensity: float = 65.0,
    output_dir: str = "ensemble_output",
    n_samples: int = 5,
    strategy: str = "best_of_n",
    mode: str = "tps",
    controlnet_checkpoint: str | None = None,
    displacement_model_path: str | None = None,
    seed: int = 42,
) -> None:
    """CLI entry point for ensemble inference."""
    from pathlib import Path

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(image_path)
    if image is None:
        print(f"ERROR: Cannot read image: {image_path}")
        return

    image = cv2.resize(image, (512, 512))

    ensemble = EnsembleInference(
        mode=mode,
        controlnet_checkpoint=controlnet_checkpoint,
        displacement_model_path=displacement_model_path,
        n_samples=n_samples,
        strategy=strategy,
        base_seed=seed,
    )
    ensemble.load()

    print(f"Generating ensemble ({n_samples} samples, strategy={strategy})...")
    result = ensemble.generate(
        image,
        procedure=procedure,
        intensity=intensity,
        seed=seed,
    )

    # Save outputs
    cv2.imwrite(str(out / "ensemble_output.png"), result["output"])
    cv2.imwrite(str(out / "original.png"), image)

    # Save individual samples
    for i, output in enumerate(result["outputs"]):
        cv2.imwrite(str(out / f"sample_{i:02d}.png"), output)
        score = result["scores"][i]
        print(
            f"  Sample {i}: score={score:.4f}"
            + (" <-- selected" if i == result.get("selected_idx") else "")
        )

    # Comparison grid
    panels = [image] + result["outputs"] + [result["output"]]
    # Resize to 256 for compact grid
    panels_small = [cv2.resize(p, (256, 256)) for p in panels]
    grid = np.hstack(panels_small)
    cv2.imwrite(str(out / "comparison_grid.png"), grid)

    print(f"\nEnsemble output saved: {out / 'ensemble_output.png'}")
    if result.get("selected_idx", -1) >= 0:
        print(
            f"Selected sample: {result['selected_idx']} "
            f"(score={result['scores'][result['selected_idx']]:.4f})"
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ensemble inference")
    parser.add_argument("image", help="Input face image")
    parser.add_argument("--procedure", default="rhinoplasty")
    parser.add_argument("--intensity", type=float, default=65.0)
    parser.add_argument("--output", default="ensemble_output")
    parser.add_argument("--n_samples", type=int, default=5)
    parser.add_argument(
        "--strategy",
        default="best_of_n",
        choices=["pixel_average", "weighted_average", "best_of_n", "median"],
    )
    parser.add_argument("--mode", default="tps", choices=["controlnet", "img2img", "tps"])
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--displacement-model", default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ensemble_inference(
        args.image,
        args.procedure,
        args.intensity,
        args.output,
        args.n_samples,
        args.strategy,
        args.mode,
        args.checkpoint,
        args.displacement_model,
        args.seed,
    )
