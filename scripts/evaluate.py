#!/usr/bin/env python3
"""Eval harness - run inference on test set, compute all metrics, dump JSON report."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from landmarkdiff.evaluation import (
    EvalMetrics,
    classify_fitzpatrick_ita,
    compute_fid,
    compute_lpips,
    compute_nme,
    compute_ssim,
    compute_identity_similarity,
    evaluate_batch,
)
from landmarkdiff.landmarks import extract_landmarks
from landmarkdiff.inference import LandmarkDiffPipeline


def load_test_pairs(
    test_dir: Path,
    max_samples: int = 0,
) -> list[dict]:
    """Load test pairs. Supports both NNNNNN_input/target format and inputs/targets subdirs."""
    pairs = []

    # Format 1: Paired files with prefix
    input_files = sorted(test_dir.glob("*_input.*"))
    if input_files:
        for inp_path in input_files:
            prefix = inp_path.name.rsplit("_input", 1)[0]
            target_path = inp_path.parent / f"{prefix}_target{inp_path.suffix}"
            if target_path.exists():
                pairs.append({
                    "input_path": str(inp_path),
                    "target_path": str(target_path),
                    "id": prefix,
                })

    # Format 2: Separate directories
    if not pairs:
        inputs_dir = test_dir / "inputs"
        targets_dir = test_dir / "targets"
        if inputs_dir.exists() and targets_dir.exists():
            for inp_path in sorted(inputs_dir.iterdir()):
                if inp_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                    target_path = targets_dir / inp_path.name
                    if target_path.exists():
                        pairs.append({
                            "input_path": str(inp_path),
                            "target_path": str(target_path),
                            "id": inp_path.stem,
                        })

    # Load metadata if available
    meta_path = test_dir / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            metadata = json.load(f)
        for pair in pairs:
            if pair["id"] in metadata:
                pair.update(metadata[pair["id"]])

    if max_samples > 0:
        pairs = pairs[:max_samples]

    return pairs


def run_evaluation(
    test_dir: str,
    output_dir: str,
    checkpoint: str | None = None,
    mode: str = "tps",
    num_samples: int = 0,
    compute_fid_score: bool = False,
    compute_identity: bool = False,
    ip_adapter_scale: float = 0.6,
) -> EvalMetrics:
    """Run full eval pipeline on test pairs.
    """
    test_path = Path(test_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    (out_path / "generated").mkdir(exist_ok=True)
    (out_path / "targets").mkdir(exist_ok=True)

    pairs = load_test_pairs(test_path, num_samples)
    if not pairs:
        print(f"ERROR: No test pairs found in {test_dir}")
        sys.exit(1)

    print(f"Found {len(pairs)} test pairs")

    # Load pipeline if not TPS-only
    pipe = None
    if mode != "tps":
        pipe = LandmarkDiffPipeline(
            mode=mode,
            ip_adapter_scale=ip_adapter_scale,
        )
        if checkpoint:
            pipe.base_model_id = checkpoint
        pipe.load()

    predictions = []
    targets = []
    pred_landmarks_list = []
    target_landmarks_list = []
    procedures = []
    results_log = []

    start_time = time.time()

    for i, pair in enumerate(pairs):
        input_img = cv2.imread(pair["input_path"])
        target_img = cv2.imread(pair["target_path"])

        if input_img is None or target_img is None:
            print(f"  Skipping {pair['id']}: could not load images")
            continue

        input_img = cv2.resize(input_img, (512, 512))
        target_img = cv2.resize(target_img, (512, 512))

        procedure = pair.get("procedure", "rhinoplasty")
        intensity = pair.get("intensity", 50.0)

        try:
            if pipe is not None:
                result = pipe.generate(
                    input_img,
                    procedure=procedure,
                    intensity=intensity,
                    seed=42,
                )
                pred_img = result["output"]
            else:
                # TPS-only mode
                from landmarkdiff.inference import mask_composite
                from landmarkdiff.synthetic.tps_warp import warp_image_tps

                face = extract_landmarks(input_img)
                if face is None:
                    print(f"  Skipping {pair['id']}: no face detected")
                    continue
                from landmarkdiff.manipulation import apply_procedure_preset
                from landmarkdiff.masking import generate_surgical_mask

                manip = apply_procedure_preset(face, procedure, intensity, image_size=512)
                mask = generate_surgical_mask(face, procedure, 512, 512)
                warped = warp_image_tps(input_img, face.pixel_coords, manip.pixel_coords)
                pred_img = mask_composite(warped, input_img, mask)

        except Exception as e:
            print(f"  Skipping {pair['id']}: {e}")
            continue

        predictions.append(pred_img)
        targets.append(target_img)
        procedures.append(procedure)

        # Extract landmarks for NME
        pred_face = extract_landmarks(pred_img)
        tgt_face = extract_landmarks(target_img)
        if pred_face is not None and tgt_face is not None:
            pred_landmarks_list.append(pred_face.pixel_coords)
            target_landmarks_list.append(tgt_face.pixel_coords)
        else:
            pred_landmarks_list.append(np.zeros((478, 2)))
            target_landmarks_list.append(np.zeros((478, 2)))

        # Save generated image for FID computation
        cv2.imwrite(str(out_path / "generated" / f"{pair['id']}.png"), pred_img)
        cv2.imwrite(str(out_path / "targets" / f"{pair['id']}.png"), target_img)

        if (i + 1) % 10 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            print(f"  [{i+1}/{len(pairs)}] {rate:.1f} img/s")

    if not predictions:
        print("ERROR: No valid predictions generated")
        sys.exit(1)

    print(f"\nComputing metrics on {len(predictions)} samples...")

    # Compute batch metrics with full stratification
    metrics = evaluate_batch(
        predictions=predictions,
        targets=targets,
        pred_landmarks=pred_landmarks_list if pred_landmarks_list else None,
        target_landmarks=target_landmarks_list if target_landmarks_list else None,
        procedures=procedures,
        compute_identity=compute_identity,
    )

    # FID (requires directories)
    if compute_fid_score and len(predictions) >= 50:
        try:
            metrics.fid = compute_fid(
                str(out_path / "targets"),
                str(out_path / "generated"),
            )
        except Exception as e:
            print(f"FID computation failed: {e}")

    # Save results
    elapsed = time.time() - start_time
    report = {
        "metrics": metrics.to_dict(),
        "config": {
            "test_dir": str(test_dir),
            "checkpoint": checkpoint,
            "mode": mode,
            "num_samples": len(predictions),
            "elapsed_seconds": round(elapsed, 1),
        },
        "summary": metrics.summary(),
    }

    report_path = out_path / "eval_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n{'='*60}")
    print(metrics.summary())
    print(f"{'='*60}")
    print(f"Report saved to {report_path}")
    print(f"Total time: {elapsed:.1f}s ({len(predictions)/elapsed:.1f} img/s)")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LandmarkDiff Evaluation Harness")
    parser.add_argument("--test-dir", required=True, help="Directory with test pairs")
    parser.add_argument("--output", default="eval_results", help="Output directory")
    parser.add_argument("--checkpoint", default=None, help="ControlNet checkpoint path")
    parser.add_argument(
        "--mode", default="tps",
        choices=["tps", "controlnet", "controlnet_ip", "img2img"],
    )
    parser.add_argument("--num-samples", type=int, default=0, help="Max samples (0=all)")
    parser.add_argument("--compute-fid", action="store_true")
    parser.add_argument("--compute-identity", action="store_true")
    parser.add_argument("--ip-adapter-scale", type=float, default=0.6)
    args = parser.parse_args()

    run_evaluation(
        test_dir=args.test_dir,
        output_dir=args.output,
        checkpoint=args.checkpoint,
        mode=args.mode,
        num_samples=args.num_samples,
        compute_fid_score=args.compute_fid,
        compute_identity=args.compute_identity,
        ip_adapter_scale=args.ip_adapter_scale,
    )
