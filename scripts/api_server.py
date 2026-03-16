#!/usr/bin/env python3
"""FastAPI REST server for LandmarkDiff surgical outcome prediction.

Exposes the full pipeline via a REST API for integration with clinical
software, web applications, and evaluation tools.

Endpoints:
  POST /predict       -- Full pipeline: upload image, get predicted outcome
  POST /analyze       -- Analyze face: landmarks, Fitzpatrick type, face view
  POST /batch         -- Batch processing for multiple images
  GET  /health        -- Health check
  GET  /procedures    -- List available procedures

Usage:
    # Start server
    python scripts/api_server.py --port 8000

    # With neural post-processing
    python scripts/api_server.py --port 8000 --neural

    # Client usage
    curl -X POST http://localhost:8000/predict \
        -F "image=@photo.jpg" \
        -F "procedure=rhinoplasty" \
        -F "intensity=65"
"""

from __future__ import annotations

import argparse
import base64
import io
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from landmarkdiff.evaluation import classify_fitzpatrick_ita, compute_ssim
from landmarkdiff.inference import mask_composite
from landmarkdiff.landmarks import extract_landmarks, render_landmark_image, visualize_landmarks
from landmarkdiff.manipulation import PROCEDURE_LANDMARKS, apply_procedure_preset
from landmarkdiff.masking import generate_surgical_mask
from landmarkdiff.synthetic.tps_warp import warp_image_tps

PROCEDURES = list(PROCEDURE_LANDMARKS.keys())


def _decode_image(image_bytes: bytes) -> np.ndarray:
    """Decode an image from bytes to BGR numpy array."""
    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")
    return img


def _encode_image(img: np.ndarray, fmt: str = ".png") -> bytes:
    """Encode a BGR numpy array to image bytes."""
    _, encoded = cv2.imencode(fmt, img)
    return encoded.tobytes()


def _image_to_base64(img: np.ndarray, fmt: str = ".png") -> str:
    """Encode image to base64 string."""
    return base64.b64encode(_encode_image(img, fmt)).decode("utf-8")


def create_app(use_neural: bool = False, displacement_model_path: str | None = None):
    """Create FastAPI application."""
    try:
        from fastapi import FastAPI, File, Form, HTTPException, UploadFile
        from fastapi.responses import JSONResponse, StreamingResponse
    except ImportError:
        raise ImportError(
            "FastAPI required. Install with: pip install fastapi uvicorn python-multipart"
        )

    app = FastAPI(
        title="LandmarkDiff API",
        description="Surgical outcome prediction using landmark-conditioned diffusion",
        version="0.2.0",
    )

    @app.get("/health")
    async def health():
        return {"status": "ok", "version": "0.2.0", "neural": use_neural}

    @app.get("/procedures")
    async def procedures():
        return {
            "procedures": PROCEDURES,
            "default_intensity": 65.0,
            "intensity_range": [0, 100],
        }

    @app.post("/predict")
    async def predict(
        image: UploadFile = File(...),
        procedure: str = Form("rhinoplasty"),
        intensity: float = Form(65.0),
        return_intermediates: bool = Form(False),
        return_format: str = Form("base64"),
    ):
        """Run full surgical prediction pipeline.

        Args:
            image: Input face image (JPEG/PNG).
            procedure: Surgical procedure type.
            intensity: Intensity 0-100.
            return_intermediates: Include intermediate pipeline outputs.
            return_format: "base64" for embedded images, "binary" for raw PNG.
        """
        if procedure not in PROCEDURES:
            raise HTTPException(400, f"Unknown procedure. Choose from: {PROCEDURES}")
        if not 0 <= intensity <= 100:
            raise HTTPException(400, "Intensity must be 0-100")

        t0 = time.time()

        # Read image
        contents = await image.read()
        try:
            img = _decode_image(contents)
        except ValueError:
            raise HTTPException(400, "Could not decode image")

        img_512 = cv2.resize(img, (512, 512))

        # Extract landmarks
        face = extract_landmarks(img_512)
        if face is None:
            raise HTTPException(422, "No face detected in image")

        # Pipeline
        manip = apply_procedure_preset(
            face,
            procedure,
            intensity,
            image_size=512,
            displacement_model_path=displacement_model_path,
        )
        mask = generate_surgical_mask(face, procedure, 512, 512)
        warped = warp_image_tps(img_512, face.pixel_coords, manip.pixel_coords)
        composited = mask_composite(warped, img_512, mask)

        # Neural post-processing
        enhanced = composited
        identity_check = {}
        if use_neural:
            try:
                from landmarkdiff.postprocess import full_postprocess

                pp = full_postprocess(
                    generated=composited,
                    original=img_512,
                    mask=mask,
                    restore_mode="codeformer",
                    codeformer_fidelity=0.7,
                    use_realesrgan=True,
                    use_laplacian_blend=True,
                    sharpen_strength=0.25,
                    verify_identity=True,
                    identity_threshold=0.6,
                )
                enhanced = pp["image"]
                identity_check = pp.get("identity_check", {})
            except Exception:
                pass

        elapsed = time.time() - t0

        # Compute metrics
        ssim_val = float(compute_ssim(enhanced, img_512))
        try:
            fitz = classify_fitzpatrick_ita(img_512)
        except Exception:
            fitz = "?"

        if return_format == "binary":
            # Return the enhanced image directly as PNG
            return StreamingResponse(
                io.BytesIO(_encode_image(enhanced)),
                media_type="image/png",
                headers={"X-SSIM": str(ssim_val), "X-Fitzpatrick": fitz},
            )

        # Build response with base64 images
        result = {
            "procedure": procedure,
            "intensity": intensity,
            "ssim": ssim_val,
            "fitzpatrick": fitz,
            "identity_check": identity_check,
            "elapsed_seconds": round(elapsed, 2),
            "neural_postprocess": use_neural,
            "output": _image_to_base64(enhanced),
        }

        if return_intermediates:
            landmark_img = render_landmark_image(manip, 512, 512)
            result["original"] = _image_to_base64(img_512)
            result["warped"] = _image_to_base64(warped)
            result["composited"] = _image_to_base64(composited)
            result["mask"] = _image_to_base64((mask * 255).astype(np.uint8))
            result["conditioning"] = _image_to_base64(landmark_img)
            result["landmarks"] = _image_to_base64(visualize_landmarks(img_512, face, radius=2))

        return JSONResponse(result)

    @app.post("/analyze")
    async def analyze(image: UploadFile = File(...)):
        """Analyze a face image without running the full pipeline.

        Returns landmark positions, Fitzpatrick type, and face view.
        """
        contents = await image.read()
        try:
            img = _decode_image(contents)
        except ValueError:
            raise HTTPException(400, "Could not decode image")

        img_512 = cv2.resize(img, (512, 512))
        face = extract_landmarks(img_512)
        if face is None:
            raise HTTPException(422, "No face detected in image")

        try:
            fitz = classify_fitzpatrick_ita(img_512)
        except Exception:
            fitz = "?"

        from landmarkdiff.inference import estimate_face_view

        view_info = estimate_face_view(face)

        return {
            "detected": True,
            "num_landmarks": face.landmarks.shape[0],
            "image_size": [face.image_width, face.image_height],
            "confidence": float(face.confidence),
            "fitzpatrick": fitz,
            "view": view_info,
            "landmarks_preview": _image_to_base64(visualize_landmarks(img_512, face, radius=2)),
        }

    @app.post("/batch")
    async def batch_predict(
        images: list[UploadFile] = File(...),
        procedure: str = Form("rhinoplasty"),
        intensity: float = Form(65.0),
    ):
        """Process multiple images in a batch."""
        if procedure not in PROCEDURES:
            raise HTTPException(400, f"Unknown procedure. Choose from: {PROCEDURES}")

        results = []
        for img_file in images:
            contents = await img_file.read()
            try:
                img = _decode_image(contents)
                img_512 = cv2.resize(img, (512, 512))
                face = extract_landmarks(img_512)
                if face is None:
                    results.append(
                        {
                            "filename": img_file.filename,
                            "success": False,
                            "error": "No face detected",
                        }
                    )
                    continue

                manip = apply_procedure_preset(face, procedure, intensity, image_size=512)
                mask = generate_surgical_mask(face, procedure, 512, 512)
                warped = warp_image_tps(img_512, face.pixel_coords, manip.pixel_coords)
                composited = mask_composite(warped, img_512, mask)

                results.append(
                    {
                        "filename": img_file.filename,
                        "success": True,
                        "output": _image_to_base64(composited),
                        "ssim": float(compute_ssim(composited, img_512)),
                    }
                )
            except Exception as e:
                results.append(
                    {
                        "filename": img_file.filename,
                        "success": False,
                        "error": str(e),
                    }
                )

        return {"results": results, "total": len(results)}

    return app


def main():
    parser = argparse.ArgumentParser(description="LandmarkDiff API server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--neural", action="store_true", help="Enable neural post-processing")
    parser.add_argument(
        "--displacement-model", default=None, help="Path to data-driven displacement model"
    )
    args = parser.parse_args()

    try:
        import uvicorn
    except ImportError:
        print("ERROR: uvicorn required. Install with: pip install uvicorn")
        sys.exit(1)

    app = create_app(
        use_neural=args.neural,
        displacement_model_path=args.displacement_model,
    )
    print(f"Starting LandmarkDiff API at http://{args.host}:{args.port}")
    print(f"Neural post-processing: {args.neural}")
    print(f"Docs: http://{args.host}:{args.port}/docs")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
