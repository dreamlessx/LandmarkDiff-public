"""Export utilities for LandmarkDiff outputs."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from landmarkdiff.landmarks import FaceLandmarks

logger = logging.getLogger(__name__)

# Default frame count and duration for progressive preview
_DEFAULT_N_FRAMES = 20
_DEFAULT_FRAME_DURATION_MS = 100


def export_before_after_gif(
    original: np.ndarray,
    prediction: np.ndarray,
    output_path: str | Path,
    duration_ms: int = 800,
    loop: int = 0,
    add_labels: bool = True,
) -> Path:
    """Export a before/after comparison as an animated GIF.

    Toggles between the original and predicted images at the given interval.

    Args:
        original: BGR original image.
        prediction: BGR predicted image (same dimensions as original).
        output_path: Path to save the GIF.
        duration_ms: Display time per frame in milliseconds.
        loop: Number of loops (0 = infinite).
        add_labels: If True, overlay "Before"/"After" text on frames.

    Returns:
        Path to the saved GIF.

    Raises:
        ImportError: If Pillow is not installed.
        ValueError: If images have different shapes.
    """
    from PIL import Image

    if original.shape != prediction.shape:
        raise ValueError(f"Image shapes must match: {original.shape} vs {prediction.shape}")

    frames = []
    for img, label in [(original, "Before"), (prediction, "After")]:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if add_labels:
            canvas = img.copy()
            h = canvas.shape[0]
            font_scale = max(0.5, h / 512.0 * 0.8)
            thickness = max(1, int(h / 512.0 * 2))
            # Black outline + white text
            cv2.putText(
                canvas,
                label,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 0, 0),
                thickness + 2,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                label,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA,
            )
            rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(rgb))

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        str(out),
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=loop,
    )
    logger.info("Saved animated GIF: %s", out)
    return out


def generate_progressive_frames(
    original: np.ndarray,
    prediction: np.ndarray,
    n_frames: int = _DEFAULT_N_FRAMES,
    add_labels: bool = True,
) -> list[np.ndarray]:
    """Generate frames that morph smoothly from original to prediction.

    Uses linear alpha blending between the original and prediction images
    to simulate a gradual intensity progression from 0 to the target.

    Args:
        original: BGR original image.
        prediction: BGR predicted image (same dimensions).
        n_frames: Number of intermediate frames (including start and end).
        add_labels: If True, overlay intensity percentage on each frame.

    Returns:
        List of BGR frames from 0% to 100% intensity.

    Raises:
        ValueError: If images have different shapes or n_frames < 2.
    """
    if original.shape != prediction.shape:
        raise ValueError(f"Image shapes must match: {original.shape} vs {prediction.shape}")
    if n_frames < 2:
        raise ValueError(f"n_frames must be >= 2, got {n_frames}")

    frames = []
    orig_f = original.astype(np.float32)
    pred_f = prediction.astype(np.float32)

    for i in range(n_frames):
        alpha = i / (n_frames - 1)
        blended = np.clip(orig_f * (1.0 - alpha) + pred_f * alpha, 0, 255).astype(np.uint8)

        if add_labels:
            h = blended.shape[0]
            font_scale = max(0.4, h / 512.0 * 0.6)
            thickness = max(1, int(h / 512.0 * 2))
            text = f"{int(alpha * 100)}%"
            cv2.putText(
                blended,
                text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 0, 0),
                thickness + 2,
                cv2.LINE_AA,
            )
            cv2.putText(
                blended,
                text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA,
            )

        frames.append(blended)

    return frames


def export_progressive_gif(
    original: np.ndarray,
    prediction: np.ndarray,
    output_path: str | Path,
    n_frames: int = _DEFAULT_N_FRAMES,
    frame_duration_ms: int = _DEFAULT_FRAME_DURATION_MS,
    loop: int = 0,
    add_labels: bool = True,
    boomerang: bool = True,
) -> Path:
    """Export a progressive intensity animation as a GIF.

    Creates a smooth morph from original to prediction, optionally
    with a boomerang (reverse) loop for continuous playback.

    Args:
        original: BGR original image.
        prediction: BGR predicted image (same dimensions).
        output_path: Path to save the GIF.
        n_frames: Number of forward frames.
        frame_duration_ms: Duration per frame in milliseconds.
        loop: Number of loops (0 = infinite).
        add_labels: If True, overlay intensity percentage.
        boomerang: If True, append reversed frames for ping-pong effect.

    Returns:
        Path to the saved GIF.
    """
    from PIL import Image

    frames_bgr = generate_progressive_frames(
        original, prediction, n_frames=n_frames, add_labels=add_labels
    )

    if boomerang and len(frames_bgr) > 2:
        frames_bgr = frames_bgr + frames_bgr[-2:0:-1]

    pil_frames = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames_bgr]

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    pil_frames[0].save(
        str(out),
        save_all=True,
        append_images=pil_frames[1:],
        duration=frame_duration_ms,
        loop=loop,
    )
    logger.info("Saved progressive GIF (%d frames): %s", len(pil_frames), out)
    return out


# ---------------------------------------------------------------------------
# 3D mesh export (OBJ / PLY)
# ---------------------------------------------------------------------------


def _get_tessellation_triangles() -> list[tuple[int, int, int]]:
    """Extract triangle faces from MediaPipe tessellation edges.

    MediaPipe provides 2556 edges that encode 852 triangles (consecutive
    triples of edges sharing exactly 3 vertices).

    Returns:
        List of (v0, v1, v2) vertex index triples.

    Raises:
        ImportError: If mediapipe is not available.
    """
    from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarksConnections

    edges = [(c.start, c.end) for c in FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION]
    triangles = []
    for i in range(0, len(edges), 3):
        verts: set[int] = set()
        for e in edges[i : i + 3]:
            verts.update(e)
        if len(verts) == 3:
            v = sorted(verts)
            triangles.append((v[0], v[1], v[2]))
    return triangles


def export_mesh_obj(
    face: FaceLandmarks,
    output_path: str | Path,
    scale: float = 100.0,
) -> Path:
    """Export face landmarks as a Wavefront OBJ mesh.

    The 478 MediaPipe landmarks become vertices and the tessellation
    edges define 852 triangle faces.

    Args:
        face: Extracted face landmarks (normalized 0-1 coordinates).
        output_path: Path to save the .obj file.
        scale: Scale factor applied to coordinates (default 100 maps
               the 0-1 range to a 100-unit bounding box).

    Returns:
        Path to the saved OBJ file.
    """
    triangles = _get_tessellation_triangles()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, "w") as f:
        f.write("# LandmarkDiff face mesh export\n")
        f.write(f"# {face.landmarks.shape[0]} vertices, {len(triangles)} faces\n\n")

        for x, y, z in face.landmarks:
            # Flip y so "up" is positive (MediaPipe y increases downward)
            f.write(f"v {x * scale:.6f} {(1.0 - y) * scale:.6f} {z * scale:.6f}\n")

        f.write("\n")
        for v0, v1, v2 in triangles:
            # OBJ faces are 1-indexed
            f.write(f"f {v0 + 1} {v1 + 1} {v2 + 1}\n")

    n_verts = face.landmarks.shape[0]
    logger.info("Saved OBJ mesh (%d verts, %d faces): %s", n_verts, len(triangles), out)
    return out


def export_mesh_ply(
    face: FaceLandmarks,
    output_path: str | Path,
    scale: float = 100.0,
    binary: bool = False,
) -> Path:
    """Export face landmarks as a Stanford PLY mesh.

    Args:
        face: Extracted face landmarks.
        output_path: Path to save the .ply file.
        scale: Scale factor applied to coordinates.
        binary: If True, write binary little-endian PLY for smaller files.

    Returns:
        Path to the saved PLY file.
    """
    import struct

    triangles = _get_tessellation_triangles()
    n_verts = face.landmarks.shape[0]
    n_faces = len(triangles)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    fmt = "binary_little_endian" if binary else "ascii"
    header = (
        "ply\n"
        f"format {fmt} 1.0\n"
        "comment LandmarkDiff face mesh export\n"
        f"element vertex {n_verts}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        f"element face {n_faces}\n"
        "property list uchar int vertex_indices\n"
        "end_header\n"
    )

    if binary:
        with open(out, "wb") as f:
            f.write(header.encode("ascii"))
            for x, y, z in face.landmarks:
                f.write(struct.pack("<fff", x * scale, (1.0 - y) * scale, z * scale))
            for v0, v1, v2 in triangles:
                f.write(struct.pack("<Biii", 3, v0, v1, v2))
    else:
        with open(out, "w") as f:
            f.write(header)
            for x, y, z in face.landmarks:
                f.write(f"{x * scale:.6f} {(1.0 - y) * scale:.6f} {z * scale:.6f}\n")
            for v0, v1, v2 in triangles:
                f.write(f"3 {v0} {v1} {v2}\n")

    logger.info("Saved PLY mesh (%d verts, %d faces, %s): %s", n_verts, n_faces, fmt, out)
    return out
