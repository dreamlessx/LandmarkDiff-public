"""Facial landmark extraction using MediaPipe Face Mesh v2."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

logger = logging.getLogger(__name__)

# Region color map for visualization (BGR)
REGION_COLORS: dict[str, tuple[int, int, int]] = {
    "jawline": (255, 255, 255),  # white
    "eyebrow_left": (0, 255, 0),  # green
    "eyebrow_right": (0, 255, 0),
    "eye_left": (255, 255, 0),  # cyan
    "eye_right": (255, 255, 0),
    "nose": (0, 255, 255),  # yellow
    "lips": (0, 0, 255),  # red
    "iris_left": (255, 0, 255),  # magenta
    "iris_right": (255, 0, 255),
}

# MediaPipe landmark index groups by anatomical region
LANDMARK_REGIONS: dict[str, list[int]] = {
    "jawline": [
        10,
        338,
        297,
        332,
        284,
        251,
        389,
        356,
        454,
        323,
        361,
        288,
        397,
        365,
        379,
        378,
        400,
        377,
        152,
        148,
        176,
        149,
        150,
        136,
        172,
        58,
        132,
        93,
        234,
        127,
        162,
        21,
        54,
        103,
        67,
        109,
    ],
    "eye_left": [
        33,
        7,
        163,
        144,
        145,
        153,
        154,
        155,
        133,
        173,
        157,
        158,
        159,
        160,
        161,
        246,
    ],
    "eye_right": [
        362,
        382,
        381,
        380,
        374,
        373,
        390,
        249,
        263,
        466,
        388,
        387,
        386,
        385,
        384,
        398,
    ],
    "eyebrow_left": [70, 63, 105, 66, 107, 55, 65, 52, 53, 46],
    "eyebrow_right": [300, 293, 334, 296, 336, 285, 295, 282, 283, 276],
    "nose": [
        1,
        2,
        4,
        5,
        6,
        19,
        94,
        141,
        168,
        195,
        197,
        236,
        240,
        274,
        275,
        278,
        279,
        294,
        326,
        327,
        360,
        363,
        370,
        456,
        460,
    ],
    "lips": [
        61,
        146,
        91,
        181,
        84,
        17,
        314,
        405,
        321,
        375,
        291,
        308,
        324,
        318,
        402,
        317,
        14,
        87,
        178,
        88,
        95,
        78,
    ],
    "iris_left": [468, 469, 470, 471, 472],
    "iris_right": [473, 474, 475, 476, 477],
}


@dataclass(frozen=True)
class FaceLandmarks:
    """Extracted facial landmarks with metadata."""

    landmarks: np.ndarray  # (478, 3) normalized (x, y, z)
    image_width: int
    image_height: int
    confidence: float

    @property
    def pixel_coords(self) -> np.ndarray:
        """Convert normalized landmarks to pixel coordinates (478, 2).

        Coordinates are clamped to valid image bounds so that extreme
        head poses do not produce out-of-range indices.
        """
        coords = self.landmarks[:, :2].copy()
        coords[:, 0] *= self.image_width
        coords[:, 1] *= self.image_height
        coords[:, 0] = np.clip(coords[:, 0], 0, self.image_width - 1)
        coords[:, 1] = np.clip(coords[:, 1], 0, self.image_height - 1)
        return coords

    def pixel_coords_at(self, width: int, height: int) -> np.ndarray:
        """Convert normalized landmarks to pixel coordinates at a given size.

        Use this when the image has been resized after landmark extraction.
        Coordinates are clamped to [0, width-1] x [0, height-1].
        """
        coords = self.landmarks[:, :2].copy()
        coords[:, 0] *= width
        coords[:, 1] *= height
        coords[:, 0] = np.clip(coords[:, 0], 0, width - 1)
        coords[:, 1] = np.clip(coords[:, 1], 0, height - 1)
        return coords

    def rescale(self, width: int, height: int) -> FaceLandmarks:
        """Return a copy with updated image dimensions.

        Landmarks stay in normalized [0,1] space; only the stored
        width/height change, so ``pixel_coords`` returns values at
        the new resolution.
        """
        return FaceLandmarks(
            landmarks=self.landmarks.copy(),
            image_width=width,
            image_height=height,
            confidence=self.confidence,
        )

    def get_region(self, region: str) -> np.ndarray:
        """Get landmark indices for a named region."""
        indices = LANDMARK_REGIONS.get(region, [])
        return self.landmarks[indices]


def extract_landmarks(
    image: np.ndarray,
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
) -> FaceLandmarks | None:
    """Extract 478 facial landmarks from an image using MediaPipe Face Mesh.

    Args:
        image: BGR image as numpy array.
        min_detection_confidence: Minimum face detection confidence.
        min_tracking_confidence: Minimum landmark tracking confidence.

    Returns:
        FaceLandmarks if a face is detected, None otherwise.
    """
    h, w = image.shape[:2]
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Try new Tasks API first (mediapipe >= 0.10.20), fall back to legacy solutions API
    try:
        landmarks, confidence = _extract_tasks_api(rgb, min_detection_confidence)
    except Exception:
        logger.debug("Tasks API unavailable, trying Solutions API", exc_info=True)
        try:
            landmarks, confidence = _extract_solutions_api(
                rgb, min_detection_confidence, min_tracking_confidence
            )
        except Exception:
            logger.debug("Both MediaPipe APIs failed", exc_info=True)
            return None

    if landmarks is None:
        return None

    return FaceLandmarks(
        landmarks=landmarks,
        image_width=w,
        image_height=h,
        confidence=confidence,
    )


def _extract_tasks_api(
    rgb: np.ndarray,
    min_confidence: float,
) -> tuple[np.ndarray | None, float]:
    """Extract landmarks using MediaPipe Tasks API (>= 0.10.20)."""
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    RunningMode = mp.tasks.vision.RunningMode
    BaseOptions = mp.tasks.BaseOptions
    import tempfile
    import urllib.request

    # Download model if not cached
    model_path = Path(tempfile.gettempdir()) / "face_landmarker_v2_with_blendshapes.task"
    if not model_path.exists():
        url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        urllib.request.urlretrieve(url, str(model_path))

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=min_confidence,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )

    with FaceLandmarker.create_from_options(options) as landmarker:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect(mp_image)

    if not result.face_landmarks:
        return None, 0.0

    face_lms = result.face_landmarks[0]
    landmarks = np.array(
        [(lm.x, lm.y, lm.z) for lm in face_lms],
        dtype=np.float32,
    )

    # MediaPipe Tasks API doesn't expose per-landmark detection confidence;
    # return 1.0 to indicate successful detection
    return landmarks, 1.0


def _extract_solutions_api(
    rgb: np.ndarray,
    min_detection_confidence: float,
    min_tracking_confidence: float,
) -> tuple[np.ndarray | None, float]:
    """Extract landmarks using legacy MediaPipe Solutions API."""
    with mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    ) as face_mesh:
        results = face_mesh.process(rgb)

    if not results.multi_face_landmarks:
        return None, 0.0

    face = results.multi_face_landmarks[0]
    landmarks = np.array(
        [(lm.x, lm.y, lm.z) for lm in face.landmark],
        dtype=np.float32,
    )
    # Legacy API doesn't expose detection confidence; return 1.0 for success
    return landmarks, 1.0


def visualize_landmarks(
    image: np.ndarray,
    face: FaceLandmarks,
    radius: int = 1,
    draw_regions: bool = True,
) -> np.ndarray:
    """Draw colored landmark dots on image by anatomical region.

    Args:
        image: BGR image to draw on (will be copied).
        face: Extracted face landmarks.
        radius: Dot radius in pixels.
        draw_regions: If True, color by region. Otherwise all white.

    Returns:
        Annotated image copy.
    """
    canvas = image.copy()
    coords = face.pixel_coords

    if draw_regions:
        # Build index -> color mapping
        idx_to_color: dict[int, tuple[int, int, int]] = {}
        for region, indices in LANDMARK_REGIONS.items():
            color = REGION_COLORS.get(region, (255, 255, 255))
            for idx in indices:
                idx_to_color[idx] = color

        for i, (x, y) in enumerate(coords):
            color = idx_to_color.get(i, (128, 128, 128))
            cv2.circle(canvas, (int(x), int(y)), radius, color, -1)
    else:
        for x, y in coords:
            cv2.circle(canvas, (int(x), int(y)), radius, (255, 255, 255), -1)

    return canvas


def render_landmark_image(
    face: FaceLandmarks,
    width: int | None = None,
    height: int | None = None,
    radius: int = 2,
) -> np.ndarray:
    """Render MediaPipe face mesh tessellation on black canvas.

    Draws the full 2556-edge tessellation mesh that CrucibleAI/ControlNetMediaPipeFace
    was pre-trained on. This is critical -- the ControlNet expects dense triangulated
    wireframes, not sparse dots.

    Falls back to colored dots if tessellation connections aren't available.

    Args:
        face: Extracted face landmarks.
        width: Canvas width (defaults to face.image_width).
        height: Canvas height (defaults to face.image_height).
        radius: Dot radius (used for key landmark dots overlay).

    Returns:
        BGR image with face mesh on black background.
    """
    w = width or face.image_width
    h = height or face.image_height
    canvas = np.zeros((h, w, 3), dtype=np.uint8)

    coords = face.landmarks[:, :2].copy()
    coords[:, 0] *= w
    coords[:, 1] *= h
    pts = coords.astype(np.int32)

    # Draw tessellation mesh (what CrucibleAI ControlNet expects)
    try:
        from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarksConnections

        tessellation = FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION
        contours = FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS

        # Draw tessellation edges (thin, gray-white)
        for conn in tessellation:
            p1 = tuple(pts[conn.start])
            p2 = tuple(pts[conn.end])
            cv2.line(canvas, p1, p2, (192, 192, 192), 1, cv2.LINE_AA)

        # Draw contour edges on top (brighter, key features)
        for conn in contours:
            p1 = tuple(pts[conn.start])
            p2 = tuple(pts[conn.end])
            cv2.line(canvas, p1, p2, (255, 255, 255), 1, cv2.LINE_AA)

        # Supplement missing nose bridge edges (landmarks 6-8-9-168)
        for a, b in [(8, 9), (8, 168), (8, 6), (9, 168)]:
            cv2.line(canvas, tuple(pts[a]), tuple(pts[b]), (192, 192, 192), 1, cv2.LINE_AA)

    except (ImportError, AttributeError):
        # Fallback: draw colored dots if tessellation not available
        idx_to_color: dict[int, tuple[int, int, int]] = {}
        for region, indices in LANDMARK_REGIONS.items():
            color = REGION_COLORS.get(region, (128, 128, 128))
            for idx in indices:
                idx_to_color[idx] = color

        for i, (x, y) in enumerate(coords):
            color = idx_to_color.get(i, (128, 128, 128))
            cv2.circle(canvas, (int(x), int(y)), radius, color, -1)

    return canvas


def load_image(path: str | Path) -> np.ndarray:
    """Load an image from disk as BGR numpy array."""
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    return img
