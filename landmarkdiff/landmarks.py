"""Facial landmark extraction using MediaPipe Face Mesh v2."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

logger = logging.getLogger(__name__)

# Lock for thread-safe model download in _extract_tasks_api
_model_download_lock = threading.Lock()

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

# Lip contour landmarks for teeth region estimation
_UPPER_LIP_INNER = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]
_LOWER_LIP_INNER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]

# Eye contour landmarks for glasses bridge detection
_EYE_BRIDGE = [6, 168, 197, 195, 5, 4]
_EYE_OUTER_LEFT = [33, 246, 161, 160, 159, 158, 157, 173, 155, 154, 153]
_EYE_OUTER_RIGHT = [362, 398, 384, 385, 386, 387, 388, 466, 390, 373, 374]


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

    @property
    def landmark_confidence(self) -> np.ndarray:
        """Per-landmark confidence weights based on z-depth dispersion.

        Landmarks with extreme z-values relative to the face center tend
        to be less reliable (self-occlusion, profile views). Returns a
        (478,) array in [0.5, 1.0] where 1.0 = high confidence.
        """
        z = self.landmarks[:, 2]
        z_center = np.median(z)
        z_dev = np.abs(z - z_center)
        z_max = z_dev.max() if z_dev.max() > 0 else 1.0
        # Map deviation [0, max] to confidence [1.0, 0.5]
        return 1.0 - 0.5 * (z_dev / z_max)

    @property
    def face_rotation(self) -> float:
        """Estimate in-plane face rotation in degrees from eye corners.

        Uses the line between left eye outer corner (landmark 33) and
        right eye outer corner (landmark 263) to estimate roll angle.
        Returns degrees counter-clockwise; 0 means upright.
        """
        left_eye = self.landmarks[33, :2]  # left eye outer corner
        right_eye = self.landmarks[263, :2]  # right eye outer corner
        dx = (right_eye[0] - left_eye[0]) * self.image_width
        dy = (right_eye[1] - left_eye[1]) * self.image_height
        return float(np.degrees(np.arctan2(dy, dx)))

    @property
    def face_yaw(self) -> float:
        """Estimate horizontal face rotation (yaw) in degrees.

        Uses the asymmetry between left and right eye corner distances
        to the nose tip (landmark 1) to estimate yaw angle. Positive
        values mean the face is turned to the subject's left (viewer's
        right); negative means turned to the subject's right.

        Returns:
            Estimated yaw in degrees. 0 = frontal, +/-90 = full profile.
        """
        nose_tip = self.landmarks[1, :2]
        left_eye = self.landmarks[33, :2]
        right_eye = self.landmarks[263, :2]

        # Distances in pixel space
        d_left = float(
            np.sqrt(
                ((left_eye[0] - nose_tip[0]) * self.image_width) ** 2
                + ((left_eye[1] - nose_tip[1]) * self.image_height) ** 2
            )
        )
        d_right = float(
            np.sqrt(
                ((right_eye[0] - nose_tip[0]) * self.image_width) ** 2
                + ((right_eye[1] - nose_tip[1]) * self.image_height) ** 2
            )
        )

        # Asymmetry ratio: 1.0 = frontal, >>1 or <<1 = profile
        if d_left + d_right < 1e-6:
            return 0.0
        ratio = (d_right - d_left) / (d_right + d_left)
        # Map ratio [-1, 1] to approximate degrees [-90, 90]
        return float(np.clip(ratio * 90.0, -90.0, 90.0))

    @property
    def face_view(self) -> str:
        """Classify face view as frontal, three-quarter, or profile.

        Returns:
            One of "frontal", "three_quarter_left", "three_quarter_right",
            "profile_left", or "profile_right".
        """
        yaw = self.face_yaw
        abs_yaw = abs(yaw)
        if abs_yaw < 15.0:
            return "frontal"
        if abs_yaw < 45.0:
            return "three_quarter_left" if yaw > 0 else "three_quarter_right"
        return "profile_left" if yaw > 0 else "profile_right"

    @property
    def visible_side(self) -> str:
        """Return which side of the face is more visible.

        Returns:
            "both" for frontal, "left" or "right" for profile/three-quarter.
        """
        view = self.face_view
        if view == "frontal":
            return "both"
        return "left" if "left" in view else "right"

    @property
    def face_bbox(self) -> tuple[int, int, int, int]:
        """Axis-aligned bounding box with 20% padding, rotation-aware.

        For rotated faces (>10 degrees), expands padding proportionally
        to prevent tight crops from cutting off the face.

        Returns:
            (x_min, y_min, x_max, y_max) in pixel coordinates, clamped
            to image bounds.
        """
        coords = self.pixel_coords
        x_min, y_min = coords.min(axis=0)
        x_max, y_max = coords.max(axis=0)

        box_w = x_max - x_min
        box_h = y_max - y_min

        # Base padding: 20% of box size on each side
        pad_frac = 0.2

        # For rotated faces, increase padding proportionally to rotation
        rotation = abs(self.face_rotation)
        if rotation > 10.0:
            # Scale padding: +1% per degree of rotation beyond 10
            pad_frac += (rotation - 10.0) * 0.01

        pad_x = box_w * pad_frac
        pad_y = box_h * pad_frac

        return (
            int(max(0, x_min - pad_x)),
            int(max(0, y_min - pad_y)),
            int(min(self.image_width - 1, x_max + pad_x)),
            int(min(self.image_height - 1, y_max + pad_y)),
        )


def get_teeth_mask(
    face: FaceLandmarks,
    image_shape: tuple[int, int],
) -> np.ndarray:
    """Generate a binary mask for the teeth/mouth interior region.

    Uses inner lip contour landmarks to define the mouth opening where
    teeth are visible. This mask can be used to apply rigid translation
    instead of smooth TPS warping in the teeth region.

    Args:
        face: Extracted face landmarks.
        image_shape: (height, width) of the target image.

    Returns:
        Float32 mask [0-1] where 1 = teeth region.
    """
    h, w = image_shape
    coords = face.pixel_coords_at(w, h)

    upper = coords[_UPPER_LIP_INNER].astype(np.int32)
    lower = coords[_LOWER_LIP_INNER].astype(np.int32)

    # Combine into a closed polygon (upper lip forward, lower lip backward)
    mouth_contour = np.concatenate([upper, lower[::-1]], axis=0)

    mask = np.zeros((h, w), dtype=np.float32)
    cv2.fillPoly(mask, [mouth_contour], 1.0)
    return mask


def detect_glasses_region(
    face: FaceLandmarks,
    image: np.ndarray,
    threshold: float = 30.0,
) -> bool:
    """Detect whether the subject is wearing glasses.

    Uses edge density in the eye bridge region (between the eyes) to
    detect glasses frames. Glasses produce strong horizontal edges
    that skin alone does not.

    Args:
        face: Extracted face landmarks.
        image: BGR image.
        threshold: Edge density threshold for glasses detection.

    Returns:
        True if glasses are likely present.
    """
    h, w = image.shape[:2]
    coords = face.pixel_coords_at(w, h)

    # Sample the bridge region between the eyes
    bridge_pts = coords[_EYE_BRIDGE].astype(np.int32)
    x_min = max(0, int(bridge_pts[:, 0].min()) - 5)
    x_max = min(w, int(bridge_pts[:, 0].max()) + 5)
    y_min = max(0, int(bridge_pts[:, 1].min()) - 10)
    y_max = min(h, int(bridge_pts[:, 1].max()) + 10)

    if x_max <= x_min or y_max <= y_min:
        return False

    roi = image[y_min:y_max, x_min:x_max]
    if roi.size == 0:
        return False

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi
    edges = cv2.Canny(gray, 50, 150)
    edge_density = float(edges.mean())

    return edge_density > threshold


def get_accessory_mask(
    face: FaceLandmarks,
    image: np.ndarray,
    include_glasses: bool = True,
    include_teeth: bool = True,
) -> np.ndarray:
    """Generate a combined mask for accessories and rigid structures.

    Identifies regions that should receive special handling during
    deformation (rigid translation for teeth, exclusion for glasses).

    Args:
        face: Extracted face landmarks.
        image: BGR image.
        include_glasses: Check for glasses.
        include_teeth: Include teeth region.

    Returns:
        Float32 mask [0-1] where 1 = accessory/rigid region.
    """
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.float32)

    if include_teeth:
        teeth = get_teeth_mask(face, (h, w))
        mask = np.maximum(mask, teeth)

    if include_glasses and detect_glasses_region(face, image):
        coords = face.pixel_coords_at(w, h)
        # Mark eye frame regions
        for eye_indices in [_EYE_OUTER_LEFT, _EYE_OUTER_RIGHT]:
            pts = coords[eye_indices].astype(np.int32)
            hull = cv2.convexHull(pts)
            cv2.fillConvexPoly(mask, hull, 1.0)
        # Bridge
        bridge_pts = coords[_EYE_BRIDGE].astype(np.int32)
        for i in range(len(bridge_pts) - 1):
            cv2.line(mask, tuple(bridge_pts[i]), tuple(bridge_pts[i + 1]), 1.0, 5)

    return mask


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

    # Download model if not cached (thread-safe with double-check locking)
    model_path = Path(tempfile.gettempdir()) / "face_landmarker_v2_with_blendshapes.task"
    if not model_path.exists():
        with _model_download_lock:
            if not model_path.exists():
                url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
                tmp_path = model_path.with_suffix(".tmp")
                urllib.request.urlretrieve(url, str(tmp_path))
                tmp_path.rename(model_path)

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


def extract_all_landmarks(
    image: np.ndarray,
    max_faces: int = 10,
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
) -> list[FaceLandmarks]:
    """Extract landmarks for all faces in an image.

    Args:
        image: BGR image as numpy array.
        max_faces: Maximum number of faces to detect.
        min_detection_confidence: Minimum face detection confidence.
        min_tracking_confidence: Minimum landmark tracking confidence.

    Returns:
        List of FaceLandmarks, one per detected face. Empty if no faces found.
    """
    h, w = image.shape[:2]
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    faces = []

    # Use Solutions API which supports max_num_faces > 1
    try:
        with mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=max_faces,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        ) as face_mesh:
            results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            for face_lm in results.multi_face_landmarks:
                landmarks = np.array(
                    [(lm.x, lm.y, lm.z) for lm in face_lm.landmark],
                    dtype=np.float32,
                )
                faces.append(
                    FaceLandmarks(
                        landmarks=landmarks,
                        image_width=w,
                        image_height=h,
                        confidence=1.0,
                    )
                )
    except Exception:
        logger.debug("Multi-face extraction failed", exc_info=True)

    return faces


def select_largest_face(faces: list[FaceLandmarks]) -> FaceLandmarks | None:
    """Select the face with the largest bounding box from a list.

    Args:
        faces: List of detected faces.

    Returns:
        The face with the largest area, or None if the list is empty.
    """
    if not faces:
        return None

    def _face_area(face: FaceLandmarks) -> float:
        coords = face.pixel_coords
        x_range = coords[:, 0].max() - coords[:, 0].min()
        y_range = coords[:, 1].max() - coords[:, 1].min()
        return float(x_range * y_range)

    return max(faces, key=_face_area)


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

        # Supplement sparse jawline edges (landmarks 172-177)
        for a, b in [(172, 173), (173, 174), (174, 175), (175, 176), (176, 177)]:
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
