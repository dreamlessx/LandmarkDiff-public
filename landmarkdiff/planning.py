"""Surgical planning mode with measurement annotations.

Overlays calibrated distances (mm) and angles (degrees) on face images
using MediaPipe landmark positions. Measurements are calibrated from
the average intercanthal distance (ICD) as a reference scale.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from landmarkdiff.landmarks import FaceLandmarks

# Average intercanthal distance in mm (adult population mean)
# Used as reference for pixel-to-mm calibration
_MEAN_ICD_MM = 33.0

# Landmark pairs for standard facial measurements
_MEASUREMENTS = {
    "intercanthal": (133, 362),  # medial canthi
    "biocular": (33, 263),  # lateral canthi
    "nose_width": (240, 460),  # alar bases
    "nose_height": (6, 94),  # nasion to subnasale
    "philtrum": (94, 0),  # subnasale to upper lip
    "upper_lip": (0, 14),  # upper lip to stomion
    "lower_lip": (14, 17),  # stomion to lower lip
    "chin_height": (17, 152),  # lower lip to menton
    "face_width": (234, 454),  # bizygomatic
    "face_height": (10, 152),  # trichion to menton
}

# Angular measurements
_ANGLES = {
    "nasal_tip": (6, 1, 94),  # nasion-tip-subnasale
    "nasolabial": (94, 0, 2),  # subnasale-upper_lip-columella
    "mentocervical": (6, 152, 175),  # nasion-menton-pogonion
}


@dataclass
class Measurement:
    """Single facial measurement."""

    name: str
    value_mm: float
    value_px: float


@dataclass
class AngleMeasurement:
    """Angular measurement between three landmarks."""

    name: str
    degrees: float
    vertex: tuple[float, float]  # (x, y) pixel position of vertex


@dataclass
class PlanningResult:
    """Complete surgical planning measurement set."""

    measurements: list[Measurement]
    angles: list[AngleMeasurement]
    px_per_mm: float  # calibration factor

    def summary(self) -> str:
        lines = ["Surgical Planning Measurements:"]
        lines.append(f"  Scale: {self.px_per_mm:.2f} px/mm")
        lines.append("  Distances:")
        for m in self.measurements:
            lines.append(f"    {m.name}: {m.value_mm:.1f} mm ({m.value_px:.1f} px)")
        lines.append("  Angles:")
        for a in self.angles:
            lines.append(f"    {a.name}: {a.degrees:.1f} deg")
        return "\n".join(lines)


def compute_planning_measurements(
    face: FaceLandmarks,
    reference_icd_mm: float = _MEAN_ICD_MM,
) -> PlanningResult:
    """Compute calibrated facial measurements for surgical planning.

    Uses the intercanthal distance (ICD) as a reference scale to
    convert pixel measurements to approximate millimeters.

    Args:
        face: Extracted face landmarks.
        reference_icd_mm: Reference ICD in mm for calibration.

    Returns:
        PlanningResult with all measurements.
    """
    coords = face.pixel_coords

    # Calibrate: ICD in pixels
    icd_px = float(np.linalg.norm(coords[133, :2] - coords[362, :2]))
    px_per_mm = icd_px / reference_icd_mm if reference_icd_mm > 0 else 1.0

    # Distance measurements
    measurements = []
    for name, (idx_a, idx_b) in _MEASUREMENTS.items():
        dist_px = float(np.linalg.norm(coords[idx_a, :2] - coords[idx_b, :2]))
        dist_mm = dist_px / px_per_mm
        measurements.append(Measurement(name=name, value_mm=dist_mm, value_px=dist_px))

    # Angle measurements
    angles = []
    for name, (idx_a, idx_b, idx_c) in _ANGLES.items():
        a = coords[idx_a, :2]
        b = coords[idx_b, :2]  # vertex
        c = coords[idx_c, :2]

        ba = a - b
        bc = c - b
        cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
        angle_deg = float(np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0))))

        angles.append(
            AngleMeasurement(
                name=name,
                degrees=angle_deg,
                vertex=(float(b[0]), float(b[1])),
            )
        )

    return PlanningResult(
        measurements=measurements,
        angles=angles,
        px_per_mm=px_per_mm,
    )


def visualize_planning(
    image: np.ndarray,
    face: FaceLandmarks,
    planning: PlanningResult,
) -> np.ndarray:
    """Overlay measurement annotations on an image.

    Draws measurement lines with mm labels and angle arcs.

    Args:
        image: BGR face image.
        face: Extracted face landmarks.
        planning: Computed planning measurements.

    Returns:
        Annotated image copy.
    """
    canvas = image.copy()
    h, w = canvas.shape[:2]
    coords = face.pixel_coords
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.25, h / 512.0 * 0.32)
    thickness = max(1, int(h / 512.0 * 1.5))

    # Color palette for different measurement types
    line_color = (0, 255, 200)  # cyan-green for distances
    angle_color = (200, 100, 255)  # purple for angles
    text_color = (255, 255, 255)

    # Draw distance measurements
    for m in planning.measurements:
        idx_a, idx_b = _MEASUREMENTS[m.name]
        pt_a = (int(coords[idx_a, 0]), int(coords[idx_a, 1]))
        pt_b = (int(coords[idx_b, 0]), int(coords[idx_b, 1]))

        # Draw measurement line
        cv2.line(canvas, pt_a, pt_b, line_color, 1, cv2.LINE_AA)
        # Small endpoint markers
        cv2.circle(canvas, pt_a, 2, line_color, -1)
        cv2.circle(canvas, pt_b, 2, line_color, -1)

        # Label at midpoint
        mid_x = (pt_a[0] + pt_b[0]) // 2
        mid_y = (pt_a[1] + pt_b[1]) // 2
        label = f"{m.value_mm:.1f}mm"
        cv2.putText(
            canvas,
            label,
            (mid_x + 3, mid_y - 3),
            font,
            font_scale,
            text_color,
            thickness,
            cv2.LINE_AA,
        )

    # Draw angle measurements
    for a in planning.angles:
        idx_a, idx_b, idx_c = _ANGLES[a.name]
        vertex = (int(coords[idx_b, 0]), int(coords[idx_b, 1]))
        pt_a = (int(coords[idx_a, 0]), int(coords[idx_a, 1]))
        pt_c = (int(coords[idx_c, 0]), int(coords[idx_c, 1]))

        # Draw angle arms
        cv2.line(canvas, vertex, pt_a, angle_color, 1, cv2.LINE_AA)
        cv2.line(canvas, vertex, pt_c, angle_color, 1, cv2.LINE_AA)

        # Label
        label = f"{a.degrees:.0f} deg"
        cv2.putText(
            canvas,
            label,
            (vertex[0] + 5, vertex[1] - 8),
            font,
            font_scale,
            angle_color,
            thickness,
            cv2.LINE_AA,
        )

    return canvas
