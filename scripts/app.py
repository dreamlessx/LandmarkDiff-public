"""Gradio UI for LandmarkDiff - single procedure, comparison, intensity sweep."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from landmarkdiff.clinical import ClinicalFlags
from landmarkdiff.conditioning import generate_conditioning
from landmarkdiff.evaluation import classify_fitzpatrick_ita
from landmarkdiff.inference import estimate_face_view, mask_composite
from landmarkdiff.landmarks import (
    extract_landmarks,
    render_landmark_image,
    visualize_landmarks,
)
from landmarkdiff.manipulation import apply_procedure_preset
from landmarkdiff.masking import generate_surgical_mask
from landmarkdiff.synthetic.tps_warp import warp_image_tps


def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    if len(img.shape) == 3 and img.shape[2] == 3:
        return img[:, :, ::-1].copy()
    return img


def _build_clinical_flags(
    vitiligo: bool,
    bells_palsy: bool,
    bells_side: str,
    keloid: bool,
    keloid_regions_str: str,
    ehlers_danlos: bool,
) -> ClinicalFlags | None:
    """Build ClinicalFlags from UI inputs."""
    if not any([vitiligo, bells_palsy, keloid, ehlers_danlos]):
        return None
    regions = [r.strip() for r in keloid_regions_str.split(",") if r.strip()] if keloid else []
    return ClinicalFlags(
        vitiligo=vitiligo,
        bells_palsy=bells_palsy,
        bells_palsy_side=bells_side,
        keloid_prone=keloid,
        keloid_regions=regions,
        ehlers_danlos=ehlers_danlos,
    )


def process_image(
    image_rgb: np.ndarray | None,
    procedure: str,
    intensity: float,
    vitiligo: bool,
    bells_palsy: bool,
    bells_side: str,
    keloid: bool,
    keloid_regions: str,
    ehlers_danlos: bool,
) -> tuple:
    """Process a single image through the full pipeline."""
    if image_rgb is None:
        blank = np.zeros((512, 512, 3), dtype=np.uint8)
        gray = np.zeros((512, 512), dtype=np.uint8)
        return blank, blank, gray, gray, blank, blank, blank, "Upload an image to begin."

    flags = _build_clinical_flags(
        vitiligo,
        bells_palsy,
        bells_side,
        keloid,
        keloid_regions,
        ehlers_danlos,
    )

    image_bgr = image_rgb[:, :, ::-1].copy()
    image_bgr = cv2.resize(image_bgr, (512, 512))
    image_rgb_512 = bgr_to_rgb(image_bgr)

    face = extract_landmarks(image_bgr)
    if face is None:
        blank = np.zeros((512, 512, 3), dtype=np.uint8)
        gray = np.zeros((512, 512), dtype=np.uint8)
        return (
            image_rgb_512,
            blank,
            gray,
            gray,
            image_rgb_512,
            image_rgb_512,
            image_rgb_512,
            "No face detected.",
        )

    # Landmarks
    annotated_bgr = visualize_landmarks(image_bgr, face, radius=2)
    annotated_rgb = bgr_to_rgb(annotated_bgr)

    # Manipulation with clinical flags
    manipulated = apply_procedure_preset(
        face,
        procedure,
        intensity,
        image_size=512,
        clinical_flags=flags,
    )

    # Conditioning
    manip_landmark_bgr = render_landmark_image(manipulated, 512, 512)
    _, manip_canny, manip_wireframe = generate_conditioning(manipulated, 512, 512)

    # Mask with clinical flags
    mask = generate_surgical_mask(
        face,
        procedure,
        512,
        512,
        clinical_flags=flags,
        image=image_bgr,
    )
    mask_vis = (mask * 255).astype(np.uint8)

    # TPS warp + composite
    tps_warped_bgr = warp_image_tps(
        image_bgr,
        face.pixel_coords,
        manipulated.pixel_coords,
    )
    composited_bgr = mask_composite(tps_warped_bgr, image_bgr, mask)

    # Side-by-side before/after
    side_by_side = np.hstack([image_rgb_512, bgr_to_rgb(composited_bgr)])

    # Convert all to RGB
    tps_rgb = bgr_to_rgb(tps_warped_bgr)
    composited_rgb = bgr_to_rgb(composited_bgr)
    manip_landmark_rgb = bgr_to_rgb(manip_landmark_bgr)

    # Info with skin tone + view angle
    view = estimate_face_view(face)
    try:
        fitz = classify_fitzpatrick_ita(image_bgr)
        fitz_str = f"Fitzpatrick Type {fitz}"
    except Exception:
        fitz_str = "Unknown"

    displacement = np.mean(
        np.linalg.norm(
            manipulated.pixel_coords - face.pixel_coords,
            axis=1,
        )
    )

    info_parts = [
        f"Procedure: {procedure}",
        f"Intensity: {intensity:.0f}%",
        f"Skin Tone: {fitz_str}",
        f"Face View: {view['view']} (yaw={view['yaw']}deg, pitch={view['pitch']}deg)",
        f"Avg displacement: {displacement:.1f} px",
        f"Landmarks: {len(face.landmarks)}",
    ]
    if flags and flags.has_any():
        active = []
        if flags.vitiligo:
            active.append("vitiligo")
        if flags.bells_palsy:
            active.append(f"Bell's palsy ({flags.bells_palsy_side})")
        if flags.keloid_prone:
            active.append(f"keloid ({','.join(flags.keloid_regions)})")
        if flags.ehlers_danlos:
            active.append("Ehlers-Danlos")
        info_parts.append(f"Clinical: {', '.join(active)}")
    if view.get("warning"):
        info_parts.append(f"WARNING: {view['warning']}")

    info = "\n".join(info_parts)

    return (
        annotated_rgb,
        manip_landmark_rgb,
        manip_wireframe,
        mask_vis,
        tps_rgb,
        composited_rgb,
        side_by_side,
        info,
    )


def create_comparison(
    image_rgb: np.ndarray | None,
    rhinoplasty: float,
    blepharoplasty: float,
    rhytidectomy: float,
    orthognathic: float,
) -> tuple:
    """Multi-procedure comparison."""
    if image_rgb is None:
        blank = np.zeros((512, 512, 3), dtype=np.uint8)
        return blank, blank, blank, blank

    image_bgr = image_rgb[:, :, ::-1].copy()
    image_bgr = cv2.resize(image_bgr, (512, 512))

    face = extract_landmarks(image_bgr)
    if face is None:
        rgb = bgr_to_rgb(image_bgr)
        return rgb, rgb, rgb, rgb

    results = []
    for proc, inten in [
        ("rhinoplasty", rhinoplasty),
        ("blepharoplasty", blepharoplasty),
        ("rhytidectomy", rhytidectomy),
        ("orthognathic", orthognathic),
    ]:
        if inten < 1.0:
            results.append(bgr_to_rgb(image_bgr))
            continue
        manip = apply_procedure_preset(face, proc, inten, image_size=512)
        mask = generate_surgical_mask(face, proc, 512, 512)
        warped = warp_image_tps(image_bgr, face.pixel_coords, manip.pixel_coords)
        comp = mask_composite(warped, image_bgr, mask)
        results.append(bgr_to_rgb(comp))

    return tuple(results)


def build_app():
    """Build the Gradio interface."""
    import gradio as gr

    with gr.Blocks(
        title="LandmarkDiff - Surgical Outcome Prediction",
        theme=gr.themes.Soft(),
        css=".side-by-side img { max-height: 300px; }",
    ) as app:
        gr.Markdown(
            "# LandmarkDiff\n"
            "**Anatomically-conditioned facial surgery outcome prediction**\n\n"
            "Upload a face photo, select a procedure, and adjust intensity. "
            "Supports frontal, 3/4, and profile views. "
            "Clinical edge cases (vitiligo, Bell's palsy, keloid) can be toggled below."
        )

        # ---- Tab 1: Single Procedure ----
        with gr.Tab("Single Procedure"):
            with gr.Row():
                with gr.Column(scale=1):
                    input_image = gr.Image(
                        label="Upload Face Photo",
                        type="numpy",
                        height=350,
                    )
                    procedure = gr.Radio(
                        choices=[
                            "rhinoplasty",
                            "blepharoplasty",
                            "rhytidectomy",
                            "orthognathic",
                            "brow_lift",
                            "mentoplasty",
                        ],
                        value="rhinoplasty",
                        label="Surgical Procedure",
                    )
                    intensity = gr.Slider(
                        minimum=0,
                        maximum=100,
                        value=50,
                        step=1,
                        label="Intensity (%)",
                        info="0 = no change, 100 = maximum effect",
                    )

                    with gr.Accordion("Clinical Flags", open=False):
                        gr.Markdown("Enable for patients with specific conditions:")
                        cb_vitiligo = gr.Checkbox(
                            label="Vitiligo (preserve depigmented patches)", value=False
                        )
                        cb_bells = gr.Checkbox(
                            label="Bell's Palsy (disable bilateral symmetry)", value=False
                        )
                        bells_side = gr.Radio(
                            ["left", "right"],
                            value="left",
                            label="Affected side",
                            visible=True,
                        )
                        cb_keloid = gr.Checkbox(
                            label="Keloid-prone (soften mask transitions)", value=False
                        )
                        keloid_regions = gr.Textbox(
                            label="Keloid regions (comma-separated)",
                            value="jawline",
                            placeholder="jawline, nose, lips",
                        )
                        cb_ehlers = gr.Checkbox(
                            label="Ehlers-Danlos (wider deformation radii)", value=False
                        )

                    run_btn = gr.Button("Generate Preview", variant="primary", size="lg")
                    info_box = gr.Textbox(label="Analysis Info", lines=8, interactive=False)

                with gr.Column(scale=2):
                    with gr.Row():
                        out_landmarks = gr.Image(label="Detected Landmarks", height=256)
                        out_manip = gr.Image(label="Manipulated Mesh", height=256)
                    with gr.Row():
                        out_wireframe = gr.Image(label="Wireframe", height=256)
                        out_mask = gr.Image(label="Surgical Mask", height=256)
                    with gr.Row():
                        out_tps = gr.Image(label="TPS Warp (Raw)", height=256)
                        out_result = gr.Image(label="Final Result", height=256)
                    gr.Markdown("### Before / After")
                    out_sidebyside = gr.Image(
                        label="Side-by-Side Comparison",
                        height=300,
                        elem_classes="side-by-side",
                    )

            all_inputs = [
                input_image,
                procedure,
                intensity,
                cb_vitiligo,
                cb_bells,
                bells_side,
                cb_keloid,
                keloid_regions,
                cb_ehlers,
            ]
            all_outputs = [
                out_landmarks,
                out_manip,
                out_wireframe,
                out_mask,
                out_tps,
                out_result,
                out_sidebyside,
                info_box,
            ]

            run_btn.click(fn=process_image, inputs=all_inputs, outputs=all_outputs)
            for trigger in [procedure, intensity]:
                trigger.change(fn=process_image, inputs=all_inputs, outputs=all_outputs)

        # ---- Tab 2: Multi-Procedure Comparison ----
        with gr.Tab("Multi-Procedure Comparison"):
            gr.Markdown("Adjust each procedure independently. Set to 0 to skip.")
            with gr.Row():
                with gr.Column(scale=1):
                    input_image_multi = gr.Image(
                        label="Upload Face Photo", type="numpy", height=300
                    )
                    slider_rhino = gr.Slider(0, 100, 50, step=1, label="Rhinoplasty")
                    slider_bleph = gr.Slider(0, 100, 0, step=1, label="Blepharoplasty")
                    slider_rhyti = gr.Slider(0, 100, 0, step=1, label="Rhytidectomy")
                    slider_ortho = gr.Slider(0, 100, 0, step=1, label="Orthognathic")
                    compare_btn = gr.Button("Compare All", variant="primary", size="lg")
                with gr.Column(scale=2):
                    with gr.Row():
                        out_rhino = gr.Image(label="Rhinoplasty", height=256)
                        out_bleph = gr.Image(label="Blepharoplasty", height=256)
                    with gr.Row():
                        out_rhyti = gr.Image(label="Rhytidectomy", height=256)
                        out_ortho = gr.Image(label="Orthognathic", height=256)

            multi_inputs = [
                input_image_multi,
                slider_rhino,
                slider_bleph,
                slider_rhyti,
                slider_ortho,
            ]
            multi_outputs = [out_rhino, out_bleph, out_rhyti, out_ortho]
            compare_btn.click(fn=create_comparison, inputs=multi_inputs, outputs=multi_outputs)
            for slider in [slider_rhino, slider_bleph, slider_rhyti, slider_ortho]:
                slider.change(fn=create_comparison, inputs=multi_inputs, outputs=multi_outputs)

        # ---- Tab 3: Intensity Sweep ----
        with gr.Tab("Intensity Sweep"):
            gr.Markdown("See how a procedure looks across intensity levels (0-100).")
            with gr.Row():
                with gr.Column(scale=1):
                    input_image_sweep = gr.Image(
                        label="Upload Face Photo", type="numpy", height=300
                    )
                    sweep_procedure = gr.Radio(
                        choices=[
                            "rhinoplasty",
                            "blepharoplasty",
                            "rhytidectomy",
                            "orthognathic",
                            "brow_lift",
                            "mentoplasty",
                        ],
                        value="rhinoplasty",
                        label="Procedure",
                    )
                    sweep_steps = gr.Slider(3, 10, 5, step=1, label="Number of steps")
                    sweep_btn = gr.Button("Generate Sweep", variant="primary", size="lg")
                with gr.Column(scale=2):
                    sweep_gallery = gr.Gallery(label="Intensity Sweep", columns=5, height=400)

            def generate_sweep(image_rgb, procedure, n_steps):
                if image_rgb is None:
                    return []
                image_bgr = image_rgb[:, :, ::-1].copy()
                image_bgr = cv2.resize(image_bgr, (512, 512))
                face = extract_landmarks(image_bgr)
                if face is None:
                    return []
                steps = np.linspace(0, 100, int(n_steps)).astype(int)
                results = []
                for i in steps:
                    if i == 0:
                        results.append((bgr_to_rgb(image_bgr), f"{i}%"))
                        continue
                    manip = apply_procedure_preset(face, procedure, float(i), image_size=512)
                    mask = generate_surgical_mask(face, procedure, 512, 512)
                    warped = warp_image_tps(image_bgr, face.pixel_coords, manip.pixel_coords)
                    comp = mask_composite(warped, image_bgr, mask)
                    results.append((bgr_to_rgb(comp), f"{i}%"))
                return results

            sweep_btn.click(
                fn=generate_sweep,
                inputs=[input_image_sweep, sweep_procedure, sweep_steps],
                outputs=[sweep_gallery],
            )

        # ---- Tab 4: Face Analysis ----
        with gr.Tab("Face Analysis"):
            gr.Markdown(
                "Analyze face photo: skin tone classification, view angle, and landmark quality."
            )
            with gr.Row():
                with gr.Column(scale=1):
                    analysis_image = gr.Image(label="Upload Face Photo", type="numpy", height=350)
                    analyze_btn = gr.Button("Analyze", variant="primary", size="lg")
                with gr.Column(scale=1):
                    analysis_info = gr.Textbox(
                        label="Analysis Results", lines=15, interactive=False
                    )
                    analysis_landmarks = gr.Image(label="Landmarks", height=350)

            def analyze_face(image_rgb):
                if image_rgb is None:
                    return "Upload an image.", None
                image_bgr = image_rgb[:, :, ::-1].copy()
                image_bgr = cv2.resize(image_bgr, (512, 512))
                face = extract_landmarks(image_bgr)
                if face is None:
                    return "No face detected.", bgr_to_rgb(image_bgr)

                view = estimate_face_view(face)
                try:
                    fitz = classify_fitzpatrick_ita(image_bgr)
                except Exception:
                    fitz = "?"

                annotated = visualize_landmarks(image_bgr, face, radius=2)

                lines = [
                    f"Fitzpatrick Type: {fitz}",
                    f"Face View: {view['view']}",
                    f"Yaw: {view['yaw']} degrees",
                    f"Pitch: {view['pitch']} degrees",
                    f"Is Frontal: {view['is_frontal']}",
                    f"Landmarks: {len(face.landmarks)}",
                    f"Detection Confidence: {face.confidence:.2f}",
                    "",
                    "Landmark Coverage by Region:",
                ]
                from landmarkdiff.landmarks import LANDMARK_REGIONS

                for region, indices in LANDMARK_REGIONS.items():
                    coords = face.pixel_coords[indices]
                    spread = np.std(coords, axis=0).mean()
                    lines.append(f"  {region}: {len(indices)} pts, spread={spread:.1f}px")

                if view.get("warning"):
                    lines.append(f"\nWARNING: {view['warning']}")

                return "\n".join(lines), bgr_to_rgb(annotated)

            analyze_btn.click(
                fn=analyze_face,
                inputs=[analysis_image],
                outputs=[analysis_info, analysis_landmarks],
            )

        # ---- Tab 5: Guided Multi-Angle Capture ----
        with gr.Tab("Multi-Angle Capture"):
            gr.Markdown(
                "## Guided Multi-Angle Face Capture\n"
                "Upload photos from multiple angles for more accurate 3D-aware prediction.\n"
                "Follow the guide: **front -> left 45 -> right 45"
                " -> left profile -> right profile**.\n\n"
                "The system validates each angle using landmark-based pose estimation "
                "and combines all views for the final result."
            )

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Step 1: Upload Photos")
                    ma_front = gr.Image(label="Front (0°)", type="numpy", height=200)
                    ma_left45 = gr.Image(label="Left 3/4 (~45°)", type="numpy", height=200)
                    ma_right45 = gr.Image(label="Right 3/4 (~45°)", type="numpy", height=200)
                    ma_left_profile = gr.Image(
                        label="Left Profile (~90°)", type="numpy", height=200
                    )
                    ma_right_profile = gr.Image(
                        label="Right Profile (~90°)", type="numpy", height=200
                    )

                    gr.Markdown("### Step 2: Configure")
                    ma_procedure = gr.Radio(
                        choices=[
                            "rhinoplasty",
                            "blepharoplasty",
                            "rhytidectomy",
                            "orthognathic",
                            "brow_lift",
                            "mentoplasty",
                        ],
                        value="rhinoplasty",
                        label="Procedure",
                    )
                    ma_intensity = gr.Slider(
                        minimum=0,
                        maximum=100,
                        value=50,
                        step=1,
                        label="Intensity (%)",
                    )
                    ma_validate_btn = gr.Button("Validate Angles", variant="secondary", size="lg")
                    ma_generate_btn = gr.Button("Generate All Views", variant="primary", size="lg")

                with gr.Column(scale=2):
                    gr.Markdown("### Angle Validation")
                    ma_validation_info = gr.Textbox(
                        label="Validation Results", lines=10, interactive=False
                    )

                    gr.Markdown("### Results by Angle")
                    with gr.Row():
                        ma_out_front = gr.Image(label="Front Result", height=200)
                        ma_out_left45 = gr.Image(label="Left 3/4 Result", height=200)
                        ma_out_right45 = gr.Image(label="Right 3/4 Result", height=200)
                    with gr.Row():
                        ma_out_left_prof = gr.Image(label="Left Profile Result", height=200)
                        ma_out_right_prof = gr.Image(label="Right Profile Result", height=200)
                    ma_gallery = gr.Gallery(
                        label="All Views (Before -> After)", columns=5, height=250
                    )

            def _validate_angle(image_rgb, expected_view, label):
                """Validate a single image's face angle matches expected view."""
                if image_rgb is None:
                    return None, f"{label}: not uploaded"
                image_bgr = image_rgb[:, :, ::-1].copy()
                image_bgr = cv2.resize(image_bgr, (512, 512))
                face = extract_landmarks(image_bgr)
                if face is None:
                    return None, f"{label}: NO FACE DETECTED"
                view = estimate_face_view(face)
                yaw = view["yaw"]
                detected = view["view"]

                # Check if the angle is close to what we expect
                angle_ok = False
                if (
                    (expected_view == "frontal" and abs(yaw) < 20)
                    or (expected_view == "left_three_quarter" and -60 < yaw < -15)
                    or (expected_view == "right_three_quarter" and 15 < yaw < 60)
                    or (expected_view == "left_profile" and yaw < -35)
                    or (expected_view == "right_profile" and yaw > 35)
                ):
                    angle_ok = True

                status = "OK" if angle_ok else "ANGLE MISMATCH"
                return face, f"{label}: yaw={yaw}° ({detected}) [{status}]"

            def validate_all_angles(front, left45, right45, left_prof, right_prof):
                """Validate all uploaded angles."""
                checks = [
                    (front, "frontal", "Front (0°)"),
                    (left45, "left_three_quarter", "Left 3/4 (~45°)"),
                    (right45, "right_three_quarter", "Right 3/4 (~45°)"),
                    (left_prof, "left_profile", "Left Profile (~90°)"),
                    (right_prof, "right_profile", "Right Profile (~90°)"),
                ]
                lines = ["Angle Validation Report:", "=" * 40]
                n_uploaded = 0
                n_valid = 0
                for img, expected, label in checks:
                    _, msg = _validate_angle(img, expected, label)
                    lines.append(msg)
                    if img is not None:
                        n_uploaded += 1
                        if "OK" in msg:
                            n_valid += 1

                lines.append("")
                lines.append(f"Uploaded: {n_uploaded}/5  Valid: {n_valid}/{n_uploaded}")
                if n_uploaded == 0:
                    lines.append("\nUpload at least one photo to begin.")
                elif n_valid == n_uploaded:
                    lines.append("\nAll angles validated. Ready to generate!")
                else:
                    lines.append(
                        "\nSome angles don't match expected views. "
                        "Results may still work but accuracy could be reduced."
                    )
                return "\n".join(lines)

            def generate_multi_angle(
                front, left45, right45, left_prof, right_prof, procedure, intensity
            ):
                """Generate surgical outcome prediction for each uploaded angle."""
                images = [
                    (front, "Front"),
                    (left45, "Left 3/4"),
                    (right45, "Right 3/4"),
                    (left_prof, "Left Profile"),
                    (right_prof, "Right Profile"),
                ]
                results = []
                gallery_items = []

                for img_rgb, label in images:
                    if img_rgb is None:
                        results.append(None)
                        continue
                    image_bgr = img_rgb[:, :, ::-1].copy()
                    image_bgr = cv2.resize(image_bgr, (512, 512))
                    face = extract_landmarks(image_bgr)
                    if face is None:
                        results.append(bgr_to_rgb(image_bgr))
                        continue

                    manip = apply_procedure_preset(
                        face, procedure, float(intensity), image_size=512
                    )
                    mask = generate_surgical_mask(face, procedure, 512, 512)
                    warped = warp_image_tps(image_bgr, face.pixel_coords, manip.pixel_coords)
                    comp = mask_composite(warped, image_bgr, mask)
                    result_rgb = bgr_to_rgb(comp)
                    results.append(result_rgb)
                    gallery_items.append((bgr_to_rgb(image_bgr), f"{label} (Before)"))
                    gallery_items.append((result_rgb, f"{label} (After)"))

                # Pad results to 5 entries
                while len(results) < 5:
                    results.append(None)

                return results[0], results[1], results[2], results[3], results[4], gallery_items

            ma_angle_inputs = [ma_front, ma_left45, ma_right45, ma_left_profile, ma_right_profile]

            ma_validate_btn.click(
                fn=validate_all_angles,
                inputs=ma_angle_inputs,
                outputs=[ma_validation_info],
            )
            ma_generate_btn.click(
                fn=generate_multi_angle,
                inputs=ma_angle_inputs + [ma_procedure, ma_intensity],
                outputs=[
                    ma_out_front,
                    ma_out_left45,
                    ma_out_right45,
                    ma_out_left_prof,
                    ma_out_right_prof,
                    ma_gallery,
                ],
            )

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LandmarkDiff Interactive UI")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="Create public Gradio link")
    parser.add_argument("--server-name", default="127.0.0.1")
    args = parser.parse_args()

    app = build_app()
    app.launch(
        server_name=args.server_name,
        server_port=args.port,
        share=args.share,
    )
