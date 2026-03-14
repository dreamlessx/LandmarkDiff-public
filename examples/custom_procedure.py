"""Example: define and apply a custom surgical procedure."""

import numpy as np
from PIL import Image

from landmarkdiff.conditioning import render_wireframe
from landmarkdiff.landmarks import extract_landmarks
from landmarkdiff.manipulation import DeformationHandle, gaussian_rbf_deform


def main():
    # define a custom procedure: lip augmentation
    lip_handles = [
        # upper lip - push forward and slightly up
        DeformationHandle(
            landmark_index=13, displacement=np.array([0, -3, -5]), influence_radius=15.0
        ),
        DeformationHandle(
            landmark_index=14, displacement=np.array([0, -3, -5]), influence_radius=15.0
        ),
        DeformationHandle(
            landmark_index=82, displacement=np.array([0, -2, -4]), influence_radius=12.0
        ),
        DeformationHandle(
            landmark_index=312, displacement=np.array([0, -2, -4]), influence_radius=12.0
        ),
        # lower lip - push forward and slightly down
        DeformationHandle(
            landmark_index=17, displacement=np.array([0, 2, -4]), influence_radius=15.0
        ),
        DeformationHandle(
            landmark_index=15, displacement=np.array([0, 2, -4]), influence_radius=15.0
        ),
        # lip corners - slight lift
        DeformationHandle(
            landmark_index=61, displacement=np.array([0, -2, -2]), influence_radius=10.0
        ),
        DeformationHandle(
            landmark_index=291, displacement=np.array([0, -2, -2]), influence_radius=10.0
        ),
    ]

    # load image and extract landmarks
    img = Image.open("face.jpg").convert("RGB").resize((512, 512))
    img_array = np.array(img)

    landmarks = extract_landmarks(img_array)
    if landmarks is None:
        print("No face detected. Place a face image as 'face.jpg'")
        return

    # apply handles at different intensities
    import cv2

    for intensity in [30, 60, 90]:
        scale = intensity / 100.0
        deformed_pts = landmarks.landmarks.copy()

        # apply each handle with scaled displacement
        for handle in lip_handles:
            scaled_handle = DeformationHandle(
                landmark_index=handle.landmark_index,
                displacement=handle.displacement * scale,
                influence_radius=handle.influence_radius,
            )
            deformed_pts = gaussian_rbf_deform(deformed_pts, scaled_handle)

        # wrap back into FaceLandmarks for rendering
        from landmarkdiff.landmarks import FaceLandmarks

        deformed_face = FaceLandmarks(
            landmarks=deformed_pts,
            image_width=landmarks.image_width,
            image_height=landmarks.image_height,
            confidence=landmarks.confidence,
        )

        mesh = render_wireframe(deformed_face, (512, 512))
        cv2.imwrite(f"lip_augmentation_{intensity}.png", mesh)
        print(f"Saved lip augmentation at {intensity}% intensity")


if __name__ == "__main__":
    main()
