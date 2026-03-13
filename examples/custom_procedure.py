"""Example: define and apply a custom surgical procedure."""

import numpy as np

from landmarkdiff.landmarks import extract_landmarks
from landmarkdiff.manipulation import ControlHandle, gaussian_rbf_deform
from landmarkdiff.conditioning import draw_tessellation


def main():
    # Define a custom procedure: lip augmentation
    lip_augmentation = [
        # Upper lip - push forward and slightly up
        ControlHandle(anchor_index=13, displacement=np.array([0, -3, -5]), radius=15.0),
        ControlHandle(anchor_index=14, displacement=np.array([0, -3, -5]), radius=15.0),
        ControlHandle(anchor_index=82, displacement=np.array([0, -2, -4]), radius=12.0),
        ControlHandle(anchor_index=312, displacement=np.array([0, -2, -4]), radius=12.0),
        # Lower lip - push forward and slightly down
        ControlHandle(anchor_index=17, displacement=np.array([0, 2, -4]), radius=15.0),
        ControlHandle(anchor_index=15, displacement=np.array([0, 2, -4]), radius=15.0),
        # Lip corners - slight lift (vermilion border enhancement)
        ControlHandle(anchor_index=61, displacement=np.array([0, -2, -2]), radius=10.0),
        ControlHandle(anchor_index=291, displacement=np.array([0, -2, -2]), radius=10.0),
    ]

    # Apply to a face image
    landmarks = extract_landmarks("face.jpg")
    if landmarks is None:
        print("No face detected. Please provide a face image as 'face.jpg'")
        return

    # Apply at different intensities
    for intensity in [0.3, 0.6, 0.9]:
        deformed = gaussian_rbf_deform(landmarks, lip_augmentation, intensity=intensity)
        mesh = draw_tessellation(deformed, (512, 512))

        import cv2
        cv2.imwrite(f"lip_augmentation_{int(intensity * 100)}.png", mesh)
        print(f"Saved lip augmentation at {intensity:.0%} intensity")


if __name__ == "__main__":
    main()
