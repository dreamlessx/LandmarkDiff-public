# landmarkdiff.conditioning

ControlNet conditioning image generation.

## Functions

### `draw_tessellation(landmarks, image_size) -> np.ndarray`

Render the 2556-edge MediaPipe Face Mesh tessellation as a wireframe image.

**Parameters:**
- `landmarks` (FaceLandmarks): Facial landmarks
- `image_size` (tuple[int, int]): Output image dimensions (width, height)

**Returns:** Grayscale wireframe image (np.ndarray, uint8)

### `auto_canny(image, sigma=0.33) -> np.ndarray`

Compute Canny edge map with automatic threshold selection adapted to skin tone.

**Parameters:**
- `image` (np.ndarray): Input image (BGR)
- `sigma` (float): Threshold sensitivity

**Returns:** Binary edge map (np.ndarray, uint8)

### `generate_conditioning(landmarks, image, image_size) -> dict`

Generate all conditioning signals for the pipeline.

**Parameters:**
- `landmarks` (FaceLandmarks): Facial landmarks (deformed)
- `image` (np.ndarray): Original face image
- `image_size` (tuple[int, int]): Output dimensions

**Returns:** dict with:
- `"mesh"` - Tessellation wireframe (used by ControlNet)
- `"canny"` - Edge map (used in compositing)
- `"mask"` - Surgical region mask (used in compositing)
