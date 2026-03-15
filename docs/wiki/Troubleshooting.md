# Troubleshooting

## CUDA Out of Memory (OOM)

**Symptom:** `RuntimeError: CUDA out of memory. Tried to allocate ...`

**Fixes (try in order):**

1. Use `tps` mode if you only need geometric warping: no GPU needed at all.

2. The pipeline enables `model_cpu_offload()` by default, which should keep peak VRAM under 6 GB for ControlNet mode. If this still fails, check that no other process is using the GPU:
   ```bash
   nvidia-smi
   ```

3. Reduce diffusion steps (default 30, try 20):
   ```python
   result = pipe.generate(image, num_inference_steps=20)
   ```

4. Disable post-processing neural models:
   ```python
   result = pipe.generate(image, postprocess=False, use_gfpgan=False)
   ```

5. For controlnet_ip mode (highest VRAM), try controlnet mode instead (saves the IP-Adapter memory).

6. If using multi-GPU, set `CUDA_VISIBLE_DEVICES` to a single GPU:
   ```bash
   CUDA_VISIBLE_DEVICES=0 python -m landmarkdiff infer face.jpg --mode controlnet
   ```

## MediaPipe Initialization Failures

**Symptom:** `ImportError: cannot import name 'FaceLandmarker'` or `FileNotFoundError: face_landmarker.task`

**Cause:** MediaPipe version mismatch between Tasks API and Solutions API.

**Fixes:**

1. Update MediaPipe:
   ```bash
   pip install mediapipe>=0.10.9
   ```

2. The code automatically falls back from Tasks API to Solutions API. If both fail, the problem is usually a missing system dependency:
   ```bash
   # Ubuntu/Debian
   apt-get install libgl1-mesa-glx libglib2.0-0

   # CentOS/RHEL
   yum install mesa-libGL glib2
   ```

3. On headless servers, MediaPipe may fail to download the face landmarker model. Pre-download it:
   ```python
   import urllib.request
   url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
   urllib.request.urlretrieve(url, "/tmp/face_landmarker_v2_with_blendshapes.task")
   ```

## Model Loading Issues

**Symptom:** `OSError: Can't load tokenizer` or `HTTPError: 401 Client Error: Unauthorized`

**Cause:** HuggingFace authentication required for gated models.

**Fix:** Set your HuggingFace token:
```bash
export HF_TOKEN=your_token_here
# or
huggingface-cli login
```

The ControlNet model (`CrucibleAI/ControlNetMediaPipeFace`) and SD1.5 (`runwayml/stable-diffusion-v1-5`) may require accepting terms on HuggingFace first. Visit the model pages and click "Agree".

**Symptom:** `ValueError: Unrecognized displacement model format`

**Cause:** Trying to load a `.npz` file that is not a valid DisplacementModel.

**Fix:** Make sure the file was created by `DisplacementModel.save()` or the `extract_displacements.py` script. Check the file:
```python
import numpy as np
data = np.load("model.npz")
print(data.files)  # should contain __metadata__ or procedures key
```

## Image Format Problems

**Symptom:** `ValueError: No face detected in image`

**Common causes:**
1. Image is too dark or overexposed
2. Face is too small relative to image (should fill >30% of frame)
3. Face is severely rotated (>60 degrees yaw)
4. Image is corrupt or in an unsupported format
5. Image has alpha channel (PNG with transparency)

**Debug steps:**
```python
import cv2
from landmarkdiff.landmarks import extract_landmarks

img = cv2.imread("problem_image.jpg")
print(f"Shape: {img.shape}, dtype: {img.dtype}")  # should be (H, W, 3) uint8

face = extract_landmarks(img, min_detection_confidence=0.3)  # lower threshold
print(f"Face detected: {face is not None}")
```

If lowering `min_detection_confidence` to 0.3 helps, the issue is marginal face quality. Consider preprocessing (better lighting, centering).

**Symptom:** `cv2.error: (-215:Assertion failed) !_src.empty()`

**Cause:** `cv2.imread()` returned None, usually because the file path is wrong or the file is not a valid image.

**Fix:** Check the file path and that the file exists:
```python
import os
print(os.path.exists("your_image.jpg"))
```

## Landmark Extraction Returns Incorrect Results

**Symptom:** Landmarks look wrong, deformations are distorted, output has artifacts.

**Possible causes:**
1. Input image is not BGR (OpenCV default). If loading with PIL or other libraries:
   ```python
   # PIL loads as RGB, convert to BGR for OpenCV/LandmarkDiff:
   import numpy as np
   from PIL import Image
   img_rgb = np.array(Image.open("face.jpg"))
   img_bgr = img_rgb[:, :, ::-1].copy()
   ```

2. Image has unusual aspect ratio. The pipeline resizes to 512x512, which can distort very wide or tall images. Pre-crop to roughly square around the face.

## Post-Processing Failures

**Symptom:** CodeFormer or GFPGAN returns the original image unchanged.

**Cause:** The neural post-processing models (CodeFormer, GFPGAN, Real-ESRGAN) are optional dependencies. If not installed, the pipeline silently falls back.

**Fix:**
```bash
# CodeFormer
pip install codeformer-pip

# GFPGAN
pip install gfpgan

# Real-ESRGAN
pip install realesrgan basicsr
```

On first run, model weights are downloaded automatically (~300 MB for CodeFormer, ~330 MB for GFPGAN, ~66 MB for Real-ESRGAN).

## Gradio Demo Errors

**Symptom:** `ImportError: No module named 'gradio'`

**Fix:**
```bash
pip install -e ".[app]"
```

**Symptom:** Demo launches but images fail to process.

**Check:** Make sure MediaPipe is working by running standalone landmark extraction first:
```python
from landmarkdiff.landmarks import extract_landmarks, load_image
img = load_image("test.jpg")
face = extract_landmarks(img)
print(face)
```

## Slow Inference

**Symptom:** ControlNet inference takes > 30 seconds.

**Possible causes:**
1. Running on CPU instead of GPU. Check:
   ```python
   from landmarkdiff.inference import get_device
   print(get_device())  # should be cuda or mps, not cpu
   ```

2. Model is being reloaded on every call. Create the pipeline once and reuse it.

3. Too many diffusion steps. Default is 30; DPM++ 2M Karras produces good results at 20 steps too.

## Import Errors After Installation

**Symptom:** `ModuleNotFoundError: No module named 'landmarkdiff'`

**Fix:** Install in editable mode from the repo root:
```bash
cd LandmarkDiff-public
pip install -e .
```

Check it installed correctly:
```bash
python -c "import landmarkdiff; print(landmarkdiff.__version__)"
```
