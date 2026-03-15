# Frequently Asked Questions

## VRAM Requirements

| Mode | Min VRAM | Recommended | Notes |
|------|----------|-------------|-------|
| tps | 0 GB | -- | CPU-only, no GPU needed |
| img2img | 4 GB | 6 GB | SD1.5 with model_cpu_offload |
| controlnet | 6 GB | 8 GB | SD1.5 + ControlNet |
| controlnet_ip | 8 GB | 10 GB | SD1.5 + ControlNet + IP-Adapter |

With `enable_model_cpu_offload()`, CUDA VRAM usage is reduced by moving model components to CPU RAM when not in use. This is enabled by default. If you still hit OOM, see the [Troubleshooting](Troubleshooting) page.

On Apple Silicon (MPS), all models run in fp32 due to MPS backend limitations. This roughly doubles memory compared to CUDA fp16.

## How do I add a custom procedure?

See the detailed guide on the [Contributing](Contributing) page. In summary:

1. Add landmark indices to `PROCEDURE_LANDMARKS` and a radius to `PROCEDURE_RADIUS` in `landmarkdiff/manipulation.py`
2. Add displacement logic in `_get_procedure_handles()`
3. Add mask config (landmark indices, dilation, feathering) in `landmarkdiff/masking.py`
4. Add the procedure to CLI choices in `landmarkdiff/__main__.py`
5. Optionally add a procedure-specific prompt in `PROCEDURE_PROMPTS` in `landmarkdiff/inference.py`
6. Write a test in `tests/test_manipulation.py`

## What are the differences between inference modes?

**TPS (Thin Plate Spline):**
- CPU-only, instant results (~50ms)
- Pure geometric warp: moves pixels, no texture synthesis
- Good for quick previews and interactive demos
- No model download needed

**img2img:**
- Feeds TPS-warped image into SD1.5 img2img
- Better texture quality than TPS alone
- Uses a surgical mask for compositing (only the procedure region is modified)
- Moderate quality, moderate speed (~5 seconds on A100)

**ControlNet:**
- Renders deformed face mesh as conditioning
- SD1.5 generates the face from mesh wireframe + text prompt
- Best texture quality, most photorealistic results
- Requires `CrucibleAI/ControlNetMediaPipeFace` weights
- Slower (~8 seconds on A100)

**ControlNet + IP-Adapter:**
- Same as ControlNet, plus face identity embedding from h94/IP-Adapter
- Strongest identity preservation
- Slowest mode (~10 seconds on A100)
- Most VRAM hungry

## How does intensity scaling work?

Intensity is always on a 0-100 scale:
- 0: No deformation (identity transform)
- 33: Mild surgical effect
- 50: Moderate (default in most examples)
- 66: Noticeable
- 100: Maximum / aggressive

Internally, `apply_procedure_preset()` divides intensity by 100 to get a scale factor that multiplies all displacement vectors. For example, a rhinoplasty tip displacement of `(0, -2.0)` at intensity=50 becomes `(0, -2.0 * 0.5) = (0, -1.0)` pixels.

When using the `DisplacementModel` (data-driven mode), the pipeline maps 0-100 to a 0-2.0 range where 1.0 represents the average observed displacement from real surgery pairs. So intensity=50 gives average displacement, and intensity=100 gives 2x average.

## How does identity preservation work?

Three mechanisms protect identity:

1. **Mask compositing:** Only the surgical region is modified. Everything outside the feathered mask boundary comes from the original image.

2. **IP-Adapter (controlnet_ip mode):** Feeds an ArcFace embedding of the input face into the generation process, biasing SD1.5 toward preserving the person's identity.

3. **ArcFace verification (post-processing):** After generation, cosine similarity between ArcFace embeddings of input and output is computed. If it drops below 0.6, the output is flagged with an identity drift warning. This is a quality gate, not a rejection: the output is still returned.

## What face poses are supported?

The pipeline estimates face orientation from landmarks (nose tip, ear, forehead, chin positions) and classifies views:

| View | Yaw Range | Quality |
|------|-----------|---------|
| Frontal | < 15 degrees | Best |
| Three-quarter | 15-45 degrees | Good |
| Profile | > 45 degrees | Reduced accuracy |

For views beyond 30 degrees yaw, a warning is returned in `result["view_info"]["warning"]`. The CrucibleAI ControlNet was primarily trained on frontal and near-frontal faces, so profile views produce less reliable results.

## Can I use a custom base model?

Yes. Pass `base_model_id` to `LandmarkDiffPipeline`:

```python
pipe = LandmarkDiffPipeline(
    mode="controlnet",
    base_model_id="stabilityai/stable-diffusion-1-5",
)
```

Any SD1.5-compatible checkpoint should work. SD2.x and SDXL checkpoints are not supported: the ControlNet is trained specifically for SD1.5's architecture.

## What image formats are supported?

Any format readable by OpenCV: JPEG, PNG, BMP, TIFF, WebP. The input is resized to 512x512 internally. For best results, use well-lit clinical photos with the face centered and filling most of the frame.

## Known Limitations

- **Single face only:** The pipeline processes one face per image. Multi-face images will use the first detected face.
- **2D projection:** All deformations happen in 2D image space. There is no underlying 3D model (FLAME integration is planned but not yet implemented).
- **SD1.5 resolution:** Output is 512x512. Higher resolutions require tiling or upscaling.
- **Skin type bias:** The diffusion model may produce slightly different quality across Fitzpatrick skin types. The auto-Canny edge detector adapts thresholds per image, but the base SD1.5 model has its own biases.
- **No temporal consistency:** Each image is processed independently. Video-based prediction with temporal smoothing is not currently supported.
- **Symmetry assumption:** The preset displacement vectors assume bilateral symmetry. For asymmetric conditions, use clinical flags (e.g., Bell's palsy) or the data-driven DisplacementModel.
