# Social Media Drafts: LandmarkDiff

All drafts are written from the perspective of `dreamlessx`.
Links:
- GitHub: https://github.com/dreamlessx/LandmarkDiff-public
- Demo: https://huggingface.co/spaces/dreamlessx/LandmarkDiff
- Wiki: https://github.com/dreamlessx/LandmarkDiff-public/wiki

**Reminder**: These are drafts only. Do not post without review. This is a research tool, not medical advice.

---

## 1. r/MachineLearning: [Project] Post

**Flair**: [Project]

**Title**:
```
[Project] LandmarkDiff: Anatomically-Conditioned Diffusion for Surgical Outcome Prediction from Clinical Photography
```

**Body**:
```
Hi r/MachineLearning,

I've been working on LandmarkDiff, an open-source system for predicting facial surgery outcomes
from standard clinical photographs. Happy to share it here and get feedback from the ML community.

**What it does**

Given a single frontal face photo and a selected surgical procedure, LandmarkDiff synthesizes a
plausible post-operative appearance while preserving patient identity. It supports six procedures:
rhinoplasty, blepharoplasty (eyelid), rhytidectomy (facelift), orthognathic (jaw), brow lift,
and mentoplasty (chin).

**Technical contribution**

The pipeline has three stages:

1. **Landmark extraction**: MediaPipe FaceMesh extracts 478 3D facial landmarks. These define
   anatomically meaningful control points across the face.

2. **Procedure-specific deformation**: Gaussian RBF (radial basis function) interpolation drives
   landmark displacements that correspond to each procedure's anatomical effect. There is also a
   data-driven displacement mode that fits displacement fields from real before/after surgery pairs.

3. **Conditioned synthesis**: Deformed landmark positions are rendered as a wireframe and fed
   as a conditioning input to a ControlNet-conditioned Stable Diffusion 1.5 pipeline (built on
   CrucibleAI's ControlNet). The wireframe spatially grounds the diffusion process to the predicted
   anatomy rather than letting the model hallucinate structure.

Post-processing: CodeFormer face restoration -> Real-ESRGAN upscaling -> histogram matching ->
unsharp sharpening -> Laplacian blend to composite back over the original.

**Inference modes**

- `tps` (thin-plate spline, CPU-only): pure warp, no diffusion, very fast
- `img2img`: SD1.5 img2img conditioned on the warped image
- `controlnet`: ControlNet conditioning on the deformed wireframe
- `controlnet_ip`: ControlNet + IP-Adapter for stronger identity preservation

The CPU TPS demo is live without any GPU requirement, so anyone can try it immediately.

**Links**

- GitHub (MIT license): https://github.com/dreamlessx/LandmarkDiff-public
- Live demo (HF Spaces, CPU): https://huggingface.co/spaces/dreamlessx/LandmarkDiff
- Wiki / docs: https://github.com/dreamlessx/LandmarkDiff-public/wiki

**Disclaimer**: This is a research tool, not medical advice. Outputs are synthetic visualizations
intended for research and educational purposes only.

Happy to discuss the deformation model, conditioning strategy, or dataset details in the comments.
```

---

## 2. r/computervision Post

**Title**:
```
LandmarkDiff: MediaPipe 478-point mesh + TPS warping for surgical face prediction [OC]
```

**Body**:
```
Built a system that uses MediaPipe FaceMesh (478 3D landmarks) as the geometric backbone for
predicting facial surgery outcomes from a single clinical photo.

The core vision pipeline:

- Extract 478 anatomical control points with MediaPipe FaceMesh
- Apply procedure-specific Gaussian RBF displacements (rhinoplasty, brow lift, facelift, etc.)
- Fit thin-plate spline (TPS) warp from original to displaced control points
- Apply warp to the full image: fast, smooth, and anatomically consistent
- Optionally pass the warped result through ControlNet + SD1.5 for texture synthesis

The TPS CPU demo runs in-browser on HF Spaces, no GPU needed. The ControlNet GPU pipeline
produces more photorealistic outputs by grounding diffusion to the deformed wireframe.

Demo: https://huggingface.co/spaces/dreamlessx/LandmarkDiff
Code: https://github.com/dreamlessx/LandmarkDiff-public

Disclaimer: research tool, not medical advice.
```

---

## 3. r/StableDiffusion Post

**Title**:
```
Using facial mesh wireframes as ControlNet conditioning inputs for surgery outcome prediction: LandmarkDiff
```

**Body**:
```
Sharing a project that uses ControlNet conditioning in an unusual way: facial anatomy as a
spatial prior for surgical outcome synthesis.

**The ControlNet pipeline**

Standard ControlNet-based generation takes a conditioning image (edges, pose, depth, etc.) and
guides the diffusion process spatially. In LandmarkDiff, the conditioning input is a rendered
wireframe of *deformed* facial landmarks: the face mesh after procedure-specific landmark
displacements are applied.

The deformation is driven by Gaussian RBF interpolation on MediaPipe's 478-point 3D face mesh.
Procedure presets encode anatomically meaningful control point movements for six surgeries
(rhinoplasty, blepharoplasty, rhytidectomy, orthognathic, brow lift, mentoplasty).

The full pipeline:

1. MediaPipe extracts 478 3D landmarks from a clinical photo
2. Procedure displacements are applied to relevant landmark subsets (e.g., nasal tip + alar
   landmarks for rhinoplasty, orbital landmarks for blepharoplasty)
3. Deformed landmarks are projected back to 2D and rendered as a mesh wireframe
4. Wireframe -> ControlNet (CrucibleAI backbone) -> SD1.5 with the original photo as init
5. Optional IP-Adapter layer for stronger identity preservation (`controlnet_ip` mode)
6. CodeFormer + Real-ESRGAN post-processing

The wireframe conditioning means the diffusion model is spatially anchored to the predicted
anatomy rather than being free to move facial features arbitrarily. This is important for
surgical visualization, where geometric plausibility is as important as texture quality.

There is also a lightweight CPU mode (`tps`) that skips diffusion entirely and produces a pure
TPS warp: useful for fast preview without GPU.

Code (MIT): https://github.com/dreamlessx/LandmarkDiff-public
Demo (CPU TPS + GPU ControlNet): https://huggingface.co/spaces/dreamlessx/LandmarkDiff

Disclaimer: research tool, not medical advice.
```

---

## 4. r/PlasticSurgery Post

**Title**:
```
Open-source research tool for visualizing facial surgery outcomes from a single photo: LandmarkDiff
```

**Body**:
```
Hi r/PlasticSurgery,

I want to share an open-source research project that may be of interest to people here who are
curious about what technology can (and cannot) do for surgical planning visualization.

**What it is**

LandmarkDiff is a research system that takes a standard facial photo and a selected procedure,
and synthesizes a plausible visualization of what a patient might look like post-operatively.
It supports six procedures:

- Rhinoplasty (nose reshaping)
- Blepharoplasty (eyelid surgery)
- Rhytidectomy (facelift)
- Orthognathic surgery (jaw repositioning)
- Brow lift
- Mentoplasty (chin augmentation/reduction)

**How it works (non-technical)**

The system uses a 478-point face mesh to map the geometry of the face, applies anatomically
informed deformations for the selected procedure, and then uses an AI image synthesis model
(Stable Diffusion with ControlNet conditioning) to generate a photorealistic visualization of
the deformed face.

There is also a fast, lightweight mode (thin-plate spline warping) that runs without a GPU --
this is available directly in the browser demo.

**Important caveats**

This is a research prototype, not a clinical tool. The outputs are synthetic visualizations
based on generalized procedure models, not a prediction tailored to any individual's anatomy
or surgical plan. Real surgical outcomes depend on a surgeon's technique, individual anatomy,
healing, and many other factors that this model does not capture.

**This is a research tool, not medical advice.** Please consult a board-certified plastic
surgeon for any decisions about surgery.

That said, I hope the demo is interesting for those who want to explore how this kind of
technology works, and I welcome feedback from anyone in the surgical community on how this
type of system could be made more realistic or useful for patient communication.

Demo: https://huggingface.co/spaces/dreamlessx/LandmarkDiff
Code: https://github.com/dreamlessx/LandmarkDiff-public
```

---

## 5. Twitter/X Thread

**Tweet 1 (hook)**:
```
Introducing LandmarkDiff: predict facial surgery outcomes from a single clinical photo.

Uses a 478-point face mesh + Gaussian RBF deformation + ControlNet-conditioned Stable Diffusion.

Open source. Live demo. Six procedures supported.

Thread below.
```

**Tweet 2 (the face mesh)**:
```
2/ The geometric backbone is MediaPipe FaceMesh: 478 3D landmarks per frame.

These aren't just face keypoints. They cover the full facial surface: orbit, nasal cartilage,
lip, chin, forehead, zygomatic arch.

That density is what makes procedure-specific deformation anatomically meaningful.
```

**Tweet 3 (the deformation)**:
```
3/ Procedure presets move specific landmark subsets.

Rhinoplasty: nasal tip, alar base, dorsum landmarks.
Blepharoplasty: upper/lower orbital rim landmarks.
Rhytidectomy (facelift): cheek, jowl, jawline landmarks.

Gaussian RBF interpolation propagates these movements smoothly across the whole face.
```

**Tweet 4 (ControlNet conditioning)**:
```
4/ The deformed landmark positions are rendered as a wireframe image.

That wireframe becomes the ControlNet conditioning input.

Result: Stable Diffusion is spatially anchored to the predicted anatomy. It generates texture
and lighting consistent with the new geometry rather than inventing structure freely.

[PLACEHOLDER: before/after wireframe conditioning screenshot]
```

**Tweet 5 (inference modes)**:
```
5/ Four inference modes:

- tps: pure thin-plate spline warp, CPU only, instant
- img2img: SD1.5 img2img on the warped image
- controlnet: ControlNet wireframe conditioning
- controlnet_ip: ControlNet + IP-Adapter for identity preservation

The TPS mode runs in the browser demo with no GPU required.

[PLACEHOLDER: demo screenshot]
```

**Tweet 6 (demo + code)**:
```
6/ Try the demo (CPU TPS, no GPU needed):
https://huggingface.co/spaces/dreamlessx/LandmarkDiff

Full code (MIT license):
https://github.com/dreamlessx/LandmarkDiff-public

Six procedures: rhinoplasty, blepharoplasty, rhytidectomy, orthognathic, brow lift, mentoplasty.
```

**Tweet 7 (disclaimer)**:
```
7/ This is a research tool, not medical advice.

Outputs are synthetic visualizations for research and educational purposes. They do not
predict individual surgical outcomes.

Feedback, contributions, and forks welcome.
```

---

## 6. Hacker News "Show HN" Post

**Title**:
```
Show HN: LandmarkDiff: facial surgery outcome prediction with MediaPipe mesh + ControlNet SD1.5
```

**Description**:
```
LandmarkDiff is an open-source system for synthesizing plausible post-operative facial
appearances from a single clinical photograph. It uses MediaPipe FaceMesh (478 3D landmarks)
as the geometric backbone, applies procedure-specific Gaussian RBF displacements to those
landmarks, and conditions Stable Diffusion 1.5 via ControlNet on a rendered wireframe of the
deformed face mesh.

Six procedures are implemented: rhinoplasty, blepharoplasty, rhytidectomy (facelift),
orthognathic surgery, brow lift, and mentoplasty.

The pipeline has four inference modes:

- tps: thin-plate spline warp, CPU only, no diffusion: fast and useful as a geometric baseline
- img2img: standard SD1.5 img2img conditioned on the warped image
- controlnet: ControlNet conditioning on the deformed wireframe for spatially grounded synthesis
- controlnet_ip: adds IP-Adapter for stronger patient identity preservation

A CPU demo (TPS mode) is live on HuggingFace Spaces. The ControlNet pipeline requires a GPU.

There is also a data-driven displacement mode that fits displacement fields from real
before/after surgery image pairs, rather than using hand-tuned procedure presets.

The codebase is MIT licensed and includes a Python package, CLI, Docker setup, Gradio demo,
and a test suite.

Disclaimer: this is a research tool, not medical advice.

GitHub: https://github.com/dreamlessx/LandmarkDiff-public
Demo: https://huggingface.co/spaces/dreamlessx/LandmarkDiff
Wiki: https://github.com/dreamlessx/LandmarkDiff-public/wiki
```

---

## 7. LinkedIn Post

**Body**:
```
I'm releasing LandmarkDiff, an open-source research system for predicting facial surgery outcomes
from standard clinical photography.

The project addresses a real gap in surgical planning: patients often struggle to form realistic
expectations of surgical outcomes before committing to a procedure. High-quality preoperative
visualization is one area where computational tools can meaningfully assist clinical communication.

The technical pipeline:

- MediaPipe FaceMesh extracts 478 anatomically distributed 3D landmarks from a single photo
- Procedure-specific Gaussian RBF deformation fields simulate landmark movements associated
  with six facial procedures: rhinoplasty, blepharoplasty, rhytidectomy, orthognathic surgery,
  brow lift, and mentoplasty
- The deformed landmark mesh is rendered as a wireframe and used as a ControlNet conditioning
  input for Stable Diffusion 1.5, spatially constraining synthesis to the predicted anatomy
- A data-driven mode fits displacement fields from real before/after surgery image pairs,
  making the deformation model learnable from clinical data

The system supports a lightweight CPU inference mode (thin-plate spline warping) and a
full GPU pipeline with optional IP-Adapter for identity preservation.

The codebase is fully open source (MIT license), includes a live demo, Python package, CLI,
Docker support, and comprehensive documentation.

Important note: this is a research tool intended for educational and research purposes. It is
not a medical device and does not constitute medical advice. Clinical decisions should always
involve a qualified surgical specialist.

GitHub: https://github.com/dreamlessx/LandmarkDiff-public
Demo: https://huggingface.co/spaces/dreamlessx/LandmarkDiff

Interested in feedback from anyone working in surgical planning, medical imaging, or generative
model applications in clinical contexts.
```

---

*Drafts complete. Review before posting. Do not post without approval.*
