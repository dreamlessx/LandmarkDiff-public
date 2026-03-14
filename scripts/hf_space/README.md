---
title: LandmarkDiff
emoji: "\U0001F52C"
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 5.50.0
python_version: "3.11"
app_file: app.py
pinned: true
license: mit
short_description: Facial surgery outcome prediction with 6 procedures
tags:
  - medical-imaging
  - face
  - landmarks
  - thin-plate-spline
  - surgery-simulation
---

# LandmarkDiff

Anatomically-conditioned facial surgery outcome prediction from standard clinical photography.

Upload a face photo, select a surgical procedure, adjust intensity, and see the predicted outcome
in real time using thin-plate spline warping on CPU.

## Supported Procedures

| Procedure | Description |
|-----------|-------------|
| **Rhinoplasty** | Nose reshaping (bridge, tip, alar width) |
| **Blepharoplasty** | Eyelid surgery (lid position, canthal tilt) |
| **Rhytidectomy** | Facelift (midface and jawline tightening) |
| **Orthognathic** | Jaw surgery (maxilla/mandible repositioning) |
| **Brow Lift** | Brow elevation and forehead ptosis reduction |
| **Mentoplasty** | Chin surgery (projection and vertical height) |

## How It Works

1. **MediaPipe landmarks** -- 478-point facial mesh extraction
2. **Anatomical displacement** -- procedure-specific landmark shifts (intensity 0-100)
3. **TPS deformation** -- thin-plate spline warps the image smoothly
4. **Masked compositing** -- blends the surgical region back into the original

GPU modes (ControlNet, img2img) with photorealistic rendering are available in the full package.

## Links

- [GitHub](https://github.com/dreamlessx/LandmarkDiff-public)
- [Documentation](https://github.com/dreamlessx/LandmarkDiff-public/tree/main/docs)
- [Wiki](https://github.com/dreamlessx/LandmarkDiff-public/wiki)
- [Discussions](https://github.com/dreamlessx/LandmarkDiff-public/discussions)

**Version:** v0.2.2
