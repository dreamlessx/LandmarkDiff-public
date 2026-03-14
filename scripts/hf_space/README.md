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
short_description: Facial surgery prediction -- 2D foundation toward 3D
tags:
  - medical-imaging
  - face
  - landmarks
  - thin-plate-spline
  - surgery-simulation
  - facial-analysis
  - symmetry
---

# LandmarkDiff

Anatomically-conditioned facial surgery outcome prediction from standard clinical photography.

Upload a face photo, select a surgical procedure, adjust intensity, and see the predicted outcome in real time using thin-plate spline warping on CPU.

## Features

- **Single Procedure**: Predict the outcome of one procedure at a chosen intensity
- **Compare All**: See all six procedures side by side at the same intensity
- **Intensity Sweep**: View a single procedure from 0% to 100% in six steps
- **Symmetry Analysis**: Measure bilateral facial symmetry across five regions, with pre/post comparison

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
2. **Anatomical displacement** -- procedure-specific landmark shifts scaled by intensity (0-100)
3. **TPS deformation** -- thin-plate spline warps the image smoothly
4. **Masked compositing** -- blends the surgical region back into the original photo

GPU modes (ControlNet, img2img) with photorealistic rendering are available in the full package.

**Roadmap:** This 2D TPS demo is the foundation. Next up -- 3D face reconstruction from phone video for interactive surgical preview.

## Photo Tips

- Use a front-facing, well-lit photo with a neutral expression
- Remove glasses, hats, or anything covering the face
- Make sure only one face is clearly visible
- At least 256x256 resolution recommended

## Links

- [GitHub](https://github.com/dreamlessx/LandmarkDiff-public)
- [Documentation](https://github.com/dreamlessx/LandmarkDiff-public/tree/main/docs)
- [Wiki](https://github.com/dreamlessx/LandmarkDiff-public/wiki)
- [Discussions](https://github.com/dreamlessx/LandmarkDiff-public/discussions)

**Version:** v0.2.2 | **License:** MIT | For research and educational purposes only
