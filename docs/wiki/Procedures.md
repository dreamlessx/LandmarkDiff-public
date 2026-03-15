# Surgical Procedures

LandmarkDiff supports six facial surgical procedures, each defined by a set of MediaPipe landmark indices, displacement vectors, and influence radii. All procedures use a 0-100 intensity scale, where 33 is mild, 66 is moderate, and 100 is aggressive.

## Rhinoplasty (Nose Reshaping)

**Influence radius:** 30.0 px (at 512x512)

Rhinoplasty simulation targets three anatomical sub-regions:

### Alar Base Narrowing
Moves nostrils inward toward the midline. Left nostril landmarks (240, 236, 141, 363, 370) move right (+X), right nostril landmarks (460, 456, 274, 275, 278, 279) move left (-X). Displacement: 2.5 * scale per axis. Uses a tighter influence radius (0.6x) to prevent cheek distortion.

### Tip Refinement
Moves the nasal tip upward, simulating tip rotation. Landmarks 1, 2, 94, 19 receive (0, -2.0 * scale) displacement. Influence radius is 0.5x to keep the effect localized.

### Dorsum Narrowing
Bilateral squeeze of the nasal bridge. Left dorsum (195, 197, 236) moves right, right dorsum (326, 327, 456) moves left. Displacement: 1.5 * scale. Influence radius: 0.5x.

**Full landmark set (25 indices):**
`1, 2, 4, 5, 6, 19, 94, 141, 168, 195, 197, 236, 240, 274, 275, 278, 279, 294, 326, 327, 360, 363, 370, 456, 460`

**Clinical note:** At high intensities (>80), the alar narrowing can produce an unnatural pinched appearance. Consider keeping intensity around 50-70 for realistic results.

---

## Blepharoplasty (Eyelid Surgery)

**Influence radius:** 15.0 px (at 512x512)

Targets the periorbital region with three effects:

### Upper Lid Elevation (Primary)
Central upper lid landmarks move upward. Left: 159, 160, 161. Right: 386, 385, 384. Displacement: (0, -2.0 * scale). This is the primary surgical effect: removing excess skin/fat from the upper eyelid.

### Corner Tapering
Medial and lateral lid corners receive reduced displacement for a natural tapered effect. Left: 158, 157, 133, 33. Right: 387, 388, 362, 263. Displacement: (0, -0.8 * scale) at 0.7x influence radius.

### Lower Lid Tightening
Subtle upward pull on lower lid landmarks. Left: 145, 153, 154. Right: 374, 380, 381. Displacement: (0, +0.5 * scale) at 0.5x influence radius.

**Full landmark set (28 indices):**
`33, 7, 163, 144, 145, 153, 154, 155, 157, 158, 159, 160, 161, 246, 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398`

**Clinical note:** The tight influence radius (15px) is important here: the periorbital region is small and spreading deformation too far creates unnatural cheek distortion.

---

## Rhytidectomy (Facelift)

**Influence radius:** 40.0 px (at 512x512)

The most complex procedure, targeting three facial zones with different displacement patterns:

### Jowl Area (Strongest Lift)
Bilateral lift toward the ears. Left jowl (132, 136, 172, 58, 150, 176): displacement (-2.5 * scale, -3.0 * scale). Right jowl (361, 365, 397, 288, 379, 400): displacement (+2.5 * scale, -3.0 * scale). Full influence radius. The lateral+vertical pull simulates the SMAS plication direction.

### Submental / Chin
Upward-only displacement on chin landmarks (152, 148, 377, 378). Displacement: (0, -2.0 * scale) at 0.8x radius. No lateral movement: this region doesn't benefit from lateral tension.

### Temple / Upper Face
Very mild lift. Left temple (10, 21, 54, 67, 103, 109, 162, 127): displacement (-0.5 * scale, -1.0 * scale). Right temple (284, 297, 332, 338, 323, 356, 389, 454): displacement (+0.5 * scale, -1.0 * scale). Reduced radius (0.6x). The forehead and temples are mostly untouched in a standard facelift.

**Full landmark set (33 indices):**
`10, 21, 54, 58, 67, 93, 103, 109, 127, 132, 136, 150, 162, 172, 176, 187, 207, 213, 234, 284, 297, 323, 332, 338, 356, 361, 365, 379, 389, 397, 400, 427, 454`

**Clinical note:** The wide influence radius (40px) is intentional: facelifts redistribute tissue broadly. Ehlers-Danlos clinical flag scales this to 60px for hypermobile tissue.

---

## Orthognathic Surgery (Jaw Repositioning)

**Influence radius:** 35.0 px (at 512x512)

Simulates mandibular advancement, setback, and chin projection changes:

### Lower Jaw Repositioning
Landmarks along the lower mandible (17, 18, 200, 201, 202, 204, 208, 211, 212, 214) move upward. Displacement: (0, -3.0 * scale). This simulates the visible effect of sagittal split osteotomy.

### Chin Projection
Chin point landmarks (175, 170, 169, 167, 396) move upward/forward. Displacement: (0, -2.0 * scale) at 0.7x radius.

### Lateral Jaw Narrowing
Bilateral inward pull for jaw narrowing. Left jaw (57, 61, 78, 91, 95, 146, 181): displacement (+1.5 * scale, -1.0 * scale). Right jaw (291, 311, 312, 321, 324, 325, 375, 405): displacement (-1.5 * scale, -1.0 * scale). Radius: 0.8x.

**Full landmark set (47 indices):**
`0, 17, 18, 36, 37, 39, 40, 57, 61, 78, 80, 81, 82, 84, 87, 88, 91, 95, 146, 167, 169, 170, 175, 181, 191, 200, 201, 202, 204, 208, 211, 212, 214, 269, 270, 291, 311, 312, 317, 321, 324, 325, 375, 396, 405, 407, 415`

---

## Brow Lift (Brow Elevation)

**Influence radius:** 25.0 px (at 512x512)

Contributed by community member **Deepak8858** ([PR #35](https://github.com/dreamlessx/LandmarkDiff-public/pull/35)).

### Brow Elevation
Left brow: landmarks 70, 63, 105, 66, 107. Right brow: landmarks 300, 293, 334, 296, 336. Lateral brow is lifted more than medial, matching real surgical technique. Per-landmark weights increase from medial (0.7) to lateral (1.1). Displacement: (0, -4.0 * weight * scale).

### Forehead Smoothing
Forehead landmarks (9, 8, 10, 109, 67, 103, 338, 297, 332) get a mild upward pull. Displacement: (0, -1.5 * scale) at 1.2x radius for a broad smooth effect.

**Full landmark set (19 indices):**
`70, 63, 105, 66, 107, 300, 293, 334, 296, 336, 9, 8, 10, 109, 67, 103, 338, 297, 332`

---

## Mentoplasty (Chin Surgery)

**Influence radius:** 25.0 px (at 512x512)

Contributed by community member **P-r-e-m-i-u-m** ([PR #36](https://github.com/dreamlessx/LandmarkDiff-public/pull/36)).

### Chin Tip Advancement
Chin tip landmarks (152, 175) move upward/forward. Displacement: (0, -4.0 * scale). Full influence radius.

### Lower Chin Contour
Lower contour landmarks (148, 149, 150, 176, 377) follow with softer displacement: (0, -2.5 * scale) at 0.8x radius. This creates a natural transition from the advanced tip to the unchanged jaw angle.

### Jaw Angle Transition
Jaw angle landmarks (171, 396) receive minimal displacement: (0, -1.0 * scale) at 0.6x radius. Prevents an abrupt boundary between the chin modification and the natural jaw.

**Full landmark set (8 indices):**
`148, 149, 150, 152, 171, 175, 176, 377`

---

## Adding Custom Procedures

See the [Contributing](Contributing) page for instructions on adding new procedures. The process involves:

1. Identifying relevant MediaPipe landmark indices for the target anatomy
2. Defining displacement vectors that match the surgical effect
3. Choosing an influence radius appropriate for the region size
4. Adding mask configuration in `masking.py`
5. Adding procedure prompts in `inference.py` (for diffusion modes)
