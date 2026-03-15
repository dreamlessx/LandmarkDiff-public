# Clinical Flags

LandmarkDiff handles four pathological conditions that affect how surgical deformations and mask compositing should behave. These are controlled through the `ClinicalFlags` dataclass.

## Overview

```python
from landmarkdiff.clinical import ClinicalFlags

flags = ClinicalFlags(
    vitiligo=False,
    bells_palsy=False,
    bells_palsy_side="left",
    keloid_prone=False,
    keloid_regions=[],
    ehlers_danlos=False,
)
```

Pass clinical flags to the pipeline at construction or per-generation:

```python
pipe = LandmarkDiffPipeline(mode="controlnet", clinical_flags=flags)
# or per-call:
result = pipe.generate(image, clinical_flags=flags)
```

## Vitiligo

**What it does:** Preserves depigmented skin patches during mask compositing.

**Problem:** Standard compositing blends the generated face over the entire surgical region, which can overwrite characteristic vitiligo depigmentation patterns. This produces unrealistic results for patients with vitiligo.

**How it works:**

1. **Detection:** `detect_vitiligo_patches()` identifies depigmented regions within the face ROI using LAB color space analysis. It looks for pixels that are both significantly brighter than surrounding skin (high L channel) and low in color saturation (A and B channels close to 128). A minimum patch area filter (default 200 pixels) removes noise.

2. **Mask adjustment:** `adjust_mask_for_vitiligo()` reduces the compositing mask intensity over detected patches by a preservation factor (default 0.3). This lets the original vitiligo pattern show through rather than being overwritten.

**Parameters:**
- `l_threshold`: Luminance threshold for patch detection (default 85.0)
- `min_patch_area`: Minimum contour area in pixels (default 200)
- `preservation_factor`: How much to reduce blending (0 = full blend, 1 = fully preserve, default 0.3)

**Note:** Vitiligo detection requires the original image to be passed to `generate_surgical_mask()`.

## Bell's Palsy

**What it does:** Disables deformation on the paralyzed side of the face.

**Problem:** Bell's palsy causes unilateral facial paralysis. Standard bilateral symmetric deformations would attempt to move landmarks on the paralyzed side, which cannot move in real life. This produces an unrealistic prediction.

**How it works:**

1. `get_bells_palsy_side_indices()` returns landmark indices for the affected side, grouped by region (eye, eyebrow, mouth corner, jawline).

2. During `apply_procedure_preset()`, deformation handles whose `landmark_index` falls on the affected side are removed from the handle list. Only the healthy side receives deformation.

**Parameters:**
- `bells_palsy_side`: Which side is affected: `"left"` or `"right"`

**Affected landmark groups (left side example):**
- Eye: 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246
- Eyebrow: 70, 63, 105, 66, 107, 55, 65, 52, 53, 46
- Mouth corner: 61, 146, 91, 181, 84
- Jawline: 132, 136, 172, 58, 150, 176, 148, 149

## Keloid Prone

**What it does:** Softens mask transitions in regions prone to keloid scarring.

**Problem:** Hard compositing boundaries can create sharp visual transitions. For keloid-prone patients, these would correspond to incision lines with aggressive tissue manipulation, which is clinically undesirable. The prediction should reflect the gentler surgical approach used for these patients.

**How it works:**

1. `get_keloid_exclusion_mask()` generates a mask of keloid-prone regions from the specified anatomical areas. It uses the convex hull of region landmarks plus a configurable margin (default 10px).

2. `adjust_mask_for_keloid()` reduces mask intensity in keloid regions by a reduction factor (default 0.5) and applies additional Gaussian blur (kernel 31, sigma 10) for softer transitions. The blurred version is used only within keloid regions.

**Parameters:**
- `keloid_regions`: List of region names, e.g. `["jawline", "nose"]`
- `margin_px`: Extra dilation around keloid regions (default 10)
- `reduction_factor`: How much to reduce mask intensity (default 0.5)

## Ehlers-Danlos Syndrome

**What it does:** Increases deformation influence radii by 50% to simulate hypermobile tissue.

**Problem:** Patients with Ehlers-Danlos syndrome have connective tissue that stretches more than normal. Standard influence radii produce deformations that are too localized for this tissue type.

**How it works:**

In `apply_procedure_preset()`, when `clinical_flags.ehlers_danlos` is True, the procedure's influence radius is multiplied by 1.5 before building deformation handles. For example, rhinoplasty's radius goes from 30.0 to 45.0, and rhytidectomy's from 40.0 to 60.0.

This single change affects all downstream deformations: each handle's Gaussian RBF falloff covers a wider area, creating broader, more distributed tissue displacement.

## Combining Flags

Multiple flags can be active simultaneously:

```python
flags = ClinicalFlags(
    vitiligo=True,
    keloid_prone=True,
    keloid_regions=["jawline"],
    ehlers_danlos=True,
)
```

The effects stack:
1. Ehlers-Danlos widens influence radii during deformation
2. Keloid softens the compositing mask in jawline regions
3. Vitiligo preserves depigmented patches during compositing

Bell's palsy operates at the deformation level (removing handles), while vitiligo and keloid operate at the mask level (adjusting compositing weights). Ehlers-Danlos operates at the radius level (widening influence). There are no conflicts between them.

## Checking Active Flags

```python
flags = ClinicalFlags(vitiligo=True, ehlers_danlos=True)
print(flags.has_any())  # True
```

The `has_any()` method returns True if any clinical flag is set.
