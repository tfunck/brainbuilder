# Landmark-Guided Alignment

Landmark-assisted chunk localization is optional. It helps the 3D chunk-to-reference registration when a chunk is small, badly deformed, or hard to localize within the reference volume by intensity alone. Enable it by passing `landmark_dir=...` to `reconstruct()`.

## How It Works

1. You draw matching landmark labels on (a) individual 2D sections and (b) the structural reference volume.
2. During multiresolution alignment, the pipeline transforms the per-section landmark images with the current best 2D transforms and stacks them into a 3D landmark volume for the chunk (`*_acq_landmarks_itr-{resolution}mm.nii.gz`).
3. A landmark-to-landmark registration between the chunk landmark volume and the reference landmark volume initializes the 3D registration in `3.2_align_3d`.

## File Naming and Format

- Landmark images are NIfTI label images (`.nii.gz`): background is `0`, each landmark is painted with an integer label ≥ 1.
- For a section with raw file `<raw_root>.<ext>`, the pipeline looks for `{landmark_dir}/<raw_root>*.nii.gz`. The landmark image must be in the same pixel space as the raw section.
- The reference landmark volume must use the same integer labels for the same anatomical points as the section landmarks: label `n` in a section landmark image corresponds to label `n` in the reference landmark volume.

## Label Mapping Rules

- Each anatomical landmark gets one unique integer label, used consistently across all sections and the reference volume.
- A section needs at least 2 landmarks to contribute; sections with fewer are skipped with a warning.
- The number of matched landmarks determines the landmark-initialization transform:

| Landmarks | Transform |
|---|---|
| < 2 | skipped |
| 2–11 | Rigid |
| 12–19 | Affine |
| ≥ 20 | SyN (nonlinear) |

## Outputs

Landmark transforms are written per chunk and resolution iteration under the stage's `landmarks/` directory:

```
sub-{sub}_hemi-{hemi}_chunk-{chunk}_landmark_init_itr-{resolution}mm_{transform}_{mesh_size}_Composite.h5
sub-{sub}_hemi-{hemi}_chunk-{chunk}_landmark_init_itr-{resolution}mm_{transform}_{mesh_size}_InverseComposite.h5
```

## Tips

- Prefer unambiguous, widely spaced anatomical points (sulcal fundi, ventricle corners, subcortical boundaries).
- A few reliable landmarks on several sections spread along the chunk beat many landmarks on a single section.
- If you see `No landmark files found in {landmark_dir}`, check that the landmark filenames start with the exact basename (without extension) of the corresponding `raw` file.
