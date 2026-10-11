# Outputs

This page documents the output directory layout produced by `reconstruct()` and explains the main result files. The focus is on the 3D volumes and transforms that are passed between pipeline stages and on the final reconstructed volumes in `4_interp`. The many per-section 2D intermediates (one file per section, per resolution, per pass) are not documented individually.

## Directory Layout

```
output_dir/
├── valid_inputs.npz                  # cached input-validation results
├── 0_downsample/                     # per-section downsampled images ("img")
│   └── downsample_sect_info.csv
├── 1_seg/                            # per-section tissue segmentations ("seg")
│   └── downsample_sect_info_segment.csv
├── 1.5_intensity_corr/               # optional intensity/batch correction
├── 2_init_align/                     # initial rigid 2D alignment
│   ├── initalign_sect_info.csv
│   ├── initalign_chunk_info.csv
│   └── sub-{sub}/hemi-{hemi}/chunk-{chunk}/
│       └── sub-{sub}_hemi-{hemi}_chunk-{chunk}_init_align.nii.gz
├── 3_multires_align/                 # multiresolution 3D + 2D refinement
│   ├── sect_info_multiresolution_alignment.csv
│   ├── chunk_info_multiresolution_alignment.csv
│   └── sub-{sub}/hemi-{hemi}/chunk-{chunk}/{resolution}mm/pass_{N}/
│       ├── 3.1_intermediate_volume/
│       ├── 3.2_align_3d/
│       ├── 3.3_align_2d/
│       └── landmarks/
├── 4_interp/                         # missing-section interpolation and final volumes
│   └── reconstructed_chunk_info.csv
└── qc/                               # quality-control figures and reports
```

Most filenames follow a BIDS-like convention:

```
sub-{sub}_hemi-{hemisphere}_chunk-{chunk}_{resolution}mm_<description>.nii.gz
```

Each stage also writes a manifest CSV (listed above). These CSVs record the file paths of the stage's outputs in new columns appended to the section/chunk tables, and they are what downstream stages read to locate their inputs. If a manifest CSV exists, the stage is treated as complete and skipped; `csv_clobber=True` deletes only these manifests to force re-orchestration without recomputing files (see [troubleshooting.md](troubleshooting.md)).

## Stage Outputs Passed Downstream

### 2_init_align — Initial chunk volume

| File | Description |
|---|---|
| `sub-{sub}_hemi-{hemi}_chunk-{chunk}_init_align.nii.gz` | First 3D volume for a chunk, created by stacking the rigidly aligned 2D sections. This is the starting "moving" volume for the multiresolution alignment stage. |
| `initalign_sect_info.csv` | Section table with added columns `init_tfm` (per-section rigid 2D transform, `.h5`) and `init_img` (rigidly aligned section). |

### 3_multires_align — Multiresolution alignment

For each chunk, the pipeline iterates over the resolutions in `resolution_list` (coarse → fine). Each `{resolution}mm/pass_{N}/` directory contains three sub-stages. In the names below, `{prefix}` = `sub-{sub}_hemi-{hemi}_chunk-{chunk}_{resolution}mm` and `{prefix_3d}` = `sub-{sub}_hemi-{hemi}_chunk-{chunk}_itr-{resolution}_{resolution_3d}mm`, where `resolution_3d` is the 3D alignment resolution for that iteration (capped by `max_resolution_3d`).

#### 3.1_intermediate_volume

| File | Description |
|---|---|
| `{prefix_3d}_acq.nii.gz` | Intermediate acquisition volume: the current best 2D-aligned sections stacked into a 3D volume at the 3D alignment resolution. This is the moving volume for the 3D registration in `3.2_align_3d`. |
| `{prefix_3d}_acq_pad.nii.gz` | Same volume, zero-padded so that nonlinear deformations near the chunk boundary are not clipped. |

#### 3.2_align_3d — Chunk-to-reference 3D registration

| File | Description |
|---|---|
| `{prefix}_SyN_{CC\|Mattes}_Composite.h5` | **The 3D chunk-to-reference transform** (`nl_3d_tfm_fn`): a composite of the linear steps (`vol_lin_tfm_type`) and the SyN nonlinear warp. Maps the chunk's native (acquisition) space to the structural reference space. `CC` vs `Mattes` reflects the `use_3d_syn_cc` setting. |
| `{prefix}_SyN_{CC\|Mattes}_InverseComposite.h5` | Inverse of the above (`nl_3d_tfm_inv_fn`): maps the reference volume into chunk space. Used to bring the reference into section space for the 2D alignment in `3.3_align_2d`, and later to define the final interpolation targets. |
| `{prefix}_moving_rsl.nii.gz` | The chunk volume resampled into reference space with the current 3D transform (`rec_3d_rsl_fn`). Useful for visually checking the 3D registration. |
| `{prefix_3d}_mri_gm_space-rec.nii.gz` | The reference (GM) volume transformed into the chunk's reconstruction space (`ref_3d_rsl_fn`). Each section of this volume serves as the fixed target for the per-section 2D alignment. |
| `sub-{sub}_hemi-{hemi}_chunk-{chunk}_{resolution_3d}mm_ref_vol.nii.gz` | Structural reference volume resampled to the current 3D resolution (`ref_rsl_fn`). |

#### 3.3_align_2d — Section-to-reference 2D refinement

| File | Description |
|---|---|
| `{prefix}_nl_2d.nii.gz` | Volume assembled from the 2D nonlinearly aligned sections (`nl_2d_vol_fn`). This is the chunk's best reconstruction at the current resolution and becomes the input for the next (finer) resolution iteration. |
| `{prefix}_nl_2d_cls.nii.gz` | Segmentation (tissue-classification) counterpart of the above (`nl_2d_vol_cls_fn`), built from the 2D-aligned section segmentations. |
| `{prefix}_ref_space-nat.nii.gz` | Reference volume in native chunk space at the 2D resolution. |

At the final (finest) resolution, the `pass` outputs of `3.3_align_2d` — together with `nl_3d_tfm_fn` — are what the interpolation stage consumes. The per-section columns `2d_align` (aligned acquisition section), `2d_align_cls` (aligned segmentation) and `2d_align_3d_res` recorded in `sect_info_multiresolution_alignment.csv` point to the individual aligned sections used to build the thickened volumes below.

## 4_interp — Interpolation Outputs

The interpolation stage fills in the gaps between acquired sections and produces the final reconstructed volumes. Outputs are organized as:

```
4_interp/
├── reconstructed_chunk_info.csv
├── atlas.nii.gz                       # mean z-scored interpolated cls volume (all acquisitions)
├── atlas_mask.nii.gz                  # Otsu mask of the atlas
└── sub-{sub}/hemi-{hemi}/
    ├── {ref}_{resolution}mm.nii.gz    # structural reference resampled to output resolution
    ├── chunk-{chunk}/acq-{acq}/
    │   ├── acq/                       # interpolation of the acquisition intensities
    │   └── cls/                       # interpolation of the tissue segmentation
    └── acq-{acq}/
        └── sub-{sub}_hemi-{hemi}_acq-{acq}_{resolution}mm_cortex.nii.gz
```

### Per-chunk, per-acquisition volumes (`chunk-{chunk}/acq-{acq}/`)

The `acq/` and `cls/` subdirectories contain the same sequence of volumes, computed first for the acquisition intensities and then re-applied (with identical flow-field transforms) to the segmentation:

| File | Description |
|---|---|
| `sub-{sub}_hemi-{hemi}_{chunk}_{acq}_{resolution}_thickened.nii.gz` | "Thickened" volume: each acquired 2D-aligned section is expanded along the section axis to the reconstruction resolution, with empty slabs where sections are missing. Input to the interpolation. The `cls/` variant has a `_cls_thickened` suffix. |
| `*_interp-vol_orig.nii.gz` | Interpolated volume on the original (anisotropic) grid: missing sections filled in by the nonlinear deformation-flow interpolation (`morphint`). |
| `*_interp-vol_iso.nii.gz` | **Final interpolated volume in native chunk space**, resampled to isotropic voxels at the reconstruction resolution. Recorded as `interp_nat` (acq) and `interp_cls_nat` (cls) in `reconstructed_chunk_info.csv`. |
| `*_interp-vol_stx.nii.gz` | **Final interpolated volume in reference (stereotaxic) space**: the `_iso` volume resampled onto the resampled structural reference grid with the chunk's 3D transform (`nl_3d_tfm_fn`). Recorded as `interp_stx`. This is the primary per-chunk end product for analyses in reference space. |

If `final_resolution` is set, the interpolation is run at that resolution instead of the finest value of `resolution_list`.

### Combined hemisphere volumes (`sub-{sub}/hemi-{hemi}/acq-{acq}/`)

When a hemisphere has more than one chunk, the per-chunk `interp_stx` volumes are merged into a single volume per acquisition:

| File | Description |
|---|---|
| `sub-{sub}_hemi-{hemi}_acq-{acq}_{resolution}mm_cortex.nii.gz` | **Final combined reconstruction** (volumetric interpolation): all chunks of the hemisphere merged in reference space, masked by the resampled reference volume. |
| `sub-{sub}_hemi-{hemi}_acq-{acq}_{resolution}mm_l{n_depths}_cortex.nii.gz` | Final combined reconstruction when `interp_method="surface"`; values are interpolated along `n_depths` cortical depth surfaces between the WM and GM meshes before being projected back into the volume. A `*_cortex_smoothed.nii.gz` variant may also be written. |

Surface interpolation additionally writes resampled depth surfaces (`*_{resolution}mm_{depth}_rsl.npz`) and per-depth vertex profiles (`*_raw_profiles` / profile `.npz` archives) under `4_interp/surfaces/`.

### reconstructed_chunk_info.csv

The final manifest, one row per (sub, hemisphere, chunk, acquisition), with the key columns:

| Column | Description |
|---|---|
| `interp_nat` | Interpolated acquisition volume in native chunk space (`*_interp-vol_iso.nii.gz`). |
| `interp_cls_nat` | Interpolated segmentation volume in native chunk space. |
| `interp_stx` | Interpolated acquisition volume in reference space (`*_interp-vol_stx.nii.gz`). |
| `nl_3d_tfm_fn` | 3D chunk-to-reference composite transform used to produce `interp_stx`. |
| `ref_vol_rsl_fn` | Structural reference volume resampled to the output resolution (the grid of `interp_stx`). |

Use `interp_stx` (or the combined `*_cortex.nii.gz`) for downstream analyses in reference space, and `interp_nat` when working in the chunk's native acquisition space.

## See Also

- [input-files.md](input-files.md) — required input CSVs and columns
- [developer-guide.md](developer-guide.md) — how each stage adds columns to the section/chunk tables
- [troubleshooting.md](troubleshooting.md) — caching behavior, clobber flags, and common errors
