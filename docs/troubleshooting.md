# Troubleshooting

Common failures, what they mean, and how to recover.

## Rerunning and Caching

BrainBuilder caches aggressively. Every stage writes a manifest CSV and skips itself if that CSV exists; individual files are additionally gated by existence checks.

- `clobber=True` — recompute everything.
- `csv_clobber=True` — delete only the stage manifest CSVs so stages re-run their bookkeeping while reusing existing image files. Use this when a manifest references files you have moved or deleted, or after upgrading BrainBuilder.
- **Changing `resolution_list` (especially its minimum value) on an existing output directory** regenerates the downsampled/segmented images under new filenames, but some cached manifests may still reference files from the previous run. If you see dimension/shape-mismatch assertions after changing the resolution schedule, delete the stage manifest CSVs (`csv_clobber=True`) or use a fresh `output_dir`.

## Input Validation Errors

| Message | Cause / fix |
|---|---|
| `Error: invalid inputs` | One or more validation checks failed; look earlier in the log for the specific file. |
| `Missing input: file does not exist {fn}` | A path in one of the CSVs is wrong. Paths are read verbatim; use absolute paths. |
| `Incorrect format: ... expected path to volume` | The file exists but cannot be loaded by nibabel/PIL. Check the file is a valid NIfTI/PNG. |
| `Duplicate sample numbers found in {sect_info_csv}` | Two rows in the same chunk share a `sample` value. Each section needs a unique position. |
| `Duplicate raw input files found in {sect_info_csv}` | The same `raw` file appears in more than one row. |
| `Invalid section_axis value` | `section_axis` must be `0/1/2` or `x/y/z`. |

Validation results are cached in `{output_dir}/valid_inputs.npz` — delete it or pass `clobber`/`csv_clobber` to revalidate after fixing inputs.

## Alignment Stage Errors

| Message | Cause / fix |
|---|---|
| `Error: 0 denominator when interpolating missing sections` | A chunk has too few usable sections (or empty segmentations) to interpolate between. Check the `seg` images from `1_seg` for empty masks. |
| `Error: input volume sum is 0 in recenter` | An intermediate chunk volume is empty, usually meaning segmentation failed for all sections of a chunk. Inspect `1_seg` outputs and consider another `seg_method`. |
| `Inconsistent dimensions in 2d_align` / shape-mismatch assertions | Typically stale cached files after a change of `resolution_list` on an existing output directory. See "Rerunning and Caching" above. |
| `Error: cannot load file ...` from `antsApplyTransforms` | A truncated or partially written NIfTI, e.g. after an interrupted run. Delete the named file and rerun. |
| `No landmark files found in {landmark_dir}` | Landmark filenames must begin with the basename of the corresponding `raw` file; see [landmarks.md](landmarks.md). |

## Interpolation Stage Errors

| Message | Cause / fix |
|---|---|
| `Could not resolve matching acquisition/segmentation section columns` | `sect_info` lacks the expected alignment columns, usually because the multiresolution stage was skipped or its manifest is stale. Rerun with `use_3d_align_stage=True` or clear the stage manifests. |
| `Acquisition/Segmentation section file does not exist for ...` | A path recorded in the manifest no longer exists on disk. Rerun with `csv_clobber=True`. |
| `Acquisition and segmentation interpolation inputs must have the same in-plane shape` | Stale files from a previous run at a different resolution; clear manifests or use a fresh output directory. |
| `Error: Empty Output` / `Error: Output file not found` | The final transform to reference space produced an empty volume — usually a degenerate `nl_3d_tfm_fn` from a failed 3D alignment. Inspect `3.2_align_3d/*_moving_rsl.nii.gz` for that chunk. |
| `Error: no hemisphere info found` / `Error: no chunk info found` | `sub`/`hemisphere`/`chunk` values are inconsistent between the three input CSVs (including type mismatches like `"1"` vs `1`). |

## Performance

- ANTs/ITK ignores `OMP_NUM_THREADS`; BrainBuilder pins `ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=1` inside its parallel sections so that `num_cores` workers don't oversubscribe the CPU. If you wrap BrainBuilder functions in your own multiprocessing, do the same.
- For a quick first run, use a short, coarse `resolution_list` (e.g., `[4, 2]`) and reduced iteration budgets, as shown in the Quick Start in the [README](../README.md).
- nnUNet segmentation requires a GPU to run in a reasonable time; use `seg_method="otsu"` or `"triangle"` for a quick CPU-only test.

## Getting Help

If you hit an issue not covered here, run with `verbose=True` and open an issue with the log, your three input CSVs, and the BrainBuilder version.
