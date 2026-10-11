# Input Files

BrainBuilder is driven by three CSV files describing the hemispheres, tissue chunks (slabs), and individual 2D sections to reconstruct. All inputs are checked by `brainbuilder.utils.validate_inputs` before the pipeline starts.

## hemi_info.csv

One row per (subject, hemisphere).

| Column | Required | Description |
|---|---|---|
| `sub` | yes | Subject identifier. |
| `hemisphere` | yes | Hemisphere identifier (e.g., `L`, `R`, or `B`). |
| `struct_ref_vol` | yes | Path to the structural reference volume (e.g., donor T1w MRI or a stereotaxic template), NIfTI format. |
| `gm_surf` | surface interp only | Path to the gray matter (pial) surface mesh (e.g., `.surf.gii`). Required when `interp_method="surface"`. |
| `wm_surf` | surface interp only | Path to the white matter surface mesh. Required when `interp_method="surface"`. |

Example:

```csv
sub,hemisphere,struct_ref_vol,gm_surf,wm_surf
MR1,R,/data/mr1/t1w.nii.gz,/data/mr1/gray_R.surf.gii,/data/mr1/white_R.surf.gii
```

## chunk_info.csv

One row per tissue chunk (slab).

| Column | Required | Description |
|---|---|---|
| `sub` | yes | Subject identifier (must match `hemi_info.csv`). |
| `hemisphere` | yes | Hemisphere identifier. |
| `chunk` | yes | Integer chunk/slab identifier. |
| `section_thickness` | yes | Physical section thickness in mm (e.g., `0.02` for 20 µm). |
| `section_axis` | no | Axis of the reference volume along which sections were acquired: `0`/`1`/`2` or `x`/`y`/`z`. Default `1` (coronal). |

Example:

```csv
sub,hemisphere,chunk,section_thickness
MR1,R,1,0.02
MR1,R,2,0.02
```

## sect_info.csv

One row per acquired 2D section.

| Column | Required | Description |
|---|---|---|
| `sub` | yes | Subject identifier. |
| `hemisphere` | yes | Hemisphere identifier. |
| `chunk` | yes | Chunk the section belongs to. |
| `acquisition` | yes | Acquisition/modality label (e.g., ligand or stain name). Sections of the same acquisition are reconstructed into one volume. |
| `raw` | yes | Path to the raw 2D section image (NIfTI or PNG/TIFF; anything loadable by nibabel or PIL). Must be unique across rows. |
| `sample` | yes | Integer section number giving the position of the section along the sectioning axis within the chunk. Must be unique within a chunk. |

Example:

```csv
sub,hemisphere,chunk,acquisition,raw,sample
MR1,R,1,oxot,/data/raw/R_slab_1_oxot_0012.nii.gz,12
MR1,R,1,oxot,/data/raw/R_slab_1_oxot_0024.nii.gz,24
```

## Validation

`validate_inputs()` checks:

- All mandatory columns are present in each CSV.
- Every referenced file (`raw`, `struct_ref_vol`, surfaces) exists and can be loaded.
- `sample` values are unique within each chunk.
- `raw` paths are unique across the dataset.
- `section_axis`, when given, is a valid axis specifier.

Results are cached in `{output_dir}/valid_inputs.npz`; rerun with `clobber=True` (or `csv_clobber=True`) to force revalidation after fixing inputs.

## See Also

- [outputs.md](outputs.md) — what the pipeline produces from these inputs
- [landmarks.md](landmarks.md) — optional landmark files for guided alignment
