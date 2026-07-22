import os
from subprocess import run

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from skimage.transform import resize

# from brainbuilder.utils.nl_deformation_flow import nlflow_isometric
from morphint.morphint import morphint

import brainbuilder.utils.ants_nibabel as nib
from brainbuilder.align.align_2d import apply_transforms_parallel
from brainbuilder.interp.acqvolume import create_thickened_volumes
from brainbuilder.utils import utils
from brainbuilder.utils.axis_utils import DEFAULT_SECTION_AXIS, get_section, get_section_axis

logger = utils.get_logger(__name__)


def idw(vol, nearest_i, min_dist, p=2, axis=DEFAULT_SECTION_AXIS):
    # Inverse Distance Weighted interpolation (IDW) with Shepard's method
    # min_dist += 1
    weights = min_dist / np.sum(min_dist)

    print(min_dist, weights)
    interp = np.sum(
        [w * get_section(vol, nearest_i[j], axis) for j, w in enumerate(weights)],
        axis=0,
    )

    assert np.sum(np.abs(interp)) > 0, "Error: Empty Output"

    return interp


def apply_final_2d_transforms(
    curr_sect_info: pd.DataFrame,
    final_tfm_dir: str,
    final_resolution: float,
    interpolation: str = "Linear",
    num_cores: int = -1,
):
    """Apply the final 2D transforms to the raw and segmented images.
    Parameters
    ----------
    curr_sect_info : pd.DataFrame
        DataFrame containing the section information for a single chunk.
    final_tfm_dir : str
        Directory to save the final transformed images.
    final_resolution : float
        Final resolution of the images.
    interpolation : str
        Interpolation method to use. Options are 'Linear', 'NearestNeighbor', 'Gaussian', 'MultiLabel'.
    num_cores : int
        Number of cores to use for parallel processing.
    """
    os.makedirs(final_tfm_dir, exist_ok=True)

    curr_sect_info["2d_align_out"] = curr_sect_info["img"].apply(
        lambda x: f"{final_tfm_dir}/{os.path.basename(x)}"
    )
    curr_sect_info["2d_align_cls"] = curr_sect_info["seg"].apply(
        lambda x: f"{final_tfm_dir}/{os.path.basename(x)}"
    )

    Parallel(n_jobs=num_cores, backend="multiprocessing")(
        # FIXME need to put the raw images through the same padding process as in downsample
        # otherwise we cannot put them together in the same volume.
        delayed(apply_transforms_parallel)(
            final_tfm_dir,
            final_resolution,
            row,
            interpolation=interpolation,
            file_str="img",  #'raw'
        )
        for _, row in curr_sect_info.iterrows()
    )

    # if cls in curr_sect_info.columns:
    if "2d_align_cls_out" in curr_sect_info.columns:
        Parallel(n_jobs=num_cores, backend="multiprocessing")(
            delayed(apply_transforms_parallel)(
                final_tfm_dir, final_resolution, row, tissue_str="_cls", file_str="seg"
            )
            for _, row in curr_sect_info.iterrows()
        )

    return curr_sect_info


# Acquisition and segmentation sections must be interpolated from matching
# resolution columns.  The first pair is for sections already resampled into the
# 3D reconstruction grid, which is normally what stage 3 multiresolution
# alignment consumes.  The next two pairs are 2D-resolution section images, used
# when stage 4 interpolation should preserve the native in-plane section grid.
SECTION_TARGET_PAIRS = [
    ("2d_align_3d_res", "seg_rsl_tfm"),
    ("2d_align_out", "2d_align_cls_out"),
    ("2d_align", "2d_align_cls"),
    ("img", "seg"),
]


def resolve_target_sections(
    sect_info: pd.DataFrame,
    target_section: str = None,
    target_section_acq: str = None,
    target_section_cls: str = None,
) -> tuple[str, str]:
    """Choose matching acquisition and segmentation section columns.

    `target_section` is the legacy single-column selector.  It may name either
    side of a known acquisition/segmentation pair; for example, passing
    `2d_align_3d_res` resolves to (`2d_align_3d_res`, `seg_rsl_tfm`), while
    passing `2d_align_out` resolves to (`2d_align_out`, `2d_align_cls_out`).

    `target_section_acq` and `target_section_cls` are the explicit form.  Use
    these when the caller already knows which resolution should be used.  This is
    important because the 3D multiresolution alignment stage and the missing-
    intensity interpolation stage can require different section grids:

    - 3D reconstruction resolution: `2d_align_3d_res` with `seg_rsl_tfm`
    - 2D section resolution after 2D alignment: `2d_align_out` with
      `2d_align_cls_out`
    - original/init 2D alignment columns: `2d_align` with `2d_align_cls`
    - raw image/segmentation columns: `img` with `seg`

    If nothing is specified, the first available complete pair from
    `SECTION_TARGET_PAIRS` is used.  That means 3D-resolution columns are
    preferred when present, so stage 4 callers that need 2D-resolution sections
    should pass `target_section="2d_align_out"` or the explicit acq/cls columns.
    """
    if target_section_acq is not None and target_section_cls is not None:
        pass
    else:
        for acq_col, cls_col in SECTION_TARGET_PAIRS:
            if target_section in [acq_col, cls_col]:
                target_section_acq = target_section_acq or acq_col
                target_section_cls = target_section_cls or cls_col
                break

    if target_section_acq is None or target_section_cls is None:
        for acq_col, cls_col in SECTION_TARGET_PAIRS:
            if acq_col in sect_info.columns and cls_col in sect_info.columns:
                target_section_acq = target_section_acq or acq_col
                target_section_cls = target_section_cls or cls_col
                break

    missing = [
        col
        for col in [target_section_acq, target_section_cls]
        if col is None or col not in sect_info.columns
    ]
    if missing:
        raise ValueError(
            "Could not resolve matching acquisition/segmentation section columns. "
            f"target_section={target_section!r}, "
            f"target_section_acq={target_section_acq!r}, "
            f"target_section_cls={target_section_cls!r}, "
            f"available columns={list(sect_info.columns)}"
        )

    return target_section_acq, target_section_cls


def validate_target_section_pair(
    sect_info: pd.DataFrame, target_section_acq: str, target_section_cls: str
) -> None:
    """Fail early if the chosen acq/cls columns are missing files or resolutions differ."""
    for _, row in sect_info.iterrows():
        acq_fn = row[target_section_acq]
        cls_fn = row[target_section_cls]

        if not os.path.exists(acq_fn):
            raise FileNotFoundError(
                f"Acquisition section file does not exist for {target_section_acq}: {acq_fn}"
            )
        if not os.path.exists(cls_fn):
            raise FileNotFoundError(
                f"Segmentation section file does not exist for {target_section_cls}: {cls_fn}"
            )

        acq_shape = nib.load(acq_fn).shape
        cls_shape = nib.load(cls_fn).shape
        if acq_shape != cls_shape:
            raise ValueError(
                "Acquisition and segmentation interpolation inputs must have the "
                "same in-plane shape. "
                f"{target_section_acq}={acq_fn} has shape {acq_shape}; "
                f"{target_section_cls}={cls_fn} has shape {cls_shape}."
            )


def volumetric_interpolation(
    curr_sect_info: pd.DataFrame,
    curr_chunk_info: pd.DataFrame,
    output_dir: str,
    resolution: float,
    resolution_list: list,
    interpolation: str = "Linear",
    tissue_type: str = "acq",
    target_section: str = "2d_align_3d_res",
    refine_2d_alignment_flag: bool = False,
    nlflow_tfm_dict: dict = None,
    num_cores: int = -1,
    clobber: bool = False,
) -> pd.DataFrame:
    os.makedirs(output_dir, exist_ok=True)

    axis = get_section_axis(curr_chunk_info)

    print("Volumetric Interpolation")
    chunk_info_thickened_csv = create_thickened_volumes(
        output_dir,
        curr_chunk_info,
        curr_sect_info,
        resolution,
        tissue_type=tissue_type,
        target_section=target_section,
        clobber=clobber,
        width=0,
    )

    chunk_info_thickened = pd.read_csv(chunk_info_thickened_csv)

    assert (
        len(chunk_info_thickened) == 1
    ), "Error: More than one chunk in the chunk info, should only be 1."

    acq_fin = chunk_info_thickened["thickened"].values[0]

    interp_iso_fin, nlflow_tfm_dict = morphint(
        acq_fin,
        output_dir,
        resolution,
        resolution_list,
        axis=axis,
        interpolation=interpolation,
        tfm_dict=nlflow_tfm_dict,
        num_jobs=num_cores,
        refine_2d_alignment_flag = refine_2d_alignment_flag,
        base_ants_itr = 40,
        n_resolutions = 4,
        clobber=clobber,
    )

    return interp_iso_fin, nlflow_tfm_dict


def volumetric_interpolation_over_dataframe(
    sect_info: pd.DataFrame,
    chunk_info: pd.DataFrame,
    output_dir: str,
    resolution: float,
    resolution_list: list,
    clobber: bool = False,
    final_resolution: float = None,
    refine_2d_alignment_flag: bool = False,
    interpolation: str = "Linear",
    tissue_type: str = "cls",
    target_section: str = "2d_align_3d_res",
    target_section_acq: str = None,
    target_section_cls: str = None,
    num_cores: int = -1,
) -> pd.DataFrame:
    """Interpolates the volumes of the sections in the chunk_info dataframe.
    The interpolation is done in the following steps:
    1. For each chunk, interpolate the missing sections in the volume.
    2. Transform the interpolated volume to the reference volume space.
    3. Interpolate between chunks in the reference volume space.

    Parameters
    ----------
    chunk_info : pd.DataFrame
        DataFrame containing the chunk information.
    struct_ref_vol : str

        Path to the reference volume.
    output_dir : str
        Directory to save the output files.
    clobber : bool, optional
        If True, overwrite existing files. The default is False.
    """
    chunk_info_out = pd.DataFrame(
        columns=["sub", "hemisphere", "chunk", "acquisition", "interp_nat"]
    )

    if final_resolution is not None and isinstance(final_resolution, float):
        resolution_list += [final_resolution]
        resolution = final_resolution

    for (sub, hemisphere, chunk, acq), curr_sect_info in sect_info.groupby(
        ["sub", "hemisphere", "chunk", "acquisition"]
    ):
        print(
            f"Processing sub: {sub}, hemisphere: {hemisphere}, chunk: {chunk}, acquisition: {acq}"
        )
        curr_output_dir = (
            f"{output_dir}/sub-{sub}/hemi-{hemisphere}/chunk-{chunk}/acq-{acq}/"
        )
        if (
            "align_2d" not in curr_sect_info.columns
            or "align_2d_cls" not in curr_sect_info.columns
        ):
            final_resolution = float(resolution_list[-1])

        # FIXME: Not yet implemented because 'raw' images need to be padded in the same way as in downsample images, 'img'
        # one possible solution is to keep the origin fixed and only pad around the exterior of the image,
        # but this needs to be implemented and tested in downsample.py first
        # if final_resolution is not None and isinstance(final_resolution, float):
        #    final_tfm_dir = curr_output_dir + "/final_tfm_2d"
        #
        #    curr_sect_info = apply_final_2d_transforms(
        #        curr_sect_info,
        #        final_tfm_dir,
        #        final_resolution,
        #        interpolation,
        #        num_cores,
        #    )

        curr_chunk_info = chunk_info[(chunk_info["chunk"] == chunk)]

        curr_target_section_acq, curr_target_section_cls = resolve_target_sections(
            curr_sect_info,
            target_section=target_section,
            target_section_acq=target_section_acq,
            target_section_cls=target_section_cls,
        )
        validate_target_section_pair(
            curr_sect_info, curr_target_section_acq, curr_target_section_cls
        )
        print(
            "Target sections for interpolation:",
            curr_target_section_acq,
            curr_target_section_cls,
        )

        # First calculate the interpolation for the acquisition volume
        interp_acq_iso_fin, nlflow_tfm_dict = volumetric_interpolation(
            curr_sect_info,
            curr_chunk_info,
            curr_output_dir + "/acq/",
            resolution,
            resolution_list,
            interpolation=interpolation,
            target_section=curr_target_section_acq,
            num_cores=num_cores,
            clobber=clobber,
            refine_2d_alignment_flag=refine_2d_alignment_flag,
        )

        # Then apply the same transformations (stored in nlflow_tfm_dict) to the segmentation volume
        interp_cls_iso_fin, _ = volumetric_interpolation(
            curr_sect_info,
            curr_chunk_info,
            curr_output_dir + "/cls/",
            resolution,
            resolution_list,
            tissue_type=tissue_type,
            target_section=curr_target_section_cls,
            nlflow_tfm_dict=nlflow_tfm_dict,
            num_cores=num_cores,
            clobber=clobber,
        )

        row = pd.DataFrame(
            {
                "sub": [sub],
                "hemisphere": [hemisphere],
                "chunk": [chunk],
                "acquisition": [acq],
                "interp_nat": [interp_acq_iso_fin],
                "interp_cls_nat": [interp_cls_iso_fin],
            }
        )

        chunk_info_out = pd.concat([chunk_info_out, row], ignore_index=True)

    return chunk_info_out


def create_mask(fn, out_fn, clobber: bool = False):
    if not os.path.exists(out_fn) or clobber:
        img = nib.load(fn)
        data = img.get_fdata()

        from skimage.filters import threshold_otsu

        t = threshold_otsu(data)  # Use Otsu's method for thresholding
        data[data <= t] = 0
        data[data > t] = 1  # Ensure binary segmentation

        nib.Nifti1Image(data, img.affine, direction_order="lpi").to_filename(out_fn)


def chunked_percentile(
    fins,
    fout,
    p=50,
    bins=256,
    dtype=np.float32,
    vmin=None,
    vmax=None,
    background=None,
    chunk=(48, 48, 48),
):
    """Average over a set of volumes by taking the p-th percentile (median by default) for each voxel over the volume stack.
    Compute the voxel-wise p-th percentile across a set of volumes in chunks to save memory.
    """
    ref = nib.load(fins[0])

    shape, affine = ref.shape, ref.affine

    Z, Y, X = shape
    cz, cy, cx = chunk

    # Establish global value range for binning (or pass your known range)
    if vmin is None or vmax is None:
        vmin, vmax = np.inf, -np.inf
        for f in fins:
            img = nib.load(f)
            # sample sparsely for speed
            s = np.asarray(img.dataobj[::8, ::8, ::8])
            if background is not None:
                s = s[s != background]
            if s.size:
                vmin = min(vmin, float(np.nanmin(s)))
                vmax = max(vmax, float(np.nanmax(s)))

        # Sparse sampling can miss non-zero signal in thin structures.
        # Fall back to full-volume range before declaring a degenerate input.
        if (not np.isfinite(vmin)) or (not np.isfinite(vmax)) or (vmax <= vmin):
            vmin, vmax = np.inf, -np.inf
            for f in fins:
                s = np.asarray(nib.load(f).dataobj)
                if background is not None:
                    s = s[s != background]
                if s.size:
                    vmin = min(vmin, float(np.nanmin(s)))
                    vmax = max(vmax, float(np.nanmax(s)))

        if not np.isfinite(vmin):  # degenerate
            vmin, vmax = 0.0, 1.0

    value_span = float(vmax - vmin)
    if not np.isfinite(value_span) or value_span < 0:
        value_span = 0.0

    edges = np.linspace(vmin, vmax, bins + 1, dtype=np.float32)
    centers = 0.5 * (edges[:-1] + edges[1:])

    out = np.zeros(shape, dtype=np.float32)

    for z0 in range(0, Z, cz):
        for y0 in range(0, Y, cy):
            for x0 in range(0, X, cx):
                z1, y1, x1 = min(z0 + cz, Z), min(y0 + cy, Y), min(x0 + cx, X)
                sz, sy, sx = z1 - z0, y1 - y0, x1 - x0
                H = np.zeros((sz, sy, sx, bins), dtype=np.uint32)

                for f in fins:
                    print("\tIncluding", f)
                    blk = np.asarray(nib.load(f).dataobj[z0:z1, y0:y1, x0:x1])
                    if background is not None:
                        blk = np.where(blk == background, np.nan, blk)
                    # Map values to bin indices
                    # scale to [0, bins-1]
                    # mask NaNs / outside
                    m = ~np.isnan(blk)
                    idx = np.zeros(blk.shape, dtype=np.int32)
                    if value_span > 0 and np.any(m):
                        idx[m] = np.floor(
                            ((blk[m] - vmin) / value_span) * (bins - 1)
                        ).astype(np.int32)
                    idx = np.clip(idx, 0, bins - 1, out=idx)
                    # Update histograms
                    # vectorized add: one bincount per voxel would be slow,
                    # so flatten spatial dims for one big add.at
                    flat_m = m.ravel()
                    flat_idx = idx.ravel()[flat_m]
                    # Build coordinates for add.at
                    # linear voxel index (0..sz*sy*sx-1)
                    lin = np.arange(sz * sy * sx, dtype=np.int32).repeat(1)[flat_m]
                    # Faster: use np.add.at on a 2D (voxels, bins) view
                    H2 = H.reshape(-1, bins)
                    np.add.at(H2, (lin, flat_idx), 1)

                # Turn histograms into the desired percentile
                Hc = np.cumsum(H, axis=-1)
                counts = Hc[..., -1]
                target = (p / 100.0) * counts
                # first bin where cumsum >= target
                # handle empty (all NaN / all background)
                empty = counts == 0
                idxp = np.argmax(Hc >= target[..., None], axis=-1)

                val = centers[idxp]
                val[empty] = np.nan if background is None else background

                out[z0:z1, y0:y1, x0:x1] = val

    if dtype == np.uint8:
        # If values are already in uint8-like range (common for cls volumes),
        # avoid renormalizing; this prevents collapse when vmin == vmax.
        if value_span > 0 and (vmax > 255 or vmin < 0 or np.nanmax(out) <= 1.0):
            out = np.rint(255 * (out - vmin) / value_span)
        else:
            out = np.rint(out)
        out = np.clip(out, 0, 255)

    out = out.astype(dtype)

    nib.Nifti1Image(out, affine, direction_order="lpi").to_filename(fout)


def create_acq_atlas(chunk_info, output_dir, atlas_fin, clobber: bool = False):
    """To save memory, for each volume :
        1. load and z-score it,
        2. add z-score image to sum volume.
    Then :
        3) calculate mean volume
        4) otsu threshold the mean volume to create atlas mask.
        5) save the atlas mask and mean volume
    """
    mask_fin = f"{output_dir}/atlas_mask.nii.gz"

    print("Creating atlas from interpolated volumes")
    print(f"Atlas filename: {atlas_fin}")
    print(f"Atlas mask filename: {mask_fin}")

    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(atlas_fin) or clobber:
        mean_vol = None

        n = len(chunk_info)

        for i, row in chunk_info.iterrows():
            interp_vol_fin = row["interp_cls_nat"]

            img = nib.load(interp_vol_fin)
            print(i / n, interp_vol_fin)

            vol = img.get_fdata()

            vol = (vol - np.mean(vol)) / np.std(vol)

            if mean_vol is None:
                mean_vol = vol
            else:
                mean_vol += vol

        mean_vol /= n

        # normalize data between -1 and 1
        # data = gaussian_filter(data, sigma=2)
        # data = 2 * (data - np.min(data)) / (np.max(data) - np.min(data)) - 1
        # data = exposure.equalize_adapthist(data, clip_limit=0.01, kernel_size=20)
        mean_vol[mean_vol < mean_vol.max() * 0.2] = 0

        mean_vol = (
            255 * (mean_vol - np.min(mean_vol)) / (np.max(mean_vol) - np.min(mean_vol))
        )

        nib.Nifti1Image(mean_vol, img.affine, direction_order="lpi").to_filename(
            atlas_fin
        )

    # Create mask
    create_mask(atlas_fin, mask_fin, clobber=clobber)

    return atlas_fin, mask_fin


def apply_final_transform_to_files(
    chunk_info: pd.DataFrame,
    ref_vol_fin: str,
    nl_3d_tfm_fn: str,
    interpolation: str = "Linear",
    clobber: bool = False,
):
    for _, row in chunk_info.iterrows():
        interp_nat_fin = row["interp_nat"]
        interp_stx_fin = row["interp_stx"]

        print("interp_stx_fin:", interp_stx_fin)

        if not os.path.exists(interp_stx_fin) or clobber:
            cmd = f"antsApplyTransforms -d 3 -n {interpolation} -i {interp_nat_fin} -o {interp_stx_fin} -r {ref_vol_fin} -t {nl_3d_tfm_fn} --float 1"

            print(cmd)

            run(cmd, shell=True)

            assert nib.load(interp_stx_fin).get_fdata().sum() > 0, "Error: Empty Output"

    return chunk_info


def create_final_transform(
    sub,
    hemisphere,
    chunk,
    chunk_info,
    in_ref_rsl_fin,
    output_dir,
    resolution,
    resolution_list_3d,
    interpolation: str = "Linear",
    clobber: bool = False,
):
    os.makedirs(output_dir, exist_ok=True)

    print("Create Acquisition Atlas")
    atlas_vol_fin, _ = create_acq_atlas(
        chunk_info, output_dir, output_dir + "/atlas.nii.gz", clobber=clobber
    )

    chunk_output_dir = f"{output_dir}/sub-{sub}/hemi-{hemisphere}/chunk-{chunk}/"

    # drop rows with Nan
    chunk_info = chunk_info.dropna()

    ref_rsl_2d_fin = utils.resample_struct_reference_volume(
        in_ref_rsl_fin, resolution, chunk_output_dir, clobber=clobber
    )

    nl_3d_tfm_fn = chunk_info["nl_3d_tfm_fn"].values[0]

    out_nl_3d_tfm_fn = nl_3d_tfm_fn
    print("atlas_vol_fn:", atlas_vol_fin)

    chunk_info = apply_final_transform_to_files(
        chunk_info,
        ref_rsl_2d_fin,
        out_nl_3d_tfm_fn,
        interpolation=interpolation,
        clobber=True,
    )

    return chunk_info


def prepare_chunk_info_for_stx(chunk_info, curr_chunk_info):
    """Prepare and modify curr_chunk_info for final alignment to stx space."""
    merged = pd.merge(
        chunk_info, curr_chunk_info, how="left", on=["sub", "hemisphere", "chunk"]
    ).dropna()

    if "acquisition_y" in merged.columns:
        merged["acquisition"] = merged["acquisition_y"]
    if "acquisition_y" in merged.columns:
        del merged["acquisition_y"]

    merged["interp_stx"] = merged["interp_nat"].apply(
        lambda x: x.replace("_iso", "_stx")
    )
    return merged


def prepare_chunk_info_for_stx(chunk_info, acq_interp_chunk_info):
    """Prepare and modify curr_chunk_info for final alignment to stx space.
    This is necessary because chunk_info adds the 'acquisition' column and is used
    in the final transformation of each reconstrucred acquisition volume into stx space
    """
    print(chunk_info.columns)

    merged = pd.DataFrame()
    for acq, temp_acq_chunk_info in acq_interp_chunk_info.groupby("acquisition"):
        for _, row in chunk_info.iterrows():
            row["acquisition"] = acq
            row["interp_nat"] = temp_acq_chunk_info["interp_nat"].values[0]
            row["interp_cls_nat"] = temp_acq_chunk_info["interp_cls_nat"].values[0]

            merged = pd.concat([merged, pd.DataFrame([row])], ignore_index=True)

    merged["interp_stx"] = merged["interp_nat"].apply(
        lambda x: x.replace("_iso", "_stx")
    )

    return merged


def apply_interpolated_volumes_to_stx(
    curr_chunk_info,
    curr_hemi_info,
    resolution,
    output_dir,
    interpolation,
    clobber,
    sub=None,
    hemisphere=None,
    nl_3d_tfm_list=None,
):
    ref_vol_fn = curr_hemi_info["struct_ref_vol"].values[0]

    ref_rsl_dir = os.path.join(output_dir, f"sub-{sub}", f"hemi-{hemisphere}")
    os.makedirs(ref_rsl_dir, exist_ok=True)

    ref_vol_rsl_fn = utils.resample_struct_reference_volume(
        ref_vol_fn, resolution, ref_rsl_dir, clobber=clobber
    )

    curr_chunk_info["ref_vol_rsl_list"] = ref_vol_rsl_fn

    if nl_3d_tfm_list is None and "nl_3d_tfm_list" in curr_chunk_info.columns:
        nl_3d_tfm_list = curr_chunk_info["nl_3d_tfm_list"].values[0]

    for _, row in curr_chunk_info.iterrows():
        interp_nat_fin = row["interp_nat"]
        interp_stx_fin = row["interp_stx"]

        logger.info(
            "Applying final transform to stx space for file: %s", interp_nat_fin
        )
        logger.info("Reference volume for stx space: %s", ref_vol_rsl_fn)
        logger.info("Nonlinear 3D transform file: %s", nl_3d_tfm_list)
        logger.info("Output file in stx space: %s", interp_stx_fin)
        logger.info("")

        if nl_3d_tfm_list in (None, "", [], ()):
            # Without a 3D chunk-to-reference transform (e.g., when stage 3 is
            # skipped), write a reference-grid volume directly so downstream
            # consumers can always use interp_stx consistently.
            if not os.path.exists(interp_stx_fin) or clobber:
                interp_nat_img = nib.load(interp_nat_fin)
                ref_img = nib.load(ref_vol_rsl_fn)

                interp_data = interp_nat_img.get_fdata().astype(np.float32)
                interp_on_ref_grid = resize(
                    interp_data,
                    ref_img.shape,
                    order=1,
                    preserve_range=True,
                    anti_aliasing=False,
                ).astype(np.float32)

                nib.Nifti1Image(
                    interp_on_ref_grid, ref_img.affine, direction_order="lpi"
                ).to_filename(interp_stx_fin)
        else:
            utils.simple_ants_apply_tfm(
                interp_nat_fin,
                ref_vol_rsl_fn,
                nl_3d_tfm_list,
                interp_stx_fin,
                n=interpolation,
                clobber=clobber,
            )

        assert os.path.exists(
            interp_stx_fin
        ), f"Error: Output file not found: {interp_stx_fin}"

    return curr_chunk_info


def volumetric_pipeline(
    sect_info: pd.DataFrame,
    chunk_info: pd.DataFrame,
    hemi_info: pd.DataFrame,
    resolution: float,
    resolution_list: list,
    output_dir: str,
    final_resolution: float = None,
    interpolation: str = "Linear",
    refine_2d_alignment_flag: bool = False,
    target_section: str = None,
    target_section_acq: str = None,
    target_section_cls: str = None,
    num_cores: int = -1,
    use_final_transform: bool = True,
    clobber: bool = False,
):
    chunk_info_list = []

    for (sub, hemisphere), sect_info_sub_hemi in sect_info.groupby(
        ["sub", "hemisphere"]
    ):
        idx = (chunk_info["sub"] == sub) & (chunk_info["hemisphere"] == hemisphere)

        if "resolution" in chunk_info.columns:
            idx = idx & (chunk_info["resolution"] == resolution)

        curr_chunk_info = chunk_info.loc[idx]

        assert (
            len(curr_chunk_info) > 0
        ), f"Error: no chunk info found, sub: {sub}, hemisphere: {hemisphere}, resolution: {resolution}, \n{chunk_info}"

        curr_hemi_info = hemi_info.loc[
            (hemi_info["sub"] == sub) & (hemi_info["hemisphere"] == hemisphere)
        ]
        assert len(curr_hemi_info) > 0, "Error: no hemisphere info found"

        curr_target_section_acq, curr_target_section_cls = resolve_target_sections(
            sect_info_sub_hemi,
            target_section=target_section,
            target_section_acq=target_section_acq,
            target_section_cls=target_section_cls,
        )

        print(
            "Target sections for interpolation:",
            curr_target_section_acq,
            curr_target_section_cls,
        )
        
        # Volumetric interpolation
        print("Volumetric Interpolation for sub:", sub, "hemi:", hemisphere)
        acq_interp_chunk_info = volumetric_interpolation_over_dataframe(
            sect_info_sub_hemi,
            curr_chunk_info,
            output_dir,
            resolution,
            resolution_list,
            final_resolution=final_resolution,
            interpolation=interpolation,
            target_section=target_section,
            target_section_acq=curr_target_section_acq,
            target_section_cls=curr_target_section_cls,
            num_cores=num_cores,
            clobber=clobber,
            refine_2d_alignment_flag=refine_2d_alignment_flag,
        )

        curr_chunk_info = prepare_chunk_info_for_stx(chunk_info, acq_interp_chunk_info)

        chunk_info_list.append(curr_chunk_info)

        curr_chunk_info = apply_interpolated_volumes_to_stx(
            curr_chunk_info,
            curr_hemi_info,
            resolution,
            output_dir,
            interpolation,
            clobber,
            sub=sub,
            hemisphere=hemisphere,
            nl_3d_tfm_list=curr_chunk_info["nl_3d_tfm_list"].values[0]
            if use_final_transform and "nl_3d_tfm_list" in curr_chunk_info.columns
            else None,
        )

    chunk_info_out = pd.concat(chunk_info_list, ignore_index=True)

    output_csv = f"{output_dir}/chunk_info_thickened_stx.csv"

    chunk_info_out.to_csv(output_csv, index=False)

    return chunk_info_out
