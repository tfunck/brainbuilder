"""Not yet impemented. Temporarily commented out."""

import os
from subprocess import run

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import brainbuilder.utils.ants_nibabel as nib
from brainbuilder.align.align_2d import apply_transforms_parallel
from brainbuilder.interp.acqvolume import create_thickened_volumes
from brainbuilder.utils import utils
from brainbuilder.utils.nl_deformation_flow import nlflow_isometric


def idw(vol, nearest_i, min_dist, p=2):
    # Inverse Distance Weighted interpolation (IDW) with Shepard's method
    # min_dist += 1
    weights = min_dist / np.sum(min_dist)

    print(min_dist, weights)
    interp = np.sum(
        [w * vol[:, nearest_i[j], :] for j, w in enumerate(weights)], axis=0
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

    curr_sect_info["2d_align"] = curr_sect_info["2d_align"].apply(
        lambda x: f"{final_tfm_dir}/{os.path.basename(x)}"
    )
    curr_sect_info["2d_align_cls"] = curr_sect_info["2d_align_cls"].apply(
        lambda x: f"{final_tfm_dir}/{os.path.basename(x)}"
    )

    Parallel(n_jobs=num_cores, backend="multiprocessing")(
        delayed(apply_transforms_parallel)(
            final_tfm_dir,
            final_resolution,
            row,
            interpolation=interpolation,
            file_str="raw",
        )
        for _, row in curr_sect_info.iterrows()
    )

    Parallel(n_jobs=num_cores, backend="multiprocessing")(
        delayed(apply_transforms_parallel)(
            final_tfm_dir, final_resolution, row, tissue_str="_cls", file_str="seg"
        )
        for _, row in curr_sect_info.iterrows()
    )


def volumetric_interpolation(
    curr_sect_info: pd.DataFrame,
    curr_chunk_info: pd.DataFrame,
    output_dir: str,
    resolution: float,
    resolution_list: list,
    interpolation: str = "Linear",
    tissue_type: str = "acq",
    target_section: str = "2d_align",
    nlflow_tfm_dict: dict = None,
    num_cores: int = -1,
    clobber: bool = False,
) -> pd.DataFrame:
    os.makedirs(output_dir, exist_ok=True)

    chunk_info_thickened_csv = create_thickened_volumes(
        output_dir + "/" + tissue_type,
        curr_chunk_info,
        curr_sect_info,
        resolution,
        tissue_type=tissue_type,
        target_section=target_section,
        clobber=clobber,
        width=1,
    )

    chunk_info_thickened = pd.read_csv(chunk_info_thickened_csv)

    acq_fin = chunk_info_thickened["thickened"].values[0]

    interp_iso_fin, nlflow_tfm_dict = nlflow_isometric(
        acq_fin,
        output_dir,
        resolution,
        resolution_list,
        interpolation=interpolation,
        tfm_dict=nlflow_tfm_dict,
        num_jobs=num_cores,
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
    interpolation: str = "Linear",
    tissue_type: str = "cls",
    target_section: str = "2d_align",
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

    resolution_list_orig = np.array(resolution_list).copy()
    resolution_orig = resolution

    if final_resolution is not None and isinstance(final_resolution, float):
        resolution_list += [final_resolution]
        resolution = final_resolution

    for (sub, hemisphere, chunk, acq), curr_sect_info in sect_info.groupby(
        ["sub", "hemisphere", "chunk", "acquisition"]
    ):
        curr_output_dir = (
            f"{output_dir}/sub-{sub}/hemi-{hemisphere}/chunk-{chunk}/acq-{acq}/"
        )

        if final_resolution is not None and isinstance(final_resolution, float):
            final_tfm_dir = curr_output_dir + "/final_tfm_2d"
            apply_final_2d_transforms(
                curr_sect_info,
                final_tfm_dir,
                final_resolution,
                interpolation,
                num_cores,
            )

        curr_chunk_info = chunk_info[(chunk_info["chunk"] == chunk)]

        interp_acq_iso_fin, nlflow_tfm_dict = volumetric_interpolation(
            curr_sect_info,
            curr_chunk_info,
            curr_output_dir,
            resolution,
            resolution_list,
            interpolation=interpolation,
            num_cores=num_cores,
            clobber=clobber,
        )

        interp_cls_iso_fin, _ = volumetric_interpolation(
            curr_sect_info,
            curr_chunk_info,
            curr_output_dir,
            resolution,
            resolution_list,
            interpolation=interpolation,
            tissue_type=tissue_type,
            target_section=target_section,
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
            print("\n\n\n\hello file:", interp_vol_fin)

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

    # drop rows with Nan
    chunk_info = chunk_info.dropna()

    mask_out_dir = f"{output_dir}/mask/"
    atlas_out_dir = f"{output_dir}/atlas/"

    os.makedirs(mask_out_dir, exist_ok=True)
    os.makedirs(atlas_out_dir, exist_ok=True)

    ref_rsl_2d_fin = utils.resample_struct_reference_volume(
        in_ref_rsl_fin, resolution, output_dir, clobber=clobber
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


def volumetric_pipeline(
    sect_info: pd.DataFrame,
    chunk_info: pd.DataFrame,
    hemi_info: pd.DataFrame,
    resolution: float,
    resolution_list: list,
    output_dir: str,
    final_resolution: float = None,
    interpolation: str = "Linear",
    num_cores: int = -1,
    clobber: bool = False,
):
    chunk_info_list = []

    for (sub, hemisphere), sect_info_sub_hemi in sect_info.groupby(
        ["sub", "hemisphere"]
    ):
        idx = (
            (chunk_info["sub"] == sub)
            & (chunk_info["hemisphere"] == hemisphere)
            & (chunk_info["resolution"] == resolution)
        )

        curr_chunk_info = chunk_info.loc[idx]

        assert (
            len(curr_chunk_info) > 0
        ), f"Error: no chunk info found, sub: {sub}, hemisphere: {hemisphere}, resolution: {resolution}, \n{chunk_info}"

        curr_hemi_info = hemi_info.loc[
            (hemi_info["sub"] == sub) & (hemi_info["hemisphere"] == hemisphere)
        ]
        assert len(curr_hemi_info) > 0, "Error: no hemisphere info found"

        ref_vol_fn = curr_hemi_info["struct_ref_vol"].values[0]

        ref_vol_rsl_fn = utils.resample_struct_reference_volume(
            ref_vol_fn, resolution, output_dir, clobber=clobber
        )

        # Volumetric interpolation
        print("Volumetric Interpolation for sub:", sub, "hemi:", hemisphere)
        curr_chunk_info = volumetric_interpolation(
            sect_info_sub_hemi,
            curr_chunk_info,
            output_dir,
            resolution,
            resolution_list,
            final_resolution=final_resolution,
            interpolation=interpolation,
            num_cores=num_cores,
            clobber=clobber,
        )

        curr_chunk_info = pd.merge(
            chunk_info, curr_chunk_info, how="left", on=["sub", "hemisphere", "chunk"]
        ).dropna()

        if "acquisition_y" in curr_chunk_info.columns:
            curr_chunk_info["acquisition"] = curr_chunk_info["acquisition_y"]
            del curr_chunk_info["acquisition_y"]

        curr_chunk_info["interp_stx"] = curr_chunk_info["interp_nat"].apply(
            lambda x: x.replace("_iso", "_stx")
        )

        curr_chunk_info["ref_vol_rsl_fn"] = ref_vol_rsl_fn

        for _, row in curr_chunk_info.iterrows():
            interp_nat_fin = row["interp_nat"]
            interp_stx_fin = row["interp_stx"]

            nl_3d_tfm_fn = (
                curr_chunk_info["nl_3d_tfm_fn"]
                .loc[curr_chunk_info["chunk"] == row["chunk"]]
                .values[0]
            )

            if not os.path.exists(interp_stx_fin) or clobber:
                cmd = f"antsApplyTransforms -d 3 -n {interpolation} -i {interp_nat_fin} -o {interp_stx_fin} -r {ref_vol_rsl_fn} -t {nl_3d_tfm_fn} --float 1"

                print(cmd)

                run(cmd, shell=True)

                assert (
                    nib.load(interp_stx_fin).get_fdata().sum() > 0
                ), "Error: Empty Output"

        chunk_info_list.append(curr_chunk_info)

    chunk_info_out = pd.concat(chunk_info_list, ignore_index=True)

    output_csv = f"{output_dir}/chunk_info_thickened_stx.csv"

    chunk_info_out.to_csv(output_csv, index=False)

    return chunk_info_out
