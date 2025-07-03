"""Not yet impemented. Temporarily commented out."""

import os
from subprocess import run

import numpy as np
import pandas as pd

import brainbuilder.utils.ants_nibabel as nib
from brainbuilder.interp.acqvolume import create_thickened_volumes
from brainbuilder.utils import utils
from brainbuilder.utils.nl_deformation_flow import nlflow_isometric
from brainbuilder.volalign import verify_chunk_limits


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


def _interpolate_missing_sections(
    vol: np.array, order: int = 0, axis: int = 1, p: int = 2
) -> np.array:
    """Vol has a discrete subset of sections with labeled regions inside of them. This function
    will use nearest neighbour interpolation to fill in the missing sections in between.
    """
    assert vol.sum() > 0, "Error: Empty Input"

    vol_interp = np.zeros_like(vol)

    # Find indices of non-empty sections
    non_empty_indices = np.array(
        [
            i
            for i in range(vol.shape[axis])
            if np.max(vol[:, i, :]) != np.min(vol[:, i, :])
        ],
        dtype=int,
    )

    if len(non_empty_indices) == 0:
        print("No non-empty sections found")
        return vol_interp  # Return empty volume if no non-empty sections are found

    print(non_empty_indices)

    for i, idx0 in enumerate(range(non_empty_indices, max(non_empty_indices))):
        if i not in non_empty_indices:
            # Calculate distances to non-empty sections
            distances = np.abs(non_empty_indices - i)

            min_dist_i = np.argsort(distances)[0 : (order + 1)]

            min_dist = np.sort(distances)[0 : (order + 1)]

            # Find the nearest non-empty section
            nearest_i = non_empty_indices[min_dist_i]
            # print('i=',i,'-->', min_dist, min_dist_i)

            section = idw(vol, nearest_i, min_dist, p=p)
            # print(np.mean(section), np.mean(vol[:,nearest_i[0],:]), np.mean(vol[:,nearest_i[1],:]))

            vol_interp[:, i, :] = section
        else:
            vol_interp[:, i, :] = vol[:, i, :]
    # assert np.sum(vol_interp) > 0, 'Error: Empty Output'

    return vol_interp


def volumetric_interpolation(
    sect_info: pd.DataFrame,
    chunk_info: pd.DataFrame,
    struct_ref_vol: str,
    output_dir: str,
    resolution: float,
    resolution_list: list,
    clobber: bool = False,
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

    for (sub, hemisphere, chunk, acq), curr_sect_info in sect_info.groupby(
        ["sub", "hemisphere", "chunk", "acquisition"]
    ):
        curr_output_dir = f"{output_dir}/sub-{sub}/hemi-{hemisphere}/acq-{acq}/"

        os.makedirs(curr_output_dir, exist_ok=True)

        curr_chunk_info = chunk_info[(chunk_info["chunk"] == chunk)]

        print("thickened")
        chunk_info_thickened_csv = create_thickened_volumes(
            curr_output_dir,
            curr_chunk_info,
            curr_sect_info,
            resolution,
            struct_ref_vol,
            clobber=clobber,
            width=1,
        )

        chunk_info_cls_thickened_csv = create_thickened_volumes(
            curr_output_dir + "/cls/",
            curr_chunk_info,
            curr_sect_info,
            resolution,
            struct_ref_vol,
            tissue_type="cls",
            clobber=clobber,
            width=1,
        )

        chunk_info_thickened = pd.read_csv(chunk_info_thickened_csv)
        chunk_info_cls_thickened = pd.read_csv(chunk_info_cls_thickened_csv)

        print(f"Interpolation Chunk: {chunk}, Acquisition: {acq}")
        cls_fin = chunk_info_cls_thickened["thickened"].values[0]
        acq_fin = chunk_info_thickened["thickened"].values[0]

        interp_acq_iso_fin, nlflow_tfm_dict = nlflow_isometric(
            acq_fin, curr_output_dir, resolution, resolution_list, clobber=clobber
        )

        interp_cls_iso_fin, _ = nlflow_isometric(
            cls_fin,
            curr_output_dir + "/cls/",
            resolution,
            resolution_list,
            tfm_dict=nlflow_tfm_dict,
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

        # ref_rsl_2d_fin = utils.resample_struct_reference_volume(
        #    struct_ref_vol, resolution, output_dir, clobber=clobber
        # )

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
    chunk_info, ref_vol_fin, nl_3d_tfm_fn, clobber: bool = False
):
    chunk_info["interp_stx"] = chunk_info["interp_nat"].replace(
        "interp-vol_iso", "interp-vol_stx"
    )
    print(chunk_info.head())
    print(chunk_info["interp_nat"])
    print(chunk_info["acquisition"])

    for _, row in chunk_info.iterrows():
        interp_nat_fin = row["interp_nat"]
        interp_stx_fin = interp_nat_fin.replace("interp-vol_iso", "interp-vol_stx")

        print("interp_stx_fin:", interp_stx_fin)

        if not os.path.exists(interp_stx_fin) or clobber:
            cmd = f"antsApplyTransforms -d 3 -i {interp_nat_fin} -o {interp_stx_fin} -r {ref_vol_fin} -t {nl_3d_tfm_fn} --float 1"

            print(cmd)

            run(
                cmd,
                shell=True,
            )

            assert nib.load(interp_stx_fin).get_fdata().sum() > 0, "Error: Empty Output"


def create_final_transform(
    sub,
    hemisphere,
    chunk,
    chunk_info,
    in_ref_rsl_fin,
    output_dir,
    resolution,
    resolution_list_3d,
    clobber: bool = False,
):
    os.makedirs(output_dir, exist_ok=True)

    resolution_3d = min(resolution_list_3d)

    ref_rsl_3d_fin = utils.resample_struct_reference_volume(
        in_ref_rsl_fin, resolution_3d, output_dir, clobber=clobber
    )

    world_chunk_limits, vox_chunk_limits = verify_chunk_limits(
        ref_rsl_3d_fin, chunk_info
    )

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

    out_nl_3d_tfm_fn = (
        f"{mask_out_dir}/sub-{sub}_hemi-{hemisphere}_chunk-{chunk}_SyN_CC_Composite.h5"
    )

    nl_3d_tfm_inv_fn = f"{mask_out_dir}/sub-{sub}_hemi-{hemisphere}_chunk-{chunk}_SyN_CC_InverseComposite.h5"
    rec_3d_rsl_fn = (
        f"{mask_out_dir}/sub-{sub}_hemi-{hemisphere}_chunk-{chunk}_rec_3d_rsl.nii.gz"
    )
    ref_3d_rsl_fn = (
        f"{mask_out_dir}/sub-{sub}_hemi-{hemisphere}_chunk-{chunk}_ref_3d_rsl.nii.gz"
    )

    # align_3d(
    #    sub,
    #    hemisphere,
    #    chunk,
    #    atlas_vol_fin,  # moving
    #    ref_rsl_3d_fin,  # fixed
    #    mask_out_dir,
    #    out_nl_3d_tfm_fn,
    #    nl_3d_tfm_inv_fn,
    #    rec_3d_rsl_fn,
    #    ref_3d_rsl_fn,
    #    resolution_3d,
    #    resolution_list_3d,
    #    world_chunk_limits,
    #    vox_chunk_limits,
    #    init_tfm = nl_3d_tfm_fn,
    #    linear_steps = [],
    #    use_3d_syn_cc = True,
    #    use_pad_volume = True,
    #    clobber = clobber,
    # )
    out_nl_3d_tfm_fn = nl_3d_tfm_fn
    print("atlas_vol_fn:", atlas_vol_fin)

    # apply_final_transform_to_files(
    #    chunk_info, ref_rsl_2d_fin, out_nl_3d_tfm_fn, clobber=False
    # )


def volumetric_pipeline(
    sect_info,
    chunk_info,
    hemi_info,
    resolution,
    resolution_3d,
    resolution_list,
    output_dir,
    clobber: bool = False,
):
    for (sub, hemisphere, chunk), curr_sect_info in sect_info.groupby(
        ["sub", "hemisphere", "chunk"]
    ):
        idx = (
            (chunk_info["sub"] == sub)
            & (chunk_info["hemisphere"] == hemisphere)
            & (chunk_info["resolution"] == resolution)
            & (chunk_info["chunk"] == chunk)
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

        # Volumetric interpolation
        curr_chunk_info = volumetric_interpolation(
            curr_sect_info,
            curr_chunk_info,
            ref_vol_fn,
            output_dir,
            resolution,
            resolution_list,
            clobber=clobber,
        )

        curr_chunk_info = pd.merge(
            chunk_info, curr_chunk_info, how="left", on=["sub", "hemisphere", "chunk"]
        ).dropna()

        resolution_list_3d = [r for r in resolution_list if r >= resolution_3d]
        print("resolution_list_3d:", resolution_list_3d)

        curr_output_dir = f"{output_dir}/sub-{sub}/hemi-{hemisphere}/atlas/"

        print("Create final transform")
        output_csv = f"{output_dir}/chunk_info_thickened_stx.csv"

        create_final_transform(
            sub,
            hemisphere,
            chunk,
            curr_chunk_info,
            ref_vol_fn,
            curr_output_dir,
            resolution,
            resolution_list_3d,
            clobber=clobber,
        )

    return curr_chunk_info
