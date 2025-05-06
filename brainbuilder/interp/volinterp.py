"""Not yet impemented. Temporarily commented out."""

import os
from subprocess import run

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from skimage import exposure
from skimage.transform import resize

import brainbuilder.utils.ants_nibabel as nib
from brainbuilder.interp.acqvolume import create_thickened_volumes
from brainbuilder.utils.nl_deformation_flow import nl_deformation_flow_3d
from brainbuilder.volalign import align_3d, verify_chunk_limits


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


def nl_deformation_flow_nii(acq_fin, output_dir, interp_acq_fin, clobber=False):
    if not os.path.exists(interp_acq_fin) or clobber:
        """
        Process the acq volume by applying non-linear deformation flow.

        Parameters
        ----------
        acq_fin : str
            Path to the acq volume file.
        output_dir : str
            Directory to save the output files.
        clobber : bool, optional
            If True, overwrite existing files. The default is False.

        Returns
        -------
        str
            Path to the processed acq volume file.
        """
        acq_img = nib.load(acq_fin)
        acq_vol = acq_img.get_fdata()
        acq_vol[acq_vol < 0] = 0

        origin = list(acq_img.affine[[0, 2], 3])
        spacing = list(acq_img.affine[[0, 2], [0, 2]])

        interp_acq_vol = nl_deformation_flow_3d(
            acq_vol,
            output_dir + "/nl_flow/",
            origin=origin,
            spacing=spacing,
            clobber=clobber,
        )

        interp_acq_img = nib.Nifti1Image(
            interp_acq_vol, acq_img.affine, direction_order="lpi"
        )
        interp_acq_img.to_filename(interp_acq_fin)


def resample_interp_vol_to_resolution(
    interp_acq_orig_fin,
    interp_acq_iso_fin: str,
    resolution: float,
    clobber: bool = False,
) -> np.array:
    """Resample the interpolated volume to the specified resolution.
    Parameters
    ----------
    interp_acq_vol : np.array
        Interpolated volume.
    acq_img : nib.Nifti1Image

    """
    if not os.path.exists(interp_acq_iso_fin) or clobber:
        interp_acq_img = nib.load(interp_acq_orig_fin)  # type:ignore[assignment]
        interp_acq_vol = interp_acq_img.get_fdata()

        slice_thickness = interp_acq_img.affine[1, 1]

        y_new = int(np.round(interp_acq_vol.shape[1] / (resolution / slice_thickness)))

        interp_acq_vol = resize(
            interp_acq_vol,
            (interp_acq_vol.shape[0], y_new, interp_acq_vol.shape[2]),
            order=1,
        )

        aff_iso = interp_acq_img.affine

        aff_iso[1, 1] = resolution

        nib.Nifti1Image(interp_acq_vol, aff_iso, direction_order="lpi").to_filename(
            interp_acq_iso_fin
        )


def volumetric_interpolation(
    sect_info: pd.DataFrame,
    chunk_info: pd.DataFrame,
    struct_ref_vol: str,
    output_dir: str,
    resolution: float,
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

        chunk_info_thickened = pd.read_csv(chunk_info_thickened_csv)

        print(f"Interpolation Chunk: {chunk}, Acquisition: {acq}")
        acq_fin = chunk_info_thickened["thickened"].values[0]

        interp_acq_iso_fin = acq_fin.replace("thickened", "interp-vol_iso")
        interp_acq_fin = acq_fin.replace("thickened", "interp-vol_orig")

        print("Resampling", interp_acq_fin)
        nl_deformation_flow_nii(
            acq_fin, curr_output_dir, interp_acq_fin, clobber=clobber
        )

        print("Resampling", interp_acq_iso_fin)
        resample_interp_vol_to_resolution(
            interp_acq_fin, interp_acq_iso_fin, resolution, clobber=clobber
        )

        row = pd.DataFrame(
            {
                "sub": [sub],
                "hemisphere": [hemisphere],
                "chunk": [chunk],
                "acquisition": [acq],
                "interp_nat": [interp_acq_iso_fin],
            }
        )

        chunk_info_out = pd.concat([chunk_info_out, row], ignore_index=True)

    return chunk_info_out


def create_mask(fn, out_fn, clobber: bool = False):
    if not os.path.exists(out_fn) or clobber:
        img = nib.load(fn)
        data = img.get_fdata()

        t = data.max() * 0.02

        mask = data > t

        mask = mask.astype(np.uint8)

        nib.Nifti1Image(mask, img.affine, direction_order="lpi").to_filename(out_fn)


def create_acq_atlas(chunk_info, output_dir, clobber: bool = False):
    """To save memory, for each volume :
        1. load and z-score it,
        2. add z-score image to sum volume.
    Then :
        3) calculate mean volume
        4) otsu threshold the mean volume to create atlas mask.
        5) save the atlas mask and mean volume
    """
    atlas_fin = f"{output_dir}/atlas.nii.gz"

    mask_fin = f"{output_dir}/atlas_mask.nii.gz"

    print("Creating atlas from interpolated volumes")
    print(f"Atlas filename: {atlas_fin}")
    print(f"Atlas mask filename: {mask_fin}")

    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(atlas_fin) or clobber:
        mean_vol = None

        n = len(chunk_info)

        for i, row in chunk_info.iterrows():
            interp_vol_fin = row["interp_nat"]

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
        data = gaussian_filter(data, sigma=2)
        data = 2 * (data - np.min(data)) / (np.max(data) - np.min(data)) - 1
        data = exposure.equalize_adapthist(data, clip_limit=0.01, kernel_size=20)

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

    for i, row in chunk_info.iterrows():
        interp_nat_fin = row["interp_nat"]
        interp_stx_fin = interp_nat_fin.replace("interp-vol_iso", "interp-vol_stx")

        if not os.path.exists(interp_stx_fin) or clobber:
            run(
                f"antsApplyTransforms -d 3 -i {interp_nat_fin} -o {interp_stx_fin} -r {ref_vol_fin} -t {nl_3d_tfm_fn} --float 1",
                shell=True,
            )


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
    # DEBUG
    # ref_fin = "/data/receptor/macaque/templates/MEBRAINS_T1_masked_left.nii.gz"
    # ref_rsl_fin=output_dir+'/'+os.path.basename(in_ref_rsl_fin).replace('.nii.gz', f'_{resolution}mm.nii.gz')
    # import nibabel
    # from nibabel.processing import resample_from_to
    # resample_from_to(nibabel.load(ref_fin), nibabel.load(in_ref_rsl_fin)).to_filename(ref_rsl_fin)
    ref_rsl_fin = in_ref_rsl_fin

    world_chunk_limits, vox_chunk_limits = verify_chunk_limits(ref_rsl_fin, chunk_info)

    print("Create Acquisition Atlas")
    atlas_vol_fin, mask_vol_fin = create_acq_atlas(
        chunk_info, output_dir, clobber=clobber
    )

    resolution_3d = min(resolution_list_3d)

    mask_out_dir = f"{output_dir}/mask/"
    atlas_out_dir = f"{output_dir}/atlas/"

    os.makedirs(mask_out_dir, exist_ok=True)
    os.makedirs(atlas_out_dir, exist_ok=True)

    nl_3d_tfm_fn = (
        f"{mask_out_dir}/sub-{sub}_hemi-{hemisphere}_chunk-{chunk}_SyN_CC_Composite.h5"
    )
    nl_3d_tfm_inv_fn = f"{mask_out_dir}/sub-{sub}_hemi-{hemisphere}_chunk-{chunk}_SyN_CC_InverseComposite.h5"
    rec_3d_rsl_fn = (
        f"{mask_out_dir}/sub-{sub}_hemi-{hemisphere}_chunk-{chunk}_rec_3d_rsl.nii.gz"
    )
    ref_3d_rsl_fn = (
        f"{mask_out_dir}/sub-{sub}_hemi-{hemisphere}_chunk-{chunk}_ref_3d_rsl.nii.gz"
    )

    align_3d(
        sub,
        hemisphere,
        chunk,
        mask_vol_fin,
        ref_rsl_fin,
        mask_out_dir,
        nl_3d_tfm_fn,
        nl_3d_tfm_inv_fn,
        rec_3d_rsl_fn,
        ref_3d_rsl_fn,
        resolution_3d,
        resolution_list_3d,
        world_chunk_limits,
        vox_chunk_limits,
        use_3d_syn_cc=True,
        use_pad_volume=False,
        clobber=clobber,
    )

    atlas_tfm_fn = (
        f"{atlas_out_dir}/sub-{sub}_hemi-{hemisphere}_chunk-{chunk}_SyN_CC_Composite.h5"
    )
    atlas_tfm_inv_fn = f"{atlas_out_dir}/sub-{sub}_hemi-{hemisphere}_chunk-{chunk}_SyN_CC_InverseComposite.h5"
    rec_3d_rsl_fn = (
        f"{atlas_out_dir}/sub-{sub}_hemi-{hemisphere}_chunk-{chunk}_rec_3d_rsl.nii.gz"
    )
    ref_3d_rsl_fn = (
        f"{atlas_out_dir}/sub-{sub}_hemi-{hemisphere}_chunk-{chunk}_ref_3d_rsl.nii.gz"
    )

    cmd = "antsRegistration --verbose 0 --dimensionality 3 --float 0 --collapse-output-transforms 1"
    cmd += f" --output [ {atlas_tfm_fn}, {rec_3d_rsl_fn}, {ref_3d_rsl_fn} ] --interpolation Linear --use-histogram-matching 0 --winsorize-image-intensities [ 0.005,0.995 ]"
    cmd += (
        f" --transform SyN[ 0.1,3,0 ] --metric CC[ {atlas_vol_fin},{ref_rsl_fin},1,4 ]"
    )
    cmd += " --convergence [ 100,1e-6,10 ] --shrink-factors 1 --smoothing-sigmas 0.1vox"

    apply_final_transform_to_files(
        chunk_info, rec_3d_rsl_fn, nl_3d_tfm_fn, clobber=True
    )


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
        print(chunk_info.columns)

        # Volumetric interpolation
        curr_chunk_info = volumetric_interpolation(
            curr_sect_info,
            curr_chunk_info,
            ref_vol_fn,
            output_dir,
            resolution,
            clobber=clobber,
        )

        curr_chunk_info = pd.merge(
            chunk_info, curr_chunk_info, how="left", on=["sub", "hemisphere", "chunk"]
        )

        resolution_3d = 0.4
        resolution_list_3d = [r for r in resolution_list if r >= resolution_3d]
        print("resolution_list_3d:", resolution_list_3d)

        curr_output_dir = f"{output_dir}/sub-{sub}/hemi-{hemisphere}/atlas/"

        print("Create final transform")
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
