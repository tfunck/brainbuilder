"""Main function for performing interpolation between acquired 2D sections."""
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from brainbuilder.interp.batch_correction import apply_batch_correction
from brainbuilder.interp.prepare_surfaces import prepare_surfaces
from brainbuilder.interp.surfinterp import (
    create_final_reconstructed_volume,
    generate_surface_profiles,
    write_raw_profiles_to_volume,
)
from brainbuilder.utils import utils
from brainbuilder.utils.mesh_utils import smooth_surface_profiles


def plot_paired_values(
    paired_values: np.array, mid_values: np.array, png_fn: str
) -> None:
    """Plot the paired values for the batch correction step.

    :param paired_values: dataframe containing the paired values
    :param mid_values: array containing the values at the mid depth
    :param png_fn: filename to save the plot
    :return: None
    """
    plt.cla()
    plt.clf()
    plt.figure(figsize=(10, 10))
    for i, row in paired_values.iterrows():
        curr_idx = row["curr_idx"].astype(int)
        next_idx = row["next_idx"].astype(int)
        curr_values = mid_values[curr_idx]
        next_values = mid_values[next_idx]

        x = [row["curr_label"], row["next_label"]]
        y = [curr_values, next_values]
        plt.scatter(x, y, c="r")
        plt.plot(x, y, c="b", alpha=0.3)
    print(f"Saving {png_fn}")
    plt.savefig(png_fn)


def resample_struct_reference_volume(
    orig_struct_vol_fn: str, resolution: float, output_dir: str, clobber: bool = False
) -> str:
    """Resample the structural reference volume to the desired resolution.

    :param orig_struct_vol_fn: path to the original structural reference volume
    :param hemisphere: hemisphere
    :param resolution: resolution of the volume
    :param output_dir: path to output directory
    :param clobber: boolean indicating whether to overwrite existing files
    :return: path to the resampled structural reference volume
    """
    base_name = re.sub(
        ".nii", f"_{resolution}mm.nii", os.path.basename(orig_struct_vol_fn)
    )
    struct_vol_rsl_fn = f"{output_dir}/{base_name}"

    if not os.path.exists(struct_vol_rsl_fn) or clobber:
        utils.resample_to_resolution(
            orig_struct_vol_fn, [resolution] * 3, struct_vol_rsl_fn
        )

    return struct_vol_rsl_fn


def volumes_to_surface_profiles(
    chunk_info: pd.DataFrame,
    sect_info: pd.DataFrame,
    resolution: float,
    output_dir: str,
    surf_dir: str,
    ref_vol_fn: str,
    gm_surf_fn: str,
    wm_surf_fn: str,
    depth_list: np.ndarray = None,
    interp_order: int = 1,
    gaussian_sd: float = 0,
    clobber: bool = False,
) -> tuple:
    """Project volumes to surfaces and generate surface profiles.

    :param chunk_info: dataframe containing chunk information
    :param sect_info: dataframe containing section information
    :param resolution: resolution of the volume
    :param output_dir: path to output directory
    :param surf_dir: path to surfaces directory
    :param ref_vol_fn: path to the structural reference volume
    :param gm_surf_fn: path to the gray matter surface
    :param wm_surf_fn: path to the white matter surface
    :param depth_list: list of depths
    :param interp_order: order of the interpolation
    :param gaussian_sd: standard deviation of the gaussian filter
    :param clobber: boolean indicating whether to overwrite existing files
    :return: sect_info, profiles_fn
    """
    surf_depth_mni_dict, surf_depth_chunk_dict = prepare_surfaces(
        chunk_info,
        ref_vol_fn,
        gm_surf_fn,
        wm_surf_fn,
        depth_list,
        surf_dir,
        resolution,
        clobber=clobber,
    )

    struct_vol_rsl_fn = resample_struct_reference_volume(
        ref_vol_fn, resolution, output_dir, clobber=clobber
    )

    #print(sect_info['nl_2d_rsl'].values); exit(0)
    (
        profiles_fn,
        surf_raw_values_dict,
        chunk_info_thickened_csv,
    ) = generate_surface_profiles(
        chunk_info,
        sect_info,
        surf_depth_chunk_dict,
        surf_depth_mni_dict,
        resolution,
        depth_list,
        struct_vol_rsl_fn,
        output_dir,
        interp_order=interp_order,
        gaussian_sd=gaussian_sd,
        clobber=clobber,
    )

    return (
        surf_depth_mni_dict,
        surf_depth_chunk_dict,
        surf_raw_values_dict,
        profiles_fn,
        chunk_info_thickened_csv,
        struct_vol_rsl_fn,
    )


def get_profiles_with_batch_correction(
    chunk_info: pd.DataFrame,
    sect_info: pd.DataFrame,
    chunk_info_thickened_csv: str,
    sub: str,
    hemisphere: str,
    acquisition: str,
    resolution: float,
    surf_depth_chunk_dict: dict,
    surf_depth_mni_dict: dict,
    struct_vol_rsl_fn: str,
    output_dir: str,
    surf_dir: str,
    batch_correction_dir: str,
    ref_vol_fn: str,
    gm_surf_fn: str,
    wm_surf_fn: str,
    depth_list: np.ndarray,
    batch_correction_resolution: int = 0,
    clobber: bool = False,
) -> tuple:
    """Get the profiles with batch correction applied.

    Batch correction is calculated by comparing vertex values along mid depth between adjacent chunks. The difference
    between the values is calculated and the mean difference is calculated.

    :param chunk_info: dataframe containing chunk information
    :param sect_info: dataframe containing section information
    :param resolution: resolution of the volume
    :param output_dir: path to output directory
    :param surf_dir: path to surfaces directory
    :param batch_correction_dir: path to batch correction directory
    :param ref_vol_fn: path to the structural reference volume
    :param gm_surf_fn: path to the gray matter surface
    :param wm_surf_fn: path to the white matter surface
    :param depth_list: list of depths
    :param batch_correction_resolution: resolution of the batch correction
    :param clobber: boolean indicating whether to overwrite existing files
    :return: sect_info, profiles_fn
    """
    n_depths = len(depth_list)

    # get the profiles without batch correction. interp order = 0 because that way
    # we avoid mixing vertex values across chunks
    (
        batch_surf_depth_mni_dict,
        batch_surf_depth_chunk_dict,
        batch_surf_raw_values_dict,
        profiles_fn,
        chunk_info_thickened_csv,
        batch_struct_vol_rsl_fn,
    ) = volumes_to_surface_profiles(
        chunk_info,
        sect_info,
        batch_correction_resolution,
        batch_correction_dir,
        surf_dir,
        ref_vol_fn,
        gm_surf_fn,
        wm_surf_fn,
        depth_list,
        interp_order=1,
        gaussian_sd=[0.5, 0.5 / 0.02, 0.5],
        clobber=clobber,
    )

    uncorrected_reconstructed_cortex_fn = f"{batch_correction_dir}/sub-{sub}_hemi-{hemisphere}_acq-{acquisition}_{resolution}mm_l{n_depths}_cortex_uncorrected.nii.gz"

    #
    # create full resolution profiles with linear interp
    #  | if batch_correction_resolution > 0 :
    #  | ---> create profiles with nearest neighbor interp and batch correction resolution
    #  | ---> calculate batch correction
    #  | ---> create profiles based on batch correction
    #  V
    #  create full resolution final cortical reconstruction

    # create a 3D volume of uncorrected values. useful for qc
    create_final_reconstructed_volume(
        uncorrected_reconstructed_cortex_fn,
        chunk_info_thickened_csv,
        batch_struct_vol_rsl_fn,
        batch_correction_resolution,
        batch_surf_depth_mni_dict,
        profiles_fn,
        clobber=clobber,
    )

    # update sect_info with chunk-level correction factors
    sect_info, paired_values = apply_batch_correction(
        chunk_info,
        sect_info,
        chunk_info_thickened_csv,
        batch_surf_raw_values_dict,
        profiles_fn,
        batch_surf_depth_mni_dict,
        batch_surf_depth_chunk_dict,
        depth_list,
        batch_correction_resolution,
        batch_struct_vol_rsl_fn,
        batch_correction_dir,
        clobber=clobber,
    )

    # calculate the old mean of the profiles so that we can recenter the corrected profiles
    values = np.load(profiles_fn + ".npz")["data"][:]
    old_mean = np.mean(values[values > values.min()])

    # recreate profiles_fn with batch corrected values. The sect_info contains batch_offset correction factors
    profiles_fn, _ = generate_surface_profiles(
        chunk_info,
        sect_info,
        surf_depth_chunk_dict,
        surf_depth_mni_dict,
        resolution,
        depth_list,
        struct_vol_rsl_fn,
        output_dir + "corrected/",
        clobber=clobber,
    )

    # recenter the corrected profiles
    print("\tCorrected profiles: ", profiles_fn)
    profiles = np.load(profiles_fn + ".npz")["data"][:]
    new_mean = np.mean(profiles[profiles > profiles.min()])
    print("\tOld mean: ", old_mean, " New mean: ", new_mean)
    profiles = profiles - new_mean + old_mean
    profiles[profiles < 0] = 0
    np.savez(profiles_fn, data=profiles)

    # mid_depth_index = int(np.rint(len(depth_list) / 2))
    # surf_fn = surf_depth_mni_dict[depth_list[mid_depth_index]]["depth_rsl_fn"]
    # sphere_fn = surf_depth_mni_dict[depth_list[mid_depth_index]]["sphere_rsl_fn"]
    # mid_values = profiles[:, mid_depth_index]
    # cortex_coords = load_mesh_ext(surf_fn)[0]
    # sphere_coords = load_mesh_ext(sphere_fn)[0]
    # png_fn = f"{output_dir}/paired_values_corrected.png"
    # cortex_png_fn = f"{output_dir}/paired_values_cortex.png"
    # sphere_png_fn = f"{output_dir}/paired_values_sphere.png"
    # plot the paired values of the corrected values. useful for qc
    # plot_paired_values(paired_values, mid_values, png_fn)
    # plot_paired_values_surf(paired_values, cortex_coords, cortex_png_fn)
    # plot_paired_values_surf(paired_values, sphere_coords, sphere_png_fn)

    return sect_info, chunk_info_thickened_csv, batch_surf_raw_values_dict, profiles_fn


def surface_pipeline(
    chunk_info: pd.DataFrame,
    sect_info: pd.DataFrame,
    resolution: float,
    output_dir: str,
    surf_dir: str,
    ref_vol_fn: str,
    gm_surf_fn: str,
    wm_surf_fn: str,
    n_depths: int = 0,
    surface_smoothing: int = 0,
    batch_correction_resolution: int = 0,
    clobber: bool = False,
) -> str:
    """Use surface-based interpolation to fill missing sections over the cortex.

    Volumetric interpolation is used to fill missing sections in the subcortex.

    :param sect_info_csv: path to csv file containing section information
    :param chunk_info_csv: path to csv file containing chunk information
    :param resolution: resolution of the volume
    :param output_dir: path to output directory
    :param ref_vol_fn: path to the structural reference volume
    :param gm_surf_fn: path to the gray matter surface
    :param wm_surf_fn: path to the white matter surface
    :param clobber: boolean indicating whether to overwrite existing files
    :return: None
    """
    sub = sect_info["sub"].values[0]
    hemisphere = sect_info["hemisphere"].values[0]
    acquisition = sect_info["acquisition"].values[0]

    reconstructed_cortex_fn = f"{output_dir}/sub-{sub}_hemi-{hemisphere}_acq-{acquisition}_{resolution}mm_l{n_depths}_cortex.nii.gz"
    smoothed_reconstructed_cortex_fn = f"{output_dir}/sub-{sub}_hemi-{hemisphere}_acq-{acquisition}_{resolution}mm_l{n_depths}_cortex_smoothed.nii.gz"

    if (
        not os.path.exists(reconstructed_cortex_fn)
        or (
            surface_smoothing > 0
            and not os.path.exists(smoothed_reconstructed_cortex_fn)
        )
        or clobber
    ):
        os.makedirs(output_dir, exist_ok=True)

        assert len(np.unique(sect_info["sub"])) == 1, "Error: multiple subjects"
        assert (
            len(np.unique(sect_info["hemisphere"])) == 1
        ), "Error: multiple hemispheres"
        assert (
            len(np.unique(sect_info["acquisition"])) == 1
        ), "Error: multiple acquisitions"

        if n_depths == 0:
            n_depths = np.ceil(5 / resolution).astype(int)

        batch_correction_dir = output_dir + "/batch_correction"
        os.makedirs(batch_correction_dir, exist_ok=True)

        depth_list = np.round(np.linspace(0, 1, int(n_depths)), 3)

        (
            surf_depth_mni_dict,
            surf_depth_chunk_dict,
            surf_raw_values_dict,
            profiles_fn,
            chunk_info_thickened_csv,
            struct_vol_rsl_fn,
        ) = volumes_to_surface_profiles(
            chunk_info,
            sect_info,
            resolution,
            output_dir,
            surf_dir,
            ref_vol_fn,
            gm_surf_fn,
            wm_surf_fn,
            depth_list,
            interp_order=0,
            clobber=clobber,
        )
        raw_profile_volume_fn = f"{output_dir}/sub-{sub}_hemi-{hemisphere}_acq-{acquisition}_{resolution}mm_l{n_depths}_raw_profiles.nii.gz"

        if batch_correction_resolution > 0:
            (
                sect_info,
                chunk_info_thickened_csv,
                batch_surf_raw_values_dict,
                final_profiles_fn,
            ) = get_profiles_with_batch_correction(
                chunk_info,
                sect_info,
                chunk_info_thickened_csv,
                sub,
                hemisphere,
                acquisition,
                resolution,
                surf_depth_chunk_dict,
                surf_depth_mni_dict,
                struct_vol_rsl_fn,
                output_dir,
                surf_dir,
                batch_correction_dir,
                ref_vol_fn,
                gm_surf_fn,
                wm_surf_fn,
                depth_list,
                batch_correction_resolution,
                clobber=clobber,
            )
        else:
            final_profiles_fn = profiles_fn

        # Write the uninterpolated surface profiles to a volume. This is useful for QC
        write_raw_profiles_to_volume(
            surf_raw_values_dict,
            surf_depth_mni_dict,
            raw_profile_volume_fn,
            struct_vol_rsl_fn,
            resolution,
            clobber=clobber,
        )

        print(f"Creating final volume with {final_profiles_fn}")
        # do surface based interpolation to fill missing sections
        create_final_reconstructed_volume(
            reconstructed_cortex_fn,
            chunk_info_thickened_csv,
            struct_vol_rsl_fn,
            resolution,
            surf_depth_mni_dict,
            final_profiles_fn,
            clobber=clobber,
        )

        if surface_smoothing > 0:
            sigma = surface_smoothing / 2.355

            smoothed_final_profiles_fn = smooth_surface_profiles(
                final_profiles_fn, surf_depth_mni_dict, sigma, clobber=clobber
            )

            # do surface based interpolation to fill missing sections
            create_final_reconstructed_volume(
                smoothed_reconstructed_cortex_fn,
                chunk_info_thickened_csv,
                struct_vol_rsl_fn,
                resolution,
                surf_depth_mni_dict,
                smoothed_final_profiles_fn,
                clobber=clobber,
            )
        # create a mask of the subcortex that can be used for volumetric interpolation
        # FIXME NOT YET IMPLEMENTED
        # subcortex_mask_fn = utils.create_subcortex_mask(wm_surf_fn)

        # perform volumetric interpolation to fill missing sections in the subcortex
        # FIXME NOT YET IMPLEMENTED
        # subcortex_interp_fn = volinterp.volumetric_interpolation(
        #    sect_info, brain_mask_fn, clobber=clobber
        # )

        # combine the interpolated cortex and subcortex
        # FIXME NOT YET IMPLEMENTED
        # combine_volumes(interp_cortex_fn, subcortex_mask_fn)

    return reconstructed_cortex_fn, smoothed_reconstructed_cortex_fn


def volumetric_pipeline(
    chunk_info_csv: str,
    sect_info_csv: str,
    resolution: float,
    output_dir: str,
    clobber: bool = False,
) -> None:
    """Use volumetric interpolation to fill missing sections over the entire brain."""
    # chunk_info_thickened_csv = create_thickened_volumes(
    #    curr_output_dir, chunk_info, sect_info, resolution
    # )
    # do volumetric interpolation to fill missing sections through whole brain
    # final_interp_fn = volinterp.volumetric_interpolation(
    #    sect_info, brain_mask_fn, clobber=clobber
    # )
    raise NotImplementedError

    return None


def interpolate_missing_sections(
    hemi_info_csv: str,
    chunk_info_csv: str,
    sect_info_csv: str,
    resolution: float,
    output_dir: str,
    n_depths: int = 0,
    surface_smoothing: int = 0,
    batch_correction_resolution: int = 0,
    clobber: bool = False,
) -> str:
    """Interpolate missing sections in a volume.

    :param sect_info: a dataframe with the following columns:
    :param brain_mask_fn: path to the brain mask
    :param resolution: resolution of the volume
    :param clobber: if True, overwrite existing files
    :return: path to the interpolated volume
    """
    print("\n\tInterpolate Missing Sections\n")

    sect_info = pd.read_csv(sect_info_csv, index_col=False)
    chunk_info = pd.read_csv(chunk_info_csv, index_col=False)
    hemi_info = pd.read_csv(hemi_info_csv, index_col=False)

    surf_dir = f"{output_dir}/surfaces/"

    out_chunk_info = []

    for (sub, hemisphere, acquisition), curr_sect_info in sect_info.groupby(
        [
            "sub",
            "hemisphere",
            "acquisition",
        ]
    ):
        curr_output_dir = f"{output_dir}/sub-{sub}/hemi-{hemisphere}/acq-{acquisition}/"

        os.makedirs(curr_output_dir, exist_ok=True)

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

        gm_surf_fn = curr_hemi_info["gm_surf"].values[0]
        wm_surf_fn = curr_hemi_info["wm_surf"].values[0]
        ref_vol_fn = curr_hemi_info["struct_ref_vol"].values[0]

        if (os.path.exists(gm_surf_fn) and os.path.exists(wm_surf_fn)) or clobber:
            reconstructed_filename, reconstructed_smoothed_filename = surface_pipeline(
                curr_chunk_info,
                curr_sect_info,
                resolution,
                curr_output_dir,
                surf_dir,
                ref_vol_fn,
                gm_surf_fn,
                wm_surf_fn,
                n_depths,
                surface_smoothing=surface_smoothing,
                batch_correction_resolution=batch_correction_resolution,
                clobber=clobber,
            )

            curr_chunk_info["acquisition"] = acquisition
            curr_chunk_info["reconstructed_filename"] = reconstructed_filename
            curr_chunk_info[
                "reconstructed_smoothed_filename"
            ] = reconstructed_smoothed_filename

            out_chunk_info.append(curr_chunk_info)

        else:
            print("Error: Volumetric interpolation pipeline not yet implemented")
            exit(1)
            # volumetric_pipeline(sect_info_csv, chunk_info_csv, resolution, output_dir, clobber=clobber)

    chunk_info = pd.concat(out_chunk_info, ignore_index=True)
    chunk_info_csv = f"{output_dir}/reconstructed_chunk_info.csv"
    chunk_info.to_csv(chunk_info_csv, index=False)

    return chunk_info_csv


if __name__ == "__main__":
    pass
