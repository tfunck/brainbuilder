"""Main function for performing interpolation between acquired 2D sections."""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from brainbuilder.interp.prepare_surfaces import prepare_surfaces
from brainbuilder.interp.surfinterp import (
    create_final_reconstructed_volume,
    generate_surface_profiles,
    write_raw_profiles_to_volume,
)
from brainbuilder.interp.volinterp import volumetric_pipeline
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


def volumes_to_surface_profiles(
    chunk_info: pd.DataFrame,
    sect_info: pd.DataFrame,
    resolution: float,
    output_dir: str,
    surf_dir: str,
    ref_vol_fn: str,
    gm_surf_fn: str,
    wm_surf_fn: str,
    volume_type: str = "thickened",
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
    os.makedirs(surf_dir, exist_ok=True)

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

    struct_vol_rsl_fn = utils.resample_struct_reference_volume(
        ref_vol_fn, resolution, output_dir, clobber=clobber
    )

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
        volume_type=volume_type,
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


def surface_pipeline(
    sect_info,
    chunk_info,
    hemi_info,
    resolution,
    output_dir,
    surf_dir,
    ref_vol_fn,
    n_depths,
    surface_smoothing: float = 0,
    clobber: bool = False,
):
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

        (
            reconstructed_filename,
            reconstructed_smoothed_filename,
        ) = create_surface_interpolated_volume(
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
            clobber=clobber,
        )

        curr_chunk_info["acquisition"] = acquisition
        curr_chunk_info["reconstructed_filename"] = reconstructed_filename
        curr_chunk_info[
            "reconstructed_smoothed_filename"
        ] = reconstructed_smoothed_filename

    return curr_chunk_info


def create_surface_interpolated_volume(
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

    if n_depths == 0:
        n_depths = np.ceil(5 / resolution).astype(int)

    reconstructed_cortex_fn = f"{output_dir}/sub-{sub}_hemi-{hemisphere}_acq-{acquisition}_{resolution}mm_l{n_depths}_cortex.nii.gz"
    smoothed_reconstructed_cortex_fn = f"{output_dir}/sub-{sub}_hemi-{hemisphere}_acq-{acquisition}_{resolution}mm_l{n_depths}_cortex_smoothed.nii.gz"

    if (
        not os.path.exists(reconstructed_cortex_fn)
        or (
            surface_smoothing > 0
            and not os.path.exists(smoothed_reconstructed_cortex_fn)
        )
        or clobber
        # or True # DEBUG DELETEME
    ):
        os.makedirs(output_dir, exist_ok=True)

        assert len(np.unique(sect_info["sub"])) == 1, "Error: multiple subjects"
        assert (
            len(np.unique(sect_info["hemisphere"])) == 1
        ), "Error: multiple hemispheres"
        assert (
            len(np.unique(sect_info["acquisition"])) == 1
        ), "Error: multiple acquisitions"

        depth_list = np.round(np.linspace(0, 1, int(n_depths)), 3)

        (
            surf_depth_mni_dict,
            surf_depth_chunk_dict,
            surf_raw_values_dict,
            profiles_fn,
            chunk_info,
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
            interp_order=1,
            clobber=clobber,
        )
        raw_profile_volume_fn = f"{output_dir}/sub-{sub}_hemi-{hemisphere}_acq-{acquisition}_{resolution}mm_l{n_depths}_raw_profiles.nii.gz"

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
            chunk_info,
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

    return reconstructed_cortex_fn, smoothed_reconstructed_cortex_fn


global METHOD_SURFACE
global METHOD_VOLUMETRIC
METHOD_SURFACE = "surface"
METHOD_VOLUMETRIC = "volumetric"


def interpolate_missing_sections(
    hemi_info_csv: str,
    chunk_info_csv: str,
    sect_info_csv: str,
    resolution: float,
    resolution_3d: float,
    resolution_list: list,
    output_dir: str,
    n_depths: int = 0,
    surface_smoothing: int = 0,
    interp_method: str = METHOD_VOLUMETRIC,
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

    os.makedirs(output_dir, exist_ok=True)

    interp_chunk_info = volumetric_pipeline(
        sect_info,
        chunk_info,
        hemi_info,
        resolution,
        resolution_3d,
        resolution_list,
        output_dir,
        clobber=clobber,
    )
    assert 'acquisition' in interp_chunk_info.columns, "Error: 'acquisition' column not found in chunk info"

    if interp_method == METHOD_SURFACE:
        for (sub, hemisphere, acq), curr_sect_info in sect_info.groupby(
            ["sub", "hemisphere", "acquisition"]
        ):
            curr_hemi_info = hemi_info.loc[
                (hemi_info["sub"] == sub) & (hemi_info["hemisphere"] == hemisphere)
            ]
            assert len(curr_hemi_info) > 0, "Error: no hemisphere info found"

            acq_chunk_info = interp_chunk_info.loc[
                (interp_chunk_info["sub"] == sub)
                & (interp_chunk_info["hemisphere"] == hemisphere)
                & (interp_chunk_info["acquisition"] == acq)
            ]
            assert len(acq_chunk_info) > 0, "Error: no chunk info found"

            ref_vol_fn = curr_hemi_info["struct_ref_vol"].values[0]

            depth_list = np.round(np.linspace(0, 1, int(n_depths)), 3)

            gm_surf_fn = curr_hemi_info["gm_surf"].values[0]
            wm_surf_fn = curr_hemi_info["wm_surf"].values[0]

            curr_output_dir = f"{output_dir}/sub-{sub}/hemi-{hemisphere}/acq-{acq}/"

            os.makedirs(curr_output_dir, exist_ok=True)

            reconstructed_cortex_fn = f"{curr_output_dir}/sub-{sub}_hemi-{hemisphere}_acq-{acq}_{resolution}mm_l{n_depths}_cortex.nii.gz"

            (
                surf_depth_mni_dict,
                _,  # surf_depth_chunk_dict,
                _,  # surf_raw_values_dict,
                profiles_fn,
                chunk_info,
                struct_vol_rsl_fn,
            ) = volumes_to_surface_profiles(
                acq_chunk_info,
                curr_sect_info,
                resolution,
                curr_output_dir,
                output_dir + "/surfaces/",
                ref_vol_fn,
                gm_surf_fn,
                wm_surf_fn,
                depth_list=depth_list,
                volume_type="interp_nat",
                interp_order=1,
                clobber=clobber,
            )

            create_final_reconstructed_volume(
                reconstructed_cortex_fn,
                acq_chunk_info,
                struct_vol_rsl_fn,
                resolution,
                surf_depth_mni_dict,
                profiles_fn,
                volume_type="interp_stx",
                clobber=True,  # clobber
            )

    chunk_info_csv = f"{output_dir}/reconstructed_chunk_info.csv"
    interp_chunk_info.to_csv(chunk_info_csv, index=False)

    return chunk_info_csv


if __name__ == "__main__":
    pass
