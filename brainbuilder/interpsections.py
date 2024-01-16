import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from brainbuilder.interp.acqvolume import create_thickened_volumes
from brainbuilder.interp.batch_correction import apply_batch_correction
from brainbuilder.interp.prepare_surfaces import prepare_surfaces
from brainbuilder.interp.surfinterp import (
    create_final_reconstructed_volume,
    generate_surface_profiles,
    write_raw_profiles_to_volume,
)
from brainbuilder.utils import utils
from brainbuilder.utils.mesh_utils import load_mesh_ext, smooth_surface_profiles


def plot_paired_values(paired_values, mid_values, png_fn):
    """Plot the paired values for the batch correction step
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


def plot_paired_values_surf(paired_values, coords, png_fn):
    """ """
    curr_idx = paired_values["curr_idx"].astype(int)
    next_idx = paired_values["next_idx"].astype(int)

    curr_coords = coords[curr_idx]

    next_coords = coords[next_idx]

    x = coords[:, 0]
    y = coords[:, 1]
    z = coords[:, 2]

    xr = np.max(x) - np.min(x)
    yr = np.max(y) - np.min(y)
    zr = np.max(z) - np.min(z)

    dx0 = xr / 4
    dy0 = yr / 4
    dz0 = zr / 4

    dx1 = xr / 6
    dy1 = yr / 6
    dz1 = zr / 20

    for i, ((xc, yc, zc), (xn, yn, zn)) in enumerate(zip(curr_coords, next_coords)):
        print(paired_values.iloc[i])
        # plot x vs y

        xmin = min([xc, xn])
        xmax = max([xc, xn])
        ymin = min([yc, yn])
        ymax = max([yc, yn])
        zmin = min([zc, zn])
        zmax = max([zc, zn])

        xidx0 = (x > xmin - dx0) & (x < xmax + dx0)
        yidx0 = (y > ymin - dy0) & (y < ymax + dy0)
        zidx0 = (z > zmin - dz1) & (z < zmax + dz1)

        idx0 = xidx0 & yidx0 & zidx0

        xi0 = x[idx0]
        yi0 = y[idx0]
        zi0 = z[idx0]

        dist = np.sqrt((xc - xn) ** 2 + (yc - yn) ** 2 + (zc - zn) ** 2)
        print(dist)

        plt.figure(figsize=(10, 10))

        plt.subplot(2, 1, 1)
        plt.scatter([xc], [yc], c="b")
        plt.scatter([xn], [yn], c="b")
        plt.plot([xc, xn], [yc, yn], c="r", alpha=0.6)
        plt.scatter(xi0, yi0, c="k", alpha=0.01)

        # PLot y vs z

        xidx1 = (x > xmin - dx1) & (x < xmax + dx1)
        yidx1 = (y > ymin - dy0) & (y < ymax + dy0)
        zidx1 = (z > zmin - dz0) & (z < zmax + dz0)

        idx1 = xidx1 & yidx1 & zidx1

        xi1 = x[idx1]
        yi1 = y[idx1]
        zi1 = z[idx1]

        plt.subplot(2, 1, 2)
        plt.title(f"Dist: {dist}")
        plt.scatter([yc], [zc], c="b")
        plt.scatter([yn], [zn], c="b")
        plt.plot([yc, yn], [zc, zn], c="r", alpha=0.6)
        plt.scatter(yi1, zi1, c="k", alpha=0.01)

        point_fn = png_fn.replace(".png", f"_{i}.png")
        print(point_fn)

        plt.savefig(point_fn)
        plt.clf()
        plt.cla()
        if i > 10:
            break


def resample_struct_reference_volume(
    orig_struct_vol_fn, resolution, output_dir, clobber=False
):
    """Resample the structural reference volume to the desired resolution

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
):
    n_depths = len(depth_list)

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
):
    """Get the profiles with batch correction applied. Batch correction is calculated by
    by comparing vertex values along mid depth between adjacent chunks. The difference
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
    profiles_fn, _, _ = generate_surface_profiles(
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

    png_fn = f"{output_dir}/paired_values_corrected.png"
    cortex_png_fn = f"{output_dir}/paired_values_cortex.png"
    sphere_png_fn = f"{output_dir}/paired_values_sphere.png"

    # recenter the corrected profiles
    print("\tCorrected profiles: ", profiles_fn)
    profiles = np.load(profiles_fn + ".npz")["data"][:]
    new_mean = np.mean(profiles[profiles > profiles.min()])
    print("\tOld mean: ", old_mean, " New mean: ", new_mean)
    profiles = profiles - new_mean + old_mean
    profiles[profiles < 0] = 0
    np.savez(profiles_fn, data=profiles)

    mid_depth_index = int(np.rint(len(depth_list) / 2))

    mid_values = profiles[:, mid_depth_index]
    surf_fn = surf_depth_mni_dict[depth_list[mid_depth_index]]["depth_rsl_fn"]
    sphere_fn = surf_depth_mni_dict[depth_list[mid_depth_index]]["sphere_rsl_fn"]

    cortex_coords = load_mesh_ext(surf_fn)[0]
    sphere_coords = load_mesh_ext(sphere_fn)[0]

    # plot the paired values of the corrected values. useful for qc
    # plot_paired_values(paired_values, mid_values, png_fn)
    # plot_paired_values_surf(paired_values, cortex_coords, cortex_png_fn)
    # plot_paired_values_surf(paired_values, sphere_coords, sphere_png_fn)

    return sect_info, batch_surf_raw_values_dict, profiles_fn


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
    use_surface_smoothing: bool = False,
    batch_correction_resolution: int = 0,
    clobber: bool = False,
) -> None:
    """Use surface-based interpolation to fill missing sections over the cortex. Volumetric interpolation
    is used to fill missing sections in the subcortex.
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
            use_surface_smoothing
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
            struct_vol_rsl_fn,
            resolution,
            surf_depth_mni_dict,
            final_profiles_fn,
            clobber=clobber,
        )

        if use_surface_smoothing:
            sigma = 2 * resolution / 2.355

            smoothed_final_profiles_fn = smooth_surface_profiles(
                final_profiles_fn, surf_depth_mni_dict, sigma, clobber=clobber
            )

            # do surface based interpolation to fill missing sections
            create_final_reconstructed_volume(
                smoothed_reconstructed_cortex_fn,
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

        return None


def volumetric_pipeline(
    chunk_info_csv: str,
    sect_info_csv: str,
    resolution: float,
    output_dir: str,
    clobber: bool = False,
) -> None:
    """Use volumetric interpolation to fill missing sections over the entire brain"""
    chunk_info_thickened_csv = create_thickened_volumes(
        curr_output_dir, chunk_info, sect_info, resolution
    )
    # do volumetric interpolation to fill missing sections through whole brain
    final_interp_fn = volinterp.volumetric_interpolation(
        sect_info, brain_mask_fn, clobber=clobber
    )

    return None


def interpolate_missing_sections(
    hemi_info_csv: str,
    chunk_info_csv: str,
    sect_info_csv: str,
    resolution: float,
    output_dir: str,
    n_depths: int = 0,
    use_surface_smoothing: bool = False,
    batch_correction_resolution: int = 0,
    clobber: bool = False,
):
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

    for (acquisition, chunk), chunk_info_row in sect_info.groupby(
        [
            "acquisition",
            "chunk",
        ]
    ):
        print(chunk, chunk_info_row["sample"].min(), chunk_info_row["sample"].max())

    surf_dir = f"{output_dir}/surfaces/"

    for (sub, hemisphere, acquisition), curr_sect_info in sect_info.groupby(
        [
            "sub",
            "hemisphere",
            "acquisition",
        ]
    ):
        # if acquisition != 'cgp5' : continue #FIXME delete this line

        curr_output_dir = f"{output_dir}/sub-{sub}/hemi-{hemisphere}/acq-{acquisition}/"

        os.makedirs(curr_output_dir, exist_ok=True)

        curr_chunk_info = chunk_info.loc[
            (chunk_info["sub"] == sub) & (chunk_info["hemisphere"] == hemisphere)
        ]

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
            surface_pipeline(
                curr_chunk_info,
                curr_sect_info,
                resolution,
                curr_output_dir,
                surf_dir,
                ref_vol_fn,
                gm_surf_fn,
                wm_surf_fn,
                n_depths,
                use_surface_smoothing=use_surface_smoothing,
                batch_correction_resolution=batch_correction_resolution,
                clobber=clobber,
            )
        else:
            print("Error: Volumetric interpolation pipeline not yet implemented")
            exit(1)
            # volumetric_pipeline(sect_info_csv, chunk_info_csv, resolution, output_dir, clobber=clobber)

    return None


if __name__ == "__main__":
    pass
