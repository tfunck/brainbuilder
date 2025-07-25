"""Interpolate missing values over cortical surfaces."""
import argparse
import os
import re

import nibabel as nb_surf
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import brainbuilder.utils.ants_nibabel as nib
from brainbuilder.utils import utils 
from brainbuilder.utils.mesh_utils import smooth_surface_profiles
from brainbuilder.interp.prepare_surfaces import prepare_surfaces
from brainbuilder.interp.acqvolume import (
    create_thickened_volumes,
    transform_chunk_volumes,
)
from brainbuilder.utils.mesh_io import load_mesh_ext
from brainbuilder.utils.mesh_utils import (
    volume_to_mesh,
    write_mesh_to_volume,
)
from brainbuilder.utils.utils import get_section_intervals

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


def get_valid_coords(coords: np.ndarray, iw: list) -> tuple:
    """Get valid surface coordinates within a given interval.

    :param coords: surface coordinates
    :param iw: interval
    :return: valid surface coordinates
    """
    lower = min(iw)
    upper = max(iw)

    idx_bool = (coords[:, 1] >= lower) & (coords[:, 1] < upper)

    # valid_coords_idx_0  = idx_range[idx_bool]

    valid_coords_idx = np.where(idx_bool)[0]

    valid_coords_idx = valid_coords_idx.reshape(valid_coords_idx.shape[0])

    valid_coords = coords[valid_coords_idx, :]

    return valid_coords, valid_coords_idx


def get_profiles(
    profiles_fn: str,
    surf_depth_chunk_dict: dict,
    surf_depth_mni_dict: dict,
    surf_raw_values_dict: dict,
    interp_order: int = 1,
    clobber: bool = False,
) -> str:
    """Get raw surface values and interpoalte over surface to fill in missing pixel intensities.

    :param profiles_fn: path to profiles file
    :param surf_depth_chunk_dict: dictionary with surface information for each chunk
    :param surf_depth_mni_dict: dictionary with surface information for each depth
    :param surf_raw_values_dict: dictionary with raw surface values for each depth
    :param clobber: boolean indicating whether to overwrite existing files
    :return: path to profiles file
    """
    print("\tGetting profiles")
    # 3. Interpolate missing densities over surface
    if not os.path.exists(profiles_fn + ".npz") or clobber:
        depth_list = sorted(surf_depth_mni_dict.keys())
        example_depth_fn = surf_raw_values_dict[depth_list[0]]

        nrows = pd.read_csv(example_depth_fn, header=None).shape[0]
        ncols = len(depth_list)

        profiles = np.zeros([nrows, ncols])

        to_do = []
        n_elements_list = []
        element_size_list = []

        coords_dtype_itemsize = load_mesh_ext(
            surf_depth_mni_dict[depth_list[0]]["depth_rsl_fn"]
        )[0].dtype.itemsize

        # get sorted list of depths
        for depth, raw_values_fn in surf_raw_values_dict.items():
            depth_index = depth_list.index(depth)

            profiles_raw = pd.read_csv(raw_values_fn, header=None, index_col=None)

            sphere_rsl_fn = surf_depth_mni_dict[depth]["sphere_rsl_fn"]

            surface_val = profiles_raw.values.reshape(
                -1,
            )
            assert (
                np.sum(np.abs(surface_val)) >= 1
            ), f"Assert: empty file {raw_values_fn}"

            to_do.append((sphere_rsl_fn, surface_val, depth_index))

            n_elements_list = [surface_val.shape[0], surface_val.shape[0] * 3]
            element_size_list = [surface_val.dtype.itemsize, coords_dtype_itemsize]

        def interpolate_over_surface_(
            sphere_rsl_fn: str, surface_val: str, depth_index: int
        ) -> tuple:
            """Interpolate over a surface sphere.

            :param sphere_rsl_fn: path to sphere object
            :param surface_val: vector of surface values
            :param depth_index: index of depth
            :return: interpolated values
            """
            return depth_index, interpolate_over_surface(
                sphere_rsl_fn, surface_val, threshold=15, order=interp_order
            )

        num_cores = utils.get_maximum_cores(n_elements_list, element_size_list)

        results = Parallel(n_jobs=num_cores)(
            delayed(interpolate_over_surface_)(sphere_rsl_fn, surface_val, depth_index)
            for sphere_rsl_fn, surface_val, depth_index in to_do
        )

        for depth_index, profile_vector in results:
            profiles[:, depth_index] = profile_vector

        np.savez(profiles_fn, data=profiles)

    return profiles_fn


def project_values_over_section_intervals(
    coords: np.ndarray,
    vol: np.ndarray,
    starts: np.ndarray,
    steps: np.ndarray,
    dimensions: np.ndarray,
    clobber: bool = False,
) -> tuple:
    """Project values over section intervals.

    :param coords: surface coordinates
    :param vol: volume
    :param starts: start of volume
    :param steps: step size of volume
    :param dimensions: dimensions of volume
    :param clobber: boolean indicating whether to overwrite existing files
    :return: tuple of values and number of vertices
    """
    all_values = np.zeros(coords.shape[0])
    n = np.zeros(coords.shape[0])

    # get the voxel intervals along the y-axis the volume
    intervals_voxel = get_section_intervals(vol)

    assert len(intervals_voxel) > 0, "Error: the length of intervals_voxels == 0 "

    idx_range = np.arange(coords.shape[0])

    # iterate over intervals along the y-axis of the volume
    for y0, y1 in intervals_voxel:
        # convert from voxel values to real world coordinates
        y0w = y0 * steps[1] + starts[1]
        y1w = y1 * steps[1] + starts[1]

        # the voxel values should be the same along the y-axis within an interval
        valid_coords_world, valid_coords_idx = get_valid_coords(coords, [y0w, y1w])

        if valid_coords_world.shape[0] != 0:
            values, valid_idx = volume_to_mesh(
                valid_coords_world, vol, starts, steps, dimensions
            )

            section_range = idx_range[valid_coords_idx]

            valid_range = section_range[valid_idx]

            all_values[valid_range] += values

            n[valid_range] += 1

            # Also remove vertices that are less than 0

            if np.sum(np.abs(values)) > 0:
                f"Error: empty vol[x,z] in project_volume_to_surfaces for {y0}"

            assert np.sum(np.isnan(values)) == 0, "Error: nan found in values."
        else:
            print("\t\t\t\tWarning: no valid coordinates found in interval", y0, y1)

    return all_values, n


def volume_to_surface_over_chunks(
    surf_chunk_dict: dict,
    volumes_df: dict,
    interp_csv: str,
    depth: float,
    volume_type: str = "thickened",
    clobber: bool = False,
    verbose: bool = False,
) -> None:
    """Project a volume onto a surface.

    :param surf_chunk_dict: dictionary with surface information for each chunk
    :param volumes_df: dataframe with volume information
    :param interp_csv: path to csv file containing interpolated values
    :param clobber: boolean indicating whether to overwrite existing files
    :return: None
    """
    # Get surfaces transformed into chunk space
    if not os.path.exists(interp_csv) or clobber:
        surf_fn = list(surf_chunk_dict.values())[0]
        nvertices = load_mesh_ext(surf_fn)[0].shape[0]
        # the default value of the vertices is set to -100 to make it easy to distinguish
        # between vertices where a acquisition density of 0 is measured versus the default value
        all_values = np.zeros(nvertices)
        n = np.zeros(nvertices)

        n_elems_list = []
        elem_size_list = []
        to_do = []

        # Iterate over chunks within a given
        for chunk, surf_fn in surf_chunk_dict.items():
            print("\t\t\tChunk:", chunk)

            vol_fn = volumes_df[volume_type].loc[volumes_df["chunk"] == chunk].values[0]

            coords, _ = load_mesh_ext(surf_fn)
            n_vtx = coords.shape[0]

            if verbose or True:
                print("surf_fn:\n", surf_fn)
                print("vol_fn:\n", vol_fn)

            # read chunk volume
            img = nib.load(vol_fn)
            vol = img.get_fdata()
            n_vox = np.product(vol.shape)

            affine = nb_surf.load(vol_fn).affine

            assert np.max(vol) != np.min(
                vol
            ), f"Error: empty volume {vol_fn}; {np.max(vol) != np.min(vol)} "

            # set variable names for coordinate system
            starts = affine[[0, 1, 2], [3, 3, 3]]
            steps = affine[[0, 1, 2], [0, 1, 2]]

            dimensions = vol.shape
            # get the intervals along the y-axis of the volume where
            # we have voxel intensities that should be interpolated onto the surfaces

            to_do.append((coords, vol, starts, steps, dimensions))

            n_elems_list += [n_vtx, n_vtx * 3, n_vox]
            elem_size_list += [
                all_values.dtype.itemsize,
                coords.dtype.itemsize,
                vol.dtype.itemsize,
            ]

        num_cores = utils.get_maximum_cores(n_elems_list, elem_size_list)

        results = Parallel(n_jobs=num_cores)(
            delayed(volume_to_mesh)(coords, vol, starts, steps, dimensions)
            for coords, vol, starts, steps, dimensions in to_do
        )

        for chunk_values, chunk_n in results:
            if len(np.unique(vol)) > 1:
                idx = chunk_values == np.min(vol)
                chunk_values[idx] = 0

            all_values[chunk_n] += chunk_values
            n += chunk_n

        all_values[n > 1] = np.min(all_values)  # set overlap vertices to 0
        all_values[n > 0] = all_values[n > 0] / n[n > 0]

        # x=np.zeros(coords.shape[0])
        # x[chunk_n] = chunk_values
        # vol, _ = mesh_to_volume(coords, x, dimensions, starts, steps)
        # nib.Nifti1Image(vol, img.affine, direction_order='lpi').to_filename(f'{os.path.dirname(interp_csv)}/chunk_{chunk}_{depth}.nii.gz')

        assert (
            np.sum(np.abs(all_values)) > 0
        ), "Error, empty array all_values in project_volumes_to_surfaces"
        print("\tWriting surface values to", interp_csv)

        np.savetxt(interp_csv, all_values)

    return None


def generate_surface_profiles(
    chunk_info: pd.DataFrame,
    sect_info: pd.DataFrame,
    surf_depth_chunk_dict: dict,
    surf_depth_mni_dict: dict,
    resolution: float,
    depth_list: list,
    struct_vol_rsl_fn: str,
    output_dir: str,
    volume_type: str = "thickened",
    tissue_type: str = "",
    gaussian_sd: float = 0,
    interp_order: int = 1,
    clobber: bool = False,
) -> str:
    """Create surface profiles over a set of brain chunks and over multiple cortical depths.

    :param chunk_info: dataframe with chunk information
    :param sect_info: dataframe with section information
    :param resolution: resolution of the volume
    :param depth_list: list of cortical depths
    :param output_dir: output directory where results will be put
    :param clobber: if True, overwrite existing files
    :return: path to profiles file
    """
    os.makedirs(output_dir, exist_ok=True)

    if volume_type == "thickened":
        chunk_info_thickened_csv = create_thickened_volumes(
            output_dir,
            chunk_info,
            sect_info,
            resolution,
            struct_vol_rsl_fn,
            gaussian_sd=gaussian_sd,
            clobber=clobber,
        )

        chunk_info = transform_chunk_volumes(
            pd.read_csv(chunk_info_thickened_csv),
            struct_vol_rsl_fn,
            output_dir,
            clobber=clobber,
        )

    sub = sect_info["sub"].values[0]
    hemisphere = sect_info["hemisphere"].values[0]
    acquisition = sect_info["acquisition"].values[0]

    n_depths = len(depth_list)

    output_prefix = f"{output_dir}/sub-{sub}_hemi-{hemisphere}_acq-{acquisition}_{resolution}mm{tissue_type}_l{n_depths}"

    os.makedirs(output_dir, exist_ok=True)

    # Project voxel intensities onto surfaces
    # Extract profiles from the chunks using the surfaces
    (
        profiles_fn,
        surf_raw_values_dict,
    ) = project_volume_to_surfaces(
        surf_depth_chunk_dict,
        surf_depth_mni_dict,
        chunk_info,
        output_prefix,
        volume_type=volume_type,
        interp_order=interp_order,
    )

    return (
        profiles_fn,
        surf_raw_values_dict,
        chunk_info,
    )


def write_raw_profiles_to_volume(
    surf_raw_values_dict: dict,
    surf_depth_mni_dict: dict,
    raw_profile_volume_fn: str,
    ref_volume_fn: str,
    resolution: float,
    clobber: bool = False,
) -> None:
    """Write raw profiles to a volume.

    :param surf_raw_values_dict: dictionary with raw surface values for each depth
    :param surf_depth_mni_dict: dictionary with surface information for each depth
    :param raw_profile_volume_fn: path to raw profiles volume
    :param ref_volume_fn: path to reference volume
    :param resolution: resolution of the volume
    :param clobber: boolean indicating whether to overwrite existing files
    :return: None
    """
    if not os.path.exists(raw_profile_volume_fn) or clobber:
        depth_list = sorted(surf_depth_mni_dict.keys())
        surfaces = [surf_depth_mni_dict[depth]["depth_rsl_fn"] for depth in depth_list]
        ncols = len(depth_list)
        nrows = pd.read_csv(list(surf_raw_values_dict.values())[0], header=None).shape[
            0
        ]

        profiles = np.zeros([nrows, ncols])

        for depth, raw_values_fn in surf_raw_values_dict.items():
            depth_index = depth_list.index(depth)

            profiles_raw = pd.read_csv(raw_values_fn, header=None, index_col=None)

            surface_val = profiles_raw.values.reshape(
                -1,
            )
            assert (
                np.sum(np.abs(surface_val)) >= 1
            ), f"Assert: empty file {raw_values_fn}"

            profiles[:, depth_index] = surface_val

        write_mesh_to_volume(
            profiles,
            surfaces,
            ref_volume_fn,
            raw_profile_volume_fn,
            resolution,
            clobber=clobber,
        )
    return None


def project_volume_to_surfaces(
    surf_depth_chunk_dict: dict,
    surf_depth_mni_dict: dict,
    chunk_info_thickened: dict,
    output_prefix: str,
    tissue_type: str = "",
    volume_type: str = "thickened",
    interp_order: int = 1,
    clobber: bool = False,
) -> str:
    """Project the voxel values of a volume onto a mesh.

    :param profiles_fn: path to profiles file
    :param surf_depth_chunk_dict: dictionary with surface information for each chunk
    :param output_prefix: prefix for output files
    :param tissue_type: tissue type of the interpolated volumes
    :param clobber: boolean indicating whether to overwrite existing files
    :return: path to profiles file
    """
    profiles_fn = f"{output_prefix}_profiles"
    surf_raw_values_dict = {}

    # iterate over depths in cortical surface
    for depth, surf_depth_dict in surf_depth_chunk_dict.items():
        print("\t\tDepth", depth)

        interp_csv = f"{output_prefix}{tissue_type}_{depth}_interp_{volume_type}.csv"

        if not os.path.exists(interp_csv) or clobber:
            volume_to_surface_over_chunks(
                surf_depth_dict,
                chunk_info_thickened,
                interp_csv,
                depth,
                volume_type=volume_type,
                clobber=clobber,
            )

        surf_raw_values_dict[depth] = interp_csv

    if not os.path.exists(profiles_fn + ".npz") or clobber:
        profiles_fn = get_profiles(
            profiles_fn,
            surf_depth_chunk_dict,
            surf_depth_mni_dict,
            surf_raw_values_dict,
            interp_order=interp_order,
            clobber=clobber,
        )

    return profiles_fn, surf_raw_values_dict


def spherical_np(xyz: np.ndarray) -> np.ndarray:
    """Convert Cartesian coordinates to spherical coordinates.

    :param xyz: Cartesian coordinates
    :return: spherical coordinates
    """
    pts = np.zeros(xyz.shape)
    xy = xyz[:, 0] ** 2 + xyz[:, 1] ** 2
    pts[:, 0] = np.sqrt(xy + xyz[:, 2] ** 2)
    pts[:, 1] = np.arctan2(np.sqrt(xy), xyz[:, 2])
    pts[:, 2] = np.arctan2(xyz[:, 1], xyz[:, 0])
    return pts


def interpolate_over_surface(
    sphere_obj_fn: str,
    surface_val: np.array,
    threshold: float = 0,
    surface_mask: np.ndarray = None,
    order: int = 1,
) -> np.ndarray:
    """Interpolate over a surface sphere.

    :param sphere_obj_fn: path to sphere object
    :param surface_val: vector of surface values
    :param threshold: threshold for surface values
    :param surface_mask: mask of surface values
    :param order: order of interpolation
    :return: interpolated values
    """
    print("\tInterpolating Over Surface")
    print("\t\t\tSphere fn:", sphere_obj_fn)
    # get coordinates from dicitonary with mesh info
    coords = load_mesh_ext(sphere_obj_fn)[0]

    assert (
        coords.shape[0] == surface_val.shape[0]
    ), f"Error: mismatch in shape of spherical mesh and surface values {coords.shape} and {surface_val.shape}"

    spherical_coords = spherical_np(coords)

    if not isinstance(surface_mask, np.ndarray):
        # define a mask of verticies where we have receptor densitiies
        t0, t1 = np.percentile(
            surface_val[surface_val > 0], [threshold, 100 - threshold]
        )
        surface_mask = surface_val > t0  # & (surface_val < t1)

    assert np.sum(np.abs(surface_mask)) != 0, "Error, empty profiles {}".format(
        np.sum(surface_mask)
    )

    # get coordinates for vertices in mask
    spherical_coords_src = spherical_coords[surface_mask.astype(bool), :]

    # get spherical coordinates from cortical mesh vertex coordinates
    lats_src, lons_src = (
        spherical_coords_src[:, 1] - np.pi / 2,
        spherical_coords_src[:, 2],
    )

    # create mesh data structure
    import stripy as stripy

    mesh = stripy.sTriangulation(lons_src, lats_src)

    lats, lons = spherical_coords[:, 1] - np.pi / 2, spherical_coords[:, 2]

    interp_val, _ = mesh.interpolate(
        lons, lats, zdata=surface_val[surface_mask], order=order
    )

    return interp_val


def fill_in_missing_voxels(
    interp_vol: np.array,
    mask_vol: np.array,
    chunk_start: float,
    chunk_end: float,
    start: float,
    step: float,
) -> np.ndarray:
    """Fill in missing voxels in a volume.

    :param interp_vol: volume with interpolated values
    :param mask_vol: volume with mask
    :param chunk_start: start of chunk
    :param chunk_end: end of chunk
    :param start: start of volume
    :param step: step size of volume
    :return: volume with filled in missing voxels
    """
    mask_vol = np.rint(mask_vol)
    mask_vol = np.pad(mask_vol, ((1, 1), (1, 1), (1, 1)))
    interp_vol = np.pad(interp_vol, ((1, 1), (1, 1), (1, 1)))

    orig_max, _ = np.max(interp_vol), np.min(interp_vol)

    xv, yv, zv = np.meshgrid(
        np.arange(mask_vol.shape[0]),
        np.arange(mask_vol.shape[1]),
        np.arange(mask_vol.shape[2]),
    )
    xv = xv.reshape(-1, 1)
    yv = yv.reshape(-1, 1)
    zv = zv.reshape(-1, 1)

    yw = yv * step + start

    voxels_within_chunk = (yw >= chunk_start) & (yw <= chunk_end)

    missing_voxels = (
        (mask_vol[xv, yv, zv] > 0.5 * mask_vol.max())
        & (interp_vol[xv, yv, zv] == 0)
        & voxels_within_chunk
    )

    counter = 0
    last_missing_voxels = np.sum(missing_voxels) + 1

    xvv = xv[missing_voxels]
    yvv = yv[missing_voxels]
    zvv = zv[missing_voxels]

    while np.sum(missing_voxels) > 0 and np.sum(missing_voxels) < last_missing_voxels:
        last_missing_voxels = np.sum(missing_voxels)
        xvv = xv[missing_voxels]
        yvv = yv[missing_voxels]
        zvv = zv[missing_voxels]

        xvp = xvv + 1
        xvm = xvv - 1
        yvp = yvv + 1
        yvm = yvv - 1
        zvp = zvv + 1
        zvm = zvv - 1

        x0 = np.vstack([xvp, yvv, zvv]).T
        x1 = np.vstack([xvm, yvv, zvv]).T
        y0 = np.vstack([xvv, yvp, zvv]).T
        y1 = np.vstack([xvv, yvm, zvv]).T
        z0 = np.vstack([xvv, yvv, zvm]).T
        z1 = np.vstack([xvv, yvv, zvp]).T

        interp_values = np.vstack(
            [
                interp_vol[x0[:, 0], x0[:, 1], x0[:, 2]],
                interp_vol[x1[:, 0], x1[:, 1], x1[:, 2]],
                interp_vol[y0[:, 0], y0[:, 1], y0[:, 2]],
                interp_vol[y1[:, 0], y1[:, 1], y1[:, 2]],
                interp_vol[z0[:, 0], z0[:, 1], z0[:, 2]],
                interp_vol[z1[:, 0], z1[:, 1], z1[:, 2]],
            ]
        ).T

        n = np.sum(np.abs(interp_values) > 0, axis=1)

        interp_sum = np.sum(interp_values, axis=1)

        xvv = xvv[n > 0]
        yvv = yvv[n > 0]
        zvv = zvv[n > 0]

        interp_values = interp_sum[n > 0] / n[n > 0]

        assert (
            np.abs(np.max(interp_vol) - orig_max) < 0.001
        ), f"Error: max value changed {np.max(interp_vol)} {orig_max}"
        # assert np.min(interp_vol) == orig_min, "Error: min value changed"

        interp_vol[xvv, yvv, zvv] = interp_values

        missing_voxels = (
            (mask_vol[xv, yv, zv] > 0.5)
            & (interp_vol[xv, yv, zv] == 0)
            & voxels_within_chunk
        )

        # print("\t\t\t\tmissing", np.sum(missing_voxels), np.sum(interp_vol == 10000))
        counter += 1

    interp_vol = interp_vol[1:-1, 1:-1, 1:-1]
    return interp_vol


def combine_volumes(
    volume_fns: list,
    output_filename: str,
    priority: str = "equal",
    mask_fn: str = None,
    clobber: bool = False,
) -> None:
    """Combine volumes.

    :param volume_fns: list of volume filenames
    :param priority: priority of volumes
    :return: None
    """
    if not os.path.exists(output_filename) or clobber:
        img = nib.load(volume_fns[0])
        aff = img.affine

        print("\tCombining volumes")
        vol = img.get_fdata()

        n = np.zeros(vol.shape)
        n[vol > 0] = 1

        idx_replace_orig = vol <= 0

        for vol_fn in volume_fns[1:]:
            curr_vol = nib.load(vol_fn).get_fdata()
            orig_vol = curr_vol.copy()
            print("\t\tVolume files:", volume_fns)
            print("" "\t\tReplacing volume", vol_fn)

            if priority == "equal":
                idx_replace = curr_vol > 0
                vol[idx_replace] += curr_vol[idx_replace]
                n[idx_replace] += 1
            elif priority == "first":
                # fill in missing values in first volume with values from second volume
                idx_replace = idx_replace_orig & (curr_vol > 0)
                # Use max value
                ar = np.vstack([curr_vol[idx_replace], vol[idx_replace]]).T
                print(ar.shape)
                vol[idx_replace] = np.max(ar, axis=1)
            else:
                print("Error: priority not recognized")
                exit(1)

        vol[n > 0] /= n[n > 0]
        print("n = ", np.unique(n))

        if mask_fn is not None:
            mask = nib.load(mask_fn).get_fdata()
            vol[mask < mask.max() * 0.5] = 0

        print("\tWriting", output_filename)

        nib.Nifti1Image(vol, aff, direction_order="lpi").to_filename(output_filename)
    else:
        vol = nib.load(output_filename).get_fdata()

    return vol


def create_reconstructed_cortex(
    reconstructed_cortex_fn: str,
    surf_volume_fn: str,
    profiles_fn: str,
    surf_depth_dict: dict,
    cortex_mask_fn: str,
    resolution: float,
    clobber: bool = False,
):
    """Create reconstructed cortex from interpolated profiles and cortical surfaces."""
    if not os.path.exists(surf_volume_fn) or clobber:
        mask_vol = nib.load(cortex_mask_fn).get_fdata()

        depth_list = sorted(surf_depth_dict.keys())

        profiles = np.load(profiles_fn + ".npz")["data"]

        affine = nb_surf.load(cortex_mask_fn).affine
        starts = np.array(affine[[0, 1, 2], 3])
        steps = np.array(affine[[0, 1, 2], [0, 1, 2]])
        dimensions = mask_vol.shape

        surface_list = [surf_depth_dict[depth]["depth_rsl_fn"] for depth in depth_list]

        unfilled_volume_fn = re.sub(
            ".nii.gz", "_unfilled.nii.gz", reconstructed_cortex_fn
        )
        out_vol = write_mesh_to_volume(
            profiles,
            surface_list,
            cortex_mask_fn,
            unfilled_volume_fn,
            resolution,
            clobber=clobber,
        )

        # print("\t\tFilling in missing voxels")
        # out_vol = fill_in_missing_voxels(
        #    out_vol,
        #    mask_vol,
        #    starts[1],
        #    dimensions[1] * steps[1] + starts[1],
        #    starts[1],
        #    steps[1],
        # )

        affine = nib.load(cortex_mask_fn).affine
        affine[[0, 1, 2], [0, 1, 2]] = resolution

        nib.Nifti1Image(out_vol, affine, direction_order="lpi").to_filename(
            surf_volume_fn
        )


def create_final_reconstructed_volume(
    reconstructed_cortex_fn: str,
    chunk_info: str,
    cortex_mask_fn: str,
    resolution: float,
    surf_depth_mni_dict: dict,
    profiles_fn: str,
    volume_type: str = "thickened_stx",
    clobber: bool = False,
) -> None:
    """Create final reconstructed volume from interpolated profiles and cortical surfaces.

    :param reconstructed_cortex_fn: path to reconstructed cortex
    :param cortex_mask_fn: path to cortex mask
    :param resolution: resolution of the reconstructed volume
    :param surf_depth_mni_dict: dictionary with surface information for each depth
    :param profiles_fn: path to profiles file
    :param clobber: boolean indicating whether to overwrite existing files
    :return: None
    """
    print("\t Creating final reconstructed volume")
    if not os.path.exists(reconstructed_cortex_fn) or clobber:
        print("\t\tMulti-mesh to volume")
        surf_volume_fn = re.sub(".nii.gz", "_surf.nii.gz", reconstructed_cortex_fn)
        thickened_vol_stx_fn = re.sub(
            ".nii.gz", "_interp-vol_stx.nii.gz", reconstructed_cortex_fn
        )

        create_reconstructed_cortex(
            reconstructed_cortex_fn,
            surf_volume_fn,
            profiles_fn,
            surf_depth_mni_dict,
            cortex_mask_fn,
            resolution,
            clobber=clobber,
        )
        print(volume_type)
        print(chunk_info[volume_type].values)

        combine_volumes(
            chunk_info[volume_type].values,
            thickened_vol_stx_fn,
            mask_fn=cortex_mask_fn,
            clobber=clobber,
        )

        print("\tWriting", reconstructed_cortex_fn)
        combine_volumes(
            [thickened_vol_stx_fn, surf_volume_fn],
            reconstructed_cortex_fn,
            mask_fn=cortex_mask_fn,
            priority="first",
            clobber=clobber,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument(
        "--clobber", dest="clobber", type=int, default=0, help="Clobber results"
    )
    parser.add_argument("--sub", dest="sub", type=str, help="sub")
    parser.add_argument("--hemisphere", dest="hemisphere", type=str, help="hemisphere")
    parser.add_argument("--out-dir", dest="out_dir", type=str, help="Clobber results")
    parser.add_argument(
        "--lin-df-fn", dest="lin_df_fn", type=str, help="Clobber results"
    )
    parser.add_argument(
        "--chunk-str", dest="chunk_str", type=str, help="Clobber results"
    )
    parser.add_argument(
        "--n-depths", dest="n_depths", type=int, default=8, help="Clobber results"
    )
    args = parser.parse_args()

    lin_df_fn = args.lin_df_fn
    out_dir = args.out_dir
    sub = args.sub
    hemisphere = args.hemisphere
    n_depths = args.n_depths
    chunk_str = args.chunk_str
    clobber = args.clobber
