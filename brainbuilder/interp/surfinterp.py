import argparse
import os
import re
from re import sub

import debug
import h5py as h5
import matplotlib
import matplotlib.pyplot as plt
import nibabel as nb_surf
import numpy as np
import pandas as pd
import stripy as stripy
import vast.surface_tools as surface_tools
from joblib import Parallel, delayed
from scipy.interpolate import interp1d
from skimage.transform import resize


import brainbuilder.utils.ants_nibabel as nib
from brainbuilder.utils.mesh_utils import load_mesh_ext, multi_mesh_to_volume, volume_to_mesh
from brainbuilder.utils.utils import (get_section_intervals, get_thicken_width )


def get_valid_coords(coords, iw):
    lower = min(iw)
    upper = max(iw)

    idx_range = np.arange(coords.shape[0]).astype(np.uint64)
    idx_bool = (coords[:, 1] >= lower) & (coords[:, 1] < upper)

    # valid_coords_idx_0  = idx_range[idx_bool]

    valid_coords_idx = np.where(idx_bool)[0]

    valid_coords_idx = valid_coords_idx.reshape(valid_coords_idx.shape[0])

    # print(valid_coords_idx_0)
    # print(valid_coords_idx)
    # print()
    # print('Error', np.sum(valid_coords_idx_0 != valid_coords_idx))
    valid_coords = coords[valid_coords_idx, :]
    # print( )
    # print('True min', np.min(coords[(coords[:,1] >= lower) & (coords[:,1] < upper), 1]) )
    # print('Bad min', np.min(valid_coords[:,1]))

    return valid_coords, valid_coords_idx







def get_profiles(
    profiles_fn,
    surf_depth_chunk_dict,
    surf_depth_mni_dict,
    surf_raw_values_dict,
    clobber:bool=False,
)->str:
    '''
    Get raw surface values and interpoalte over surface to fill in missing pixel intensities
    :param profiles_fn: path to profiles file
    :param surf_depth_chunk_dict: dictionary with surface information for each chunk
    :param surf_depth_mni_dict: dictionary with surface information for each depth
    :param surf_raw_values_dict: dictionary with raw surface values for each depth
    :param clobber: boolean indicating whether to overwrite existing files
    :return: path to profiles file
    '''

    print("\tGetting profiles")
    # 3. Interpolate missing densities over surface
    if not os.path.exists(profiles_fn+'.npz') or clobber:
        depth_list = sorted(surf_depth_mni_dict.keys())
        example_depth_fn = surf_raw_values_dict[depth_list[0]]

        print(example_depth_fn)
        nrows = pd.read_csv(example_depth_fn, header=None ).shape[0]
        ncols = len(depth_list)
        
        profiles = np.zeros([nrows,ncols])
        
        # get sorted list of depths

        for depth, raw_values_fn in surf_raw_values_dict.items():
            depth_index = depth_list.index(depth)

            profiles_raw = pd.read_csv(raw_values_fn, header=None, index_col=None)
            
            sphere_rsl_fn = surf_depth_mni_dict[depth]["sphere_rsl_fn"]

            surface_val = profiles_raw.values.reshape(-1,)
            assert np.sum(surface_val) > 1, f'Assert: empty file {raw_values_fn}' 

            profile_vector = interpolate_over_surface(
                sphere_rsl_fn, surface_val, threshold=0.0001, order=0
            )
            profiles[:, depth_index] = profile_vector

            del profile_vector

        np.savez(profiles_fn, data=profiles)

    return profiles_fn


def project_values_over_section_intervals(
        coords:np.ndarray,
        all_values:np.ndarray,
        n:np.ndarray,
        vol:np.ndarray,
        starts:np.ndarray,
        steps:np.ndarray,
        dimensions:np.ndarray,
        clobber:bool=False
        ):

    intervals_voxel = get_section_intervals(vol)
    
    vol_min = np.min(vol)
    assert (len(intervals_voxel) > 0), "Error: the length of intervals_voxels == 0 "

    for y0, y1 in intervals_voxel:
        # convert from voxel values to real world coordinates
        y0w = y0 * steps[1] + starts[1]
        y1w = y1 * steps[1] + starts[1]

        # the voxel values should be the same along the y-axis within an interval
        valid_coords_world, valid_coords_idx = get_valid_coords(coords, [y0w, y1w])
        ymin = np.min(coords[:,1])
        ymax = np.max(coords[:,1])
        print('\t\t\t', np.round([y0w, y1w, ymin, ymax],3) )
        if valid_coords_world.shape[0] != 0:
            print('\t\t\t', y0, y1)

            values, valid_idx =  volume_to_mesh(coords, vol, starts, steps, dimensions)
            all_values[valid_idx] = values

            n[valid_idx] += 1

            # Also remove vertices that are equal to background from the receptor volume
            all_values[ all_values == vol_min ] = 0


            if np.sum(np.abs(values)) > 0:
                f"Error: empty vol[x,z] in project_volume_to_surfaces for {y0}"

            assert (
                np.sum(np.isnan(values)) == 0
            ), f"Error: nan found in values from {vol_fn}"

    return all_values, n

    


def volume_to_surface_over_chunks(
        surf_chunk_dict:dict,
        volumes_df:dict,
        interp_csv:str,
        clobber=False,
    )->None:
    """

    """

    # Get surfaces transformed into chunk space
    if not os.path.exists(interp_csv) or clobber :
        surf_fn = list(surf_chunk_dict.values())[0]
        nvertices = load_mesh_ext(surf_fn)[0].shape[0]
        # the default value of the vertices is set to -100 to make it easy to distinguish
        # between vertices where a acquisition density of 0 is measured versus the default value
        all_values = np.zeros(nvertices) 
        n = np.zeros_like(all_values)
    
        # Iterate over chunks within a given
        for chunk, surf_fn in surf_chunk_dict.items():
            vol_fn = volumes_df['thickened'].loc[ volumes_df['chunk']  == chunk].values[0]

            coords, _ = load_mesh_ext(surf_fn)

            print('surf_fn', surf_fn)
            print('vol_fn', vol_fn)
            # read chunk volume
            img = nib.load(vol_fn)
            vol = img.get_fdata()
            rec_min = np.min(vol)

            affine = nb_surf.load(vol_fn).affine

            assert np.max(vol) != np.min(vol), f"Error: empty volume {vol_fn}; {np.max(vol) != np.min(vol)} "

            # set variable names for coordinate system
            starts = affine[[0,1,2], [3,3,3]]
            steps = affine[[0,1,2], [0,1,2]]

            dimensions = vol.shape
            print( starts )
            print( steps )
            print( np.array(starts)+np.array(steps)*dimensions )
            # get the intervals along the y-axis of the volume where
            # we have voxel intensities that should be interpolated onto the surfaces
            
            all_values, n = project_values_over_section_intervals(coords, all_values, n, vol, starts, steps, dimensions, clobber=clobber)
        
        all_values[n>0] = all_values[n>0] / n[n>0]

        print('All Values', np.sum(all_values));
        np.savetxt(interp_csv, all_values)
        
        print("\tWriting surface values to", interp_csv)
        assert (
            np.sum(np.abs(all_values)) > 0
        ), "Error, empty array all_values in project_volumes_to_surfaces"
    
    return None

def project_volume_to_surfaces(
        profiles_fn:str,
        surf_depth_chunk_dict:dict,
        surf_depth_mni_dict:dict,
        chunk_info_thickened:dict,
        output_prefix:str,
        tissue_type='',
        clobber=False
    )->str:
    '''
    Project the voxel values of a volume onto a mesh
    :param profiles_fn: path to profiles file
    :param surf_depth_chunk_dict: dictionary with surface information for each chunk
    :param output_prefix: prefix for output files
    :param tissue_type: tissue type of the interpolated volumes
    :param clobber: boolean indicating whether to overwrite existing files
    :return: path to profiles file 
    '''

    if not os.path.exists(profiles_fn+'.npz') or clobber :
        surf_raw_values_dict = {}
        # iterate over depth
        for depth, surf_depth_dict in surf_depth_chunk_dict.items():
            interp_csv = f'{output_prefix}{tissue_type}_{depth}_interp.csv'

            volume_to_surface_over_chunks(
                surf_depth_dict,
                chunk_info_thickened,
                interp_csv,
                clobber=clobber
            )
            surf_raw_values_dict[depth] = interp_csv
        profiles_fn = get_profiles( 
                            profiles_fn, 
                            surf_depth_chunk_dict, 
                            surf_depth_mni_dict,
                            surf_raw_values_dict,
                            clobber)
    return profiles_fn


def interpolate_over_surface(sphere_obj_fn, surface_val, threshold=0, order=1):
    print("\tInterpolating Over Surface")
    print("\t\t\tSphere fn:", sphere_obj_fn)
    # get coordinates from dicitonary with mesh info
    coords = load_mesh_ext(sphere_obj_fn)[0]
    assert (
        coords.shape[0] == surface_val.shape[0]
    ), f"Error: mismatch in shape of spherical mesh and surface values {coords.shape} and {surface_val.shape}"

    spherical_coords = surface_tools.spherical_np(coords)

    # define a mask of verticies where we have receptor densitiies
    surface_mask = surface_val > threshold * np.max(surface_val)

    # a=1636763
    # b=1636762
    # print(coords[surface_mask.astype(bool)][a])
    # print(coords[surface_mask.astype(bool)][b])
    assert np.sum(surface_mask) != 0, "Error, empty profiles {}".format(
        np.sum(surface_mask)
    )
    # define vector with receptor densities
    surface_val_src = surface_val[surface_mask.astype(bool)]

    # define vector without receptor densities
    surface_val_tgt = surface_val[~surface_mask.astype(bool)]

    # get coordinates for vertices in mask
    spherical_coords_src = spherical_coords[surface_mask.astype(bool), :]

    # get spherical coordinates from cortical mesh vertex coordinates

    lats_src, lons_src = (
        spherical_coords_src[:, 1] - np.pi / 2,
        spherical_coords_src[:, 2],
    )
    """
    print(lats_src.shape, np.unique(lats_src).shape[0]) 
    a=13971-1
    b=13974-1
    print(coords[a])
    print(coords[b])
    print(spherical_coords_src[a])
    print(spherical_coords_src[b])
    print(lats_src[a], lons_src[a])
    print(lats_src[b], lons_src[b])
    """
    temp = np.concatenate(
        [
            (spherical_coords_src[:, 1] - np.pi / 2).reshape(-1, 1),
            spherical_coords_src[:, 2].reshape(-1, 1),
        ],
        axis=1,
    )

    # create mesh data structure

    temp = np.concatenate([lons_src.reshape(-1, 1), lats_src.reshape(-1, 1)], axis=1)

    mesh = stripy.sTriangulation(lons_src, lats_src)
    lats, lons = spherical_coords[:, 1] - np.pi / 2, spherical_coords[:, 2]
    # print(lats[a], lats[b])
    # print(lons[a], lons[b])
    interp_val, interp_type = mesh.interpolate(
        lons, lats, zdata=surface_val[surface_mask], order=order
    )

    return interp_val


class Ext:
    def __init__(self, gm_sphere, wm_sphere, gm, wm):
        self.gm_sphere = gm_sphere
        self.wm_sphere = wm_sphere
        self.gm = gm
        self.wm = wm


def get_surface_filename(surf_obj_fn, surf_gii_fn, surf_fs_fn):
    if os.path.exists(surf_gii_fn):
        surf_fn = surf_gii_fn
        ext = Ext(".surf.gii", ".surf.gii", ".surf.gii", ".surf.gii")
    elif os.path.exists(surf_fs_fn):
        surf_fn = surf_fs_fn
        ext = Ext(".pial.sphere", ".white.sphere", ".pial", ".white")
    elif os.path.exists(surf_obj_fn):
        surf_fn = surf_gii_fn
        ext = Ext(".surf.gii", "surf.gii", ".surf.gii", ".surf.gii")
        obj_to_gii(surf_obj_fn, ref_surf_fn, surf_fn)
    else:
        print("Error: could not find input GM surface (obj, fs, gii) for ", surf_gii_fn)
        exit(1)
    return surf_fn, ext


class ImageParameters:
    def __init__(self, starts, steps, dimensions):
        self.starts = starts
        self.steps = steps
        self.dimensions = dimensions
        self.affine = np.array(
            [
                [self.steps[0], 0, 0, self.starts[0]],
                [0, self.steps[1], 0, self.starts[1]],
                [0, 0, self.steps[2], self.starts[2]],
                [0, 0, 0, 1],
            ]
        )
        # print('\tStarts', self.starts)
        # print('\tSteps', self.steps)
        # print('\tDimensions:', self.dimensions)
        # print('\tAffine', self.affine)


def get_image_parameters(fn):
    img = nib.load(fn)
    affine = nb_surf.load(fn).affine

    starts = np.array(affine[[0, 1, 2], 3])
    steps = affine[[0, 1, 2], [0, 1, 2]]

    dimensions = img.shape

    imageParam = ImageParameters(starts, steps, dimensions)

    return imageParam


def combine_interpolated_sections(chunk_fn, interp_vol, ystep_lo, ystart_lo):
    img = nib.load(chunk_fn)
    vol = img.get_fdata()
    vol_min = np.ceil(np.min(vol))

    affine = nb_surf.load(chunk_fn).affine
    ystep_hires = affine[1, 1]
    ystart_hires = affine[1, 3]

    print("\tCombining interpolated sections with original sections")
    interp_vol[np.min(interp_vol) == interp_vol] = np.min(vol)

    for y in range(vol.shape[1]):
        section_not_empty = np.max(vol[:, y, :]) > vol_min
        print(np.max(vol[:, y, :]), vol_min)
        # convert from y from thickened filename
        yw = y * ystep_hires + ystart_hires
        y_lo = np.rint((yw - ystart_lo) / ystep_lo).astype(int)
        if section_not_empty and y_lo < interp_vol.shape[1]:
            # use if using mapper interpolation interp_section = interp_vol[ : , yi, : ]
            original_section = vol[:, y, :]

            interp_vol[:, y_lo, :] = original_section
        else:
            pass

    return interp_vol


def create_reconstructed_volume(
    interp_fn_list,
    acquisitionSlabData,
    profiles_fn,
    surf_depth_mni_dict,
    files,
    df_acquisition,
    scale_factors_json,
    use_mapper=True,
    clobber=False,
    gm_label=2.0,
):
    ref_fn = surf_depth_mni_dict[0.0]["depth_surf_fn"]

    profiles = None

    interp_dir = acquisitionSlabData.volume_dir
    depth_list = acquisitionSlabData.depths
    resolution = acquisitionSlabData.resolution
    chunks = acquisitionSlabData.chunks

    for interp_fn, chunk in zip(interp_fn_list, chunks):
        print(chunk, type(chunk))
        sectioning_direction = scale_factors_json[str(chunk)]["direction"]
        df_acquisition_chunk = df_acquisition.loc[df_acquisition["chunk"].astype(int) == int(chunk)]

        if not os.path.exists(interp_fn) or clobber:
            print("\tReading profiles", profiles_fn)
            if type(profiles) != type(np.array):
                profiles = np.load(profiles_fn)["data"]
            # assert np.sum(profiles>0) > 0, 'Error: profiles h5 is empty: '+profiles_fn
            # Hiad dimensions for output volume
            try:
                files_resolution = files[str(int(chunk))]
            except KeyError:
                print(files.keys())
                files_resolution = files[int(chunk)]

            resolution_list = list(files_resolution.keys())
            max_resolution = resolution_list[-1]
            chunk_fn = files_resolution[max_resolution]["nl_2d_vol_fn"]
            srv_space_rec_fn = files_resolution[max_resolution]["srv_space_rec_fn"]
            srv_iso_space_rec_fn = files_resolution[max_resolution]["cortex_fn"]

            thickened_fn = acquisitionSlabData.volumes[chunk]
            print("Thickened Filename", thickened_fn)
            print("Mask fn", srv_iso_space_rec_fn)

            # out_affine= nib.load(chunk_fn).affine
            out_affine = nib.load(srv_iso_space_rec_fn).affine

            imageParamHi = get_image_parameters(chunk_fn)
            print(chunk_fn, imageParamHi.dimensions)
            imageParamLo = get_image_parameters(srv_iso_space_rec_fn)
            print(srv_iso_space_rec_fn, imageParamLo.dimensions)

            mask_img = nib.load(srv_iso_space_rec_fn)
            mask_vol = mask_img.get_fdata()

            # commented out because was used when masks had multiple labels

            # gm_lo = gm_label * 0.75
            # gm_hi = gm_label * 1.25
            # valid_idx = (mask_vol >= gm_lo) & (mask_vol < gm_hi)
            valid_idx = mask_vol >= 0.5 * np.max(mask_vol)
            assert np.sum(valid_idx) > 0
            mask_vol = np.zeros_like(mask_vol).astype(np.uint8)
            mask_vol[valid_idx] = 1

            # Use algorithm that averages vertex values into voxels
            interp_only_fn = re.sub(".nii", "_interp-only.nii", interp_fn)
            multi_mesh_interp_fn = re.sub(".nii", "_multimesh.nii", interp_fn)
            multi_mesh_filled_fn = re.sub(".nii", "_multimesh-filled.nii", interp_fn)
            gm_mask_fn = re.sub(".nii", "_gm-mask.nii", interp_fn)
            nib.Nifti1Image(mask_vol, out_affine).to_filename(gm_mask_fn)

            if (
                not os.path.exists(multi_mesh_interp_fn)
                or not os.path.exists(multi_mesh_filled_fn)
                or clobber
            ):
                y0 = (
                    df_acquisition["chunk_order"].min() * imageParamHi.steps[1]
                    + imageParamHi.starts[1]
                )
                y1 = (
                    df_acquisition["chunk_order"].max() * imageParamHi.steps[1]
                    + imageParamHi.starts[1]
                )
                chunk_start = min(y0, y1)
                chunk_end = max(y0, y1)

                if not os.path.exists(multi_mesh_interp_fn) or clobber:
                    interp_vol = multi_mesh_to_volume(
                        profiles,
                        acquisitionSlabData.surfaces[chunk],
                        depth_list,
                        imageParamLo.dimensions,
                        imageParamLo.starts,
                        imageParamLo.steps,
                        resolution,
                        y0,
                        y1,
                        ref_fn=ref_fn,
                    )
                    print("\t\tWriting", multi_mesh_interp_fn)
                    nib.Nifti1Image(interp_vol, out_affine).to_filename(
                        multi_mesh_interp_fn
                    )
                else:
                    interp_vol = nib.load(multi_mesh_interp_fn).get_fdata()

                interp_vol = fill_in_missing_voxels(
                    interp_vol,
                    mask_vol,
                    chunk_start,
                    chunk_end,
                    imageParamLo.starts[1],
                    imageParamLo.steps[1],
                )
                print("\t\tWriting", multi_mesh_filled_fn)
                nib.Nifti1Image(interp_vol, out_affine).to_filename(
                    multi_mesh_filled_fn
                )
            else:
                interp_vol = nib.load(multi_mesh_interp_fn).get_fdata()

            ystep_interp = imageParamHi.steps[1]
            ystart_interp = imageParamHi.starts[1]

            assert (
                np.sum(interp_vol) > 0
            ), f"Error: interpolated volume is empty using {profiles_fn}"

            df_acquisition_chunk = df_acquisition.loc[df_acquisition["chunk"].astype(int) == int(chunk)]
            print("Writing", interp_only_fn)
            nib.Nifti1Image(interp_vol, mask_img.affine).to_filename(interp_only_fn)

            output_vol = combine_interpolated_sections(
                thickened_fn, interp_vol, imageParamLo.steps[1], imageParamLo.starts[1]
            )

            print("Writing", interp_fn)
            nib.Nifti1Image(output_vol, mask_img.affine).to_filename(interp_fn)
            print("\nDone.")
        else:
            print(interp_fn, "already exists")


def fill_in_missing_voxels(interp_vol, mask_vol, chunk_start, chunk_end, start, step):
    print("\tFilling in missing voxels.")
    mask_vol = np.rint(mask_vol)
    mask_vol = np.pad(mask_vol, ((1, 1), (1, 1), (1, 1)))
    interp_vol = np.pad(interp_vol, ((1, 1), (1, 1), (1, 1)))

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
        (mask_vol[xv, yv, zv] > 0.5)
        & (interp_vol[xv, yv, zv] == 0)
        & voxels_within_chunk
    )
    print(start, step)
    print(np.sum(yv > (121 - start) / step))
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

        n = np.sum(interp_values > 0, axis=1)

        interp_sum = np.sum(interp_values, axis=1)

        xvv = xvv[n > 0]
        yvv = yvv[n > 0]
        zvv = zvv[n > 0]

        interp_values = interp_sum[n > 0] / n[n > 0]

        interp_vol[xvv, yvv, zvv] = interp_values

        missing_voxels = (
            (mask_vol[xv, yv, zv] > 0.5)
            & (interp_vol[xv, yv, zv] == 0)
            & voxels_within_chunk
        )
        print("missing", np.sum(missing_voxels), np.sum(interp_vol == 10000))
        counter += 1

    interp_vol = interp_vol[1:-1, 1:-1, 1:-1]
    return interp_vol


def interpolate_between_chunks(
    combined_chunk_vol,
    surf_depth_mni_dict,
    depth_list,
    profiles_fn,
    ref_fn,
    interp_dir,
    srv_rsl_fn,
    resolution,
):
    img = nb_surf.load(srv_rsl_fn)
    starts = np.array(img.affine[[0, 1, 2], 3])
    steps = np.array(img.affine[[0, 1, 2], [0, 1, 2]])
    dimensions = img.shape

    ref_fn = surf_depth_mni_dict[0.0]["depth_surf_fn"]

    wm_upsample_fn = surf_depth_mni_dict[depth_list[0]]["depth_rsl_gii"]
    gm_upsample_fn = surf_depth_mni_dict[depth_list[-1]]["depth_rsl_gii"]
    profiles = np.load(profiles_fn)["data"]
    mask_vol = np.rint(nib.load(srv_rsl_fn).get_fdata())

    y0 = starts[1]
    y1 = starts[1] + dimensions[1] * steps[1]

    chunk_start = min(y0, y1)
    chunk_end = max(y0, y1)

    depth_rsl_dict = dict(
        [(k, v["depth_rsl_fn"]) for k, v in surf_depth_mni_dict.items()]
    )

    interp_vol = multi_mesh_to_volume(
        profiles,
        depth_rsl_dict,
        depth_list,
        dimensions,
        starts,
        steps,
        resolution,
        chunk_start,
        chunk_end,
        ref_fn=ref_fn,
    )

    # interp_vol[ combined_chunk_vol != 0 ] = combined_chunk_vol[ combined_chunk_vol != 0 ]

    mask_vol = resize(mask_vol, interp_vol.shape, order=0)
    mask_vol[mask_vol < 0.5 * np.max(mask_vol)] = 0

    interp_vol = fill_in_missing_voxels(
        interp_vol, mask_vol, chunk_start, chunk_end, starts[1], steps[1]
    )
    return interp_vol, mask_vol


def combine_chunks_to_volume(interp_fn_mni_list, output_fn):
    ref_img = nib.load(interp_fn_mni_list[0])
    output_volume = np.zeros(ref_img.shape)

    if not os.path.exists(output_fn):
        n = np.zeros_like(output_volume)
        for interp_cortex_vol_fn in interp_fn_mni_list:
            vol = nib.load(interp_cortex_vol_fn).get_fdata()
            output_volume += vol
            n[vol > 0] += 1
        # FIXME taking the average can attenuate values in overlapping areas
        output_volume[n > 0] /= n[n > 0]
        output_volume[np.isnan(output_volume)] = 0
        nib.Nifti1Image(
            output_volume, ref_img.affine, direction_order="lpi"
        ).to_filename(output_fn)

    return output_fn

def create_final_reconstructed_volume(
    reconstructed_cortex_fn,
    cortex_mask_fn,
    resolution,
    surf_depth_mni_dict,
    profiles_fn,
    clobber:bool=False
    ):

    
    if not os.path.exists(reconstructed_cortex_fn) or clobber :
    
        mask_vol = nib.load(cortex_mask_fn).get_fdata()

        depth_list = sorted(surf_depth_mni_dict.keys())

        profiles = np.load(profiles_fn+'.npz')["data"]

        affine = nb_surf.load(cortex_mask_fn).affine
        starts = np.array(affine[[0, 1, 2], 3])
        steps = np.array(affine[[0, 1, 2], [0, 1, 2]])
        dimensions = mask_vol.shape

        out_vol = multi_mesh_to_volume(
            profiles,
            surf_depth_mni_dict,
            depth_list,
            dimensions,
            starts,
            steps,
            resolution)

        
        out_vol = fill_in_missing_voxels(
            out_vol,
            mask_vol,
            starts[1],
            dimensions[1]*steps[1]+starts[1],
            starts[1],
            steps[1],
        )

        affine[[0,1,2],[0,1,2]] = resolution

        nib.Nifti1Image(out_vol, affine, direction_order="lpi").to_filename(
            reconstructed_cortex_fn
        )


def surface_interpolation(
        sect_info:str,
        surf_depth_mni_dict:dict,
        surf_depth_chunk_dict:dict,
        chunk_info_thickened_csv:str,
        cortex_mask_fn:str,
        output_dir:str,
        resolution:float,
        depth_list:list[float],
        tissue_type = '',
        clobber:bool = False,
):
    sub = sect_info['sub'].values[0]
    hemisphere = sect_info['hemisphere'].values[0]
    acquisition = sect_info['acquisition'].values[0]


    chunk_info_thickened = pd.read_csv(chunk_info_thickened_csv, index_col=None)

    n_depths = len(depth_list)
    chunks = np.unique(sect_info['chunk'])

    output_prefix = f"{output_dir}/sub-{sub}_hemi-{hemisphere}_acq-{acquisition}_{resolution}mm{tissue_type}_l{n_depths}"

    profiles_fn = f"{output_prefix}_profiles"
    
    reconstructed_cortex_fn = f"{output_prefix}_cortex.nii.gz"
    
    os.makedirs(output_dir, exist_ok=True)

    # Project autoradiograph densities onto surfaces
    # Extract profiles from the chunks using the surfaces
    project_volume_to_surfaces(
        profiles_fn,
        surf_depth_chunk_dict,
        surf_depth_mni_dict,
        chunk_info_thickened,
        output_prefix,
    )

    # interpolate from surface to volume over entire brain
    print("\tCreate final reconstructed volume")
     
    create_final_reconstructed_volume(
        reconstructed_cortex_fn,
        cortex_mask_fn,
        resolution,
        surf_depth_mni_dict,
        profiles_fn,
        clobber=clobber
        )
   

    return reconstructed_cortex_fn


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
    parser.add_argument("--chunk-str", dest="chunk_str", type=str, help="Clobber results")
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
