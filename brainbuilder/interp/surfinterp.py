import argparse
import os
import re
from re import sub

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

from brainbuilder.interp.acqvolume import create_thickened_volumes

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
    Project a volume onto a surface
    :param surf_chunk_dict: dictionary with surface information for each chunk
    :param volumes_df: dataframe with volume information
    :param interp_csv: path to csv file containing interpolated values
    :param clobber: boolean indicating whether to overwrite existing files
    :return: None
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


def generate_surface_profiles(
        chunk_info: pd.DataFrame,
        sect_info: pd.DataFrame,
        surf_depth_chunk_dict:dict,
        surf_depth_mni_dict:dict,
        resolution: float,
        depth_list: list,
        struct_vol_rsl_fn: str,
        output_dir: str,
        tissue_type: str = "",
        clobber: bool = False) -> str:
    '''
    Create surface profiles over a set of brain chunks and over multiple cortical depths
    :param chunk_info: dataframe with chunk information
    :param sect_info: dataframe with section information
    :param resolution: resolution of the volume
    :param depth_list: list of cortical depths
    :param output_dir: output directory where results will be put
    :param clobber: if True, overwrite existing files
    :return: path to profiles file
    '''

    chunk_info_thickened_csv = create_thickened_volumes(
        output_dir, chunk_info, sect_info, resolution
    )

    sub = sect_info["sub"].values[0]
    hemisphere = sect_info["hemisphere"].values[0]
    acquisition = sect_info["acquisition"].values[0]

    n_depths = len(depth_list)
    chunks = np.unique(sect_info['chunk'])

    chunk_info_thickened = pd.read_csv(chunk_info_thickened_csv, index_col=None)


    output_prefix = f"{output_dir}/sub-{sub}_hemi-{hemisphere}_acq-{acquisition}_{resolution}mm{tissue_type}_l{n_depths}"

    profiles_fn = f"{output_prefix}_profiles"
    
    
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

    return profiles_fn, chunk_info_thickened_csv




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

    # iterate over depth
    for depth, surf_depth_dict in surf_depth_chunk_dict.items():

        interp_csv = f'{output_prefix}{tissue_type}_{depth}_interp.csv'

        if not os.path.exists(interp_csv) or clobber:
            volume_to_surface_over_chunks(
                surf_depth_dict,
                chunk_info_thickened,
                interp_csv,
                clobber=clobber
            )

            surf_raw_values_dict[depth] = interp_csv

    if not os.path.exists(profiles_fn+'.npz') or clobber :

        surf_raw_values_dict = {}

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
    
    interp_val, interp_type = mesh.interpolate(
        lons, lats, zdata=surface_val[surface_mask], order=order
    )

    return interp_val


def fill_in_missing_voxels(interp_vol, mask_vol, chunk_start, chunk_end, start, step):
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

        #print("\t\t\t\tmissing", np.sum(missing_voxels), np.sum(interp_vol == 10000))
        counter += 1

    interp_vol = interp_vol[1:-1, 1:-1, 1:-1]
    return interp_vol


def create_final_reconstructed_volume(
    reconstructed_cortex_fn,
    cortex_mask_fn,
    resolution,
    surf_depth_mni_dict,
    profiles_fn,
    clobber:bool=False
    ):

    print('\t Creating final reconstructed volume')
    if not os.path.exists(reconstructed_cortex_fn) or clobber :
    
        mask_vol = nib.load(cortex_mask_fn).get_fdata()

        depth_list = sorted(surf_depth_mni_dict.keys())

        profiles = np.load(profiles_fn+'.npz')["data"]

        affine = nb_surf.load(cortex_mask_fn).affine
        starts = np.array(affine[[0, 1, 2], 3])
        steps = np.array(affine[[0, 1, 2], [0, 1, 2]])
        dimensions = mask_vol.shape
        
        print('\t\tMulti-mesh to volume')
        out_vol = multi_mesh_to_volume(
            profiles,
            surf_depth_mni_dict,
            depth_list,
            dimensions,
            starts,
            steps,
            resolution)

        print('\t\tFilling in missing voxels') 
        out_vol = fill_in_missing_voxels(
            out_vol,
            mask_vol,
            starts[1],
            dimensions[1]*steps[1]+starts[1],
            starts[1],
            steps[1],
        )

        affine[[0,1,2],[0,1,2]] = resolution


        print('\tWriting', reconstructed_cortex_fn )

        nib.Nifti1Image(out_vol, affine, direction_order="lpi").to_filename(
            reconstructed_cortex_fn
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
