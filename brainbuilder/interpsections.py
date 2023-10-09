
import pandas as pd
import numpy as np
import os
import re

from brainbuilder.utils import utils
from brainbuilder.interp.surfinterp import generate_surface_profiles, create_final_reconstructed_volume
from brainbuilder.interp.acqvolume import create_thickened_volumes
from brainbuilder.interp.volinterp import volumetric_interpolation
from brainbuilder.interp.prepare_surfaces import prepare_surfaces
from brainbuilder.interp.batch_correction import apply_batch_correction 

def resample_struct_reference_volume(orig_struct_vol_fn, resolution, output_dir, clobber=False):
    '''
    resample the structural reference volume to the desired resolution

    :param orig_struct_vol_fn: path to the original structural reference volume 
    :param hemisphere: hemisphere
    :param resolution: resolution of the volume
    :param output_dir: path to output directory
    :param clobber: boolean indicating whether to overwrite existing files
    :return: path to the resampled structural reference volume
    '''

    base_name = re.sub('.nii', f'_{resolution}mm.nii', os.path.basename(orig_struct_vol_fn))
    struct_vol_rsl_fn = f'{output_dir}/{base_name}'
    
    if not os.path.exists(struct_vol_rsl_fn) or clobber :
        utils.resample_to_resolution(orig_struct_vol_fn, [ resolution ] * 3, struct_vol_rsl_fn )

    return struct_vol_rsl_fn


def surface_pipeline(
    chunk_info: pd.DataFrame,
    sect_info: pd.DataFrame,
    resolution: float,
    output_dir: str,
    ref_vol_fn: str,
    gm_surf_fn: str,
    wm_surf_fn: str,
    n_depths: int = 0,
    batch_correction: bool = False,
    clobber: bool = False,
) -> None:
    """
    Use surface-based interpolation to fill missing sections over the cortex. Volumetric interpolation
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
    assert len(np.unique(sect_info["sub"])) == 1, "Error: multiple subjects"
    assert len(np.unique(sect_info["hemisphere"])) == 1, "Error: multiple hemispheres"
    assert len(np.unique(sect_info["acquisition"])) == 1, "Error: multiple acquisitions"

    if n_depths == 0:
        n_depths = np.ceil(5 / resolution).astype(int)
    
    depth_list = np.linspace(0, 1, int(n_depths))

    surf_dir = f'{output_dir}/surfaces/'

    surf_depth_mni_dict, surf_depth_chunk_dict = prepare_surfaces(
        chunk_info,
        ref_vol_fn,
        gm_surf_fn,
        wm_surf_fn,
        depth_list,
        surf_dir,
        resolution,
        clobber=clobber
    )

    struct_vol_rsl_fn = resample_struct_reference_volume(ref_vol_fn,  resolution, output_dir, clobber=clobber)

    profiles_fn, chunk_info_thickened_csv = generate_surface_profiles(
            chunk_info, 
            sect_info, 
            surf_depth_chunk_dict,
            surf_depth_mni_dict,
            resolution, 
            depth_list,
            struct_vol_rsl_fn,
            output_dir, 
            clobber = clobber )

    if batch_correction :
        sect_info, chunk_info = apply_batch_correction(
                sect_info,
                profiles_fn,
                surf_depth_mni_dict,
                surf_depth_chunk_dict,
                chunk_info,
                depth_list,
                resolution,
                struct_vol_rsl_fn,
                output_dir, 
                clobber=clobber)
        
    # do surface based interpolation to fill missing sections
    sub = sect_info["sub"].values[0]
    hemisphere = sect_info["hemisphere"].values[0]
    acquisition = sect_info["acquisition"].values[0]

    reconstructed_cortex_fn = f"{output_dir}/sub-{sub}_hemi-{hemisphere}_acq-{acquisition}_{resolution}mm_l{n_depths}_cortex.nii.gz"
    
    create_final_reconstructed_volume(
        reconstructed_cortex_fn,
        struct_vol_rsl_fn,
        resolution,
        surf_depth_mni_dict,
        profiles_fn,
        clobber=clobber
        )

    # create a mask of the subcortex that can be used for volumetric interpolation
    #FIXME NOT YET IMPLEMENTED
    #subcortex_mask_fn = utils.create_subcortex_mask(wm_surf_fn)

    # perform volumetric interpolation to fill missing sections in the subcortex
    #FIXME NOT YET IMPLEMENTED
    #subcortex_interp_fn = volinterp.volumetric_interpolation(
    #    sect_info, brain_mask_fn, clobber=clobber
    #)

    # combine the interpolated cortex and subcortex
    #FIXME NOT YET IMPLEMENTED
    #combine_volumes(interp_cortex_fn, subcortex_mask_fn)

    return None


def volumetric_pipeline(
    chunk_info_csv: str,
    sect_info_csv: str,
    resolution: float,
    output_dir: str,
    clobber: bool = False,
) -> None:
    """
    Use volumetric interpolation to fill missing sections over the entire brain
    """

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
        batch_correction: bool = False,
        clobber: bool = False,
):
    """
    Interpolate missing sections in a volume.
    :param sect_info: a dataframe with the following columns:
    :param brain_mask_fn: path to the brain mask
    :param resolution: resolution of the volume
    :param clobber: if True, overwrite existing files
    :return: path to the interpolated volume
    """

    sect_info = pd.read_csv(sect_info_csv, index_col=False)
    chunk_info = pd.read_csv(chunk_info_csv, index_col=False)
    hemi_info = pd.read_csv(hemi_info_csv, index_col=False)

    for (sub, hemisphere,acquisition), curr_sect_info in sect_info.groupby(["sub", "hemisphere", "acquisition"]):
        curr_output_dir = f"{output_dir}/sub-{sub}/hemi-{hemisphere}/acq-{acquisition}/"

        os.makedirs(curr_output_dir, exist_ok=True)

        curr_chunk_info = chunk_info.loc[
            (chunk_info["sub"] == sub) & (chunk_info["hemisphere"] == hemisphere) & (chunk_info["resolution"] == resolution)
        ]

        curr_hemi_info = hemi_info.loc[
            (hemi_info["sub"] == sub) & (hemi_info["hemisphere"] == hemisphere) 
        ]
        gm_surf_fn = curr_hemi_info['gm_surf'].values[0]
        wm_surf_fn = curr_hemi_info['wm_surf'].values[0] 
        ref_vol_fn = curr_hemi_info['struct_ref_vol'].values[0]
        
        print(gm_surf_fn, os.path.exists(gm_surf_fn))
        print(wm_surf_fn, os.path.exists(wm_surf_fn) )
        if (os.path.exists(gm_surf_fn) and os.path.exists(wm_surf_fn)) or clobber:
            surface_pipeline(
                 curr_chunk_info, 
                 curr_sect_info, 
                 resolution, 
                 curr_output_dir, 
                 ref_vol_fn,
                 gm_surf_fn,
                 wm_surf_fn,
                 n_depths,
                 batch_correction=batch_correction,
                 clobber=clobber
            )
        else:
            print("Error: Volumetric interpolation pipeline not yet implemented")
            exit(1)
            # volumetric_pipeline(sect_info_csv, chunk_info_csv, resolution, output_dir, clobber=clobber)

    return None


if __name__ == "__main__":
    pass
