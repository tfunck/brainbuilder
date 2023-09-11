import os
from glob import glob

import pandas as pd
import numpy as np
from skimage.transform import resize

import brainbuilder.utils as utils
import brainbuilder.utils.ants_nibabel as nib
from brainbuilder.utils.utils import shell


def volumetric_interpolation(
        sect_info:pd.DataFrame,
        chunk_info:pd.DataFrame,
        reference_volume_fn:str,
        output_dir,
    ):
    for (sub, hemisphere, chunk), curr_df in sect_info.groupby(["sub", "hemisphere", "chunk"]):
        
        vol_dir = f"{output_dir}/{sub}/{hemisphere}/{chunk}"
        os.makedirs(vol_dir, exist_ok=True)

        
    # mask_mni_fn=f'{output_dir}/{brain}_{hemi}_{resolution}mm_subcortex_mask.nii.gz'

    # shell(f'{wm_surf_mni_fn} {mask_fn}')
    chunk_list = list(chunk_info.keys())

    # 1. transform subcortex mask from mni space to chunk space
    for chunk in chunk_list:

        chunk_interp_fn= f'{output_dir}/{brain}_{hemi}_{chunk}_{resolution}mm_space-chunk_interp.nii.gz'

        if not os.path.exists(chunk_interp_fn) or clobber:
            img = nib.load(thickened_fn)
            vol = img.get_fdata()
            #vol = interpolate_missing_sections(vol)
            
            #nib.Nifti1Image(vol, img.affine,direction_order='lpi').to_filename(chunk_interp_fn)


        tfm_fn = chunk_info[chunk]["nl_3d_tfm_inv_fn"]
        ref_fn = chunk_info[chunk]["srv_iso_space_rec_fn"]
        atlas_chunk_fn = f"{vol_dir}/{brain}_{hemi}_{chunk}_{resolution}mm_space-chunk_subcortex_mask.nii.gz"
        shell(
            f"antsApplyTransforms -n NearestNeighbor -i {subcortex_mask_fn} -r {ref_fn} -t {tfm_fn} -o {atlas_chunk_fn}"
        )

        # 2. use non-linear 2d alignment between sections
        # from rat.process import align_nl
        # align_nl(df,nl_dir, mask_fn=mask_chunk_fn)
        # exit(0)

        # 3. interpolate between chunks
        mask_img = nib.load(atlas_chunk_fn)
        mask_vol = mask_img.get_fdata()

        try:
            thickened_string = f"{vol_dir}/thickened_{chunk}_{acquisition}_{resolution}*[0-9]_l{ndepths}.nii.gz"
            print(thickened_string)
            thickened_fn = glob(thickened_string)[0]
        except IndexError:
            print("Skipping", acquisition)
            continue



        if np.sum(mask_vol.shape) != np.sum(vol.shape):
            mask_vol = resize(mask_vol, vol.shape, order=0)

        vol[mask_vol == 0] = 0

        vol_interp_fn = f"{vol_dir}/{brain}_{hemi}_{chunk}_{acquisition}_{resolution}mm_space-chunk_vol_interp.nii.gz"
        nib.Nifti1Image(vol, img.affine).to_filename(vol_interp_fn)
        print(vol_interp_fn)


def volinterp(hemi_df, args, files, highest_resolution, chunk_files_dict, interp_dir, sub, hemisphere, scale_factors, norm_df_csv=None):
    if not args.no_surf:
        print("\tSurface-based reconstruction")
        chunkData, final_acquisition_dict = surface_based_reconstruction(
            hemi_df,
            args,
            files,
            highest_resolution,
            chunk_files_dict,
            interp_dir,
            sub,
            hemisphere,
            scale_factors,
            norm_df_csv=args.norm_df_csv,
        )
