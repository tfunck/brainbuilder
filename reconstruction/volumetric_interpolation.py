from reconstruction.receptor_segment import interpolate_missing_sections
from utils.utils import shell
from glob import glob
from skimage.transform import resize
import numpy as np
import utils.ants_nibabel as nib
import os


def volumetric_interpolation(brain, hemi, resolution, slab_dict, ligand_list, subcortex_mask_fn, vol_dir, ndepths):
    os.makedirs(vol_dir, exist_ok=True)
    
    #mask_mni_fn=f'{output_dir}/{brain}_{hemi}_{resolution}mm_subcortex_mask.nii.gz'

    #shell(f'{wm_surf_mni_fn} {mask_fn}')
    slab_list = list(slab_dict.keys())

    #1. transform subcortex mask from mni space to slab space
    for slab in slab_list :
        tfm_fn = slab_dict[slab]['nl_3d_tfm_inv_fn'] 
        ref_fn = slab_dict[slab]['srv_iso_space_rec_fn']
        atlas_slab_fn=f'{vol_dir}/{brain}_{hemi}_{slab}_{resolution}mm_space-slab_subcortex_mask.nii.gz'
        shell(f'antsApplyTransforms -n NearestNeighbor -i {subcortex_mask_fn} -r {ref_fn} -t {tfm_fn} -o {atlas_slab_fn}')

        # 2. use non-linear 2d alignment between sections
        #from rat.process import align_nl
        #align_nl(df,nl_dir, mask_fn=mask_slab_fn)
        #exit(0)

        # 3. interpolate between slabs
        mask_img = nib.load(atlas_slab_fn)
        mask_vol = mask_img.get_fdata()

        for ligand in ligand_list :
            try: 
                thickened_string = f'{vol_dir}/thickened_{slab}_{ligand}_{resolution}*[0-9]_l{ndepths}.nii.gz'
                print(thickened_string)
                thickened_fn=glob(thickened_string)[0]
            except IndexError:
                print('Skipping', ligand)
                continue

            img = nib.load(thickened_fn)
            vol = img.get_fdata()
            vol = interpolate_missing_sections(vol)

            if np.sum(mask_vol.shape) != np.sum(vol.shape):
                mask_vol = resize(mask_vol, vol.shape, order=0)

            vol[mask_vol == 0 ] = 0

            
            

            vol_interp_fn=f'{vol_dir}/{brain}_{hemi}_{slab}_{ligand}_{resolution}mm_space-slab_vol_interp.nii.gz'
            nib.Nifti1Image(vol, img.affine).to_filename(vol_interp_fn)
            print(vol_interp_fn)



