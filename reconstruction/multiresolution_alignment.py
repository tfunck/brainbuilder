import os
import json
import shutil
import re
import argparse
import pandas as pd
import numpy as np
import nibabel
import utils.ants_nibabel as ants_nibabel
from glob import glob
from scipy.ndimage import label
from scipy.ndimage import binary_dilation, binary_closing, binary_fill_holes
from scipy.ndimage.filters import gaussian_filter
from utils.utils import shell, create_2d_sections, run_stage, prefilter_and_downsample, resample_to_autoradiograph_sections, kill_python_threads
from utils.ANTs import ANTs
from nibabel.processing import smooth_image
from utils.mesh_io import load_mesh_geometry, save_mesh_data, save_obj, read_obj
from reconstruction.align_slab_to_mri import align_slab_to_mri
from reconstruction.receptor_segment import classifyReceptorSlices, resample_transform_segmented_images
from reconstruction.nonlinear_2d_alignment import receptor_2d_alignment, concatenate_sections_to_volume
from reconstruction.init_alignment import receptorRegister, apply_transforms_to_landmarks
from reconstruction.crop import crop, process_landmark_images
from validation.validate_alignment import validate_alignment

def multiresolution_alignment( slab_df,  hemi_df, brain, hemi, slab, slab_index, args, files, resolution_list, resolution_list_3d, init_align_fn, max_resolution=0.3):
    '''
    About:
        Mutliresolution scheme that a. segments autoradiographs, b. aligns these to the donor mri in 3D,
        and c. aligns autoradiograph sections to donor MRI in 2d. This schema is repeated for each 
        resolution in the resolution heiarchy.  

    Inputs:
        slab_df:    data frame with info for autoradiographs in current slab
        hemi_df     data frame with info for autoradiographs in current hemisphere
        brain:       name of current subject id
        hemi:        current hemisphere
        slab:        current slab
        args:        user input arguments
        files:       dictionary with all the filenames used in reconstruction
        resolution_list:    heirarchy of resolutions  
        init_align_fn:      filename for initial alignement of autoradiographs
        max_resolution:     maximum 3d resolution that can be used

    Outputs:
        slab_df: updated slab_df data frame with filenames for nonlinearly 2d aligned autoradiographs

    '''
    slab_list = [int(i) for i in  files[brain][hemi].keys() ]
    slab_files = files[brain][hemi][str(slab)]
    

    ### Iterate over progressively finer resolution
    for resolution_itr, resolution in enumerate(resolution_list) :
        print('\tMulti-Resolution Alignement:',resolution)
        cfiles = files[brain][hemi][str(slab)][str(resolution)] #Current files
        
        slab_info_fn = files[brain][hemi][str(slab)][str(resolution)]['slab_info_fn']
        
        cur_out_dir = cfiles['cur_out_dir']
        seg_dir = cfiles['seg_dir'] 
        srv_dir = cfiles['srv_dir'] 
        align_to_mri_dir = cfiles['align_to_mri_dir'] 
        nl_2d_dir = cfiles['nl_2d_dir']

        for dir_name in [cur_out_dir, align_to_mri_dir , seg_dir, nl_2d_dir, srv_dir ] :
            os.makedirs(dir_name, exist_ok=True)

        srv_rsl_fn = cfiles['srv_rsl_fn']  
        srv_crop_rsl_fn = cfiles['srv_crop_rsl_fn']  

        seg_rsl_fn = cfiles['seg_rsl_fn']
        nl_3d_tfm_fn = cfiles['nl_3d_tfm_fn']
        nl_3d_tfm_inv_fn = cfiles['nl_3d_tfm_inv_fn']
        rec_3d_rsl_fn = cfiles['rec_3d_rsl_fn']
        srv_3d_rsl_fn = cfiles['srv_3d_rsl_fn']
        srv_space_rec_fn = cfiles['srv_space_rec_fn']
        srv_iso_space_rec_fn = cfiles['srv_iso_space_rec_fn']
        nl_2d_vol_fn = cfiles['nl_2d_vol_fn']
        nl_2d_cls_fn = cfiles['nl_2d_vol_cls_fn']

        resolution_3d = max(float(resolution), max_resolution)
        if resolution_3d == max_resolution :
            resolution_itr_3d = [ i for i, res in enumerate(resolution_list) if float(res) >= max_resolution ][-1]
        else :
            resolution_itr_3d = resolution_itr

        prev_resolution=resolution_list[resolution_itr-1]

        last_nl_2d_vol_fn = files[brain][hemi][str(slab)][str(resolution_list[resolution_itr-1])]['nl_2d_vol_fn']
        #DEBUG
        #if resolution_itr > 0 : 
        #    align_fn = last_nl_2d_vol_fn
        #else : 
        #    align_fn = init_align_fn

        ###
        ### Stage 1.25 : Downsample SRV to current resolution
        ###
        print('\t\tStage 1.25' )
        crop_srv_rsl_fn = files[brain][hemi][str(int(slab))][str(resolution)]['srv_crop_rsl_fn']
        if run_stage([args.srv_fn], [srv_rsl_fn, crop_srv_rsl_fn]) or args.clobber :
            # downsample the original srv gm mask to current 3d resolution
            prefilter_and_downsample(args.srv_fn, [resolution_3d]*3, srv_rsl_fn)
            
            # if this is the fiest slab, then the cropped version of the gm mask
            # is the same as the downsampled version because there are no prior slabs to remove.
            # for subsequent slabs, the cropped srv file is created at stage 3.5
            if int(slab) == int(slab_list[0]) : shutil.copy(srv_rsl_fn, crop_srv_rsl_fn)

        ### Stage 1.5 : combine aligned sections from previous iterations into a volume
        #Combine 2d sections from previous resolution level into a single volume
        if resolution != resolution_list[0] and not os.path.exists(last_nl_2d_vol_fn)  :
            last_nl_2d_dir = files[brain][hemi][slab][prev_resolution]['nl_2d_dir']
            slab_df = concatenate_sections_to_volume(slab_df, init_align_fn, last_nl_2d_dir, last_nl_2d_vol_fn)

        ###
        ### Stage 2 : Autoradiograph segmentation
        ###
        print('\t\tStep 2: Autoradiograph segmentation')
        stage_2_outputs=[seg_rsl_fn]
        if not os.path.exists(seg_rsl_fn) or args.clobber  :
            print('\t\t\tResampling segemented sections')
            resample_transform_segmented_images(slab_df, resolution_itr, resolution, resolution_3d, seg_dir+'/2d/' )
            #write 2d segmented sections at current resolution. apply initial transform
            print('\t\t\tInterpolating between segemented sections')
            classifyReceptorSlices(slab_df, init_align_fn, seg_dir+'/2d/', seg_dir, seg_rsl_fn, resolution_3d, resolution )

        ###
        ### Stage 3 : Align slabs to MRI
        ###
        print('\t\tStep 3: align slabs to MRI')
        stage_3_outputs = [nl_3d_tfm_fn, nl_3d_tfm_inv_fn, rec_3d_rsl_fn, srv_3d_rsl_fn]

        if run_stage(stage_2_outputs, stage_3_outputs) or args.clobber  :
            scale_factors_json = json.load(open(args.scale_factors_fn,'r'))
            slab_direction = scale_factors_json[brain][hemi][slab]['direction']
            align_slab_to_mri(  brain, hemi, slab, seg_rsl_fn, crop_srv_rsl_fn, align_to_mri_dir, 
                                hemi_df, args.slabs, nl_3d_tfm_fn, nl_3d_tfm_inv_fn, rec_3d_rsl_fn, srv_3d_rsl_fn, 
                                resolution_3d, resolution_itr_3d, 
                                resolution_list, slab_direction, cfiles['manual_alignment_points'], cfiles['manual_alignment_affine'], use_masks=False )
        
        ###
        ### Stage 4 : 2D alignment of receptor to resample MRI GM vol
        ###
        if run_stage([srv_rsl_fn], [srv_iso_space_rec_fn, srv_space_rec_fn]) or args.clobber :
            resample_to_autoradiograph_sections(brain, hemi, slab, float(resolution), srv_rsl_fn, seg_rsl_fn, nl_3d_tfm_inv_fn, srv_iso_space_rec_fn, srv_space_rec_fn)
        
        create_2d_sections( slab_df, srv_space_rec_fn, float(resolution), nl_2d_dir, dtype=np.uint8 )
            
        print('\t\tStep 4: 2d nl alignment')
        stage_4_outputs=[nl_2d_vol_fn, nl_2d_cls_fn]
        #if run_stage(stage_3_outputs, stage_4_outputs)  or args.clobber:
        slab_df = receptor_2d_alignment( slab_df, init_align_fn, srv_space_rec_fn,seg_dir+'/2d/', nl_2d_dir,  resolution, resolution_itr, resolution_list)
        kill_python_threads()
        
        # Concatenate 2D nonlinear aligned sections into output volume
        slab_df = concatenate_sections_to_volume( slab_df, srv_space_rec_fn, nl_2d_dir, nl_2d_vol_fn)

        # Concatenate 2D nonlinear aligned cls sections into an output volume
        slab_df = concatenate_sections_to_volume( slab_df, srv_space_rec_fn, nl_2d_dir, nl_2d_cls_fn, target_str='cls_rsl')
        kill_python_threads()

        slab_df.to_csv(slab_info_fn)
        print('\t\tWriting', slab_info_fn)
    return slab_df

