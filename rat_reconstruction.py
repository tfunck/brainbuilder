import os
import numpy as np
import imageio
import matplotlib.pyplot as plt
import shutil
import multiprocessing
import re
import pandas as pd
import nibabel 
import utils.ants_nibabel as nib
from nibabel.processing import resample_to_output
from reconstruction.surface_interpolation import surface_interpolation
from utils.utils import prefilter_and_downsample, resample_to_autoradiograph_sections
from reconstruction.align_slab_to_mri import *
from utils.utils import get_section_intervals
from reconstruction.nonlinear_2d_alignment import create_2d_sections, receptor_2d_alignment, concatenate_sections_to_volume
from reconstruction.crop import crop
from reconstruction.receptor_segment import classifyReceptorSlices, interpolate_missing_sections, resample_transform_segmented_images
from skimage.transform import resize
from scipy.ndimage.filters import gaussian_filter
from utils.ANTs import ANTs
from joblib import Parallel, delayed
from scipy.ndimage import binary_dilation
from glob import glob
from preprocessing.preprocessing import fill_regions
from utils.utils import safe_imread, points2tfm 
from skimage.filters import threshold_otsu, threshold_li , threshold_sauvola, threshold_yen
from scipy.ndimage import label, center_of_mass


def reconstruct(subject_id, auto_dir, template_fn, scale_factors_json_fn, out_dir, csv_fn, native_pixel_size=0.02163,brain = '11530', hemi='B', pytorch_model='', ligands_to_exclude=[], resolution_list=[4,3,2,1], lowres=0.4, flip_dict={} ):
    mask_dir = f'{auto_dir}/mask_dir/'
    subject_dir=f'{out_dir}/{subject_id}/' 
    crop_dir = f'{out_dir}/{subject_id}/crop/'
    init_dir = f'{out_dir}/{subject_id}/init_align/'
    downsample_dir = f'{out_dir}/{subject_id}/downsample'
    init_3d_dir = f'{out_dir}/{subject_id}/init_align_3d/'
    ligand_dir = f'{subject_dir}/ligand/'

    srv_max_resolution_fn=template_fn #might need to be upsampled to maximum resolution
    surf_dir = f'{auto_dir}/surfaces/'
    interp_dir = f'{subject_dir}/interp/'
    
    for dirname in [subject_dir, crop_dir, init_dir, downsample_dir, init_3d_dir, ligand_dir] : os.makedirs(dirname, exist_ok=True)

    
    affine=np.array([[lowres,0,0,-90],[0,0.02,0,0.0],[0,0,lowres,-70],[0,0,0,1]])
    
    df = pd.read_csv(csv_fn) 

    # Output files
    affine_fn = f'{init_3d_dir}/{subject_id}_affine.mat'
    volume_fn = f'{out_dir}/{subject_id}/{subject_id}_volume.nii.gz'
    volume_init_fn = f'{out_dir}/{subject_id}/{subject_id}_init-align_volume.nii.gz'
    volume_seg_fn = f'{subject_dir}/{subject_id}_segment_volume.nii.gz'
    volume_seg_iso_fn = f'{subject_dir}/{subject_id}_segment_iso_volume.nii.gz'

    scale_factors_json = json.load(open(scale_factors_json_fn,'r'))[brain][hemi]

    ### 1. Section Ordering
    print('1. Section Ordering')
    if ligands_to_exclude != [] :
        df = df.loc[ df['ligand'].apply(lambda x : not x in ligands_to_exclude  ) ] 
 
    section_scale_factor = get_section_scale_factor(out_dir, template_fn, df['raw'].values)

    pixel_size = section_scale_factor * native_pixel_size  
    
    ### 2. Crop
    print('2. Cropping')
    #crop_to_do = [ (y, raw_fn, crop_fn, ligand) for y, raw_fn, crop_fn, ligand in zip(df['repeat'], df['raw'], df['crop'],df['ligand']) if not os.path.exists(crop_fn) ]
    num_cores = min(1, multiprocessing.cpu_count() )
    
   
    # Crop non-cellbody stains
    crop(crop_dir, mask_dir, df.loc[df['ligand'] != 'cellbody'], scale_factors_json_fn, resolution=[60,45], remote=False, pad=0, clobber=False, create_pseudo_cls=False, brain_str='brain', crop_str='crop', lin_str='raw', flip_axes_dict=flip_dict, pytorch_model=pytorch_model )
    
    # Crop cellbody stains
    crop(crop_dir, mask_dir, df.loc[ df['ligand'] == 'cellbody' ], scale_factors_json_fn, resolution=[91,91], remote=False, pad=0, clobber=False,create_pseudo_cls=False, brain_str='brain', crop_str='crop', lin_str='raw', flip_axes_dict=flip_dict, pytorch_model=pytorch_model)
    df['crop_raw_fn'] = df['crop']

    df = df.loc[ df['crop'].apply(lambda fn : os.path.exists(fn)) ]
    assert df.shape[0] > 0 , 'Error: empty data frame'

    ### 3. Resample images
    df = downsample(df, downsample_dir, lowres)

    ### 4. Align
    print('3. Init Alignment')
    concat_section_to_volume(df, affine, volume_fn, file_var='crop' )

    
    #TODO: ligand_contrast_order = get_ligand_contrast_order()
    ligand_contrast_order=['oxot'] #DEBUG just for rat!
    
    aligned_df = align(df, init_dir, ligand_contrast_order )
    concat_section_to_volume(aligned_df, affine, volume_init_fn, file_var='init' )

    

