import os
import numpy as np
import imageio
import matplotlib.pyplot as plt
import shutil
import multiprocessing
import re
import pandas as pd
import h5py as h5
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
from utils.ANTs import ANTs
from joblib import Parallel, delayed
from scipy.ndimage import binary_dilation
from glob import glob
from preprocessing.preprocessing import fill_regions
from utils.utils import safe_imread, points2tfm 
from skimage.filters import threshold_otsu, threshold_li , threshold_sauvola, threshold_yen
from scipy.ndimage import label, center_of_mass
from macaque.process import align, get_section_scale_factor, downsample, concat_section_to_volume, multires_align_3d, align_2d


def load_warp(fn):
    F = h5.File(fn,'r')
    transform_parameters = F['TransformGroup']['1']['TransformParameters'][:]
    transform_fixed_parameters = F['TransformGroup']['1']['TransformFixedParameters'][:]
    return transform_parameters, transform_fixed_parameters

def write_warp(tfm, tfm_fx, ref_fn, out_fn):
    shutil.copy(ref_fn, out_fn)
    F = h5.File(out_fn,'a')
    F['TransformGroup']['1']['TransformParameters'][:] = tfm
    F['TransformGroup']['1']['TransformFixedParameters'][:] = tfm_fx

    F.close()

    

def average_warp(fn0, fn1, out_fn):
    tfm0, tfm_fx0 = load_warp(fn0)
    tfm1, tfm_fx1 = load_warp(fn1)
    tfm_avg = (tfm0 + tfm1) / 2.
    tfm_fx_avg = (tfm_fx0 + tfm_fx1) / 2.
    write_warp(tfm_avg, tfm_fx_avg, fn0, out_fn)




def align_nl(df,nl_dir):

    n=df.shape[0]
    file_to_align='init'
    # 0.1, 0.2, 0.4, 0.8 
    metrics=[ ['Mattes'], ['Mattes'], ['Mattes'], ['CC']] 
    tfm_type=['SyN']
    iterations=[ ['750'], ['500'],['250'],['150']]
    shrink_factors=[ ['4'], ['3'], ['2'], ['1']]
    smoothing_sigmas=[ [str(np.power(2,float(f[0])-1)/np.pi)] for f in shrink_factors  ]
    
    nl_tfm_files={}

    example_fn = df[file_to_align].iloc[0]
    xdim, zdim = np.rint(np.array(nib.load(example_fn).shape)).astype(int)
    order_min = df['order'].min()
    ydim=int(df['order'].max()  + 1)
    vol=np.zeros([xdim,ydim,zdim])

    df['lin']=df[file_to_align]
    for itr, factor, sigma, metric in zip(iterations, shrink_factors, smoothing_sigmas, metrics):
        img = nib.load(df[file_to_align].iloc[0])
        mult_factor = np.power(2,float(factor[0])-1)
        xres = img.affine[0,0] 
        zres = img.affine[1,1] 
        affine=np.array([[xres,0,0,-90],[0,0.02,0,0.0],[0,0,zres,-70],[0,0,0,1]])
       
        curr_itr_fn = f'{nl_dir}/nl_align_{factor[0]}.nii.gz'

        new_nl_files={}
        for i in range(1,n-1):
            i0=i-1
            i1=i+1

            prev_row = df.iloc[i0]
            curr_row = df.iloc[i]
            next_row = df.iloc[i1]

            outprefix0 = f'{nl_dir}/{factor[0]}/{prev_row["order"]}_{prev_row["ligand"]}/'
            outprefix1 = f'{nl_dir}/{factor[0]}/{curr_row["order"]}_{curr_row["ligand"]}/'
            out_tfm_fn = f'{nl_dir}/{factor[0]}/align_tfm_{factor[0]}_{curr_row["order"]}_{curr_row["ligand"]}.h5'
            out_nii_fn = f'{nl_dir}/{factor[0]}/nl_aligned_{factor[0]}_{curr_row["order"]}_{curr_row["ligand"]}.nii.gz'
            os.makedirs(outprefix0, exist_ok=True)
            os.makedirs(outprefix1, exist_ok=True)

            #   prev_tfm_fn
            #       |
            # prev <-- curr <-- next
            #                 |
            #           curr_tfm_inv_fn

            prev_dim = nib.load(prev_row[file_to_align]).shape
            curr_dim = nib.load(curr_row[file_to_align]).shape
            next_dim = nib.load(next_row[file_to_align]).shape

            assert prev_dim == curr_dim == next_dim, f'Error: {prev_dim} {curr_dim} {next_dim}'
            
            prev_tfm_fn, prev_tfm_inv_fn, prev_rsl_fn = ANTs(tfm_prefix=outprefix0,
                    fixed_fn=prev_row[file_to_align], moving_fn=curr_row[file_to_align], moving_rsl_prefix=outprefix0, 
                    metrics=metric, tfm_type=tfm_type, iterations=itr, shrink_factors=factor,
                    smoothing_sigmas=sigma, init_tfm='', no_init_tfm=False, dim=2, nbins=32,
                    sampling_method='Regular',sampling=1, verbose=0, generate_masks=False, clobber=0)
            
            curr_tfm_fn, curr_tfm_inv_fn, curr_rsl_fn = ANTs(tfm_prefix=outprefix1,
                    fixed_fn=curr_row[file_to_align], moving_fn=next_row[file_to_align], moving_rsl_prefix=outprefix1, 
                    metrics=metric, tfm_type=tfm_type, iterations=itr, shrink_factors=factor,
                    smoothing_sigmas = sigma, init_tfm='', no_init_tfm=False, dim=2, nbins=32,
                    sampling_method='Regular',sampling=1, verbose=0, generate_masks=False, clobber=0)

            assert os.path.exists(prev_tfm_fn), 'Error: files does not exist: ' + prev_tfm_fn
            assert os.path.exists(curr_tfm_inv_fn), 'Error: files does not exist: '+ curr_tfm_inv_fn
           
            if not os.path.exists(out_tfm_fn) :
                #getdim = lambda fn : nib.load(fn).shape
                average_warp(prev_tfm_fn, curr_tfm_inv_fn, out_tfm_fn)   

            try :
                nl_tfm_files[i] = [out_tfm_fn] + nl_tfm_files[i]
            except KeyError :
                nl_tfm_files[i] = [out_tfm_fn]

            transformations = ' '.join([ ' -t ' + fn for fn in nl_tfm_files[i] ])

            if not os.path.exists(out_nii_fn)  :
                shell(f'antsApplyTransforms -v 0 -d 2 -i {curr_row["lin"]} -r {curr_row[file_to_align]} {transformations} -o {out_nii_fn}')
                i_ss = np.sum(np.power(nib.load(curr_row["lin"]).dataobj,2))
                o_ss = np.sum(np.power(nib.load(out_nii_fn).dataobj,2)) 
                if i_ss == o_ss: print("\nWarning: no transformation applied\n")

            vol[:,curr_row['order'],:] = nib.load(out_nii_fn).get_fdata()


            if i0 == 0 : new_nl_files[i0] = outprefix0 + f'_level-0_{metric[0]}_SyN_inverse.nii.gz'
            if i1 == n-1 : new_nl_files[i1] = outprefix1 + f'_level-0_{metric[0]}_SyN.nii.gz' 
            new_nl_files[i] = out_nii_fn


        for i in range(0,n): df[file_to_align].iloc[i] = new_nl_files[i]
        nib.Nifti1Image(vol, affine).to_filename(curr_itr_fn)
        #concat_section_to_volume(df, affine, curr_itr_fn, file_var=file_to_align )
       
    return  df, curr_itr_fn


def reconstruct(subject_id, auto_dir, template_fn, scale_factors_json_fn, out_dir, csv_fn, native_pixel_size=0.02163,brain = '11530', hemi='B', pytorch_model='', ligands_to_exclude=[], resolution_list=[4,3,2,1], lowres=0.4, flip_dict={} ):
    mask_dir = f'{auto_dir}/mask_dir/'
    subject_dir=f'{out_dir}/{subject_id}/' 
    crop_dir = f'{out_dir}/{subject_id}/crop/'
    init_dir = f'{out_dir}/{subject_id}/init_align/'
    nl_dir = f'{out_dir}/{subject_id}/nl_align/'
    downsample_dir = f'{out_dir}/{subject_id}/downsample'
    init_3d_dir = f'{out_dir}/{subject_id}/init_align_3d/'
    ligand_dir = f'{subject_dir}/ligand/'

    srv_max_resolution_fn=template_fn #might need to be upsampled to maximum resolution
    surf_dir = f'{auto_dir}/surfaces/'
    interp_dir = f'{subject_dir}/interp/'
    
    for dirname in [subject_dir, crop_dir, init_dir, downsample_dir, init_3d_dir, ligand_dir, interp_dir] : os.makedirs(dirname, exist_ok=True)

    
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
    
    print('1')
    #TODO: ligand_contrast_order = get_ligand_contrast_order()
    ligand_contrast_order=['oxot'] #DEBUG just for rat!
    
    print('2')
    aligned_df = align(df, init_dir, ligand_contrast_order, metric='GC' )

    print('3')
    concat_section_to_volume(aligned_df, affine, volume_init_fn, file_var='init' )

    print('4')
    file_to_align='init'
    for itr, curr_res in enumerate(resolution_list):
        # Define some variables
        resolution_2d = curr_res
        # in human reconstruction, the 3d resolution isn't necessarily the same as 2d because very high 3d alignment
        # resolution eats too much RAM. 
        resolution_3d = curr_res 

        # Define directories
        seg_dir = f'{out_dir}/{subject_id}/{curr_res}mm/segment/'
        seg_2d_dir = f'{seg_dir}/2d/'
        align_3d_dir=f'{out_dir}/{subject_id}/{curr_res}mm/align_3d/'
        align_2d_dir=f'{out_dir}/{subject_id}/{curr_res}mm/align_2d/'
        
        # Create directories
        multires_dirs = [seg_dir, seg_2d_dir, align_2d_dir, align_3d_dir]
        for dirname in multires_dirs : os.makedirs(dirname, exist_ok=True)
       
        # Define volume filenames
        volume_seg_fn = f'{seg_dir}/{subject_id}_segment_volume.nii.gz'
        volume_align_2d_fn = f'{align_2d_dir}/{subject_id}_align_2d_space-nat.nii.gz'
        current_template_fn = f'{align_2d_dir}/'+os.path.basename(re.sub('.nii',f'_{curr_res}mm.nii', template_fn))
        template_rec_space_fn = f'{align_2d_dir}/'+os.path.basename(re.sub('.nii',f'_{curr_res}mm_y-0.02_space-nat.nii', template_fn))
        template_iso_rec_space_fn = f'{align_2d_dir}/'+os.path.basename(re.sub('.nii',f'_{curr_res}mm_space-nat.nii', template_fn))

        ### 5. Segment
        print('\t5. Segment')
        resample_transform_segmented_images(aligned_df, itr, resolution_2d, resolution_3d, seg_2d_dir, file_to_align='crop')
        classifyReceptorSlices(aligned_df, volume_init_fn, seg_2d_dir, seg_dir, volume_seg_fn, resolution=resolution_3d, interpolation='linear', file_to_align='crop', flip_axes=())

        ### 6. 3D alignment
        if not os.path.exists(current_template_fn):
            prefilter_and_downsample(template_fn, [resolution_3d]*3, current_template_fn)

        print('\tAlign 3D')
        tfm_3d_fn, tfm_3d_inv_fn = multires_align_3d(subject_id, align_3d_dir, volume_seg_fn, current_template_fn, resolution_list, curr_res, affine_fn, metric='Mattes')
        
        print(file_to_align); 
        ### 7. 2d alignement
        if not os.path.exists(template_rec_space_fn) : 
            resample_to_autoradiograph_sections(subject_id, '', '', float(curr_res), current_template_fn, volume_seg_fn, tfm_3d_inv_fn, template_iso_rec_space_fn, template_rec_space_fn)
       
        create_2d_sections(aligned_df, template_rec_space_fn, float(curr_res), align_2d_dir )
        aligned_df = align_2d(aligned_df, align_2d_dir, volume_init_fn, template_rec_space_fn, seg_2d_dir, resolution_2d, itr, file_to_align='crop', use_syn=False)
        
        aligned_df = concatenate_sections_to_volume( aligned_df, template_rec_space_fn, align_2d_dir, volume_align_2d_fn)

  
    for ligand, ligand_df in aligned_df.groupby(['ligand']):
        #ligand_df, nl_vol_fn = align_nl(ligand_df, nl_dir)
        
        interp_fn=f'{interp_dir}/{subject_id}_{ligand}_vol_interp.nii.gz'
        interp_stx_fn=f'{interp_dir}/{subject_id}_{ligand}_vol_interp_space-stx.nii.gz'
        img = nib.load(volume_align_2d_fn)
        vol = img.get_fdata()
        vol = interpolate_missing_sections(vol, dilate_volume=False)
        nib.Nifti1Image(vol, img.affine).to_filename(interp_fn)

        shell(f'antsApplyTransforms -v 1 -d 3 -i {interp_fn} -r {current_template_fn} -t {tfm_3d_fn} -o {interp_stx_fn}')


    
    

