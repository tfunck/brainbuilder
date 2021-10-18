from scipy.ndimage import gaussian_filter
import pandas as pd
import numpy as np
import json
import nibabel as nib
import ants
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
import shutil
import re
import h5py as h5 
from glob import glob
from utils.ANTs import ANTs
from utils.utils import shell, splitext
from nibabel.processing import resample_from_to, resample_to_output
from utils.utils import splitext  
from scipy.ndimage.filters import gaussian_filter1d
from scipy.io import loadmat
from numpy.linalg import det
from scipy.interpolate import interp1d
import matplotlib
#matplotlib.use('TKAgg')


def w2v(c, step, start):
    return np.round( (c-start)/step ).astype(int)
def v2w(i, step, start) :
    return start + i * step

def find_vol_min_max(vol) :
    '''
    Finds the min and max spatial coordinate of the srv image
    '''
    profile = np.max(vol, axis=(0,2))
    if np.sum(profile) == 0 :
        print("Error : empty srv file")
        exit(1)
    srvMin = np.argwhere(profile>=.1)[0][0]
    srvMax =  np.argwhere(profile>=.1)[-1][0]
    return srvMin, srvMax


def get_slab_start_end(df,slabs,ystep,ystart, cur_slab, srv_max, srv_width, srv_ystep, srv_ystart, verbose=False):
    slab_minima={}
    slab_maxima={}
    slab_width={}
    total_slab_width=0

    slabs.sort()

    slab_processing_order = range(len(slabs)) #Could be changed to a different ordering eventually
    unaligned_slab_list = [ i+1 for i in slab_processing_order if i+1 <= cur_slab ]

    for slab_i in slabs :
        slab_vmin = df.loc[ df['slab']==int(slab_i) ]['global_order'].min()
        slab_vmax = df.loc[ df['slab']==int(slab_i) ]['global_order'].max()

        slab_min = v2w(slab_vmin, ystep, ystart)
        slab_max = v2w(slab_vmax, ystep, ystart)

        #print('{}\t{}\t{}\t{}'.format(slab_i,slab_vmin,slab_vmax,slab_vmax-slab_vmin) )

        slab_minima[slab_i] = slab_min
        slab_maxima[slab_i] = slab_max
        slab_width[slab_i]  = (slab_vmax-slab_vmin)*ystep
        
        if slab_i in unaligned_slab_list :
            total_slab_width += slab_width[slab_i]

    total_slab_gap = srv_width - total_slab_width
    n_slabs = len(slabs)
    average_slab_gap = total_slab_gap / n_slabs
    print('average slab gap', average_slab_gap)
    slab_start = srv_max - slab_width[str(cur_slab)] - average_slab_gap / 2 #1.2

    y1w = srv_max
    y0w = slab_start

    y0 = w2v(y0w, srv_ystep, srv_ystart)
    y1 = w2v(y1w, srv_ystep, srv_ystart)

    if verbose : print(y0w,y1w,y0,y1) 
    return y0, y1

def pad_volume(vol, max_factor, affine, min_voxel_size=29):
    xdim, ydim, zdim = vol.shape


    def padded_dim(dim, max_factor, min_voxel_size) :
        # min_voxel_size < dim / 2 ** (max_factor-1) 
        # min_voxel_size * 2 ** (max_factor-1) < dim
        downsampled_dim = np.ceil(dim / 2 ** (max_factor-1))
        if downsampled_dim < min_voxel_size :
            return np.ceil( (min_voxel_size-downsampled_dim)/2).astype(int)
        else :
            return 0

    x_pad = padded_dim(xdim, max_factor, min_voxel_size) 
    y_pad = padded_dim(ydim, max_factor, min_voxel_size)
    z_pad = padded_dim(zdim, max_factor, min_voxel_size)

    #print(x_pad,y_pad,z_pad)
    #print(vol.shape)
    vol_padded = np.pad(vol, ((x_pad, x_pad), (y_pad, y_pad),(z_pad, z_pad))) 
    #print(vol_padded.shape)
    
    affine[0,3] -= x_pad * affine[0,0]
    affine[1,3] -= y_pad * affine[1,1]
    affine[2,3] -= z_pad * affine[2,2]
    return vol_padded, affine

def align_slab_to_mri(seg_rsl_fn, srv_rsl_fn, slab, out_dir, df, slabs, out_tfm_fn, out_tfm_inv_fn, out_fn, out_inv_fn, resolution, resolution_itr, resolution_list, clobber=False, verbose=True ) :
    print('\tRunning')
    # Load super-resolution GM volume extracted from donor MRI.
    slab=int(slab)
    print('SRV rsl fn:', srv_rsl_fn)
    srv_img = nib.load(srv_rsl_fn)
    srv_vol = srv_img.get_fdata()
    ymax = srv_vol.shape[1]

    aff=srv_img.affine

    srv_ystep = srv_img.affine[1,1] 
    srv_ystart = srv_img.affine[1,3] 
    srv_min, srv_max = list(map(lambda x: v2w(x,srv_ystep,srv_ystart), find_vol_min_max(srv_vol) ))
    srv_width = srv_max - srv_min
    print('srv min max', srv_min, srv_max)

    cur_resolution = float(resolution_list[resolution_itr])
    max_resolution = float(resolution_list[0])

    # max_downsample_factor = 2 ^ ( L - 1) 
    max_downsample_factor = np.floor( max_resolution / cur_resolution ).astype(int)

    # log2(max_downsample_factor) + 1 = L
    max_downsample_level = np.int( np.log2(max_downsample_factor) + 1 )
    seg_img = nib.load(seg_rsl_fn)
    seg_vol = seg_img.get_fdata() 


    seg_vol, pad_affine = pad_volume(seg_vol, max_downsample_level, seg_img.affine )
    seg_rsl_pad_fn=re.sub('.nii','_padded.nii', seg_rsl_fn)
    nib.Nifti1Image(seg_vol, pad_affine).to_filename(seg_rsl_pad_fn)
    print('\t\tPadded segmented autoradiographs', seg_rsl_pad_fn)
    seg_rsl_fn = seg_rsl_pad_fn

    ystep = 0.02
    ystart = seg_img.affine[1,3]

    metric_list=[]
    tfm_list=[]
  
    y0, y1 = get_slab_start_end(df, slabs, ystep, ystart, slab, srv_max, srv_width, srv_ystep, srv_ystart, verbose=True)
    exit(0)
    print('Max Downsample Factor:',max_downsample_factor)
    print('Max Downsample Level:',max_downsample_level)
    
    f_list=[ str(f) for f in range(max_downsample_level, 0, -1)]# if smallest_dimension / 2**(f-1) > 29 ]
   
    assert len(f_list) != 0, 'Error: no smoothing factors' 
    print(f_list)

    f_str='x'.join([ str(f) for f in f_list ])
    s_list = [int(f)*.2 for f in f_list ] 
    s_str='x'.join( [str(i) for i in s_list] ) + 'vox'

    min_nl_itr = len( [ resolution_list[i] for i in range(resolution_itr) if  float(resolution_list[i]) <= .1 ] )

    base_lin_itr= 500
    base_nl_itr = 100
    max_lin_itr = base_lin_itr * (len(f_list)+1)
    max_nl_itr  = base_nl_itr * (resolution_itr-min_nl_itr+1)
    lin_step = base_lin_itr 
    nl_step  = base_nl_itr 

    lin_itr_str='x'.join([str(max_lin_itr - i*lin_step) for i in range(len(f_list))])
    nl_itr_str='x'.join([str(max_nl_itr - i*nl_step) for i in range(len(f_list))  ])



    
    srv_slab_fn=f'{out_dir}/srv_{y0}_{y1}.nii.gz' 
    temp_out_fn=f'{out_dir}/partial_rec_space-mri.nii.gz'

    prefix=re.sub('_SyN_Composite.h5','',out_tfm_fn)
    prefix_rigid=prefix+'_Rigid_'
    prefix_similarity=prefix+'_Similarity_'
    prefix_affine=prefix+'_Affine_'
    prefix_syn=prefix+'_SyN_'

    if not os.path.exists(temp_out_fn) or not os.path.exists(out_tfm_fn) or clobber:
        # write srv slab if it does not exist
        aff[1,3] = v2w(y0, srv_ystep, srv_ystart)
        srv_slab = srv_vol[:,y0:y1,:]
        srv_slab, pad_aff = pad_volume(srv_slab, max_downsample_level, aff )
        nib.Nifti1Image(srv_slab, pad_aff).to_filename(srv_slab_fn)

        # set initial transform
        init_moving=f'--initial-moving-transform [{srv_slab_fn},{seg_rsl_fn},1]'
        
        # calculate rigid registration
        if not os.path.exists(f'{prefix_rigid}Composite.h5'):
            shell(f'/usr/bin/time -v antsRegistration -v 1 -a 1 -d 3 {init_moving} -t Rigid[.05] -m GC[{srv_slab_fn},{seg_rsl_fn},1,20,Regular,1]  -s {s_str} -f {f_str}  -c {lin_itr_str}   -o [{prefix_rigid},{temp_out_fn},{out_inv_fn}] ', verbose=True)

        # calculate similarity registration
        if not os.path.exists(f'{prefix_similarity}Composite.h5'):
            shell(f'/usr/bin/time -v antsRegistration -v 1 -a 1 -d 3 --initial-moving-transform {prefix_rigid}Composite.h5    -t Similarity[.05]  -m Mattes[{srv_slab_fn},{seg_rsl_fn},1,20,Regular,1]  -s {s_str} -f {f_str}  -c {lin_itr_str} -o [{prefix_similarity},{temp_out_fn},{out_inv_fn}] ', verbose=True)

        #calculate affine registration
        if not os.path.exists(f'{prefix_affine}Composite.h5'):
            shell(f'/usr/bin/time -v antsRegistration -v 1 -a 1 -d 3 --initial-moving-transform {prefix_similarity}Composite.h5 -t Affine[.01]  -m Mattes[{srv_slab_fn},{seg_rsl_fn},1,20,Regular,1]  -s {s_str} -f {f_str}  -c {lin_itr_str}  -o [{prefix_affine},{temp_out_fn},{out_inv_fn}] ', verbose=True)

        #calculate SyN
        if float(resolution) >= 1.0 :
            nl_metric = f'CC[{srv_slab_fn},{seg_rsl_fn},1,2,Regular,1]'
        else :
            nl_metric=f'Mattes[{srv_slab_fn},{seg_rsl_fn},1,20,Regular,1]'

        if not os.path.exists(f'{prefix_syn}Composite.h5'):
            shell(f'/usr/bin/time -v antsRegistration -v 1 -a 1 -d 3  --initial-moving-transform {prefix_affine}Composite.h5 -t SyN[.01] -m {nl_metric}  -s {s_str} -f {f_str}  -c {nl_itr_str}   -o [{prefix_syn},{temp_out_fn},{out_inv_fn}] ', verbose=True)
        #hi_res_align(seg_rsl_fn, srv_slab_fn, resolution, out_dir, f'{prefix_syn}_init_Composite.h5', prefix_syn,out_fn, out_inv_fn)

    if not os.path.exists(out_fn):
        shell(f'antsApplyTransforms -i {seg_rsl_fn} -r {srv_rsl_fn}  -t {prefix_syn}Composite.h5  -o {out_fn}')

    return 0

def write_block(fn, start, end, out_fn) :
    img = nib.load(fn)
    vol = img.get_fdata()
    block = vol[start[0]:end[0],start[1]:end[1],start[2]:end[2]]
    nib.Nifti1Image(block, affine).to_filename(out_fn)
    return block
'''
def hi_res_align(moving_fn, fixed_fn, resolution, tfm_dir, init_tfm, prefix, out_fn, out_inv_fn):
    kernel_dim = [10/resolution, 10/resolution, 10/resolution] 
    step = [ int(kernel_dim[0]/3) ] * 3
    out_tfm_fn = f'{prefix}Composite.h5'
    img = nib.load(moving_fn)
    image_dim = img.shape

    f = h5.File('/tmp/deformation_field.h5', 'w')
    dfield = f.create_dataset('field',(np.product(image_dim.shape[0])*3,),dtype=np.float64)

    for x in range(0,image_dim.shape[0],step[0]) :
        for y in range(0,image_dim.shape[1],step[1]) :
            for z in range(0,image_dim.shape[2],step[2]) :
                #
                start=[x,y,z]
                end=[x+step[0],y+step[1],z+step[2]]

                block_fn = f'{out_dir}/block_{x}-{end[0]}_{y}-{end[1]}_{z}-end[2]'
                # extract block from moving image, save to tmp directory 
                write_block(moving_fn, start, end, '/tmp/moving.nii.gz')
                # extract block from fixed image, save to tmp directory
                write_block(fixed_fn, start, end, '/tmp/fixed.nii.gz')
                
                # non-linear alignment     
                shell(f'/usr/bin/time -v antsRegistration -v 1 -a 1 -d 3  --initial-moving-transform {init_tfm}  -t SyN[.1] -c [500] -m GC[/tmp/fixed.nii.gz,/tmp/moving.nii.gz,1,20,Regular,1] -s 0vox -f 1  -o [/tmp/,/tmp/tfm.nii.gz, /tmp/tfm_inv.nii.gz ] ', verbose=True)
            
                # use h5 to load deformation field and save
                block_dfield = ants.read_transform(block_tfm_fn)
                dfield['field'][x:x+step,y:y+step,z:z+step] = block_dfield
                

    final_tfm = ants.transform_from_deformation_field(dfield['field'])
    ants.write_transform(final_tfm, out_tfm_fn)

'''
        




