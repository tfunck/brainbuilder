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


def get_start_end_slab_world_coordinates(df,slabs,ystep,ystart):
    slab_minima={}
    slab_maxima={}
    slab_width={}
    
    slabs.sort()
    for slab_i in slabs :
        slab_vmin = df.loc[ df['slab']==int(slab_i) ]['global_order'].min()
        slab_vmax = df.loc[ df['slab']==int(slab_i) ]['global_order'].max()

        slab_min = v2w(slab_vmin, ystep, ystart)
        slab_max = v2w(slab_vmax, ystep, ystart)

        #print('{}\t{}\t{}\t{}'.format(slab_i,slab_vmin,slab_vmax,slab_vmax-slab_vmin) )

        slab_minima[slab_i] = slab_min
        slab_maxima[slab_i] = slab_max
        slab_width[slab_i]  = (slab_vmax-slab_vmin)*ystep

    return slab_minima, slab_maxima, slab_width


def align_slab_to_mri(seg_rsl_fn, srv_rsl_fn, slab, out_dir, df, slabs, out_tfm_fn, out_tfm_inv_fn, out_fn, out_inv_fn, resolution, resolution_itr, resolution_list, clobber=False, verbose=True ) :
    print('\tRunning')
    # Load super-resolution GM volume extracted from donor MRI.
    slab=int(slab)
    srv_img = nib.load(srv_rsl_fn)
    srv_vol = srv_img.get_fdata()
    ymax = srv_vol.shape[1]
    aff=srv_img.affine

    srv_ystep = srv_img.affine[1,1] 
    srv_ystart = srv_img.affine[1,3] 
    srv_min, srv_max = list(map(lambda x: v2w(x,srv_ystep,srv_ystart), find_vol_min_max(srv_vol) ))
    srv_width = srv_max - srv_min

    seg_img = nib.load(seg_rsl_fn)
    seg_vol = seg_img.get_fdata() 
    ystep = 0.02
    ystart = seg_img.affine[1,3]

    metric_list=[]
    tfm_list=[]
  
    slab_minima, slab_maxima, slab_width = get_start_end_slab_world_coordinates(df,slabs,ystep,ystart)

    seg_width =  np.sum(list(slab_width.values())) 
    n_slabs = len(slab_width.keys())
    widthDiff = (srv_width-seg_width)/ max(n_slabs-1,1)
    slab_prior_position={}
    prevWidth=0
    for i,s in enumerate(slabs) :
        slab_prior_position[s] = srv_max - i*widthDiff - slab_width[s]/2.0 - prevWidth
        prevWidth += slab_width[s]

    cur_slab_width = slab_width[str(slab)]

    print('Current slab width:',cur_slab_width)

    min_nl_itr = len( [ resolution_list[i] for i in range(resolution_itr) if  float(resolution_list[i]) <= .1 ] )

    base_lin_itr= 500
    base_nl_itr = 100
    max_lin_itr = base_lin_itr * (resolution_itr+1)
    max_nl_itr  = base_nl_itr * (resolution_itr-min_nl_itr+1)
    lin_step = -base_lin_itr 
    nl_step  = -base_nl_itr 

    lin_itr_str='x'.join([str(i) for i in range(max_lin_itr,0,lin_step)])
    nl_itr_str='x'.join([str(i) for i in range(max_nl_itr,0,nl_step)  ])

    f_str='x'.join([ str(i) for i in range(resolution_itr+1,0,-1)])
    f = lambda x : x/2 if x > 1  else 0
    s_list = map(f,  range(resolution_itr+1,0,-1) ) 
    s_str='x'.join( [str(i) for i in s_list] ) + 'vox'

    y0w= ( slab_prior_position[str(slab)] - cur_slab_width/2 * 1.3)
    y1w= ( slab_prior_position[str(slab)] + cur_slab_width/2 * 1.3)
    y0=w2v(y0w, srv_ystep, srv_ystart)
    y1=w2v(y1w, srv_ystep, srv_ystart)

    if verbose : print(y0w,y1w,y0,y1) 

    srv_slab_fn=f'{out_dir}/srv_{y0}_{y1}.nii.gz' 

    prefix=re.sub('_SyN_Composite.h5','',out_tfm_fn)
    prefix_rigid=prefix+'_Rigid_'
    prefix_similarity=prefix+'_Similarity_'
    prefix_affine=prefix+'_Affine_'
    prefix_syn=prefix+'_SyN_'

    if not os.path.exists(out_fn) or not os.path.exists(out_tfm_fn) or clobber:
        # write srv slab if it does not exist
        if not os.path.exists(srv_slab_fn) :
            aff[1,3] = v2w(y0, srv_ystep, srv_ystart)
            srv_slab=srv_vol[:,y0:y1,:]
            nib.Nifti1Image(srv_slab, aff).to_filename(srv_slab_fn)
        # set initial transform
        init_moving=f'--initial-moving-transform [{srv_slab_fn},{seg_rsl_fn},1]'
        
        # calculate rigid registration
        if not os.path.exists(f'{prefix_rigid}Composite.h5'):
            shell(f'/usr/bin/time -v antsRegistration -v 1 -a 1 -d 3 {init_moving} -t Rigid[.1] -m GC[{srv_slab_fn},{seg_rsl_fn},1,20,Regular,1]  -s {s_str} -f {f_str}  -c {lin_itr_str}   -o [{prefix_rigid},{out_fn},{out_inv_fn}] ', verbose=True)

        # calculate similarity registration
        if not os.path.exists(f'{prefix_similarity}Composite.h5'):
            shell(f'/usr/bin/time -v antsRegistration -v 1 -a 1 -d 3 --initial-moving-transform {prefix_rigid}Composite.h5    -t Similarity[.1]  -m Mattes[{srv_slab_fn},{seg_rsl_fn},1,20,Regular,1]  -s {s_str} -f {f_str}  -c {lin_itr_str} -o [{prefix_similarity},{out_fn},{out_inv_fn}] ', verbose=True)
        
        #calculate affine registration
        if not os.path.exists(f'{prefix_affine}Composite.h5'):
            shell(f'/usr/bin/time -v antsRegistration -v 1 -a 1 -d 3 --initial-moving-transform {prefix_similarity}Composite.h5 -t Affine[.1]  -m Mattes[{srv_slab_fn},{seg_rsl_fn},1,20,Regular,1]  -s {s_str} -f {f_str}  -c {lin_itr_str}  -o [{prefix_affine},{out_fn},{out_inv_fn}] ', verbose=True)

        #calculate SyN
        if not os.path.exists(f'{prefix_syn}Composite.h5'):
                
            shell(f'/usr/bin/time -v antsRegistration -v 1 -a 1 -d 3  --initial-moving-transform {prefix_affine}Composite.h5 -t SyN[.1] -m Mattes[{srv_slab_fn},{seg_rsl_fn},1,20,Regular,1]  -s {s_str} -f {f_str}  -c {nl_itr_str}   -o [{prefix_syn},{out_fn},{out_inv_fn}] ', verbose=True)
        #hi_res_align(seg_rsl_fn, srv_slab_fn, resolution, out_dir, f'{prefix_syn}_init_Composite.h5', prefix_syn,out_fn, out_inv_fn)


    return 0

def write_block(fn, start, end, out_fn) :
    img = nib.load(fn)
    vol = img.get_fdata()
    block = vol[start[0]:end[0],start[1]:end[1],start[2]:end[2]]
    nib.Nifti1Image(block, affine).to_filename(out_fn)
    return block

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


        




