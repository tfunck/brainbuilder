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
from glob import glob
from ANTs import ANTs
from utils.utils import shell, splitext
from nibabel.processing import resample_from_to, resample_to_output
from utils.utils import splitext  
from scipy.ndimage.filters import gaussian_filter1d
from scipy.io import loadmat
from numpy.linalg import det
from scipy.interpolate import interp1d
import matplotlib
matplotlib.use('TKAgg')


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


def align_slab_to_mri(seg_rsl_fn, srv_rsl_fn, slab, out_dir, df, slabs, tfm=None, clobber=False, verbose=True ) :
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
        print(srv_max,widthDiff,slab_width[s], prevWidth)
        slab_prior_position[s] = srv_max - i*widthDiff - slab_width[s]/2.0 - prevWidth
        prevWidth += slab_width[s]

    cur_slab_width = slab_width[str(slab)]

    print('Current slab width:',cur_slab_width)

    print(slab_prior_position)
    print(slab)
    y0w= ( slab_prior_position[str(slab)] - cur_slab_width/2 * 1.3)
    y1w= ( slab_prior_position[str(slab)] + cur_slab_width/2 * 1.3)
    y0=w2v(y0w, srv_ystep, srv_ystart)
    y1=w2v(y1w, srv_ystep, srv_ystart)
    if verbose :
        print(y0w,y1w,y0,y1) 
    srv_slab_fn=f'{out_dir}/srv_{y0}_{y1}.nii.gz' 
    prefix=f'{out_dir}/{y0}_{y1}_'
    out_fn=f'{prefix}rec_seg.nii.gz'
    out_inv_fn=f'{prefix}srv.nii.gz'
    out_tfm_fn=f'{prefix}Composite.h5'
    out_tfm_inv_fn=f'{prefix}InverseComposite.h5'

    if not os.path.exists(out_fn) or not os.path.exists(out_tfm_fn) or clobber:
        # write srv slab if it does not exist
        if not os.path.exists(srv_slab_fn) :
            aff[1,3] = v2w(y0, srv_ystep, srv_ystart)
            srv_slab=srv_vol[:,y0:y1,:]
            nib.Nifti1Image(srv_slab, aff).to_filename(srv_slab_fn)
        # set initial transform
        #if tfm == None:
        init_moving=f'--initial-moving-transform [{srv_slab_fn},{seg_rsl_fn},1]'
        #else :
        #    init_moving=f'--initial-moving-transform {tfm}'
        
        # calculate registration
        # 
        shell(f'antsRegistration -v 0 -a 1 -d 3 {init_moving} -t Rigid[.1] -m GC[{srv_slab_fn},{seg_rsl_fn},1,20,Regular,1] -c [1000] -s 0vox -f 1  -t Similarity[.1] -c [500] -m Mattes[{srv_slab_fn},{seg_rsl_fn},1,20,Regular,1] -s 0vox -f 1  -t Affine[.1] -c [500] -m GC[{srv_slab_fn},{seg_rsl_fn},1,20,Regular,1] -s 0vox -f 1 -t SyN[.1] -c [500] -m GC[{srv_slab_fn},{seg_rsl_fn},1,20,Regular,1] -s 0vox -f 1  -o [{prefix},{out_fn},{out_inv_fn}]', verbose=True)
    return out_tfm_fn, out_tfm_inv_fn, out_fn, out_inv_fn




        



