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

def calc_distance_metric(fixed_fn, moving_rsl_fn):
    metric_val=0
    if  os.path.exists(fixed_fn) and os.path.exists(moving_rsl_fn) :
            try :
                print('fixed_fn',fixed_fn) 
                print('moving_rsl_fn', moving_rsl_fn)
                fixed  = ants.from_numpy( nib.load(fixed_fn).get_fdata())
                moving = ants.from_numpy( nib.load(moving_rsl_fn).get_fdata())
                #ants_metric = ants.create_ants_metric(fixed,moving, metric_type='MattesMutualInformation')
                ants_metric = ants.create_ants_metric(fixed,moving, metric_type='Correlation')
                metric_val = ants_metric.get_value()
                del ants_metric
                del fixed
                del moving
            except RuntimeError :
                pass
    
    return  metric_val

def align_slab_to_mri(seg_rsl_fn, srv_rsl_fn, slab, out_dir, df, tfm=None, clobber=False ) :
    # Load super-resolution GM volume extracted from donor MRI.
    print('srv',srv_rsl_fn)
    slab=int(slab)
    srv_img = nib.load(srv_rsl_fn)
    srv_vol = srv_img.get_fdata()
    ymax = srv_vol.shape[1]
    aff=srv_img.affine
    srv_ystep = srv_img.affine[1,1] 
    srv_ystart = srv_img.affine[1,3] 
    srv_min, srv_max = list(map(lambda x: v2w(x,srv_ystep,srv_ystart), find_vol_min_max(srv_vol) ))
    srv_width = srv_max - srv_min

    print('seg',seg_rsl_fn)
    seg_img = nib.load(seg_rsl_fn)
    seg_vol = seg_img.get_fdata() 
    ystep = 0.02
    ystart = seg_img.affine[1,3]

    metric_list=[]
    tfm_list=[]

    slab_minima = np.array( list( map(lambda x: v2w(x,ystep,ystart), df.groupby(['slab'])['global_order'].min() ) ))
    slab_maxima = np.array( list( map(lambda x: v2w(x,ystep,ystart), df.groupby(['slab'])['global_order'].max() ) ))
    slab_width = slab_maxima-slab_minima
    seg_width =  np.sum(slab_width) 
    n_slabs = len(slab_width)
    widthDiff = (srv_width-seg_width)/(n_slabs-1)
    slab_prior_position={}
    for i in range(n_slabs ):
        s=i+1
        prevWidth=0
        if s > 1 : prevWidth = np.sum(slab_width[(i-1)::-1])

        slab_prior_position[s] = srv_max - i*widthDiff - slab_width[i]/2.0 - prevWidth

    cur_slab_width = slab_width[ slab-1]
    step = 2 # int(cur_slab_width / (4* srv_ystep))

    #for y0 in range(0,ymax,step) :
    print(slab_prior_position)
    y0w= ( slab_prior_position[slab] - cur_slab_width/2 * 1.3)
    y1w= ( slab_prior_position[slab] + cur_slab_width/2 * 1.3)
    print(srv_ystep, srv_ystart)
    y0=w2v(y0w, srv_ystep, srv_ystart)
    y1=w2v(y1w, srv_ystep, srv_ystart)
    y = int((y0+y1)/2)
    #yw = v2w(y, srv_ystep, srv_ystart)
    #if np.abs(yw - slab_prior_position[slab]) < cur_slab_width and yw > slab_prior_position[slab+1] + slab_width[slab+1]/2   :

    srv_slab_fn=f'{out_dir}/srv_{y0}_{y1}.nii.gz' 
    prefix=f'{out_dir}/{y0}_{y1}_'
    out_fn=f'{prefix}rec_seg.nii.gz'
    out_inv_fn=f'{prefix}srv.nii.gz'
    out_tfm_fn=f'{prefix}Composite.h5'

    if not os.path.exists(out_fn) or not os.path.exists(out_tfm_fn) or clobber:
        # write srv slab if it does not exist
        if not os.path.exists(srv_slab_fn) :
            aff[1,3] = v2w(y0, srv_ystep, srv_ystart)
            srv_slab=srv_vol[:,y0:y1,:]
            nib.Nifti1Image(srv_slab, aff).to_filename(srv_slab_fn)
        print('\tCreate',out_fn)
        print('\tCreate',out_tfm_fn)
        # set initial transform
        #if tfm == None:
        init_moving=f'--initial-moving-transform [{srv_slab_fn},{seg_rsl_fn},1]'
        #else :
        #    init_moving=f'--initial-moving-transform {tfm}'
        
        # calculate registration
        shell(f'antsRegistration -v 0 -a 1 -d 3 {init_moving} -o [{prefix},{out_fn},{out_inv_fn}]   -t Rigid[.1] -m GC[{srv_slab_fn},{seg_rsl_fn},1,20,Regular,1] -c [1000] -s 0vox -f 1  -t Affine[.1] -c [500] -m Mattes[{srv_slab_fn},{seg_rsl_fn},1,20,Regular,1] -s 0vox -f 1 ', verbose=True)
        #shell(f'antsRegistration -v 0 -a 1 -d 3 {init_moving} -o [{prefix},{out_fn},{out_inv_fn}]   -t Rigid[.1] -m Mattes[{srv_slab_fn},{seg_rsl_fn},1,20,Regular,1] -c [1000] -s 0vox -f 1  -t Affine[.1] -c [10]  -m Mattes[{srv_slab_fn},{seg_rsl_fn},1,20,Regular,1] -s 0vox -f 1')
    
    #metric_fn=f'{out_dir}/{y0}_{y1}_metric.txt'
    
    #if not os.path.exists(metric_fn) :
    #    metric = calc_distance_metric(srv_slab_fn, out_fn)
    #    with open(metric_fn,'w+') as f:
    #        f.write(f'{metric}')
    #else :
    #    with open(metric_fn, 'r') as f:
    #        for l in f.readlines():
    #            metric=float(l.rstrip())

    #tfm_list.append(out_tfm_fn)
    #metric_list.append(metric_fn)

    #metric_argmin = np.argmin(metric_list)
    #best_tfm = tfm_list[ metric_argmin ]
    #print('Best transform', best_tfm, metric_list[metric_argmin])
    return out_tfm_fn, out_fn, out_inv_fn




        



