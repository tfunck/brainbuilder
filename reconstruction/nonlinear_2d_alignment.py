import numpy as np
import nibabel as nib
import pandas as pd
import sys
import os
import json
import re
import pandas as pd
import time
import shutil
import tempfile
from section_2d import section_2d
from sys import argv
from glob import glob
from utils.ANTs import ANTs
from utils.utils import splitext, shell, w2v , v2w

def gen_2d_fn(prefix,x):
    return f'{prefix}_{x}_0.2mm.nii.gz'

def save_sections(file_list, vol, aff) :
    for fn, y in file_list:
        # Create 2D srv section
        nib.Nifti1Image(vol[ :, int(y), : ] , aff).to_filename(fn)

def create_2d_sections( df, rec_fn, rec_rsl_fn, srv_fn, output_dir,clobber=False) :
    fx_to_do=[]
    mv_to_do=[]
    
    tfm_dir=output_dir + os.sep + 'tfm'
    os.makedirs(tfm_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    for idx, (i, row) in enumerate(df.iterrows()):
        y=row['volume_order'] 
        prefix=f'{tfm_dir}/y-{y}' 
        mv_rsl_fn=gen_2d_fn(prefix,'mv')
        fx_fn=gen_2d_fn(prefix,'fx')

        if not os.path.exists(fx_fn) : fx_to_do.append( [fx_fn, y])
        if not os.path.exists(mv_rsl_fn) : mv_to_do.append( [mv_rsl_fn, y])

    if len(mv_to_do + fx_to_do) > 0 :

        rec_hires_img = nib.load(rec_fn)
        srv_img = nib.load(srv_fn)

        rec_hires_vol = rec_hires_img.get_fdata()
        srv = srv_img.get_fdata()
        
        xstep = rec_hires_img.affine[0,0]
        zstep = rec_hires_img.affine[2,2]

        aff_hires = np.array([[xstep,  0, 0, 0],
                        [0, zstep, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])

        save_sections(fx_to_do, srv, aff_hires)
        save_sections(mv_to_do, rec_hires_vol, aff_hires)

def receptor_2d_alignment( df, rec_fn, rec_rsl_fn, srv_fn, output_dir, resolution, resolution_itr, batch_processing=False, direction='rostral_to_caudal', clobber=False): 
    
    tfm_dir=output_dir + os.sep + 'tfm'

    #don't remember what resolution_level does, so setting to 3 to cancel out it's effect
    #avg_step = (xstep+zstep)/2 
    resolution_level = 1 # np.ceil(np.log2(3/avg_step )+ 1 ).astype(int)

    #Set strings for alignment parameters
    base_lin_itr= 100
    base_nl_itr = 10
    max_lin_itr = resolution_level * base_lin_itr * (resolution_itr+1)
    max_nl_itr  = resolution_level * base_nl_itr * (resolution_itr+1)
    lin_step = -base_lin_itr #*(resolution_itr)
    nl_step  = -base_nl_itr #*(resolution_itr)

    lin_itr_str='x'.join([str(i) for i in range(max_lin_itr,0,lin_step)])
    nl_itr_str='x'.join([str(i) for i in range(max_nl_itr,0,nl_step)])

    f_str='x'.join([ str(i) for i in range(resolution_itr+1,0,-1)])
    f = lambda x : x/2 if x > 1  else 0
    s_list = map(f,  range(resolution_itr+1,0,-1) ) 
    s_str='x'.join( [str(i) for i in s_list] ) + 'vox'


    for idx, (i, row) in enumerate(df.iterrows()):
        y=row['volume_order'] 
        prefix=f'{tfm_dir}/y-{y}' 
        # Create 2d rec section
        mv_rsl_fn=gen_2d_fn(prefix,'mv')
        # Create 2D srv section
        fx_fn=gen_2d_fn(prefix,'fx')

        out_vol_fn=f'{prefix}.nii.gz'
        out_inv_fn=f'{prefix}_inv.nii.gz'
        tfm_fn=f'{prefix}_Composite.h5'

        if not os.path.exists(tfm_fn) or not os.path.exists(out_vol_fn) or clobber : 
            print('\t\t',y)
            args=[ prefix, mv_rsl_fn, fx_fn, out_vol_fn, s_str, f_str, lin_itr_str, nl_itr_str]
            if not batch_processing :
                section_2d(*args)
            else : 
                shell('sbatch section_2d.sh '+' '.join(args))
                exit(0)

def concatenate_sections_to_volume(df, rec_fn, output_dir, out_fn):
    
    tfm_dir=output_dir + os.sep + 'tfm'

    hires_img = nib.load(rec_fn)
    out_vol=np.zeros(hires_img.shape)

    for idx, (i, row) in enumerate(df.iterrows()):
        prefix=f'{tfm_dir}/y-{y}'
        y=row['volume_order'] 
        fn=f'{tfm_dir}/y-{y}.nii.gz' 

        out_vol[:,int(y),:] = nib.load(fn).get_fdata()

    print('\t\tWriting 3D non-linear:', out_fn)
    nib.Nifti1Image(out_vol, hires.affine).to_filename(out_fn)

