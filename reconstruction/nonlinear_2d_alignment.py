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

def gen_2d_fn(prefix,suffix,ext='.nii.gz'):
    return f'{prefix}{suffix}{ext}'

def save_sections(file_list, vol, aff) :
    for fn, y in file_list:
        # Create 2D srv section
        nib.Nifti1Image(vol[ :, int(y), : ] , aff).to_filename(fn)

def get_to_do_list(df,out_dir,str_var,ext='.nii.gz'):
    to_do_list=[]
    for idx, (i, row) in enumerate(df.iterrows()):
        y=row['volume_order'] 
        prefix=f'{out_dir}/y-{y}' 
        fn=gen_2d_fn(prefix,str_var,ext=ext)
        if not os.path.exists(fn) : to_do_list.append( [fn, y])
    return to_do_list

def create_2d_sections( df, rec_fn, srv_fn, output_dir,clobber=False) :
    fx_to_do=[]
    mv_to_do=[]
    
    tfm_dir=output_dir + os.sep + 'tfm'
    os.makedirs(tfm_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    fx_to_do = get_to_do_list(df, tfm_dir, '_fx_0.2mm') 
    mv_to_do = get_to_do_list(df, tfm_dir, '_mv_0.2mm') 

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

def receptor_2d_alignment( df, rec_fn, srv_fn, output_dir, resolution, resolution_itr, batch_processing=False, clobber=False): 
    
    tfm_dir=output_dir + os.sep + 'tfm'

    out_to_do=get_to_do_list(df, tfm_dir, '')
    tfm_to_do=get_to_do_list(df, tfm_dir, '_Composite', ext='.h5')

    if len(out_to_do) != len(tfm_to_do) :
        print('Error: number of output 2d nii.gz does not match number of tfm .h5 files, {} vs {}', len(out_to_do), len(tfm_to_do))
        exit(1)

    if len(out_to_do + tfm_to_do) != 0 :


        #don't remember what resolution_level does, so setting to 3 to cancel out it's effect
        #This is done because you always want to start the registration from 5mm and end at 250um
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


        for (out_fn, y), (tfm_fn,y1) in zip(out_to_do,tfm_to_do):
            if y != y1 :
                print('Error: mismatched y for {} {} and {} {}',y, out_fn, y1, tfm_fn)
            prefix=f'{tfm_dir}/y-{y}' 
            fx_fn = gen_2d_fn(prefix,'_fx_0.2mm')
            mv_fn = gen_2d_fn(prefix,'_mv_0.2mm')

            print('\t\t',y)
            args=[ prefix, mv_fn, fx_fn, out_fn, s_str, f_str, lin_itr_str, nl_itr_str]
            batch_fn=None
            if batch_processing :
                batch_fn=f'{tfm_dir}/batch_{y}.sh'
            args.append(batch_fn)

            section_2d(*args)

        if batch_processing == True : 
            print('\nCompleted all processing up to nl 2d alignment. This should be run remotely with with --nl-2d-only argument\n')
            exit(0)
def concatenate_sections_to_volume(df, rec_fn, output_dir, out_fn):
    
    tfm_dir=output_dir + os.sep + 'tfm'

    hires_img = nib.load(rec_fn)
    out_vol=np.zeros(hires_img.shape)

    for idx, (i, row) in enumerate(df.iterrows()):
        y=row['volume_order'] 
        prefix=f'{tfm_dir}/y-{y}'
        y=row['volume_order'] 
        fn=f'{tfm_dir}/y-{y}.nii.gz' 

        out_vol[:,int(y),:] = nib.load(fn).get_fdata()

    print('\t\tWriting 3D non-linear:', out_fn)
    nib.Nifti1Image(out_vol, hires_img.affine).to_filename(out_fn)

