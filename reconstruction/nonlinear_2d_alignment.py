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
import multiprocessing
from nibabel.processing import resample_to_output
from joblib import Parallel, delayed
from section_2d import section_2d
from sys import argv
from glob import glob
from utils.ANTs import ANTs
from utils.utils import *




def align_2d_parallel(tfm_dir, mv_dir, resolution_itr, resolution, row):
    #Set strings for alignment parameters
    f_list = [ '1', '2', '4', '8', '16', '24']
    s_list = [ '0', '1', '2', '4', '8', '16']
    max_itr = min(resolution_itr, len(f_list))
    f_list = f_list[0:(max_itr+1)]
    s_list = s_list[0:(max_itr+1)]
    s_list.reverse()
    f_list.reverse()

    base_lin_itr= 1000
    base_nl_itr = 200
    max_lin_itr = base_lin_itr * (resolution_itr+1)
    max_nl_itr  = base_nl_itr * (resolution_itr+1)
    lin_step = -base_lin_itr 
    nl_step  = -base_nl_itr 

    lin_itr_str='x'.join([str(base_lin_itr *(max_itr+1) + i * lin_step) for i in range(max_itr+1)])
    nl_itr_str='x'.join( [str(base_nl_itr *(max_itr+1) + i * nl_step) for i in range(max_itr+1)])

    f_str='x'.join( [ f_list[i] for i in range(max_itr+1)])
    s_str='x'.join( [ s_list[i] for i in range(max_itr+1)]) + 'vox'

    f_cc = f_list[-1]
    s_cc = s_list[-1]

    s_str_final = s_str.split(',')[-1]
    f_str_final = f_str.split(',')[-1]
    
    n0 = len(lin_itr_str.split('x'))
    n1 = len(f_str.split('x'))
    n2 = len(s_str.split('x'))-1
    n3 = len(nl_itr_str.split('x'))
    assert n0==n1==n2==n3 , "Error: Incorrect lengths for ANTs parameters"

    y=row['volume_order']

    prefix=f'{tfm_dir}/y-{y}' 
    fx_fn = gen_2d_fn(prefix,'_fx')

    #mv_fn_list = glob('{}/y-*{}'.format(mv_dir, os.path.basename(row['seg_fn'])) )
    #if len(mv_fn_list) == 0 : 
    #    print('Could not find file for ', row['seg_fn'] )
    #    exit(1)
    #mv_fn = mv_fn_list[0]

    mv_fn = get_seg_fn(mv_dir, y, resolution, row['seg_fn'], suffix='_rsl')

    print('\t\t',y)
    
    init_tfm = row['init_tfm']
    init_str = f'[{fx_fn},{mv_fn},1]'
    if type(init_tfm) == str :
        init_str = init_tfm

    command_str = f'time antsRegistration -v 0 -d 2 --write-composite-transform 1  --initial-moving-transform {init_str} -o [{prefix}_,{prefix}_mv_rsl.nii.gz,/tmp/out_inv.nii.gz] -t Rigid[.1] -c {lin_itr_str}  -m Mattes[{fx_fn},{mv_fn},1,20,Regular,1] -s {s_str} -f {f_str}  -c {lin_itr_str} -t Similarity[.1]  -m Mattes[{fx_fn},{mv_fn},1,20,Regular,1] -s {s_str} -f {f_str} -t Affine[.1] -c {lin_itr_str} -m Mattes[{fx_fn},{mv_fn},1,20,Regular,1] -s {s_str} -f {f_str} -t SyN[0.1] -m Mattes[{fx_fn},{mv_fn},1,20,Regular,1] -c {nl_itr_str} -s {s_str} -f {f_str}  -t SyN[0.1]  -m CC[{fx_fn},{mv_fn},1,20,Regular,1] -c 20 -s {s_cc}  -f {f_cc} '

    with open(prefix+'_command.txt','w') as f : f.write(command_str)

    shell(command_str)
    #init_tfm=row['init_tfm']
    #if init_tfm != None :
    #    print(f'antsApplyTransforms -v 1 -d 2 -i {mv_fn} -r {mv_fn}  -t {prefix}_nl_Composite.h5 -t {init_tfm} -o [{prefix}_Composite.h5]')
    #    shell(f'antsApplyTransforms -v 1 -d 2 -i {mv_fn} -r {mv_fn}  -t {prefix}_nl_Composite.h5 -t {init_tfm} -o [{prefix}_Composite.h5]')
    #    assert os.path.exists(f'{prefix}_Composite.h5'), f'Error concatenating initial rigid tfm with nl tfm:\n -t {prefix}_nl_Composite.h5 \n-t {init_tfm} \n -o [{prefix}_Composite.h5]' 
    #else :
    #    shutil.copy(f'{prefix}_nl_Composite.h5', f'{prefix}_Composite.h5')
    return 0
    
def apply_transforms_parallel(tfm_dir, mv_dir, resolution_itr, row):
    y=row['volume_order']
    prefix=f'{tfm_dir}/y-{y}' 
    out_fn=prefix+'_rsl.nii.gz'
    fx_fn = gen_2d_fn(prefix,'_fx')

    crop_fn = row['crop_fn']
    shell(f'antsApplyTransforms -v 1 -d 2  -i {crop_fn} -r {fx_fn} -t {prefix}_Composite.h5 -o {out_fn} ')
    assert os.path.exists(f'{out_fn}'), 'Error apply nl 2d tfm to cropped autoradiograph'
    print('\nDone.\n')
    return 0

def receptor_2d_alignment( df, rec_fn, srv_fn, mv_dir, output_dir, resolution, resolution_itr, batch_processing=False, clobber=False): 
    df.reset_index(drop=True,inplace=True)
    df.reset_index(drop=True,inplace=True)

    tfm_dir = output_dir + os.sep + 'tfm'
    os.makedirs(tfm_dir,exist_ok=True)

    num_cores = min(14, multiprocessing.cpu_count() )
    print('num cores', num_cores)
    print(df.shape, df.shape)
    to_do_df = pd.DataFrame([])
    to_do_resample_df = pd.DataFrame([])
    for i, row in df.iterrows() :
        y = row['volume_order']
        prefix = f'{tfm_dir}/y-{y}' 
        out_fn = prefix+'_rsl.nii.gz'
        tfm_fn = prefix+'_Composite.h5'
        df['tfm'].loc[ df['volume_order'] == df['volume_order'] ] = tfm_fn

        try :
            init_tfm = df['init_tfm']
        except IndexError :
            init_tfm=None
        
        if not os.path.exists(tfm_fn) :
            to_do_df = to_do_df.append(row)
        
        if not os.path.exists(out_fn) :
            to_do_resample_df = to_do_resample_df.append(row)
    
    if to_do_df.shape[0] > 0 :
        Parallel(n_jobs=num_cores)(delayed(align_2d_parallel)(tfm_dir, mv_dir, resolution_itr, resolution, row) for i, row in  to_do_df.iterrows()) 
        
    if to_do_resample_df.shape[0] > 0 :
        Parallel(n_jobs=num_cores)(delayed(apply_transforms_parallel)(tfm_dir, mv_dir, resolution_itr, row) for i, row in  to_do_resample_df.iterrows()) 

    return df

def concatenate_sections_to_volume(df, rec_fn, output_dir, out_fn):
    exit_flag=False
    tfm_dir=output_dir + os.sep + 'tfm'

    hires_img = nib.load(rec_fn)
    out_vol=np.zeros(hires_img.shape)

    for idx, (i, row) in enumerate(df.iterrows()):
        y = row['volume_order'] 
        prefix = f'{tfm_dir}/y-{y}'
        y = row['volume_order'] 
        fn = f'{tfm_dir}/y-{y}_rsl.nii.gz' 
        try : 
            out_vol[:,int(y),:] = nib.load(fn).get_fdata()
        except EOFError :
            print('Error:', fn)
            os.remove(fn)
            exit_flag=True

        if exit_flag : exit(1)
    print('\t\tWriting 3D non-linear:', out_fn)
    nib.Nifti1Image(out_vol, hires_img.affine).to_filename(out_fn)

