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
from utils.utils import *


def align_2d_parallel(tfm_dir, mv_dir, resolution_itr, resolution, row):
    #Set strings for alignment parameters
    f_list = [ '1', '2', '3', '4', '6', '8', '10', '12', '14', '16', '18', '24']
    s_list = [ '0', '1', '1.5', '2', '3', '4', '5', '6', '7', '8', '9', '16']
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

    prefix = f'{tfm_dir}/y-{y}' 
    fx_fn = gen_2d_fn(prefix,'_fx')

    mv_fn = get_seg_fn(mv_dir, y, resolution, row['seg_fn'], suffix='_rsl')

    init_tfm = row['init_tfm']
    init_str = f'[{fx_fn},{mv_fn},1]'
    #if type(init_tfm) == str :
    #    init_str = init_tfm

    #if float(resolution) >= 1.0 :
    #    nl_metric = f'CC[{fx_fn},{mv_fn},1,2,Regular,1]'
    #else :
    nl_metric=f'Mattes[{fx_fn},{mv_fn},1,20,Regular,1]'

    fix_affine(fx_fn)
    fix_affine(mv_fn)

    affine_command_str = f'antsRegistration -n NearestNeighbor -v 0 -d 2 --initial-moving-transform {init_str} --write-composite-transform 1 -o [{prefix}_Affine_,{prefix}_affine_cls_rsl.nii.gz,/tmp/out_inv.nii.gz] -t Rigid[.1] -c {lin_itr_str}  -m Mattes[{fx_fn},{mv_fn},1,20,Regular,1] -s {s_str} -f {f_str}  -c {lin_itr_str} -t Similarity[.1]  -m Mattes[{fx_fn},{mv_fn},1,20,Regular,1] -s {s_str} -f {f_str} -t Affine[.1] -c {lin_itr_str} -m Mattes[{fx_fn},{mv_fn},1,20,Regular,1] -s {s_str} -f {f_str} '

    syn_command_str = f'antsRegistration -n NearestNeighbor -v 0 -d 2  --initial-moving-transform {prefix}_Affine_Composite.h5 --write-composite-transform 1 -o [{prefix}_,{prefix}_cls_rsl.nii.gz,/tmp/out_inv.nii.gz]  -t SyN[0.1] -m {nl_metric} -c {nl_itr_str} -s {s_str} -f {f_str}' # -t SyN[0.1]  -m CC[{fx_fn},{mv_fn},1,20,Regular,1] -c 100 -s {s_cc}  -f {f_cc}'
    with open(prefix+'_command.txt','w') as f : f.write(affine_command_str)
    with open(prefix+'_command.txt','w') as f : f.write(syn_command_str)
    shell(affine_command_str)
    shell(syn_command_str)
    assert os.path.exists(f'{prefix}_cls_rsl.nii.gz') , f'Error: output does not exist {prefix}_cls_rsl.nii.gz'
    return 0
    
def apply_transforms_parallel(tfm_dir, mv_dir, resolution_itr, resolution, row):
    y=row['volume_order']
    prefix=f'{tfm_dir}/y-{y}' 
    crop_rsl_fn=f'{prefix}_{resolution}mm.nii.gz'
    cls_rsl_fn = prefix+'_cls_rsl.nii.gz'
    out_fn=prefix+'_rsl.nii.gz'
    fx_fn = gen_2d_fn(prefix,'_fx')

    crop_fn = row['crop_raw_fn']
    
    img = nib.load(crop_fn)
    img_res = np.array([img.affine[0,0], img.affine[1,1] ])
    vol = img.get_fdata()
    sd = np.array( (float(resolution)/img_res) / np.pi )
    vol = gaussian_filter(vol, sd )
    nib.Nifti1Image(vol, img.affine).to_filename(crop_rsl_fn)
    
    fix_affine(crop_rsl_fn)
    fix_affine(fx_fn)
    print(f'antsApplyTransforms -v 1 -d 2 -n NearestNeighbor -i {crop_rsl_fn} -r {fx_fn} -t {prefix}_Composite.h5 -o {out_fn} ')
    shell(f'antsApplyTransforms -v 1 -d 2 -n NearestNeighbor -i {crop_rsl_fn} -r {fx_fn} -t {prefix}_Composite.h5 -o {out_fn} ')
    
    #Commented out masking because GM receptor mask not good enough at the moment.
    #rsl_img = nib.load(out_fn)
    #rsl_vol = rsl_img.get_fdata()

    #cls_rsl_img = nib.load(cls_rsl_fn)
    #cls_rsl_vol = cls_rsl_img.get_fdata()

    #rsl_vol[ cls_rsl_vol < 1  ] = 0

    #nib.Nifti1Image(rsl_vol, rsl_img.affine ).to_filename(out_fn)

    assert os.path.exists(f'{out_fn}'), 'Error apply nl 2d tfm to cropped autoradiograph'
    return 0

def receptor_2d_alignment( df, rec_fn, srv_fn, mv_dir, output_dir, resolution, resolution_itr, batch_processing=False, clobber=False): 
    df.reset_index(drop=True,inplace=True)
    df.reset_index(drop=True,inplace=True)

    tfm_dir = output_dir + os.sep + 'tfm'
    os.makedirs(tfm_dir,exist_ok=True)

    num_cores = min(14, multiprocessing.cpu_count() )
    print('num cores', num_cores)
    to_do_df = pd.DataFrame([])
    to_do_resample_df = pd.DataFrame([])
    df['tfm_affine']=['']*df.shape[0]
    for i, row in df.iterrows() :
        y = row['volume_order']
        prefix = f'{tfm_dir}/y-{y}' 
        cls_fn = prefix+'_cls_rsl.nii.gz'
        out_fn = prefix+'_rsl.nii.gz'
        tfm_fn = prefix+'_Composite.h5'
        tfm_affine_fn = prefix+'_Affine_Composite.h5'
        df['tfm'].loc[ df['volume_order'] == y ] = tfm_fn
        df['tfm_affine'].loc[ df['volume_order'] == y ] = tfm_affine_fn

        
        if not os.path.exists(tfm_fn) or not os.path.exists(cls_fn):
            to_do_df = to_do_df.append(row)

        if not os.path.exists(out_fn)  :
            to_do_resample_df = to_do_resample_df.append(row)

    if to_do_df.shape[0] > 0 :
        Parallel(n_jobs=num_cores)(delayed(align_2d_parallel)(tfm_dir, mv_dir, resolution_itr, resolution, row) for i, row in  to_do_df.iterrows()) 
        
    if to_do_resample_df.shape[0] > 0 :
        Parallel(n_jobs=num_cores)(delayed(apply_transforms_parallel)(tfm_dir, mv_dir, resolution_itr, resolution, row) for i, row in  to_do_resample_df.iterrows()) 
    
    for i, row in df.iterrows() :
        y = row['volume_order']
        prefix = f'{tfm_dir}/y-{y}' 
        out_fn = prefix+'_rsl.nii.gz'
        tfm_fn = prefix+'_Composite.h5'
        tfm_affine_fn = prefix+'_Affine_Composite.h5'
        df['tfm'].loc[ df['volume_order'] == y ] = tfm_fn
        df['tfm_affine'].loc[ df['volume_order'] == y ] = tfm_affine_fn

    #df['init_tfm'] = df['tfm']

    return df

def concatenate_sections_to_volume(df, rec_fn, output_dir, out_fn, target_str='rsl'):
    exit_flag=False
    tfm_dir=output_dir + os.sep + 'tfm'

    hires_img = nib.load(rec_fn)
    out_vol=np.zeros(hires_img.shape)
    target_name = 'nl_2d_'+target_str

    df[target_name] = [''] * df.shape[0]

    for idx, (i, row) in enumerate(df.iterrows()):
        y = row['volume_order'] 
        prefix = f'{tfm_dir}/y-{y}'
        y = row['volume_order'] 
        fn = f'{tfm_dir}/y-{y}_{target_str}.nii.gz' 

        df[target_name].loc[i] = fn
        try : 
            out_vol[:,int(y),:] = nib.load(fn).get_fdata()
        except EOFError :
            print('Error:', fn)
            os.remove(fn)
            exit_flag=True

        if exit_flag : exit(1)
    print('\t\tWriting 3D non-linear:', out_fn)
    nib.Nifti1Image(out_vol, hires_img.affine).to_filename(out_fn)
    return df 
