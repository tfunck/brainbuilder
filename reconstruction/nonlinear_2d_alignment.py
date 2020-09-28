import numpy as np
import nibabel as nib
import pandas as pd
import sys
import os
import json
import re
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import time
import stripy as stripy
import shutil
from sys import argv
from glob import glob
from ants import  apply_transforms, from_numpy, from_numpy, image_read, registration
from utils.ANTs import ANTs
from utils.utils import splitext, shell, w2v , v2w
from scipy.ndimage.measurements import center_of_mass

#matplotlib.use('TKAgg')


def receptor_2d_alignment( df, rec_fn, rec_rsl_fn, srv_fn, output_dir,  out_3d_fn, resolution, resolution_itr, direction='rostral_to_caudal', clobber=False): 
    if not os.path.exists(output_dir) :
        os.makedirs(output_dir)

    temp_hires_fn=re.sub('.nii.gz','_hires_temp.nii.gz', out_3d_fn) 

    tfm_dir=output_dir + os.sep + 'tfm'
    if not os.path.exists(tfm_dir) : os.makedirs(tfm_dir)

    srv_img = nib.load(srv_fn)
    srv = srv_img.get_fdata()
    
    rec_hires_img = nib.load(rec_fn)
    rec_hires_vol = rec_hires_img.get_fdata()
    xstep = rec_hires_img.affine[0,0]
    zstep = rec_hires_img.affine[2,2]

    out_hires = np.zeros_like(rec_hires_vol)
    
    avg_step = (xstep+zstep)/2 
    resolution_level = np.ceil(np.log2(3/avg_step )+ 1 ).astype(int)

    #Set strings for alignment parameters
    lin_itr_str='x'.join([str(i) for i in range(resolution_level*100,0,-100)])
    nl_itr_str='x'.join([str(i) for i in range(resolution_level*20*(resolution_itr+1),0,-20*(resolution_itr+1))])

    f_str='x'.join([ str(i) for i in range(resolution_level,0,-1)])
    f = lambda x : x/2 if x > 1  else 0
    s_list = map(f,  range(resolution_level,0,-1) ) 
    s_str='x'.join( [str(i) for i in s_list] ) + 'vox'

    fx_fn='/tmp/fixed.nii.gz'

    aff_hires = np.array([[xstep,  0, 0, 0],
                        [0, zstep, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])
    
    for idx, (i, row) in enumerate(df.iterrows()):
        y=row['volume_order'] 
        prefix=f'{tfm_dir}/y-{y}' 
        
        out_hires_fn=f'{prefix}.nii.gz'
        out_inv_fn=f'{prefix}_inv.nii.gz'
        tfm_fn=f'{prefix}_Composite.h5'
        if not os.path.exists(tfm_fn) or not os.path.exists(out_hires_fn) or clobber : 
            # Create 2d rec section
            mv_rsl_fn=f'{prefix}_mv_0.2mm.nii.gz'
            rec_np = rec_hires_vol[:,int(y),:]
            img = nib.Nifti1Image(rec_np, aff_hires )
            img.to_filename(mv_rsl_fn)

            # Create 2D srv section
            srv_np = srv[ :, int(y), : ]
            nib.Nifti1Image(srv_np, aff_hires).to_filename(fx_fn)

            # calculate 2d nonlinear transform between receptor slice and mri gm srv volume
            if not os.path.exists(tfm_fn)  or clobber : 
                print('\t\t',y)

                stout, sterr, errorcode = shell(f'antsRegistration -v 0 -d 2 --write-composite-transform 1  --initial-moving-transform [{fx_fn},{mv_rsl_fn},1] -o [{prefix}_,{out_hires_fn},/tmp/out_inv.nii.gz] -t Similarity[.1] -c {lin_itr_str}  -m Mattes[{fx_fn},{mv_rsl_fn},1,20,Regular,1] -s {s_str} -f {f_str}   -t Affine[.1]   -c {lin_itr_str}  -m Mattes[{fx_fn},{mv_rsl_fn},1,20,Regular,1] -s {s_str} -f {f_str} -t SyN[0.1] -m Mattes[{fx_fn},{mv_rsl_fn},1,20,Regular,1] -c [{nl_itr_str}] -s {s_str} -f {f_str}', exit_on_failure=True,verbose=False)

                #Create QC image
                out_2d_np = nib.load(out_hires_fn).get_fdata()
                plt.imshow(srv_np,cmap='gray',origin='lower')
                plt.imshow(out_2d_np,cmap='hot',alpha=0.45,origin='lower')
                plt.tight_layout()
                plt.savefig(f'{prefix}qc.png',fc='black')
                plt.clf()
                plt.cla()

        img_hires = nib.load(out_hires_fn)
        warpedmovout_hires = img_hires.get_fdata()
        out_hires[:,int(y),:] = warpedmovout_hires
        del warpedmovout_hires

        if idx % 50 == 0 :
            print(f'\r{idx} writing temp file to {temp_hires_fn}')
            nib.Nifti1Image(out_hires, rec_hires_img.affine).to_filename(temp_hires_fn)

    print('\t\tWriting 3D non-linear:', out_3d_fn)
    nib.Nifti1Image(out_hires, srv_img.affine).to_filename(out_3d_fn)

