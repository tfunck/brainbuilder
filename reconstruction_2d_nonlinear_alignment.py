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
from sys import argv
from glob import glob
from ants import  apply_transforms, from_numpy
from ants import  from_numpy
from ANTs import ANTs
from ants import image_read, registration
from utils.utils import splitext, shell
matplotlib.use('TKAgg')


def receptor_2d_alignment( df_fn, srv_fn, output_dir,  out_3d_fn, direction='rostral_to_caudal', clobber=False): 
    if not os.path.exists(output_dir) :
        os.makedirs(output_dir)
    temp_fn=re.sub('.nii.gz','_temp.nii.gz', out_3d_fn) 
    df = pd.read_csv(df_fn)

    print("\t\tNonlinear alignment of coronal sections")
    tfm_dir=output_dir + os.sep + 'tfm'
    if not os.path.exists(tfm_dir) : os.makedirs(tfm_dir)
    
    print("srv_fn:", srv_fn)
    srv_img = nib.load(srv_fn)
    srv = srv_img.get_data()
    out = np.zeros_like(srv)

    ymax = srv.shape[1]

    for idx, (i, row) in enumerate(df.iterrows()):
        #print(row)
        y=row['volume_order'] 
        prefix=f'{tfm_dir}/y-{y}' 
        
        out_fn=f'{prefix}.nii.gz'
        out_inv_fn=f'{prefix}_inv.nii.gz'
        tfm_fn=f'{prefix}_Composite.h5'
        print(idx)
        print('Out fn', out_fn, os.path.exists(out_fn) )
        print('TFM fn', tfm_fn, os.path.exists(tfm_fn) )
        if not os.path.exists(tfm_fn) or not os.path.exists(out_fn) or clobber : 
            # Create 2D receptor section
            rec_img = nib.load( row['filename_rsl']  )
            rec_np = rec_img.get_fdata()
            if direction == "rostral_to_caudal":
                rec_np = np.flip(rec_np, 1)
            rec_np = np.flip(rec_np, 0)
            nib.Nifti1Image(rec_np, rec_img.affine).to_filename('/tmp/moving.nii.gz')

            # Create 2D src section
            srv_np = srv[ :, int(y), : ]
            nib.Nifti1Image(srv_np, rec_img.affine).to_filename('/tmp/fixed.nii.gz')

            if not os.path.exists(tfm_fn)  or clobber : 
                #init_tfm=row['init_tfm']
                #if type(init_tfm) == str :
                #    init_moving=f'--initial-moving-transform [ {init_tfm}]'
                #else : init_moving=''

                shell(f'antsRegistration -v 0 -d 2 --write-composite-transform 1  --initial-moving-transform [/tmp/fixed.nii.gz,/tmp/moving.nii.gz,1] -o [{prefix}_,{out_fn},{out_inv_fn}]   -t Rigid[.1] -m Mattes[/tmp/fixed.nii.gz,/tmp/moving.nii.gz,1,20,Regular,1] -c [500] -s 0vox -f 1  -t Affine[.1]   -c [250]  -m Mattes[/tmp/fixed.nii.gz,/tmp/moving.nii.gz,1,20,Regular,1] -s 0vox -f 1 -t SyN[.1] -m Mattes[/tmp/fixed.nii.gz,/tmp/moving.nii.gz,1,20,Regular,1] -c [100] -s 0vox -f 1 ') 
            elif not os.path.exists(out_fn) or clobber :
                shell(f'antsApplyTransforms   -d 2 -t {tfm_fn} -t {init_tfm} -i {rec_slice} -r {rec_slice} -o [{prefix}_,{out_fn},{out_inv_fn}]') 
        else : 
            print('Reading', out_fn)
        img = nib.load(out_fn)
        warpedmovout = img.get_fdata()
        out[:,y,:] = warpedmovout
        if idx % 50 == 0 :
            print('Writing intermediate file', temp_fn)
            nib.Nifti1Image(out, srv_img.affine).to_filename(temp_fn)

    print(out_3d_fn)
    nib.Nifti1Image(out, srv_img.affine).to_filename(out_3d_fn)

if __name__ == '__main__':
    for i in range(len(argv)) : print(i,argv[i])
    receptor_2d_alignment(argv[1],argv[2],argv[3],argv[4],argv[5])
