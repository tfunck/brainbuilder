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


def receptorInterpolate( df_fn, srv_fn, output_dir, out_fn, direction='rostral_to_caudal', clobber=False): 
    if not os.path.exists(output_dir) :
        os.makedirs(output_dir)


    df = pd.read_csv(df_fn)

    print("\t\tNonlinear alignment of coronal sections")
    tfm_dir=output_dir + os.sep + 'tfm'
    if not os.path.exists(tfm_dir) : os.makedirs(tfm_dir)
    
    print("srv_fn:", srv_fn)
    srv_img = nib.load(srv_fn)
    srv = srv_img.get_data()
    out = np.zeros_like(srv)

    ymax = srv.shape[1]
    for idx, df_ligand in df.groupby(['ligand']) :
        print(idx)
        for i, row in df_ligand.iterrows() :
            y=row['volume_order'] 
            prefix=f'{tfm_dir}/y-{y}' 
            if not os.path.exists(prefix) : os.makedirs(prefix)
            
            if ( not os.path.exists(prefix+"1Warp.nii.gz") or not os.path.exists(prefix+'_SyN.nii.gz'))  or clobber : 
                srv_np = srv[ :, int(y), : ]
                rec_img=nib.load( row['filename_rsl']  )
                rec_np = rec_img.get_fdata()
                if direction == "rostral_to_caudal":
                    rec_np = np.flip(rec_np, 1)
                rec_np = np.flip(rec_np, 0)
                srv_slice = from_numpy( srv_np )
                rec_slice = from_numpy( rec_np )

                if not os.path.exists(prefix+"1Warp.nii.gz")  or clobber : 
                    reg = registration(fixed=srv_slice, moving=rec_slice, init_transforms=[row['init_tfm']], type_of_transform='SyNAggro',reg_iterations=(1000,750,500), aff_metric='Mattes', syn_metic='Mattes', outprefix=prefix, dimensions=2  )
                    warpedmovout = reg['warpedmovout'].numpy()
                    nib.Nifti1Image(warpedmovout, rec_img.affine).to_filename(prefix+'_SyN.nii.gz')
                elif not os.path.exists(prefix+'_SyN.nii.gz') or clobber :
                    warpedmovout=apply_transforms(srv_slice, rec_slice, [ prefix+"1Warp.nii.gz", prefix+"0GenericAffine.mat" ] ).numpy()
                    nib.Nifti1Image(warpedmovout, rec_img.affine).to_filename(prefix+'_SyN.nii.gz')
            
            warpedmovout = nib.load(prefix+'_SyN.nii.gz').get_fdata()

            #print(warpedmovout.shape)
            #plt.subplot(2,1,1)
            #plt.imshow(warpedmovout[:,:,0,0,0])
            #plt.subplot(2,1,2)
            #plt.imshow(warpedmovout[:,:,0,0,1])
            #plt.show()
            out[:,y,:] =  warpedmovout


        nib.Nifti1Image(out, srv_img.affine).to_filename(out_fn)

if __name__ == '__main__':
    receptorInterpolate(argv[1],argv[2],argv[3],argv[4])
