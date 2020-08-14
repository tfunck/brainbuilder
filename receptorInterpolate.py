import numpy as np
import nibabel as nib
import pandas as pd
import sys
import os
import json
import re
import matplotlib.pyplot as plt
import pandas as pd
import time
import stripy as stripy
#import psutil
import vast.surface_tools as surface_tools
import vast.surface_tools as io_mesh
from glob import glob
from ants import  apply_transforms, from_numpy
from ants import  from_numpy
from ANTs import ANTs
from ants import image_read, registration
from utils.utils import splitext, shell


def receptorInterpolate(df_fn, rec_fn, srv_fn, output_dir, out_fn, clobber=False): 
    if not os.path.exists(output_dir) :
        os.makedirs(output_dir)

    df = pd.read_csv(df_fn)

    print("\t\tNonlinear alignment of coronal sections for ligand",ligand,"for slab",int(slab))
    tfm_dir=output_dir + os.sep + 'tfm'
    if not os.path.exists(tfm_dir) : os.makedirs(tfm_dir)
    
    print("srv_fn:", srv_fn)
    srv_img = nib.load(srv_fn)
    srv = srv_img.get_data()
    
    rec_img = nib.load(rec_fn)
    rec = rec_img.get_data()

    out = np.zeros_like(rec.shape)

    ymax = srv.shape[1]
    
    for row in df.iterrows():
        y=row['volume_order'].values[0]
        init_tfm_fn=row['init_tfm'].values[0]

        prefix=f'{tfm_dir}/slab-{slab}_ligand-{ligand}_y-{_y}' 
        if not os.path.exists(prefix) : os.makedirs(prefix)
        
        if (not os.path.exists(prefix+"1Warp.nii.gz") and np.sum(rec[:,y,:]) != 0 ) or clobber : 
            srv_slice = from_numpy( srv[ :, int(y), : ] )
            rec_slice = from_numpy( rec[ :, int(y), : ] )
            reg = registration(fixed=srv_slice, moving=rec_slice, init_transform=init_tfm, type_of_transform=tfm_type_2d,reg_iterations=(1000,750,500), aff_metric='Mattes', syn_metic='Mattes', outprefix=prefix  )
            print(reg)
            out[:,y,:] = reg['warpedmovout'].numpy()
            del reg

    nib.Nifti1Image(out, rec_img.affine).to_filename(out_fn)
