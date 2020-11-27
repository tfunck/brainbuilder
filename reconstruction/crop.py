import matplotlib
import shutil
import argparse
import os
import json
import h5py as h5
import numpy as np
import pandas as pd
import numpy as np
import imageio
import gc 
import matplotlib.pyplot as plt
from scipy.ndimage import rotate
from utils.utils import safe_imread, downsample
#from utils.mouse_click import click
from glob import glob
import h5py as h5
import nibabel as nib
import pandas as pd
from skimage import exposure
from nibabel.processing import *
from sklearn.cluster import KMeans
from re import sub
#matplotlib.use("TkAgg")

def get_base(fn) :
    fn = os.path.splitext(os.path.basename(fn))[0] 
    return fn


def crop(src_dir, out_dir, df, remote=False,clobber=False):
    pad = 50
    p = int(pad/2)
    
    if not False in df['crop_fn'].apply(lambda x: os.path.exists(x) ) : return 0
    if not os.path.exists(out_dir) : os.makedirs(out_dir)
    files = glob('{}/*'.format(src_dir))
    basenames = [ get_base(fn) for fn in files ]
    n=len(files)

    if not remote :
        for index, (fn, base) in enumerate(zip(files, basenames)) : 

            if 'MR1' not in base : continue #WILL NEED TO REMOVE
            crop_fn = '{}/{}.nii.gz'.format(out_dir , base)
            if not os.path.exists(crop_fn) or clobber :

                mask_fn=glob(f'crop/combined_final/mask/{base}*.png')
                #print(mask_fn)
                if len(mask_fn) > 0 : 
                    mask_fn = mask_fn[0]
                else : 
                    print('Skipping', mask_fn)
                    mask_fn = f'crop/combined_final/mask/{base}.png'
                
                
                mask = imageio.imread(mask_fn)
                img = imageio.imread(fn)
                img[ mask != np.max(mask) ] = 0
                img = img.reshape(img.shape[0], img.shape[1])
                section=df['lin_fn'] == os.path.splitext(os.path.basename(fn))[0]
               
                img_pad = np.zeros([img.shape[0]+pad,img.shape[1]+pad])

                img_pad[p:p+img.shape[0],p:p+img.shape[1]] =img
                img = img_pad
                try :
                    section_rotation =  df['rotate'].loc[  section ].values[0] 
                    if section_rotation != 0 : img = rotate(img, section_rotation, reshape=False)
                except IndexError : 
                    pass

                nib.Nifti1Image(img, [[0.2,0,0,0],[0,0.2,0,0],[0,0,0.2,0],[0,0,0,1]]).to_filename(crop_fn)
    return 0





