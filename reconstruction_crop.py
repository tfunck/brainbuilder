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
from utils.mouse_click import click
from glob import glob
import h5py as h5
import tensorflow as tf
import nibabel as nib
import pandas as pd
from skimage import exposure
from nibabel.processing import *
from sklearn.cluster import KMeans
from re import sub
matplotlib.use("TkAgg")

def get_base(fn) :
    fn = os.path.splitext(os.path.basename(fn))[0] 
    #if fn[-2:] == '#L':
    #    fn = fn[0:-2]
    return fn

if __name__ == '__main__' :
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--clobber', dest='clobber', action='store_true', default=False, help='Clobber')
    parser.add_argument('-s', dest='src_dir', default='crop/combined_final/mask/', help='source dir')
    parser.add_argument('-o', dest='out_dir', default='reconstruction_output/crop/', help='source dir')
    parser.add_argument('-d', dest='auto_info_fn', default='autoradiograph_info.csv', help='source dir')
    pad = 50
    p = int(pad/2)
    args = parser.parse_args()

    df = pd.read_csv(args.auto_info_fn)
    df['lin_fn'] = df['lin_fn'].apply( lambda x: os.path.splitext(os.path.basename(x))[0] ) 

    if not os.path.exists(args.out_dir) : os.makedirs(args.out_dir)
    files = glob('{}/*'.format(args.src_dir))
    basenames = [ get_base(fn) for fn in files ]
    n=len(files)
    for index, (fn, base) in enumerate(zip(files, basenames)) : 

        if 'MR1' not in base : continue
        if index % 100 == 0 : print(index/n,end='\r')
        crop_fn = '{}/{}.nii.gz'.format(args.out_dir , base)

        if not os.path.exists(crop_fn) or args.clobber :
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
            #imageio.imsave(crop_fn, img)





