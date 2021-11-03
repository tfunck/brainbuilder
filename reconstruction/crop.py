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
import h5py as h5
import nibabel as nib
import pandas as pd
import multiprocessing
from scipy.ndimage import rotate
from utils.utils import safe_imread, downsample
from scipy.ndimage.filters import gaussian_filter
#from utils.mouse_click import click
from skimage.filters import threshold_otsu, threshold_li
from glob import glob
from skimage import exposure
from nibabel.processing import *
from sklearn.cluster import KMeans
from re import sub
from skimage.transform import resize
from joblib import Parallel, delayed

#matplotlib.use("TkAgg")

def get_base(fn) :
    fn = os.path.splitext(os.path.basename(fn))[0] 
    return fn

def gen_affine(row, scale,global_order_min):
    brain = row['mri']
    hemi = row['hemisphere']
    slab = row['slab']

    direction = scale[brain][hemi][str(slab)]["direction"]
    z_mm = scale[brain][hemi][str(slab)]["size"]
    xstep = 0.02
    zstep = z_mm / 4164. # pixel size in z-axis is variable depending on microscope magnification
    ystep = 0.02 
    slab_ymin = -126 + global_order_min + 0.02 * row["global_order"] 
    
    affine=np.array([[xstep, 0, 0, -90],
                            [0,  zstep, 0, -72],
                            [0, 0 , ystep, slab_ymin],
                            [0, 0, 0, 1]])
    return affine


def threshold(img,sd=1):
    img = gaussian_filter(img, sd)
    out = np.zeros(img.shape)
    out[ img > threshold_li(img[img>0]) ] = 1
    return out

def process_image(img, mask_fn, row, scale, pad, affine): 
    mask_lores = imageio.imread(mask_fn)
    mask = resize(mask_lores, [img.shape[0],img.shape[1]],order=0).astype(int)
    #plt.subplot(4,1,1)
    #plt.imshow(img)
    #plt.imshow(mask,alpha=0.5)
    brain = row['mri']
    hemi = row['hemisphere']
    slab = row['slab']

    p = int(pad/2)
    direction = scale[brain][hemi][str(slab)]["direction"]

    # apply mask to image
    img[ mask < np.max(mask)*0.5 ] = 0
    img = img.reshape(img.shape[0], img.shape[1])

    #plt.subplot(4,1,2)
    #plt.imshow(img)
    #plt.imshow(mask,alpha=0.5)

    if direction == "rostral_to_caudal": 
        img = np.flip( img, 1 )
    img = np.flip(img, 0)

    #plt.subplot(4,1,3)
    #plt.imshow(img)
   
    ## pad image in case it needs to be rotated
    # all images are padded regardless of whether they are rotated because that way they all 
    # have the same image dimensions
    img_pad = np.zeros([img.shape[0]+pad,img.shape[1]+pad])

    img_pad[p:p+img.shape[0],p:p+img.shape[1]] =img
    try :
        section_rotation =  row['rotate']
        if section_rotation != 0 : img_pad = rotate(img_pad, section_rotation, reshape=False)
    except IndexError : 
        pass

    #plt.subplot(4,1,4)
    #plt.imshow(img_pad)
    #plt.savefig(os.path.basename(mask_fn))

    return img_pad

def crop_parallel(row, mask_dir, scale,global_order_min, pad = 1000, clobber=True ):

   
    fn = row['lin_fn']
    base = row['lin_base_fn'] # os.path.splitext( os.path.basename(fn) )[0]

    crop_fn = row['crop_fn'] #'{}/{}.nii.gz'.format(out_dir , base)

    if not os.path.exists(crop_fn) or clobber : 
        print('\t crop_fn', crop_fn) 

        # identify mask image filename
        mask_fn=glob(f'{mask_dir}/combined_final/mask/{base}*.png')
        
        if len(mask_fn) > 0 : 
            mask_fn = mask_fn[0]
        else : 
            print('Skipping', fn,f'{mask_dir}/combined_final/mask/{base}*.png' )
            mask_fn = f'{mask_dir}/combined_final/mask/{base}.png'
            
        print('\t\tMask fn:', mask_fn)
        
        # load mask image 
        img = imageio.imread(fn)

        affine = gen_affine(row, scale, global_order_min)
        
        img = process_image(img, mask_fn, row, scale, pad, affine)
        nib.Nifti1Image(img, affine ).to_filename(crop_fn)
    
    seg_fn = row['seg_fn']
    if not os.path.exists(seg_fn) or clobber :
        print('\t seg_fn', seg_fn) 
        img_hd = nib.load(crop_fn)
        img = img_hd.get_fdata()
        img = threshold(img)
        nib.Nifti1Image(img, img_hd.affine ).to_filename(seg_fn)

def crop(src_dir, mask_dir, out_dir, df, scale_factors_json, remote=False,clobber=False):
    '''take raw linearized images and crop them'''
    df = df.loc[ (df['hemisphere'] == 'R') & (df['mri'] == 'MR1' )  ] #FIXME, will need to be removed

    with open(scale_factors_json) as f : scale=json.load(f)
    def fn_check(fn_list) : return [os.path.exists(fn) for fn in fn_list ]
    crop_check = fn_check(df['crop_fn'])
    seg_check = fn_check(df['seg_fn'])

    if False in crop_check or False in seg_check : 
        pass
    else : return 0

    os.makedirs(out_dir,exist_ok=True)
    
    global_order_min = df["global_order"].min()
    print('Cropping')
    Parallel(n_jobs=14)(delayed(crop_parallel)(row, mask_dir, scale, global_order_min) for i, row in  df.iterrows()) 
    

            
    return 0


