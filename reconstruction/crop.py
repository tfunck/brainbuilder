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
from skimage.measure import label
from sklearn.mixture import GaussianMixture
from scipy.stats import entropy, kurtosis, skew
from scipy.interpolate import griddata

#matplotlib.use("TkAgg")

def pseudo_classify_autoradiograph(autoradiograph_fn, mask_fn, out_fn, y, resolution):

    n=3
    #load autoradiograph
    img = nib.load(autoradiograph_fn)
    vol = img.get_fdata()

    voxel_size = img.affine[0,0] * img.affine[1,1] * img.affine[2,2]

    #load mask
    mask_img = nib.load(mask_fn)
    mask_vol = mask_img.get_fdata()

    #blur to 200um
    vol = gaussian_filter(vol, (0.5/0.02)/np.pi)

    xdim, zdim = vol.shape

    # Define a mesh grid over which we can interpolate label values
    zz, xx = np.meshgrid(range(zdim), range(xdim))

    xx_valid = xx[mask_vol==1]
    zz_valid = zz[mask_vol==1]

    locations = np.vstack( [ xx_valid[0:n * 3], zz_valid[0:n*3] ]).T
    
    # set size of sliding kernel 
    mm=3
    kernel_dim = int(1/0.02) * mm
    kernel_pad = int( (kernel_dim+1) / 2 )

    features=[]
    step=int(kernel_dim*1.5)
    for x,z in locations:
        if mask_vol[x,z] > 0 :
            i0=max(0,x-kernel_pad)
            j0=max(0,z-kernel_pad)
            
            i1=min(xdim,x+kernel_pad)
            j1=min(zdim,z+kernel_pad)

            section = vol[i0:i1, j0:j1 ]
            mask_section=mask_vol[i0:i1, j0:j1] 

            #plt.imshow(section*mask_section); plt.show()
            section = section[mask_section>0]

            m=np.mean(section)
            s=np.std(section)
            k=kurtosis(section.reshape(-1,1))[0]
            sk=skew(section.reshape(-1,1))[0]
            e=entropy( section.reshape(-1,1) )[0]
            vector = [x,z,m,s,k,sk,e]
            features.append(vector)
        else :
            print('Error: point must be in mask')
            exit(0)
    #normalize columns of feature metrics
    features_std = np.std(features, axis=0)
    features_std[ features_std < 0.001 ] = 1
    features = (np.array(features) - np.mean(features, axis=0)) / features_std
    #print('\t\tn features =', features.shape[0], autoradiograph_fn)
    
    # Use a gaussian mixture model to classify the features into n class labels
    if features.shape[0] < n :
        print(f'Warning: When creating pseudo-classified image, {features.shape[0]} features found but at least {n} required. Occured for image:\n {autoradiograph_fn}')
        return mask_vol

    labels = GaussianMixture(n_components=n).fit_predict(features) +1
    
    # Apply nearest neighbour interpolation
    out = griddata(locations, labels, (xx, zz), method='nearest')
    out[ mask_vol == 0 ] = 0
    out = label(out)
    
    #Get rid of labeled regions that are less than the minimum label size
    out_unique = np.unique(out)[1:]
    out_label_sizes =np.bincount(out.reshape(-1,))[1:].astype(float)
    out_label_sizes = out_label_sizes[ out_label_sizes > 0 ]
    out_label_sizes *= voxel_size

    assert len(out_label_sizes) == len(out_unique) , 'Error: unique labels doesnt equal size of sums'
  
    for l, t in zip(out_unique, out_label_sizes) :
        if t < resolution :
            #print('-->',np.sum(out==l), np.sum(out==l)*voxel_size, t, voxel_size)
            out[ out == l ] = 0


    #print('Hello!', np.min(np.bincount(out.reshape(-1,)).astype(float) * voxel_size) ) 
    #exit(0) 
    offset = 100000 if np.max(out_unique) < 100 else 1000000
    out[ out > 0 ] = y + offset * out[ out > 0 ]

    if np.sum(out) == 0 : out = mask_vol
    # save classified image as nifti
    nib.Nifti1Image(out.astype(np.uint32), img.affine).to_filename(out_fn)


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

def crop_parallel(row, mask_dir, scale,global_order_min, resolution, pad = 1000, clobber=True ):
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

    pseudo_cls_fn = row['pseudo_cls_fn']
    if not os.path.exists(pseudo_cls_fn) :
        print('\t pseudo-cls_fn', pseudo_cls_fn) 
        pseudo_classify_autoradiograph( crop_fn, seg_fn, pseudo_cls_fn, int(row['volume_order']), resolution )

def crop(src_dir, mask_dir, out_dir, df, scale_factors_json, resolution, remote=False,clobber=False):
    '''take raw linearized images and crop them'''
    df = df.loc[ (df['hemisphere'] == 'R') & (df['mri'] == 'MR1' )  ] #FIXME, will need to be removed
    
    with open(scale_factors_json) as f : scale=json.load(f)
   
    file_check = lambda x : not os.path.exists(x)
    crop_check = df['crop_fn'].apply( file_check ).values
    seg_check =  df['seg_fn'].apply( file_check ).values
    cls_check =  df['pseudo_cls_fn'].apply( file_check ).values

    missing_files = crop_check + seg_check + cls_check
    if np.sum( missing_files ) > 0 : 
        pass
    else : return 0

    os.makedirs(out_dir,exist_ok=True)
    df_to_process = df.loc[ missing_files ]  

    global_order_min = df["global_order"].min()
    print('Cropping')
    Parallel(n_jobs=14)(delayed(crop_parallel)(row, mask_dir, scale, global_order_min, resolution) for i, row in  df_to_process.iterrows()) 
    

            
    return 0


