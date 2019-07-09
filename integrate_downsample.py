from utils.utils import shell
from scipy.integrate import simps
from scipy.integrate import simps
from scipy.interpolate import interp2d, griddata
from glob import glob
from re import sub
from os.path import basename, splitext
import json
import re
import os
import nibabel as nib
import numpy as np
import imageio
import matplotlib.pyplot as plt


def integrate_downsample_tif(img, step, affine, out_image):
    xstep = affine[0,0] 
    zstep = affine[2,2] 
    xmax=xstep * img.shape[0]
    zmax=zstep * img.shape[1]
    
    xlo = np.arange(0, xmax, step) # img.shape[0]) 
    zlo = np.arange(0, zmax, step) #img.shape[1]) 
    
    xlo_int = range(0,len(xlo))
    zlo_int = range(0,len(zlo))

    dwn_img=np.zeros([len(xlo),len(zlo)])

    xhi = np.arange(0, xmax, xstep)
    zhi = np.arange(0, zmax, zstep)
   
    for x0 , x_int in zip(xlo, xlo_int) :
        x1 = x0 
        xi0 = int(np.round(x0 / xstep))
        xi1=xi0
        while x1 < x0 + step -xstep and xi1 < img.shape[0] : 
            x1 += xstep         
            xi1 += 1

        for z0, z_int in zip(zlo, zlo_int ):
            zi0 = int(np.round(z0 / zstep))
            z1 = z0 
            zi1 = zi0 
           
            while z1 < z0 + step-zstep  and zi1 < img.shape[1] : 
                z1 += zstep
                zi1 += 1
            dwn_img[x_int,z_int] += np.sum(img[ xi0:xi1,  zi0:zi1 ])
    
    #imageio.imwrite(out_image, dwn_img)
    nib.Nifti1Image(dwn_img, affine).to_filename(out_image)

    
def downsample_and_crop(source_lin_dir, lin_dwn_dir,crop_dir, affine, step=0.2, clobber=False):

    for f in glob(source_lin_dir+"/*.TIF") :
        dwn_fn = lin_dwn_dir + splitext(basename(f))[0] + '.nii.gz'
        if not os.path.exists(dwn_fn) or clobber :
            try :
                base=sub('#L','',basename(splitext(f)[0]))
                path_string=crop_dir+"/detailed/**/"+base+"_bounding_box.png"
                print("path:", path_string)

                crop_fn = glob(path_string)[0]
            except IndexError :
                print("\t\tDownsample & Crop : Skipping ", f)
                continue
            print(crop_fn)
            print(f)
            img = imageio.imread(f)
            if len(img.shape) == 3 : img = np.mean(img,axis=2)

            bounding_box = imageio.imread(crop_fn) 
            if np.max(bounding_box) == 0 : 
                bounding_box = np.ones(img.shape)
            else : 
                bounding_box = bounding_box / np.max(bounding_box)
            img = img * bounding_box 
            integrate_downsample_tif(img, step, affine, dwn_fn)
            nib.processing.resample_to_output(nib.Nifti1Image(img, affine), step, order=5).to_filename(dwn_fn)
