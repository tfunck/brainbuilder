import bisect
import contextlib
import re
import scipy
import os
import pandas as pd
import imageio
import PIL
import matplotlib
import time
import ants
import numpy as np
import nibabel as nb

class Nifti1Image():
    def __init__(self, dataobj, affine):
        self.affine = affine
        self.dataobj= dataobj
        self.shape = dataobj.shape

    def to_filename(self, filename):
        print('to_filename')
        write_nifti(self.dataobj, self.affine, filename)

    def get_fdata(self):
        return self.dataobj

    def get_data(self):
        return self.dataobj

def safe_image_read(fn) :

    try :
        img = ants.image_read(fn)
    except RuntimeError :
        print("Error: cannot load file", fn)
        exit(1)

    return img

def read_affine_antspy(fn): 
    img = safe_image_read(fn)
    spacing = img.spacing
    origin = img.origin
    #print('spacing', spacing)
    #print('origin',origin)
    
    affine = np.eye(4)

    for i, (s, o) in enumerate(zip(spacing,origin)):
        affine[i,i]=s
        affine[i,3]=o

    if len(img.shape) == 3 and img.shape[-1] != 1 :
        pass #assert img.orientation =='RAS', f'Error: file has {orientation}, not RAS. {fn}'

    return affine

def read_affine(fn, use_antspy=True):
    if use_antspy :
        affine = read_affine_antspy(fn)
    else : 
        affine = nb.load(fn).affine
        #print('reading', affine) 

    return affine

def load(fn) :
    affine = read_affine(fn)
    vol = safe_image_read(fn).numpy()

    return Nifti1Image(vol,affine)

def write_nifti(vol, affine, out_fn):

    ndim = len(vol.shape)
    idx0 = list(range(0,ndim))
    idx1 = [3]*ndim

    origin = list(affine[ idx0, idx1 ])
    spacing = list( affine[ idx0, idx0 ])
    if len(spacing) == 2 :
        direction=[[1., 0.], [0., 1.]]
    else :
        # Force to write in RAS coordinates
        direction=[[1., 0., 0.], [0., 1., 0.], [0., 0., -1.]]
    print('--> spacing', spacing)
    print('--> origin', origin)
    print('--> direction', direction)
    ants_image = ants.from_numpy(vol, origin=origin, spacing=spacing, direction=direction)
    ants.image_write(ants_image, out_fn)
