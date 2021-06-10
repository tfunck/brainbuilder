import imageio
import os
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib

from re import sub

from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage import label, center_of_mass
from skimage.filters import threshold_otsu, threshold_li , threshold_sauvola, threshold_yen
from sys import argv
from glob import glob

os.makedirs('tif_lowres_split', exist_ok=True)

for fn in glob('tif_lowres/*'):
    print(fn)
    img = imageio.imread(fn).astype(float)
    img = gaussian_filter(img, 1)
    img = img.max()-img
    
    left_fn = 'tif_lowres_split/' + os.path.basename(sub('.tif','_left.nii.gz',fn))
    right_fn ='tif_lowres_split/' + os.path.basename(sub('.tif','_right.nii.gz',fn))

    cx , cz = np.rint(center_of_mass(img)).astype(int)
    
    img_left = np.zeros_like(img)
    img_right = np.zeros_like(img)

    img_right[:, :int(cz) ] = img[:, :int(cz)]
    img_left[:, int(cz): ] = img[:, int(cz):]

    img_left =  np.rot90(np.flipud(img_left) )
    img_right =  np.rot90(np.flipud(img_right) )

    img_left = img_left.reshape([img_left.shape[0],img_left.shape[1]])
    img_right = img_right.reshape([img_right.shape[0],img_right.shape[1]])

    affine=np.array([[0.2,0,0,0],[0,0.2,0,0.0],[0,0,0.02,0],[0,0,0,1]])
    nib.Nifti1Image(img_left, affine).to_filename(left_fn)
    print(left_fn)
    nib.Nifti1Image(img_right, affine).to_filename(right_fn)
    print(right_fn)
    #plt.cla()
    #plt.clf()
    #plt.imshow(img_left, cmap='Greys')
    #plt.imshow(img_right, alpha=0.5)
    #plt.show()

    
    
    





