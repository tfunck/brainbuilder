import imageio
import os
import matplotlib.pyplot as plt
import numpy as np

from re import sub

from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage import label, center_of_mass
from skimage.filters import threshold_otsu, threshold_li , threshold_sauvola, threshold_yen
from sys import argv
from glob import glob


example_fn = glob('img_lin/*')[0]
shape = nib.load(example_fn).shape

os.makedirs('tif_lowres_split', exist_ok=True)
for fn in glob('tif_lowres/*'):
    print(fn)
    img = imageio.imread(fn).astype(float)
    img = img.max()-img

    left_fn = sub('.tif','_left.tif',fn)
    right_fn = sub('.tif','_right.tif',fn)

    cx , cz = np.rint(center_of_mass(img)).astype(int)
    
    img_left = np.zeros_like(img)
    img_right = np.zeros_like(img)

    img_left[:, :int(cz) ] = img[:, :int(cz)]
    img_right[:, int(cz): ] = img[:, int(cz):]

    imageio.imsave( left_fn, img_left)
    imageio.imsave( right_fn, img_right)
    #plt.cla()
    #plt.clf()
    #plt.imshow(img_left, cmap='Greys')
    #plt.imshow(img_right, alpha=0.5)
    #plt.show()

    
    
    





