import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.ndimage.morphology import binary_dilation, binary_erosion
from skimage.morphology import skeletonize
from skimage.util import invert
from nipy.labs.viz_tools import maps_3d
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage import label
from nibabel.processing import resample_to_output, resample_from_to
from skimage.filters import threshold_otsu, threshold_li
from sklearn.feature_extraction import image
from sklearn.cluster import spectral_clustering
from sklearn.cluster import KMeans

def myKMeans(s):
    upper=np.max(s)
    #mid = np.median(s[s>0])
    if np.max(s) == np.min(s) : return np.zeros(s.shape)
    t = threshold_otsu(s)
    mid = np.percentile(s[s>t], [10])[0]
    init=np.array([0, mid, upper]).reshape(-1,1)
    cls = KMeans(3, init=init).fit_predict(s.reshape(-1,1)).reshape(s.shape)
    cls[ cls != 2 ] = 0
    cls[cls == 2] = 1
    return cls


def denoise(s) :
    cc, nlabels = label(s, structure=np.ones([3,3,3]))
    cc_unique = np.unique(cc)
    numPixels = np.bincount(cc.reshape(-1,))

    numPixels = numPixels[1:]
    cc_unique = cc_unique[1:]
    
    idx = numPixels < (np.max(numPixels) * 0.01)
    labels_to_remove = cc_unique[ idx ]
    labels_to_keep = cc_unique[ ~ idx ]
    
    out = np.zeros(cc.shape)
    for i in labels_to_keep :
        out[ cc == i ] = 1
    
    return out
mri_img = nib.load("civet_out/mr1/final/mr1_t1_tal.mnc")
cls_img = nib.load("civet_out/mr1/classify/mr1_pve_classify.mnc")
mask_img = nib.load("civet_out/mr1/mask/mr1_skull_mask.mnc")
skull_img = nib.load("civet_out/bet_skull.nii.gz")


if not os.path.exists("skull.nii.gz") :
    skull = gaussian_filter(skull_img.get_data()*1000., 15)
    idx = skull > np.max(skull) *.1
    skull[ idx  ] = 1
    skull[ ~idx  ] = 0
    skull= binary_erosion(skull, iterations=15).astype(float)

    skull_out=np.zeros(skull.shape)
    for z in range(skull.shape[2]) : skull_out[:,:,z]+=skeletonize(skull[:,:,z])
    for y in range(skull.shape[1]) : skull_out[:,y,:]+=skeletonize(skull[:,y,:])
    for x in range(skull.shape[0]) : skull_out[x,:,:]+=skeletonize(skull[x,:,:])
    skull_out = gaussian_filter(skull_out, 5)
    idx = skull_out > 0.10
    skull_out[ idx ] = 1
    skull_out[ ~idx ] = 0
    nib.Nifti1Image(skull_out, skull_img.affine).to_filename("skull.nii.gz")
    out=skull_out


out = resample_from_to( nib.load("skull.nii.gz"), mri_img).get_data()

cls = cls_img.get_data()
mask = binary_dilation(mask_img.get_data(), iterations=2).astype(int)
mri = mri_img.get_data()

out[  ((cls==2) | (cls==3) | (cls==4)) & (mask==1) ] = 2
out[  (cls==1 ) & (mask==1) ] = 3
#out[  (out == 0 ) & ((cls == 2)|(cls==3)) ] = 2
#sums = np.bincount(out.reshape(-1,))[1:]
#max_index = np.argmax(sums) + 1

#print(max_index)
#out[ out != max_index ] = 0 
#out[ out > 0 ] = 1 
hdr1 = nib.Nifti1Header()  
hdr1.set_dim_info(0,1,2)
out = np.swapaxes(out, 0,2)
nib.AnalyzeImage(out, mri_img.affine, hdr1).to_filename("mr1_attenuation_map.img")
#nib.Nifti1Image(out, mri_img.affine, hdr1).to_filename("mr1_attenuation_map.nii")


