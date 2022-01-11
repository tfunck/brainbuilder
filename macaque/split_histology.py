import imageio
import os
import matplotlib.pyplot as plt
import numpy as np
import utils.ants_nibabel as nib
#import nibabel as nib
from sklearn.cluster import KMeans
from scipy.ndimage import label
from re import sub
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage import label, center_of_mass
from skimage.filters import threshold_otsu, threshold_li , threshold_sauvola, threshold_yen
from sys import argv
from glob import glob

os.makedirs('tif_lowres_split', exist_ok=True)

def segment(im):
    t = threshold_otsu(im)
    im_bin = np.zeros_like(im)
    im_bin[ im >= t] = 1
    im_label, nlabels = label(im_bin)
    plt.subplot(2,1,1)
    plt.imshow(im)
    plt.subplot(2,1,2)
    plt.imshow(im_label);

    #plt.show()

    return im_label, nlabels 

def get_com_list(im_label, nlabels):
    com_list = [] 

    for i in range(1,nlabels) :
        label_image = np.zeros_like(im_label)
        label_image[ im_label == i ] = 1
        cx , cz = np.rint(center_of_mass(label_image)).astype(int)

        com_list.append( [cx,cz] )
    com_np = np.array(com_list)
    return com_np

def split_image(im, qc_fn):

    im_labels, nlabels = segment(im)
    com_np = get_com_list(im_labels, nlabels)[:,1].reshape(-1,1).astype(np.float64)
    com_np += np.random.normal(0,0.1,com_np.shape)

    xmid = im_labels.shape[0]/2.
    cluster_0 = np.mean(com_np[com_np < xmid])
    cluster_1 = np.mean(com_np[com_np >= xmid])
    print(com_np)

    cluster_labels = KMeans(n_clusters=2, init=np.array([[cluster_0],[cluster_1]]) ).fit_predict(com_np)
    
    split_im = np.zeros_like(im)
    for l, c, com in zip(range(1,nlabels+1), cluster_labels, com_np) :
        split_im[im_labels==l] = c+1


    # resplit the image if only a single piece of brain tissue
    if np.sum(split_im==1) < 0.3*np.sum(split_im==2) or np.sum(split_im==2) < 0.3*np.sum(split_im==1):
        split_im[split_im>0]=1
        com = center_of_mass(split_im)
        split_im[ :, int(com[1]): ] *= 2

    mask_left = np.zeros_like(split_im) 
    mask_right = np.zeros_like(split_im)

    mask_left[split_im == 1] = 1 

    mask_right[split_im == 2] = 1   

    plt.subplot(3,1,1)
    plt.imshow(split_im)
    plt.subplot(3,1,2)
    plt.imshow(mask_left)
    plt.subplot(3,1,3)
    plt.imshow(mask_right)
    plt.savefig(qc_fn)


    return mask_left, mask_right
    
    

affine=np.array([[0.2,0,0,0],[0,0.2,0,0.0],[0,0,0.02,0],[0,0,0,1]])
os.makedirs('hist_split',exist_ok=True)

for fn in glob('png_0.02mm/*.png'):
    left_fn = 'hist_split/' + os.path.basename(sub('.png','_left.png',fn))
    right_fn ='hist_split/' + os.path.basename(sub('.png','_right.png',fn))
    qc_fn ='hist_split/' + os.path.basename(sub('.png','_qc.png',fn))

    print(fn)
    if not os.path.exists(qc_fn) :
    
        img = imageio.imread(fn).astype(float)
        img = gaussian_filter(img, 1)
        img = img.max()-img
        if img.shape[0] > img.shape[1] : img = np.flipud(img.T)
        
        mask_left, mask_right = split_image(img, qc_fn)
        
        img_left = img * mask_left
        img_right = img * mask_right
        
        #img_left =  np.rot90(np.flipud(img_left) )
        #img_right =  np.rot90(np.flipud(img_right) )
        
        #img_left = img_left.reshape([img_left.shape[0],img_left.shape[1]])
        #img_right = img_right.reshape([img_right.shape[0],img_right.shape[1]])
        
        print(left_fn)
        print(right_fn)
        imageio.imsave(left_fn, img_left)
        imageio.imsave(right_fn, img_right )


    left_nii_fn='hist_split/' + os.path.basename(sub('.png','_left.nii.gz',fn))
    right_nii_fn='hist_split/' + os.path.basename(sub('.png','_right.nii.gz',fn))
    print(left_nii_fn)

    if not os.path.exists(left_nii_fn) or True:
        img = imageio.imread(left_fn)
        nib.Nifti1Image(np.rot90(np.flipud(img)), affine).to_filename(left_nii_fn)

    if not os.path.exists(right_nii_fn) or True:
        img = imageio.imread(right_fn)
        nib.Nifti1Image(np.rot90(np.flipud(img)), affine).to_filename(right_nii_fn)
