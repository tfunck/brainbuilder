from skimage.transform import (hough_line, hough_line_peaks, probabilistic_hough_line)
import numpy as np
import scipy.misc
from sklearn.cluster import spectral_clustering, DBSCAN, KMeans
from glob import glob
from sklearn.feature_extraction import image
from sys import exit, argv
import os
import matplotlib.pyplot as plt
import imageio
from sklearn.metrics import normalized_mutual_info_score
from scipy.ndimage import label
from scipy.ndimage.measurements import center_of_mass
from scipy.ndimage.morphology import binary_dilation, binary_erosion, binary_fill_holes
from skimage.morphology import remove_small_holes, reconstruction
from skimage.restoration import inpaint
import pandas as pd
import argparse
from utils.mouse_click import click
from utils.utils import *
from skimage import filters
from cropp_img import cropp_img

### Required libraries :
### pip3 install --upgrade --user imageio scikit-image opencv-python

def curvature(img):
    d0 = np.gradient(img,edge_order=2, axis=0)
    d00 = np.gradient(d0,edge_order=2, axis=0)
    d1 = np.gradient(img,edge_order=2, axis=1)
    d11 = np.gradient(d1,edge_order=2, axis=1)
    d10 = np.gradient(d1,edge_order=2, axis=0)

    num = (d00*d11**2 + d11*d00**2 - 2*d10*d1*d0) 
    den = (d0**2 + d1**2)**(3/2)
    den[den==0]=1
    num[den==0]=0
    k =np.abs(num/ den)
    return(k)




def get_kmeans_img(img_dwn, nMeans):
    init=np.percentile(img_dwn, [0.5,0.99]).reshape(2,1)
    db = KMeans(nMeans, init=init).fit_predict(img_dwn.reshape(-1,1))
    clustering = db.reshape(img_dwn.shape)
    #In order for the "label" function to work, the background
    #must be equal to 0 in the clustering array. To ensure this,
    #the following bit of code switches the labeled region with the lowest
    #mean with that of the 0 labeled region, if the former is lower than the latter
    pixel_measure = np.mean(img_dwn[ clustering == 0])
    measure_n=0
    for n in range(1, nMeans) :
        cur_pixel_measure = np.mean(img_dwn[ clustering == n])
        if cur_pixel_measure > pixel_measure :
            max_pixel_measure = cur_pixel_measure
            measure_n =n

    if measure_n != 0 :
        idx0 = clustering == 0
        idx1 = clustering == measure_n
        clustering[idx0] = max_pixel_measure
        clustering[idx1] = 0
    
    #Perform erosion and dilation on all values in clustered image
    for n in range(1, nMeans) :
        temp = np.zeros(img_dwn.shape)
        temp[clustering == n ] =1 
        temp = binary_erosion(temp, iterations=1)
        temp = binary_dilation(temp, iterations=1)
        clustering[ clustering == n ] = temp[clustering==n]

    return(clustering)

def adjust_mask_region( thr, img_dwn, clustered, img_mask) :
    img_thr=np.copy(img_dwn)

    #img_mask_d = binary_dilation(img_mask, iterations=2).astype(int)
    #mask_border = img_mask_d - img_mask
    #img_mask = img_mask_d

    #temp=np.copy(img_dwn)
    #temp[ img_mask == 1 ] = np.median(img_dwn[mask_border == 1]) 
    #temp = gaussian_filter(temp, sigma=2, mode="reflect")
    #img_thr[ img_mask == 1 ] = temp[img_mask == 1] # np.max(img_dwn[img_mask == 0])
    m = np.median( img_dwn[(img_mask == 0) & (clustered == 0)] )
    img_thr[  (clustered == 0)] = m
    img_thr[ (img_mask == 1)  ] = m 
    return img_thr


def crop(img, img_dwn, bounding_box_dwn) :
    #Create downsampled cropped image
    cropped_dwn = img_dwn * bounding_box_dwn 
    #cropped_dwn[(bounding_box_dwn==0) | (mask==1)]=np.median( img_dwn[(bounding_box_dwn == 0) | (mask==1) ])
    cropped_dwn[(bounding_box_dwn==0) ]=np.median( img_dwn[(bounding_box_dwn == 0) ])
    
    #Create cropped image at full resolution
    bounding_box=scipy.misc.imresize(bounding_box_dwn,size=(img.shape[0],img.shape[1]),interp="nearest")
    cropped = np.zeros(img.shape)
    cropped[bounding_box != 0 ] = img[bounding_box != 0]
    cropped[bounding_box == 0 ] = np.median(img[bounding_box == 0 ])
    return cropped, cropped_dwn


def save_qc(img_dwn, im_hist_adjust,img_lines_removed, im_thr, cc, bb, cropped_dwn,out_fn,manual_check):
    dim0=img_dwn.shape[0]
    dim1=img_dwn.shape[1]
    qc=np.zeros([ dim0, dim1*2 ])
    plt.clf()
    plt.Figure()
    plt.title("Image Downsample")
    plt.subplot(2,4,1)
    plt.imshow(img_dwn / img_dwn.max())
    
    print("QC :", out_fn)
    plt.subplot(2,4,2)
    plt.title("Histogram Adjust")
    plt.imshow(im_hist_adjust)

    if np.sum(img_lines_removed) != 0 :
        plt.subplot(2,4,3)
        plt.title("Lines Removed")
        plt.imshow(img_lines_removed)

    plt.subplot(2,4,4)
    plt.title("Thesholded Image")
    plt.imshow(im_thr)

    plt.subplot(2,4,5)
    plt.title("Connected Voxels")
    plt.imshow(cc)

    plt.subplot(2,4,6)
    plt.title("Bounding Box")
    plt.imshow(bb)

    if np.sum(cropped_dwn) != 0 :
        plt.subplot(2,4,8)
        plt.title("Cropped Downsample")
        plt.imshow(cropped_dwn/cropped_dwn.max())
    
    plt.savefig(out_fn)
    
    if manual_check :
        plt.show()
        # Create figure and axes
        # Display the image
        #plt.imshow(qc,cmap='gray' )
        #plt.show()
        while True :
            response = input("Does image pass QC? (y/n)")
            if response == "y" : return 0
            elif response == "n" : return 1
            print("Click and drag with left mouse to draw rectangle.")
            print("Warning: can crash if you don't start rectangle from top left corner!")
            print("If the 'r' key is pressed, reset the cropping region.")
            print("If the 'c' key is pressed, break from the loop.")
    else : plt.clf()

    return 0

def crop_gui(subject_output_base, img_dwn, img, mask, qc_fn):
    refPts = click(subject_output_base+"img_dwn.jpg")
    try :
        x0=refPts[0][0]
        y0=refPts[0][1]
        x1=refPts[1][0]
        y1=refPts[1][1]
        manual_crop_dwn = np.zeros(img_dwn.shape)
        manual_crop_dwn[y0:y1,x0:x1] = 1
        #cropped = manual_crop * img
        cropped, cropped_dwn = crop(img, img_dwn, manual_crop_dwn, mask )
        
        scipy.misc.imsave(qc_fn, cropped_dwn)
    except TypeError :
        return [], []
    return cropped, cropped_dwn


from PIL import Image, ImageDraw
def crop_source_files(source_files, output_dir, downsample_step=0.5, clobber=False, manual_check=False, manual_only=False, histogram_threshold=False, use_remove_lines=False) :
    qc_status=0
    for f in source_files :
        #Set output filename
        fsplit = os.path.splitext(os.path.basename(f))
        qc_dir = output_dir + os.sep + "qc" + os.sep 
        subject_output_dir = output_dir + os.sep + "detailed" + os.sep + fsplit[0] + os.sep
        subject_output_base = subject_output_dir+fsplit[0]+'_'
        fout = output_dir + os.sep + fsplit[0]+'_cropped'+fsplit[1]
        qc_fn=qc_dir+os.sep+fsplit[0]+'_cropped_qc.png'
        #if not "RD#HG#MR1s3#R#rx82#5791#04" in f : continue 
        if (not os.path.exists(fout) or not os.path.exists(qc_fn)) or clobber :
        #if (not os.path.exists(fout) ) or clobber :
            print("Input:", f)
            if not os.path.exists(subject_output_dir) : os.makedirs(subject_output_dir)
            if not os.path.exists(qc_dir) : os.makedirs(qc_dir)

            #Load image and convert from rgb to grayscale
            img = imageio.imread(f)
            if len(img.shape) > 2 : img = rgb2gray(img)

            #Downsampled the image to make processing faster
            img_dwn = downsample(img, subject_output_base, downsample_step)

            if not manual_only :
                bounding_box_dwn, im_hist_adjust, img_lines_removed, im_thr, cc = cropp_img(img_dwn, use_remove_lines)
                #Crop image
                cropped, cropped_dwn = crop(img, img_dwn, bounding_box_dwn)

                #Quality Control
                qc_status = save_qc(img_dwn, im_hist_adjust, img_lines_removed, im_thr, cc, bounding_box_dwn,  cropped_dwn, qc_fn, manual_check)

            if qc_status != 0 or manual_only : 
                #Automated cropping failed to pass QC, use manual QC
                cropped, cropped_dwn = crop_gui(subject_output_base,img_dwn,img,mask,qc_fn)
            
            if cropped != [] :
                print("Cropped:", fout,"\n")
                scipy.misc.imsave(fout, cropped)
            else : 
                print("No cropped image to save")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--source', dest='source_dir',  help='Directory with raw images')
    parser.add_argument('--output', dest='output_dir',  help='Directory name for outputs')
    parser.add_argument('--ext', dest='ext', default=".tif", help='File extension for input files (default=.tif)')
    parser.add_argument('--step', dest='downsample_step', default=0.1, type=float, help='File extension for input files (default=.tif)')
    parser.add_argument('--method', dest='method', default="bounding_box", type=str, help='Method for automated cropping (default = bounding_box). Implemented methods: bounding_box, largest_region')
    parser.add_argument('--manual', dest='manual_check', action='store_true', default=False, help='Do QC and manually crop region if automated method fails')
    parser.add_argument('--remove-lines', dest='use_remove_lines', action='store_true', default=False, help='Remove lines from image')
    parser.add_argument('--manual-only', dest='manual_only', action='store_true', default=False, help='Only do manual cropping (default=False)')
    parser.add_argument('--histogram-threshold', dest='histogram_threshold', action='store_true', default=False, help='Only do manual cropping (default=False)')
    parser.add_argument('--clobber', dest='clobber', action='store_true', default=False, help='Clobber results')

    args = parser.parse_args()
    if not os.path.exists(args.output_dir) : os.makedirs(args.output_dir)
    source_files = glob(args.source_dir+"**"+os.sep+"*"+args.ext)
    if source_files == [] : print("Warning: Could not find source files.")
    crop_source_files(source_files, args.output_dir, downsample_step=args.downsample_step, clobber=args.clobber, manual_check=args.manual_check, manual_only=args.manual_only, histogram_threshold=args.histogram_threshold, use_remove_lines=args.use_remove_lines)

