import scipy.misc
import os
import matplotlib.pyplot as plt
import imageio
import pandas as pd
import argparse
import numpy as np
import matplotlib.pyplot as plt 
import scipy.signal
from skimage.transform import resize
from glob import glob
from sklearn.feature_extraction import image
from sys import exit, argv
from utils.mouse_click import click
from utils.utils import *
from scipy.ndimage.morphology import binary_dilation, binary_erosion
from utils.utils import *
from scipy.ndimage import label
from skimage.filters import threshold_otsu, threshold_li
from scipy.ndimage.filters import gaussian_filter
from skimage.exposure import  equalize_hist
from scipy.ndimage.morphology import binary_closing, binary_dilation, binary_fill_holes
from skimage.exposure import equalize_adapthist

def myKMeans(s):
    s = np.max(s) - s 
    upper=np.max(s)
    #mid = np.median(s[s>0])
    if np.max(s) == np.min(s) : return np.zeros(s.shape)
    mid = np.percentile(s[s>0], [10])[0]
    init=np.array([0, mid, upper]).reshape(-1,1)
    cls = KMeans(3, init=init).fit_predict(s.reshape(-1,1)).reshape(s.shape)
    cls[ cls != 2 ] = 0
    cls[cls == 2] = 1
    return cls

def threshold(img,sd=3):
    #img = equalize_hist(img)
    img =  rank.equalize(img,disk(200)) 
    img = gaussian_filter(img, sd  )
    out = np.zeros(img.shape)
    out[ img < threshold_otsu(img) ] = 1
    return out

def binary_mask(im5):

    out = np.zeros(im5.shape) 
    for i in np.unique(im5)[1:] :
        temp = np.zeros(im5.shape)
        temp[im5==i] = 1
        ymin, ymax, xmin, xmax, yi, xi = find_min_max(temp)
        im6 = np.zeros(temp.shape)   
        for y0, y1, i in zip(ymin, ymax, xi ) :
            if y0 != y1 :
                im6[y0:y1,i] =  1 #1
        for x0, x1, i in zip(xmin, xmax, yi ) :
            if x0 != x1 :
                im6[i,x0:x1] = 1  #+= 1
        im6 = binary_erosion(im6, iterations=2)
        idx= im6 > 0 
        out[idx ] = i

    return out


def remove_border_regions(cc) :
    border_box = np.zeros(cc.shape)
    border_box[:,(cc.shape[1]-3):(cc.shape[1]) ]=1
    border_box[:,0:3]=1
    border_box[0:3,:]=1
    border_box[(cc.shape[0]-3):(cc.shape[0]),:]=1
    border_box = border_box * cc
    cc_unique = np.unique(cc)[1:]
    max_label = np.max(cc_unique)
    numPixels = np.array([np.sum(cc == l) for l in cc_unique  ])

    numPixels_sort=np.sort(numPixels)
    largest_region=cc_unique[numPixels.argmax()]
    
    #If there is more than one labelled region
    if len(numPixels_sort) > 1 :
        ratio =   numPixels_sort[-2] / numPixels_sort[-1]
    else : 
        ratio=1
    print(ratio)
    #for each label in the set of labels
    for l in cc_unique :
        #if there is overlap between the bounding box and the labelled region
        if True in (border_box == l) :
            #If the current label is that for the largest region
            #and the next largest region is much smaller than the current region,
            #then ignore the overlap
            if  ratio < 0.05  and (l == largest_region )  : continue
            #Otherwise set the labelled region to 0
            cc[cc == l] = -max_label
    return cc


def find_bb_overlap(cc):
    temp=np.zeros(cc.shape)
    overlap=np.zeros(cc.shape)

    for i in np.unique(cc)[1:] :
        temp *= 0
        temp[cc==i]=1
        ymin, ymax, xmin, xmax, yi, xi = find_min_max(temp)
        overlap[min(ymin) : max(ymax), min(xmin) : max(xmax)] = 1

    overlap, labels=label(overlap,structure=np.ones([3,3]))
    return overlap

from skimage.morphology import disk
from skimage.filters import rank

def cropp_img(img, line, no_frame=False, low_contrast=False, plot=False, linearized=True):
    '''
    Use histogram thresholding and morphological operations to create a binary image that is equal to 1 for pixels which will be kept after cropping.

    Input : 
        img --  image to crop
    Output :
        im5 --  bounding box
        cc0  --  connected labeled regions
    '''
    if plot :
        plt.subplot(2,2,1)
        plt.imshow(img)
    
    if linearized : 
        img = np.max(img) - img
        print("Linearized Inversing")
    else :
        print("NOT Linearized Inversing")
    
    img = (img - img.min()) / (img.max() - img.min())
    img_hist =  rank.equalize(img,disk(100)) 
    
    img_thr_0 = threshold(img, sd=0)
    img_thr_2 = threshold(img, sd=2)
    img_thr_4 = threshold(img, sd=4)
    prob_thr = (img_thr_0 + img_thr_2 + img_thr_4   )/ 3.
    img_thr = np.zeros(prob_thr.shape)
    img_thr[ prob_thr > 0.5 ] = 1

    if line.T.shape == img_thr.shape :
        line = line.T
    
    if no_frame : 
        img_thr[ line != 0 ] = 1
    else : 
        img_thr[ line != 0 ] = 0

    img_thr = binary_dilation(binary_erosion(img_thr, iterations=3), iterations=3).astype(int)

    cc, nlabels = label(img_thr, structure=np.array([[0,1,0],[1,1,1],[0,1,0]]))
    #plt.subplot(3,1,1); plt.imshow(cc) 
    cc = binary_mask(cc)
    #plt.subplot(3,1,2); plt.imshow(cc) 
    cc, nlabels =label(cc,  structure=np.array([[0,1,0],[1,1,1],[0,1,0]]))
    
    cc = remove_border_regions(cc)
    cc_with_border = cc
    cc[cc<0]=0

    bounding_box = cc #binary_mask(cc)
    bounding_box[ bounding_box > 0 ] = 1
    bounding_box = binary_dilation(binary_erosion(bounding_box, iterations=3), iterations=3).astype(int)
    return bounding_box, img_hist, img_thr, cc_with_border

def crop(img, img_dwn, bounding_box_dwn) :
    if np.sum(bounding_box_dwn) > 0 : 
        #Create downsampled cropped image
        cropped_dwn = img_dwn * bounding_box_dwn 
        
        #ymin, ymax, xmin, xmax, yi, xi = find_min_max(bounding_box_dwn)
        #y0=min(ymin)
        #y1=max(ymax)
        #x0=min(xmin) 
        #x1=max(xmax)
        #cropped_dwn = cropped_dwn[y0:y1,x0:x1]
        
        #Create cropped image at full resolution
        bounding_box=resize(bounding_box_dwn, (img.shape[0],img.shape[1]), order=0)
        #bounding_box=downsample(bounding_box_dwn, (img.shape[0],img.shape[1]),interp=0)
        bounding_box = bounding_box /  np.max(bounding_box)
        ymin, ymax, xmin, xmax, yi, xi = find_min_max(bounding_box)
        y0=min(ymin)
        y1=max(ymax)
        x0=min(xmin) 
        x1=max(xmax)
        cropped = img * bounding_box
        #plt.subplot(3,1,1)
        #plt.imshow(bounding_box)
        #plt.subplot(3,1,2)
        #plt.imshow(img)
        #plt.subplot(3,1,3)
        #plt.imshow(cropped)
        #plt.show()
        #cropped = cropped[y0:y1,x0:x1]
    else :
        cropped = img
        cropped_dwn = img_dwn
        bounding_box=np.zeros(cropped.shape)
    return cropped, cropped_dwn, bounding_box


def save_qc(img_dwn, img_smoothed, im_thr, cc, bb, cropped_dwn,out_fn,manual_check):
    dim0=img_dwn.shape[0]
    dim1=img_dwn.shape[1]
    qc=np.zeros([ dim0, dim1*2 ])
    plt.clf()
    plt.Figure()
    plt.title("Image Downsample")
    plt.subplot(2,3,1)
    plt.imshow(img_dwn / img_dwn.max())
    
    print("QC :", out_fn)
    plt.subplot(2,3,2)
    plt.title("Histogram Normalized")
    plt.imshow(img_smoothed)

    plt.subplot(2,3,3)
    plt.title("Segmentation")
    plt.imshow(im_thr)

    plt.subplot(2,3,4)
    plt.title("Labels")
    plt.imshow(cc)

    plt.subplot(2,3,5)
    plt.title("Bounding Box")
    plt.imshow(bb)

    if np.sum(cropped_dwn) != 0 :
        plt.subplot(2,3,6)
        plt.title("Cropped")
        plt.imshow(cropped_dwn/cropped_dwn.max())
    
    plt.savefig(out_fn, figsize=(20, 20) )
    
    if manual_check :
        #plt.show()
        # Create figure and axes
        # Display the image
        #plt.imshow(qc,cmap='gray' )
        plt.show()
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

def crop_gui(subject_output_base, img_dwn, img, qc_fn):
    refPts = click(subject_output_base+"img_dwn.jpg")
    try :
        x0=refPts[0][0]
        y0=refPts[0][1]
        x1=refPts[1][0]
        y1=refPts[1][1]
        manual_crop_dwn = np.zeros(img_dwn.shape)
        manual_crop_dwn[y0:y1,x0:x1] = 1
        #cropped = manual_crop * img
        cropped, cropped_dwn, bounding_box = crop(img, img_dwn, manual_crop_dwn )
        
        imageio.imsave(qc_fn, cropped_dwn)
    except TypeError :
        return [], []
    return cropped, cropped_dwn, bounding_box


def crop_source_files(source_dir, output_dir, downsample_step=0.5, clobber=False, manual_check=False, manual_only=False, no_frame=False, linearized=True, ext='.png') :
    qc_status=0

    if not os.path.exists(output_dir) : os.makedirs(output_dir)
    source_files = glob(source_dir+os.sep+"final"+os.sep+"*"+ext)
    line_files = glob(source_dir+os.sep+"lines"+os.sep+"*.png")
    if source_files == [] : print("Warning: Could not find source files in ", source_dir+os.sep+"*"+ext )

    for f in source_files :
        #Set output filename
        fsplit = os.path.splitext(os.path.basename(f))
        qc_dir = output_dir + os.sep + "qc" + os.sep 
        subject_output_dir = output_dir + os.sep + "detailed" + os.sep + fsplit[0] + os.sep
        subject_output_base = subject_output_dir+fsplit[0]+'_'
        fout = output_dir + os.sep + fsplit[0]+'_cropped.png'
        qc_fn=qc_dir+os.sep+fsplit[0]+'_cropped_qc.png'
        bounding_box_fn=subject_output_base + 'bounding_box.png' 
        if (not os.path.exists(bounding_box_fn) or not os.path.exists(qc_fn)) or clobber :
            line_fn = [ i for i in line_files if fsplit[0] in i  ] 
            if line_fn != [] : line_fn = line_fn[0]
            else : 
                print("Error: could not find file", f)
                exit(0)
            print("Input:", f)
            print("Line:", line_fn)
            if not os.path.exists(subject_output_dir) : os.makedirs(subject_output_dir)
            if not os.path.exists(qc_dir) : os.makedirs(qc_dir)

            #Load image and convert from rgb to grayscale
            img = imageio.imread(f)
            if len(img.shape) > 2 : img = rgb2gray(img)
            
            iline = imageio.imread(line_fn)
            iline=binary_dilation(iline,iterations=3).astype(int)

            low_contrast=False
            low_contrast_ligands=["oxot", "epib", "ampa", "uk14", "mk80" ]
            for ligand in low_contrast_ligands :
                if ligand in fsplit[0] :
                    low_contrast=True
                    break

            #Downsampled the image to make processing faster
            img_dwn =   downsample(img, subject_fn=subject_output_base+"img_dwn.jpg", step=downsample_step, interp=1)
            iline_dwn = downsample(iline, subject_fn="", step=downsample_step, interp=1)

            if not manual_only :
                bounding_box_dwn, img_smoothed,  im_thr, cc = cropp_img(img_dwn,iline_dwn,no_frame,low_contrast=low_contrast, linearized=linearized)
                #Crop image
                cropped, cropped_dwn, bounding_box = crop(img, img_dwn, bounding_box_dwn)

                #Quality Control
                qc_status = save_qc(img_dwn, img_smoothed, im_thr, cc, bounding_box_dwn,  cropped_dwn, qc_fn, manual_check)

            if qc_status != 0 or manual_only : 
                #Automated cropping failed to pass QC, use manual QC
                cropped, cropped_dwn, bounding_box = crop_gui(subject_output_base,img_dwn,img,qc_fn)
            
            if cropped != [] :
                print("Cropped:", fout,"\n")
                imageio.imsave(fout, cropped)
                imageio.imsave(bounding_box_fn, bounding_box)
            else : 
                print("No cropped image to save")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--source', dest='source_dir',  help='Directory with raw images')
    parser.add_argument('--output', dest='output_dir',  help='Directory name for outputs')
    parser.add_argument('--ext', dest='ext', default=".TIF", help='File extension for input files (default=.tif)')
    parser.add_argument('--step', dest='downsample_step', default=0.1, type=float, help='File extension for input files (default=.tif)')
    parser.add_argument('--manual', dest='manual_check', action='store_true', default=False, help='Do QC and manually crop region if automated method fails')
    parser.add_argument('--manual-only', dest='manual_only', action='store_true', default=False, help='Only do manual cropping (default=False)')
    parser.add_argument('--clobber', dest='clobber', action='store_true', default=False, help='Clobber results')
    parser.add_argument('--not-linearized', dest='linearized', action='store_false', default=True, help='Autoradiographs have not been linearized')
    parser.add_argument('--no-frame', dest='no_frame', action='store_true', default=False, help='No frames in line images.')

    args = parser.parse_args()

    crop_source_files(args.source_dir, args.output_dir, downsample_step=args.downsample_step, clobber=args.clobber, manual_check=args.manual_check, manual_only=args.manual_only, no_frame=args.no_frame, linearized=args.linearized, ext=args.ext)

