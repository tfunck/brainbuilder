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
from utils.utils import safe_imread, downsample
from utils.mouse_click import click
import h5py as h5
import tensorflow as tf
import nibabel as nib
import pandas as pd
from skimage import exposure
from nibabel.processing import *
from sklearn.cluster import KMeans
from tensorflow.keras.models import load_model
from re import sub
from scipy.ndimage.morphology import binary_closing, binary_dilation, binary_erosion, binary_fill_holes
from skimage import exposure
from glob import glob
from utils.utils import downsample, safe_imread, find_min_max
from skimage.transform import rotate, resize 
from glob import glob
from scipy.ndimage.filters import gaussian_filter
from skimage.filters import rank
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import backend as K
from skimage.filters import threshold_otsu, threshold_li , threshold_sauvola, threshold_yen
from skimage.segmentation import clear_border
from skimage.exposure import equalize_hist
from skimage.morphology import disk
from scipy.ndimage import label
from utils.utils import shell 
matplotlib.use("TkAgg")

def qc(img_list, name_list, qc_fn, dpi=200):
    '''
    Purpose : Create figure showing images in img_list and with titles in name_list
    Inputs :
        img_list        -   list of 2D numpy arrays
        name_list       -   list of strings giving plot titles
        qc_fn          -   string containing output filename
        dpi             -   dots per inch
    Outputs :
        None        
    '''
    row_img = np.floor(np.sqrt(len(img_list))).astype(int)
    col_img = np.ceil(np.sqrt(len(img_list))).astype(int)
    row_img += max(len(img_list) - row_img * col_img  ,0)
    #plt.clf()
    fig = plt.figure()

    for j, (name, x) in enumerate(zip(name_list,img_list)) :
        plt.subplot(row_img, col_img, j+1)
        plt.imshow(x.astype(float))
        #plt.colorbar()
        plt.title(name)
        plt.axis('off')
    #plt.show()
    plt.tight_layout()
    plt.savefig(qc_fn, dpi=dpi)

    plt.close()
    plt.cla()
    del img_list
    gc.collect()

def remove_overlap_regions(img, lines, region_label=3):
    '''
    Purpose : Remove regions that overlap with 
    Inputs :
        img     -   input binary segmentation image (2d np array, int)
        lines   -   line segmentation (2d np array, int)

    Outputs :
        None
    '''
    labels, nlabels = label(img)
    roi = np.zeros_like(img)
    lines_2 = np.zeros_like(img)
    lines_2[lines == region_label] = 1
    #lines_2 = binary_dilation(lines_2, iterations=1).astype(int)
    for l in np.unique(labels)[1:] :
        roi *= 0
        roi[ labels == l ] = 1
        if np.sum( roi * lines_2 ) > 0  :
            img[labels==l] = 0
    
    #remove small regions
    labels, nlabels = label(img)
    if not nlabels == 0 :
        unique_labels = np.unique(labels)[1:]
        counts = np.bincount(labels.reshape(-1,))[1:]
        largest_region_size = np.max(counts)
        for l, c in zip(unique_labels, counts) :
            if c < .05 * largest_region_size :
                print('removing', l, c)
                img[ labels == l ] = 0

    del lines_2
    del roi
    del labels
    return img

def fill_regions(im, use_labels=True):
    '''
    Purpose : Fill in binary segementation. Fills holes but also open spaces between contiguous segmented regions
    Inputs :
        im  -   input image to be filled
    Outputs :
        out -   output image with filled regions
    '''
    if use_labels : 
        cc, nlabels = label(im, structure=np.array([[0,1,0],[1,1,1],[0,1,0]]))
    else :
        cc = im
        nlabels=[0,1]

    out = np.zeros(cc.shape) 
    temp = np.zeros(cc.shape)
    for i in np.unique(cc)[1:] :
        #Create temporary array with region with label i filled in
        temp[:,:] = 0 
        temp[cc==i] = 1
        #find the min/max indices for each row/col of the region
        ymin, ymax, xmin, xmax, yi, xi = find_min_max(temp)
        #im6 = np.zeros(temp.shape)   
        for y0, y1, i in zip(ymin, ymax, xi ) :
            if y0 != y1 :
                temp[y0:y1,i] =  1 #1
        for x0, x1, i in zip(xmin, xmax, yi ) :
            if x0 != x1 :
                temp[i,x0:x1] = 1  #+= 1
        temp = binary_erosion(temp, iterations=2)
        idx= temp > 0 
        out[idx ] = 1
    del cc
    del temp
    return out

def denoise(bin_img, verbose=False) :
    '''
    Purpose : Remove noise (small islands of 1s) from binary segmentation.
    Inputs :
        bin_img     -   2D numpy array, binary segmentation
    Outputs :
        out         -   a denoised version of bin_img
    '''
    cc, nlabels = label(bin_img, structure=np.array([[0,1,0],[1,1,1],[0,1,0]]) )
    cc_unique = np.unique(cc)
    cc_unique = cc_unique[1:]

    numPixels = np.bincount(cc.reshape(-1,))
    numPixels = numPixels[1:]

    for threshold in [ 500, 100, 10 ] :
        idx = numPixels < (threshold)
        if len(idx) != 0 : break
    if len(idx) == 0 : return bin_img

    labels_to_remove = cc_unique[ idx ]
    labels_to_keep = cc_unique[ ~ idx ]
    if verbose : print('\t\tRemoving regions smaller than', threshold,':','removing', len(labels_to_remove), 'of regions, keeping', len(labels_to_keep), 'regions.')  
    out = np.zeros(cc.shape)
    for i in labels_to_keep :
        out[ cc == i ] = 1
    del cc
    return out

def segment_tissue(im,version) :
    '''
    Purpose : Segement tissue in image with Otsu thresholding
    Inputs : 
        im - image to be segmented
    Outputs :
        out_otsu - segmented image
    '''
    out_otsu =np.zeros_like(im)
    if np.max(im) != np.min(im) :
        out_otsu[ im > threshold_otsu(im) ] = 1
    return out_otsu

def remove_lines(im,  base, version) :
    '''
    Purpose : Remove lines and replace with min of image
    Inputs :
        im - image from which lines are to be removed
    Outputs :
        im_out - image without lines
    '''

    label_list = glob('artefact_dictionary/set_*/mask/%s*'%base)
    if len(label_list) == 1 :
        label_fn = label_list[0]
        lines_im = safe_imread(label_fn)
        print('Artefact file:', label_fn)
    else : 
        #print('No label image found in dictionary for',fn,base)
        return im, np.zeros_like(im) 

    im_out = np.copy(im)
    lines_binary =np.copy(lines_im)
    lines_binary[lines_im == 2] = 1
    if version == 1 or version == 2 :
        lines_binary = binary_dilation(lines_binary, iterations=1).astype(int)
    lines_binary[lines_im == 3] = 1
    im_out[ lines_binary == 1 ] =  np.min(im) 
    del lines_binary
    return im_out, lines_im

def calculate_contrast(im) :
    '''
    Purpose : Calculate image contast using 10th and 90th percentile to avoid outliers/artefacts
    Inputs :
        im          -   input image (2D numpy array)
    Outputs :
        im_contrast -   measure of image contrast (float)  
    '''
    im_min , im_max=np.percentile(im[im>np.min(im)],[10,90])
    im_contrast = (im_max-im_min) / 255.
    return im_contrast


def get_base(fn) :
    fn = os.path.splitext(os.path.basename(fn))[0] 
    if fn[-2:] == '#L':
        fn = fn[0:-2]
    return fn

def crop_gui(img_dwn):
    #img_dwn = safe_imread(img_fn)
    refPts = click(img_dwn)
    try :
        x0=refPts[0][0]
        y0=refPts[0][1]
        x1=refPts[1][0]
        y1=refPts[1][1]
        manual_crop_dwn = np.zeros(img_dwn.shape)
        manual_crop_dwn[y0:y1,x0:x1] = 1
    except TypeError :
        return np.zeros_like(img_dwn)
    return manual_crop_dwn

def preprocess(fn,base,version,qc_fn,temp_crop_fn):
    print(fn)
    im = safe_imread(fn)
    im_no_lines, lines = remove_lines(im, base,version)
    if np.max(im_no_lines) == np.min(im_no_lines) : return 1
    if version == 4 : 
        im_no_lines = exposure.equalize_adapthist(im, clip_limit=0.03)
        #im_no_lines =  equalize_hist(gaussian_filter(im_no_lines,1))
        crop_lo = crop_gui(np.copy(im_no_lines))  
    else :
        ### 4) Adjust image histogram if the contrast is low
        im_contrast = calculate_contrast(im_no_lines)
        #contrast_df = pd.concat([contrast_df,pd.DataFrame({'ligand':[ligand],'contrast':[im_contrast]}) ])
        if version == 3:
            im_no_lines = exposure.equalize_adapthist(im_no_lines, clip_limit=0.05)
        elif im_contrast < 0.1 or version == 2 :
            im_no_lines = equalize_hist(gaussian_filter(im_no_lines,3))
        
        ### 5) K means segmentation
        if verbose : print('\tSegment Tissue')
        seg = segment_tissue(im_no_lines,version)

        ### 6) Add visual cues to keep onto the segmented image
        #lines_2 = np.zeros_like(seg)
        #lines_2[lines == 2] = 1
        #lines_2 = binary_dilation(lines_2, iterations=10)
        seg[lines == 3] = 1

        ### 8) Denoise the segmented image
        if verbose : print('\tDenoise tissue segmentation')
        seg = denoise(seg,  verbose=verbose)

        if verbose : print('\tFill in tissue regions')
        if version != 5 :
            seg_fill = fill_regions(seg)
        else :
            seg_fill = seg
        ### 9) Remove pieces of tissue that are connected to border
        if verbose : print('\tClear border')
        seg_fill_no_border = clear_border(seg_fill, buffer_size=3)


        ### 10) Remove pieces of tissue that are connceted to lines with label 2
        if verbose : print('\tRemove tissue regions connected to artefacts with label = 3')
        print('Unique elements in lines:', np.unique(lines))
        crop_lo = remove_overlap_regions(seg_fill_no_border, lines, region_label=3)
        if version == 3 :
            crop_lo = fill_regions(binary_dilation(crop_lo,iterations=10).astype(int), False)
        else :
            crop_lo = fill_regions(fill_regions(crop_lo,False), False)

        if version != 5 :
            crop_lo = binary_dilation(crop_lo, iterations=10).astype(int)

    if version == 2 or version == 3 :
        im = im_no_lines
        img_crop = crop_lo * im
    else :
        img_crop = crop_lo * im

    ### 11) Upsample the cropping frame and apply to lin img and tissue segmentation at full resolution
    #crop_hi = resize(crop_lo, im_hi.shape, order=0)
    #im_hi_cropped = crop_hi*im_hi
    #seg_hi_cropped = crop_hi * seg

    #img_list=[im, lines,im_no_lines, seg, seg_fill, seg_fill_no_border,  crop_lo]
    #name_list=['original','lines','no lines-%1.2f'%im_contrast, 'seg','seg filled','border removed', 'cropped']
    img_list=[im, img_crop]
    name_list=['original',  'cropped']
    print('Output Function:',qc_fn)
    qc(img_list, name_list, qc_fn, dpi=300)
    imageio.imsave(temp_crop_fn,crop_lo)
    del im
    del crop_lo
    del img_crop
    #del seg_fill
    #del seg
    #del im_no_lines
    #del lines



if __name__ == '__main__' :
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--process', dest='process', action='store_true', default=False, help='Do QC processing')
    parser.add_argument('--passed', dest='passed', action='store_true', default=False, help='Do QC processing')
    parser.add_argument('--final-qc', dest='final_qc', action='store_true', default=False, help='Do QC processing')
    parser.add_argument('--final-check', dest='final_check', action='store_true', default=False, help='Do QC processing')
    parser.add_argument('--v', dest='version', default=1, type=int, help='version')
    args = parser.parse_args()


    qc_dir ='crop/v%s/qc' %args.version
    pass_dir ='crop/v%s/pass' %args.version
    final_dir ='crop/v%s/final'%args.version 
    temp_dir ='crop/v%s/temp'%args.version 
    fail_dir ='crop/v%s/fail'%args.version 
    if not os.path.exists(qc_dir) : os.makedirs(qc_dir)
    if not os.path.exists(pass_dir) : os.makedirs(pass_dir)
    if not os.path.exists(final_dir) : os.makedirs(final_dir)
    if not os.path.exists(temp_dir) : os.makedirs(temp_dir)
    if not os.path.exists(fail_dir) : os.makedirs(fail_dir)

    files = glob('receptor_dwn/*')
    basenames = [ get_base(fn) for fn in files ]
    verbose=True
    n=len(files)
    for index, (fn, base) in enumerate(zip(files, basenames)) : 
        qc_fn = '%s/qc_%s.png'%(qc_dir, os.path.splitext(os.path.basename(fn))[0])
        pass_fn = '%s/qc_%s.png'%(pass_dir, os.path.splitext(os.path.basename(fn))[0])
        temp_crop_fn = '%s/%s.png'%(temp_dir, os.path.splitext(os.path.basename(fn))[0])
        final_crop_fn = '%s/%s.png'%(final_dir, os.path.splitext(os.path.basename(fn))[0])
        
        qc_str = 'crop/v%s/qc/*%s*'%(args.version,base)
        pass_str = 'crop/*/pass/*%s*'%(base)
        fail_str = 'crop/v%s/fail/*%s*'%(args.version,base)
        
        pass_list = glob(pass_str)
        qc_list = glob(qc_str)
        fail_list = glob(fail_str)
        if len(qc_list) == 1 and len(glob(pass_list)) == 1 : os.remove(qc_fn)
        
        if args.version == 5 :
            #Version 5 of the preprocessing is just to copy the autoradiograph 
            # to final_crop_fn and then use an external software (e.g., GIMP)
            #to manually crop the image. Last resort if nothing else has worked.
            #
            if len(pass_list) == 0 :
                shutil.copy(fn, final_crop_fn)
                shutil.copy(fn, pass_fn)
            continue


        if args.process :
            if len(pass_list) > 0 or len(qc_list) > 0 : continue
            print(pass_str)
            print(pass_list)
            preprocess(fn, base, args.version, qc_fn, temp_crop_fn)

        if args.passed :
            if len(pass_list) > 0 : 
                current_version = int(pass_fn.split('/')[1][1]) 
                #print(current_version, pass_fn,'>',final_crop_fn)
                if not os.path.exists(temp_crop_fn) :
                    pass
                    #print('\tpreprocess')
                    #preprocess(fn, base, current_version, qc_fn, temp_crop_fn)
                if not os.path.exists(final_crop_fn) and os.path.exists(temp_crop_fn) :
                    print('\tcopy')
                    shutil.copy(temp_crop_fn, final_crop_fn) 
            else :
                if os.path.exists(final_crop_fn) :
                    os.remove(final_crop_fn)

        if args.final_qc or args.final_check:
            temp_qc_dir=f'crop/combined_final/qc/temp'
            if not os.path.exists(temp_qc_dir) :
                os.makedirs(temp_qc_dir)

            mask_fn=glob(f'crop/combined_final/mask/{base}*.png')
            if len(mask_fn) > 0 : mask_fn = mask_fn[0]
            else : mask_fn = f'crop/combined_final/mask/{base}.png'

            qc_fn = glob( f'crop/combined_final/qc/{base}*.png' )
            if len(qc_fn) > 0 : qc_fn = qc_fn[0]
            else : qc_fn = f'{temp_qc_dir}/{base}.png'
            #else : qc_fn = f'crop/combined_final/qc/{temp_qc_dir}{base}.png'
            
            v=int(pass_fn.split('/')[1][1])
            if (not os.path.exists(qc_fn) or not os.path.exists(mask_fn)) and args.final_check : 
                preprocess(fn, base, v, qc_fn, mask_fn)
            if not os.path.exists(qc_fn) and os.path.exists(mask_fn) : 
                #print(np.round(100.*index/n,3), qc_fn)
                mask = imageio.imread(mask_fn)
                img = imageio.imread(fn)
                plt.subplot(1,2,1)
                plt.imshow(img)
                plt.subplot(1,2,2)
                img[ mask != np.max(mask) ] = 0
                plt.imshow(img)
                plt.tight_layout()
                plt.savefig(qc_fn)
                plt.cla()
                plt.clf()



