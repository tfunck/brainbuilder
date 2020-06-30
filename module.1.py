import argparse
import os
import json
import h5py as h5
import numpy as np
import pandas as pd
from utils.utils import safe_imread, downsample
import numpy as np
import matplotlib.pyplot as plt
import h5py as h5
import tensorflow as tf
import nibabel as nib
import pandas as pd
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
from skimage.filters import threshold_otsu, threshold_li , threshold_sauvola
from skimage.segmentation import clear_border
from skimage.exposure import equalize_hist
from skimage.morphology import disk
from scipy.ndimage import label

def remove_lines(im, lines_in) :
    '''
    Purpose : Remove lines and replace with min of image
    Inputs :
        im - image from which lines are to be removed
        lines_in - image with lines
    Outputs :
        im_out - image without lines
    '''
    im_out = np.copy(im)
    lines =np.copy(lines_in)
    lines[lines > 0] = 1
    im_out[ lines == 1 ] = np.min(im)

    return im_out

def segment_tissue(im) :
    '''
    Purpose : Segement tissue in image with Otsu thresholding
    Inputs : 
        im - image to be segmented
    Outputs :
        out_otsu - segmented image
    '''
    out_otsu =np.zeros_like(im)
    out_otsu[ im > threshold_otsu(im) ] = 1
    return out_otsu

def downsample_min(bin_img, dim, threshold=0.5, order=1, clobber=False):
    '''
    Purpose : Downsample binary image and set all values below threshold to 0. Allows preservation of space
                between sulci.
    Inputs : 
        bin_img     -    binary image to be resampled
        dim         -   dimensions to which image will be resampled
        threshold   -   value below which pixels set to 0
    Outputs :
        bin_img     -   resampled and threshold binary image
    '''
    bin_img = resize(bin_img.astype(np.float64), dim, order=order,anti_aliasing=True)
    if np.max(bin_img)-np.min(bin_img) == 0 :
        print('Error downsampling. Image min == Image max ')
        exit(1)
    bin_img = (bin_img - np.min(bin_img))/(np.max(bin_img)-np.min(bin_img))
    bin_img[ bin_img < 0.5 ] = 0
    bin_img[ bin_img > 0] = 1
    return bin_img
   
def qc(img_list, name_list, out_fn, dpi=200):
    '''
    Purpose : Create figure showing images in img_list and with titles in name_list
    Inputs :
        img_list        -   list of 2D numpy arrays
        name_list       -   list of strings giving plot titles
        out_fn          -   string containing output filename
        dpi             -   dots per inch
    Outputs :
        None        
    '''
    row_img = np.floor(np.sqrt(len(img_list))).astype(int)
    col_img = np.ceil(np.sqrt(len(img_list))).astype(int)
    row_img += max(len(img_list) - row_img * col_img  ,0)
    plt.clf()
    plt.figure()
    for j, (name, x) in enumerate(zip(name_list,img_list)) :
        plt.subplot(row_img, col_img, j+1)
        plt.imshow(x.astype(float))
        plt.title(name)
        plt.axis('off')
    plt.savefig(out_fn, dpi=dpi)

def fill_regions(im):
    '''
    Purpose : Fill in binary segementation. Fills holes but also open spaces between contiguous segmented regions
    Inputs :
        im  -   input image to be filled
    Outputs :
        out -   output image with filled regions
    '''
    cc, nlabels = label(im, structure=np.array([[0,1,0],[1,1,1],[0,1,0]]))
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
    return out

def apply_line_detection_model(img, model):
    '''
    Purpose : Apply tensorflow neural network model to input image
    Inputs :
        img     -   input image at 100um (2D numpy array)
        model   -   neural network model stored in hdf5 file
    Outputs :
        lines   -   3 class segmentation of lines in input image
    '''
    # Normalize
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    lines = np.argmax( model.predict( img.reshape([1,img.shape[0],img.shape[1],1])  , batch_size=1), axis=3)
    lines = lines.reshape(lines.shape[1:3])
    for i in [1,2] :
        lines_keep = np.zeros_like(lines)
        lines_keep[ lines == i] = 1
        lines_keep = binary_dilation(lines_keep, iterations=5)
        lines[ lines_keep == 1] = i
    return lines

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



def downsample_lin(dwn_vol_fn, lin_fn_list, step=0.1, clobber=False):
    '''
    Purpose : Downsample set of TIF files and store them in a common hdf5 file. 
    Inputs :
        dwn_vol_fn      -   filename for hdf5 file (string)
        lin_fn_list     -   list of TIF filenames to read and downsample (list)
        step            -   pixel size to which to downsample the images (float)
        clobber         -   overwrite outputs (bool)
    Outputs :
        None
    '''
    img0 = imresize(safe_imread(lin_fn_list[0]), (312,416), interp=3) 

    if not os.path.exists(dwn_vol_fn) or args.clobber :
        dwn_vol = h5.File(dwn_vol_fn,'w')
        dwn_vol.create_dataset("data", (lin_fn_list.shape[0], img0.shape[0], img0.shape[1]), dtype='float32')
    else :
        dwn_vol = h5.File(dwn_vol_fn,'a')

    for idx , lin_fn in enumerate(lin_fn_list) :
        if np.max(dwn_vol['data'][idx,:,:]) == 0 or clobber:
            img = downsample(safe_imread(lin_fn), step=step,  interp=3)
            if img.T.shape == img0.shape :
                img=img.T
            dwn_vol['data'][idx,:,:] = img

    
def remove_overlap_regions(img, lines, region_label=2):
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
    for l in np.unique(labels)[1:] :
        roi *= 0
        roi[ labels == l ] = 1
        if np.sum( roi * lines_2 ) > 0 :
            img[labels==l] = 0
    return img

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--input','-i', dest='source', default='data/',  help='Directory with raw images')
    parser.add_argument('--csv','-c', dest='auto_info_fn', default='autoradiograph_info.csv',  help='csv file with autoradiograph information for each section')
    parser.add_argument('--model-fn','-m', dest='model_fn', default='model.h5', help='h5py file with neural network model to remove visual artefacts from autoradiographs')
    parser.add_argument('--line-fn','-l', dest='line_hdf5_fn', default='model.h5', help='h5py file with labeled line data')
    parser.add_argument('--output','-o', dest='output', default='output/', help='Directory name for outputs')
    parser.add_argument('--slab','-s', dest='slab', type=int, help='Brain slab')
    parser.add_argument('--hemi','-H', dest='hemi', help='Brain hemisphere')
    parser.add_argument('--brain','-b', dest='brain', help='Brain number')
    parser.add_argument('--clobber', dest='clobber', action='store_true', default=False, help='Clobber results')
    parser.add_argument('--verbose','-v', dest='verbose', action='store_true', default=False, help='Print verbose')

    args = parser.parse_args()

    ###################
    # Setup variables #
    ###################
    verbose=args.verbose
    scale_factors_json = args.source + os.sep + "scale_factors.json"
    scale = json.load(open(scale_factors_json,'r'))
    z_mm = scale[args.brain][args.hemi][str(args.slab)]["size"]
    auto_affine = np.array([[z_mm/4164.,0,0,0],[0.0,z_mm/4164.,0,0],[0,0,1.0,0],[0,0,0,1]])
    input_raw = args.source + 'lin' 

    dwn_vol_fn = '%s/vol_brain-%s_hemi-%s_slab-%s_lin_100um.h5'  % (args.output, args.brain, args.hemi, args.slab)
    seg_vol_fn = '%s/vol_brain-%s_hemi-%s_slab-%s_seg_100um.h5'  % (args.output, args.brain, args.hemi, args.slab)
    if not os.path.exists(args.output) : os.makedirs(args.output)
    
    df = pd.read_csv(args.auto_info_fn)
    df = df.loc[ (df['mri'] == args.brain) &  (df['slab'] == args.slab) & ( df['hemisphere'] == args.hemi) ]

    #downsample_lin(dwn_vol_fn, df['lin_fn'].values,  step=0.1, clobber=False)
    #exit(0)
    #vol = h5.File(dwn_vol_fn, 'r')
    #line_data = h5.File(args.line_hdf5_fn, 'r')
    #model = load_model(args.model_fn)
    contrast_df=pd.DataFrame({'ligand':[],'contrast':[]})
    
    qc_dir ='%s/qc'%(args.output)
    if not os.path.exists(qc_dir) : os.makedirs(qc_dir)

    for i in range(0, vol['data'].shape[0]):
        lin_fn = df['lin_fn'].iloc[i]
        out_fn = '%s/qc_%s.png'%(qc_dir, os.path.splitext(os.path.basename(lin_fn))[0])
        print(out_fn)
        if os.path.exists(out_fn) or args.clobber : continue

        im_lo = vol['data'][i]
        im_hi = safe_imread( lin_fn)  
        if im_hi.shape == (4164, 3120) :
            im_hi = im_hi.T
        ligand = df['ligand'].iloc[i]  
        
        ### 1) Apply model to detect lines in im_loage
        if not im_hi in line_data['x_fn'] :
            if verbose : print('\tApply line detection')
            lines = apply_line_detection_model(im_lo, model)    

            ### 2) Resample line array to high resolution
            lines_hi = np.round( resize(lines.astype(np.float32), (3120,4164), order=1), 0)
        else : 
            lines_hi = safe_imread( data['y_fn'][ data['x_fn'] == lin_fn ])

        ### 3) Remove the lines from the image at high resolution
        if verbose : print('\tRemove lines')
        im_no_lines = remove_lines(im_hi, lines_hi)

        ### 4) Adjust image histogram if the contrast is low
        im_contrast = calculate_contrast(im_no_lines)
        contrast_df = pd.concat([contrast_df,pd.DataFrame({'ligand':[ligand],'contrast':[im_contrast]}) ])
        if im_contrast < 0.1 :
            im_no_lines = equalize_hist(gaussian_filter(im_no_lines,3))

        ### 5) K means segmentation
        if verbose : print('\tSegment Tissue')
        seg = segment_tissue(im_no_lines)

        ### 6) Add visual cues to keep onto the segmented image
        lines_2 = np.zeros_like(seg)
        lines_2[lines_hi == 2] = 1
        lines_2 = binary_dilation(lines_2, iterations=10)
        seg[lines_2 == 1] = 1

        ### 7) Downsample segmented image 
        if verbose : print('\tDownsample Segmentation')
        seg_dwn = downsample_min(seg,  im_lo.shape, order=1)
        if im_contrast < 0.1 :
            #seg_dwn = binary_closing(seg_dwn, iterations=5)
            seg_dwn = binary_dilation(seg_dwn, iterations=5)

        ### 8) Denoise the segmented image
        if verbose : print('\tDenoise tissue segmentation')
        seg_dwn = denoise(seg_dwn,  verbose=verbose)

        if verbose : print('\tFill in tissue regions')
        seg_dwn = fill_regions(seg_dwn)
        seg_fill = seg_dwn

        ### 9) Remove pieces of tissue that are connected to border
        if verbose : print('\tClear border')
        seg_dwn = clear_border(seg_dwn, buffer_size=10)

        ### 10) Remove pieces of tissue that are connceted to lines with label 2
        if verbose : print('\tRemove tissue regions connected to artefacts with label = 2')
        crop_lo = remove_overlap_regions(seg_dwn, lines, region_label=2)

        ### 11) Upsample the cropping frame and apply to lin img and tissue segmentation at full resolution
        crop_hi = resize(crop_lo, im_hi.shape, order=0)
        im_hi_cropped = crop_hi*im_hi
        seg_hi_cropped = crop_hi * seg

        img_list=[im_lo, im_no_lines, lines_hi, seg, seg_fill,  seg_dwn]
        name_list=['original-%s'%ligand, 'no lines-%1.2f'%im_contrast, 'lines (hi res)', 'seg. (hi res)','seg - filled', 'seg. - no border (lo res)']
        print('Output Function:',out_fn)
        qc(img_list, name_list, out_fn, dpi=100)

    contrast_df.to_csv('contrast.csv')
    contrast_df = contrast_df.groupby(['ligand']).mean()
    contrast_df.sort_values(['contrast'],inplace=True)
    print(contrast_df)
