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
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage import label
from scipy.ndimage.measurements import center_of_mass
from scipy.ndimage.morphology import binary_dilation, binary_erosion, binary_fill_holes
from skimage.morphology import remove_small_holes, reconstruction
from skimage.restoration import inpaint
import pandas as pd
import argparse
from utils.mouse_click import click
from skimage import filters
### Required libraries :
### pip3 install --upgrade --user imageio scikit-image opencv-python


def rgb2gray(rgb): return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

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

def downsample(img, subject_fn, step=1):
    #Calculate length of image based on assumption that pixels are 0.02 x 0.02 microns
    l0 = img.shape[0] * 0.2 
    l1 = img.shape[1] * 0.2
    #Calculate the length for the downsampled image
    dim0=int(np.ceil(l0 / step))
    dim1=int(np.ceil(l1 / step))
    #Calculate the standard deviation based on a FWHM (pixel step size) of downsampled image
    sd0 = step / 2.634 
    sd1 = step / 2.634 
    #Gaussian filter
    img_blr = gaussian_filter(img, sigma=[sd0, sd1])
    #Downsample
    img_dwn = scipy.misc.imresize(img_blr,size=(dim0, dim1),interp='cubic' )
    print("Downsampled:", subject_fn +'img_dwn.jpg')
    scipy.misc.imsave(subject_fn +'img_dwn.jpg', img_dwn)
    return(img_dwn)


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

from skimage.filters import frangi, threshold_otsu, threshold_yen, threshold_li, threshold_minimum, threshold_niblack, threshold_triangle 
from scipy.signal import argrelextrema
def threshold_lines(img_dwn, clustered, subject_fn):
    ar = np.arange(np.min(img_dwn), np.max(img_dwn), np.max(img_dwn)/20)
    t=[]
    val, bin_width  = np.histogram(img_dwn.reshape(-1,1), 10)
    dval = np.diff(val)
    minima = argrelextrema(val, np.less)[0]
    if len(minima) > 0 :
        print(minima)
        print(len(minima))
        min_idx = np.min(minima)
        thr = bin_width[min_idx]
    else : thr = threshold_li(img_dwn)
    #thr =  threshold_otsu(img_dwn)
    #thr =  threshold_li(img_dwn)
    #f_list=[threshold_otsu, threshold_yen, threshold_li, threshold_minimum,  threshold_triangle]
    img_mask=np.zeros(clustered.shape)
    if thr > 0 : img_mask[ img_dwn < thr ] = 1 

    #thr = bin_width[2]
    img_thr = adjust_mask_region( thr, img_dwn, clustered, img_mask)
    
    print("Threshold: ", thr) 
    print('Thresholded Image:', subject_fn +'thr.jpg')
    scipy.misc.imsave(subject_fn +'thr.jpg',  img_thr)
    return img_mask, img_thr, thr


import matplotlib.pyplot as plt
def cluster(img_dwn, subject_fn, nMeans=2, histogram_threshold=False):
    dim1=img_dwn.shape[1]
    dim0=img_dwn.shape[0]

    if histogram_threshold :
        clustered_init = get_kmeans_img(img_dwn, nMeans)
        img_mask, img_thr, thr = threshold_lines(img_dwn, clustered_init, subject_fn)
        img_dwn = img_thr
    else :
        img_thr=img_dwn
        mask_dwn = np.zeros(img_dwn.shape)
        img_mask = np.zeros(img_dwn.shape)
        thr=0
    #Use KMeans clustering with 2 classes
    clustered = get_kmeans_img(img_thr, nMeans)
    clustered_original = np.copy(clustered)
    print('Clustering:', subject_fn +'kmeans.jpg')
    scipy.misc.imsave(subject_fn +'kmeans.jpg',  clustered)
   
    bg = np.median(img_dwn[ clustered == 0 ])

    #Separate clusters into labels
    labels, nlabels = label(clustered, structure=np.ones([3,3]))
    nlabels += 1
    print('Labels:', subject_fn +'labels.jpg')
    plt.imsave(subject_fn +'labels.jpg', labels)
    return img_thr, labels, clustered, nlabels, img_mask, thr, bg



def get_bounding_boxes(labels, nlabels, method):
    temp=np.zeros(labels.shape)
    ignore_list=[]
    maybe_list=[]
    probably_not_list=[]
    okay_list=[]
    dim0=labels.shape[0]
    dim1=labels.shape[1]
    boxes=np.zeros([nlabels] + list(labels.shape))
    boxSum = np.zeros(labels.shape)
    area=np.zeros(nlabels)

    d=int(0.005 * max(labels.shape))
    #print(d)
    for l in range(1,nlabels) :
        temp *= 0
        temp[ labels == l ]=1
        y0, y1, x0, x1 = find_min_max(temp)
    
        start_check = x0 > d and y0 > d
        end_check = x1 < (dim1-d) and y1 < (dim0-d)
        if method == "bounding_box" :  area_temp = (x1-x0)*(y1-y0)
        elif method == "largest_region" : area_temp=np.sum(temp)
        if area_temp > 0 :
            boxes[l,y0:y1, x0:x1] = 1.
            boxSum += boxes[l]
            area[l] = area_temp
            if start_check and end_check and area_temp < 0.95 * np.prod(labels.shape) :
                #print("okay")
                okay_list.append(l)
            elif area_temp > 0.95 * np.prod(labels.shape) : 
                probably_not_list.append(l)
            #else :
                #print("maybe",x0,y0, start_check, end_check)
                #maybe_list.append(l)
        else :
            ignore_list.append(l)
    return okay_list, maybe_list, ignore_list, probably_not_list, boxes, boxSum, area 

def concat_bounding_boxes(okay_list, boxes, area):
    for i in range(len(okay_list)) :
        l=okay_list[i]
        for l0 in okay_list[i:] :
            if l0 != l :
                overlap = np.sum(boxes[l] * boxes[l0] )
                if overlap != 0 :
                    boxes[l] = boxes[l] +boxes[l0]
                    idx = boxes > 0
                    boxes[ idx ] = 1
                    boxes[ ~idx] = 0
                    area[l] = np.sum(boxes[l][ boxes[l] > 0 ] )
                    boxes[l0] = boxes[l] 
                    area[l0] = area[l]
    return boxes, area

def check_extra_boxes(okay_list, maybe_list, probably_not_list, area, boxes) : 
    if okay_list != [] :
        max_area = okay_list[ area[okay_list].argmax()]
    elif maybe_list != [] : 
        max_area = maybe_list[ area[maybe_list].argmax()] 
    else :
        max_area = probably_not_list[ area[probably_not_list].argmax()] 
    temp0=np.zeros(boxes[0].shape)
    temp1=np.zeros(boxes[0].shape)
    for l in maybe_list :
        temp0 *= 0
        temp1 *= 0
        temp0[ boxes[l] > 0 ] = 1
        temp1[ boxes[max_area] > 0 ] = 1
        overlap = np.sum(temp0*temp1)/np.sum(temp1)
        #print(l, np.sum(temp1), np.sum(temp0), overlap)
        if overlap > 0.50 :
            boxes[max_area] = boxes[l] + boxes[max_area]
    return max_area, boxes

def get_largest_region(labels, nlabels,method):
    okay_list, maybe_list, ignore_list, probably_not_list, boxes, boxSum, area=get_bounding_boxes(labels,nlabels, method)
    if okay_list != [] :
        max_area = okay_list[ area[okay_list].argmax()]
    elif maybe_list != [] : 
        max_area = maybe_list[ area[maybe_list].argmax()] 
    elif  probably_not_list != []:
        max_area = probably_not_list[ area[probably_not_list].argmax()]
    else : max_area=0 

    bounding_box = np.zeros(labels.shape)
    bounding_box[labels == max_area] = 1
    return bounding_box, bounding_box


def get_bounding_box(labels, nlabels,method):
    okay_list, maybe_list, ignore_list, probably_not_list, boxes, boxSum, area=get_bounding_boxes(labels,nlabels, method)
    boxes, area = concat_bounding_boxes(okay_list, boxes, area)
    max_area, boxes = check_extra_boxes(okay_list, maybe_list, probably_not_list, area, boxes)
    bounding_box = np.zeros(labels.shape)
    bounding_box[ boxes[max_area] > 0 ] = 1
    bounding_box= binary_dilation(bounding_box, iterations=2).astype(bounding_box.dtype)
    return bounding_box, boxSum

def crop(img, img_dwn, bounding_box_dwn, mask_dwn, thr=0, bg=0) :
    mask=None
    #Create downsampled cropped image
    if np.sum(bounding_box_dwn) != 0 :
        y0, y1, x0, x1 =  find_min_max(bounding_box_dwn)
    else : 
        y0=x0=0
        y1=bounding_box_dwn.shape[0]
        x1=bounding_box_dwn.shape[1]

    yr = y1 - y0
    xr = x1 - x0
    cropped_dwn = np.zeros([yr,xr])   
    cropped_dwn = img_dwn[y0:y1,x0:x1] * bounding_box_dwn[y0:y1,x0:x1]
    cropped_dwn[bounding_box_dwn[y0:y1,x0:x1]== 0 ] = bg
    #Create cropped image at full resolution
    bounding_box = scipy.misc.imresize(bounding_box_dwn,size=(img.shape[0],img.shape[1]) )

    if np.sum(bounding_box) != 0 :
        q0, q1, r0, r1 =  find_min_max(bounding_box)
    else : 
        q0=r0=0
        q1=bounding_box.shape[0]
        r1=bounding_box.shape[1]

    qr = q1 - q0
    rr = r1 - r0
    cropped = np.zeros([qr,rr])   
    cropped = img[q0:q1,r0:r1] * bounding_box[q0:q1,r0:r1]
    cropped[bounding_box[q0:q1,r0:r1]== 0 ] = bg
    if np.sum(mask_dwn) > 0 : 
        mask = scipy.misc.imresize(mask_dwn,size=(img.shape[0],img.shape[1]), interp="nearest" )
        mask_cropped = mask[q0:q1,r0:r1]
        mask_dwn_cropped = mask_dwn[y0:y1,x0:x1]
        cropped = adjust_mask_region(thr, cropped, np.array(bounding_box[q0:q1,r0:r1]),np.zeros([qr,rr]) )
        cropped_dwn = adjust_mask_region(thr, cropped_dwn, np.array(bounding_box_dwn[y0:y1,x0:x1]),np.zeros([yr,xr]))

    return cropped, cropped_dwn



def find_min_max(seg):
    #Flatten series along axis 1
    sum_series_0 = np.sum(seg, axis=1)
    #Flatten series along axis 0
    sum_series_1 = np.sum(seg, axis=0)
    #Create two arrays that number from 0 to length of the flattened series
    av0=np.arange(len(sum_series_0))
    av1=np.arange(len(sum_series_1))
    #Get the range values that are larger than 0 in flattened series
    ar0 = av0[ sum_series_0 > 0]
    ar1 = av1[ sum_series_1 > 0]
    #Calculate min/max
    max0 = int(np.max(ar0))
    min0 = int(np.min(ar0))
    max1 = int(np.max(ar1))
    min1 = int(np.min(ar1))
    return min0,max0,min1,max1

def save_qc(img_dwn, img_thr, clustered, labels, boxSum, cropped_dwn,out_fn,manual_check):
    dim0=img_dwn.shape[0]
    dim1=img_dwn.shape[1]
    qc=np.zeros([ dim0*2, dim1*3 ])
    qc[0:dim0,0:dim1]            = img_dwn / img_dwn.max()
    qc[0:dim0,(1*dim1):(2*dim1)] = img_thr / img_thr.max()
    qc[0:dim0,(2*dim1):(3*dim1)] = clustered/clustered.max()

    qc[dim0:(dim0*2),0:(1*dim1) ] = labels/labels.max()
    qc[dim0:(dim0*2),(1*dim1):(2*dim1) ] = boxSum/boxSum.max()
    if np.sum(cropped_dwn) != 0 :
        qc[dim0:(cropped_dwn.shape[0]+dim0),(2*dim1):(2*dim1+cropped_dwn.shape[1]) ]=cropped_dwn/cropped_dwn.max()
    print("QC :", out_fn)
    scipy.misc.imsave(out_fn, qc)

    if manual_check :
        # Create figure and axes
        plt.clf()
        # Display the image
        plt.imshow(qc,cmap='gray' )
        plt.show()
        while True :
            response = input("Does image pass QC? (y/n)")
            if response == "y" : return 0
            elif response == "n" : return 1
            print("Click and drag with left mouse to draw rectangle.")
            print("Warning: can crash if you don't start rectangle from top left corner!")
            print("If the 'r' key is pressed, reset the cropping region.")
            print("If the 'c' key is pressed, break from the loop.")

    return 0

def crop_gui(subject_output_base, img_dwn, img, qc_fn ):
    refPts = click(subject_output_base+"img_dwn.jpg")
    try :
        x0=refPts[0][0]
        y0=refPts[0][1]
        x1=refPts[1][0]
        y1=refPts[1][1]
        manual_crop_dwn = np.zeros(img_dwn.shape)
        manual_crop_dwn[y0:y1,x0:x1] = 1
        #cropped = manual_crop * img
        cropped, cropped_dwn = crop(img, img_dwn, manual_crop_dwn ) 
        scipy.misc.imsave(qc_fn, cropped_dwn)
    except TypeError :
        return [], []
    return cropped, cropped_dwn


from PIL import Image, ImageDraw
def crop_source_files(source_files, output_dir, downsample_step=1, clobber=False, manual_check=False, manual_only=False, histogram_threshold=False, method="bounding_box") :
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
                #Cluster the image with KMeans
                img_thr,labels,clustered,nlabels,mask_dwn,thr,bg=cluster(img_dwn,subject_output_base, histogram_threshold = histogram_threshold)

                if method == "bounding_box" :
                    #Get bounding box
                   bounding_box_dwn, boxSum = get_bounding_box(labels, nlabels,method)
                elif method == "largest_region" :
                   bounding_box_dwn, boxSum = get_largest_region(labels, nlabels,method)

                #Crop image
                cropped, cropped_dwn = crop(img, img_dwn, bounding_box_dwn, mask_dwn, bg=bg)

                #Quality Control
                qc_status = save_qc(img_dwn, img_thr, clustered, labels, boxSum, cropped_dwn, qc_fn, manual_check)

            if qc_status != 0 or manual_only : 
                #Automated cropping failed to pass QC, use manual QC
                cropped, cropped_dwn = crop_gui(subject_output_base, img_dwn, img, qc_fn)
            
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
    parser.add_argument('--step', dest='downsample_step', default="1", type=int, help='File extension for input files (default=.tif)')
    parser.add_argument('--method', dest='method', default="bounding_box", type=str, help='Method for automated cropping (default = bounding_box). Implemented methods: bounding_box, largest_region')
    parser.add_argument('--manual', dest='manual_check', action='store_true', default=False, help='Do QC and manually crop region if automated method fails')
    parser.add_argument('--manual-only', dest='manual_only', action='store_true', default=False, help='Only do manual cropping (default=False)')
    parser.add_argument('--histogram-threshold', dest='histogram_threshold', action='store_true', default=False, help='Only do manual cropping (default=False)')
    parser.add_argument('--clobber', dest='clobber', action='store_true', default=False, help='Clobber results')

    args = parser.parse_args()
    if not os.path.exists(args.output_dir) : os.makedirs(args.output_dir)
    source_files = glob(args.source_dir+os.sep+"*"+args.ext)
    crop_source_files(source_files, args.output_dir, downsample_step=args.downsample_step, clobber=args.clobber, manual_check=args.manual_check, manual_only=args.manual_only, histogram_threshold=args.histogram_threshold, method=args.method)

