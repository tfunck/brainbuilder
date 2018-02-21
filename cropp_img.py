import cv2
from skimage.filters import  threshold_otsu
from scipy.ndimage.morphology import binary_dilation, binary_erosion, binary_fill_holes
import bisect
from utils.utils import *
import numpy as np
from scipy.ndimage import label
from utils.bounding_box import get_bounding_box
from utils.lines import get_lines, fill_lines
from skimage.filters import threshold_otsu, threshold_yen, threshold_li
from scipy.ndimage.filters import gaussian_filter

def imadjust(src, tol=1, vin=[0,255], vout=(0,255)):
    # src : input one-layer image (numpy array)
    # tol : tolerance, from 0 to 100.
    # vin  : src image bounds
    # vout : dst image bounds
    # return : output img

    dst = src.copy()
    tol = max(0, min(100, tol))

    if tol > 0:
        # Compute in and out limits
        # Histogram
        hist = np.zeros(256, dtype=np.int)
        for r in range(src.shape[0]):
            for c in range(src.shape[1]):
                hist[src[r,c]] += 1
        # Cumulative histogram
        cum = hist.copy()
        for i in range(1, len(hist)):
            cum[i] = cum[i - 1] + hist[i]

        # Compute bounds
        total = src.shape[0] * src.shape[1]
        low_bound = total * tol / 100
        upp_bound = total * (100 - tol) / 100
        vin[0] = bisect.bisect_left(cum, low_bound)
        vin[1] = bisect.bisect_left(cum, upp_bound)

    # Stretching
    scale = (vout[1] - vout[0]) / (vin[1] - vin[0])
    for r in range(dst.shape[0]):
        for c in range(dst.shape[1]):
            vs = max(src[r,c] - vin[0], 0)
            vd = min(int(vs * scale + 0.5) + vout[0], vout[1])
            dst[r,c] = vd
    return dst

def get_connected_regions(img):

    return cc, nlabels
import matplotlib.pyplot as plt 

def binary_mask(im5):
    ymin, ymax, xmin, xmax = find_min_max(im5)

    im6 = np.zeros(im5.shape)   
    for y0, y1, i in zip(ymin, ymax, range(len(xmax)) ) :
        if y0 != y1 :
            im6[y0:y1,i] = 1
    for x0, x1, i in zip(xmin, xmax, range(len(ymax)) ) :
        if x0 != x1 :
            im6[i,x0:x1] += 1

    idx= im6 > 0 
    im6[idx ] = 1
    im6[~idx]=0
    return im6


def extrema(labels) :
    xx, yy = np.meshgrid(range(labels.shape[1]), range(labels.shape[0]))
    xxVals = (xx[labels > 0])
    yyVals = (yy[labels > 0])
    y0 = min(yyVals)
    y1 = max(yyVals) 
    x0 = min(xxVals)
    x1 = max(xxVals)
    return([x0,y0,x1,y1])

def remove_border_regions(cc) :
    border_box = np.zeros(cc.shape)
    border_box[:,cc.shape[1]-1]=1
    border_box[:,0]=1
    border_box[0,:]=1
    border_box[cc.shape[0]-1,:]=1
    border_box = border_box * cc
    for l in np.unique(cc) :
        if True in (border_box == l) : 
            cc[cc == l] = 0
    return cc

'''
def remove_lines(img,n=100, thr=1) :
    v, bin_widths = np.histogram(img, n)
    v = v / np.median(v)
    med = np.median(v)
    for i in range(n) :
        if v[i] < thr*med : 
            print(bin_widths[i], thr*med )
            img[ img < bin_widths[i] ] = np.median(img)
            break
    return img
'''

def cropp_img(img, use_remove_lines=False):
    '''
    Use histogram thresholding and morphological operations to create a binary image that is equal to 1 for pixels which will be kept after cropping.

    Input : 
        img --  image to crop
    Output :
        im5 --  bounding box
        cc0  --  connected labeled regions
    '''
    img = gaussian_filter(img, sigma=1)
    use_remove_lines=True
    if use_remove_lines :
        lines = get_lines(img)
        img = fill_lines(lines, img)
        

    im1 = imadjust(img)
    im_hist_adj = im1 #cv2.equalizeHist(im1)
    thr = threshold_li(im_hist_adj)
    im_thr = np.zeros(img.shape)
    im_thr[ im_hist_adj < thr] = 1
    cc, nlabels = label(im_thr, structure=np.ones([3,3]))
    nlabels += 1

    cc = remove_border_regions(cc)

    cc_unique = np.unique(cc)
    if len(cc_unique) >= 2 :
        cc_unique = np.delete(cc_unique, 0)
    numPixels = np.array([np.sum(cc == l) for l in cc_unique  ])
    min_area = max(numPixels)*0.1
    idx = cc_unique[numPixels > min_area]#+1
    im4=np.zeros(img.shape);
    for i in idx :
        im4[ cc == i  ] = 1;
    
    bounding_box = binary_mask(im4)

    return bounding_box, im_hist_adj, im_thr, cc
    
