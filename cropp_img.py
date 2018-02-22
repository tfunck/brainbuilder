import cv2
from skimage.filters import  threshold_otsu
from scipy.ndimage.morphology import binary_dilation, binary_erosion, binary_fill_holes
import bisect
from utils.utils import *
import numpy as np
from scipy.ndimage import label
from utils.bounding_box import get_bounding_box
from utils.lines import  fill_lines
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

    im6 = binary_dilation(im6, iterations=3).astype(int)

    return im6


from utils.anisotropic_diffusion import *
def remove_border_regions(cc) :
    border_box = np.zeros(cc.shape)
    border_box[:,cc.shape[1]-1]=1
    border_box[:,0]=1
    border_box[0,:]=1
    border_box[cc.shape[0]-1,:]=1
    border_box = border_box * cc

    cc_unique = np.unique(cc)
    cc_unique = cc_unique[ cc_unique != 0 ]
    numPixels = np.array([np.sum(cc == l) for l in cc_unique  ])

    #for c, n in zip(range(len(cc_unique)), numPixels):
    #    print(c,n)

    for l in cc_unique :
        if True in (border_box == l) and (l != numPixels.argmax()+1)  : 
            cc[cc == l] = 0
    return cc


def cropp_img(img, use_remove_lines=False):
    '''
    Use histogram thresholding and morphological operations to create a binary image that is equal to 1 for pixels which will be kept after cropping.

    Input : 
        img --  image to crop
    Output :
        im5 --  bounding box
        cc0  --  connected labeled regions
    '''
    img = imadjust(img) #cv2.equalizeHist(im1)
    img_hist_adj = np.copy(img)

    if use_remove_lines :
        img, mask = fill_lines(np.copy(img))
    img_lines_removed=np.copy(img)

    thr = threshold_otsu(img)
    img_thr = np.zeros(img.shape)
    img_thr[ img < thr] = 1
    
    cc, nlabels = label(img_thr, structure=np.ones([3,3]))
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

    return bounding_box, img_hist_adj, img_lines_removed, img_thr, cc
    
