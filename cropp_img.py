import cv2
from skimage.filters import  threshold_otsu
from scipy.ndimage.morphology import binary_dilation, binary_erosion, binary_fill_holes
import bisect
from utils.utils import *
import numpy as np
from scipy.ndimage import label

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


def cropp_img(img):
    im1 = imadjust(img)
    im2 = cv2.equalizeHist(im1)
    thr = threshold_otsu(im2)
    im4 = np.zeros(im2.shape)
    im4[ img > thr] = 1

    #se = strel('disk',er_size);
    #se1 = strel('disk',di_size);

    im5 = binary_dilation(binary_erosion(im4, iterations=5), iterations=5)

    #im5 = (imdilate(imerode(im4,se),se1));

    #cc=bwconncomp(im5);
    cc, nlabels = label(im5, structure=np.ones([3,3]))
    nlabels += 1

    im6=np.zeros(im5.shape);

    #numPixels = cellfun(@numel,cc.PixelIdxList);
    #[biggest,idx] = max(numPixels);

    numPixels = np.array([np.sum(cc[cc == l]) for l in range(1,nlabels)])
    biggest = np.max(numPixels)
    idx = numPixels.argmax()+1

    im6[ cc == idx ] = 1;

    #[a,b]=find(im6==1);
    ymin, ymax, xmin, xmax = find_min_max(im6)

    im7 = np.zeros(img.shape)   
    for y0, y1, i in zip(ymin, ymax, range(len(xmax)) ) :
        if y0 != y1 :
            im7[y0:y1,i] = 1
    for x0, x1, i in zip(xmin, xmax, range(len(ymax)) ) :
        if x0 != x1 :
            im7[i,x0:x1] += 1
    idx= im7 > 0
    im7[idx ] = 1
    im7[~idx]=0

    return im7
    
