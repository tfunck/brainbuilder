import numpy as np
from scipy.ndimage.filters import gaussian_filter
import scipy

def downsample(img, subject_fn="", step=0.1):
    #Calculate length of image based on assumption that pixels are 0.02 x 0.02 microns
    l0 = img.shape[0] * 0.02 
    l1 = img.shape[1] * 0.02
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
    if subject_fn != "" : 
        print("Downsampled:", subject_fn +'img_dwn.jpg')
        scipy.misc.imsave(subject_fn +'img_dwn.jpg', img_dwn)
    return(img_dwn)

#def rgb2gray(rgb): return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
def rgb2gray(rgb): return np.mean(rgb, axis=2)

def find_min_max(seg):

    fmin = lambda a : min(np.array(range(len(a)))[a == 1])
    fmax = lambda a : max(np.array(range(len(a)))[a == 1])
    xmax = [ fmax(seg[i,:]) if np.sum(seg[i,:]) != 0 else 0  for i in range(seg.shape[0]) ]
    xmin = [ fmin(seg[i,:]) if np.sum(seg[i,:]) != 0 else 0  for i in range(seg.shape[0]) ]

    ymax = [ fmax(seg[:,i])  if np.sum(seg[:,i]) != 0 else 0  for i in range(seg.shape[1]) ]
    ymin = [ fmin(seg[:,i])  if np.sum(seg[:,i]) != 0 else 0 for i in range(seg.shape[1]) ]

    return ymin,ymax,xmin,xmax

def extrema(labels) :
    xx, yy = np.meshgrid(range(labels.shape[1]), range(labels.shape[0]))
    xxVals = (xx[labels > 0])
    yyVals = (yy[labels > 0])
    y0 = min(yyVals)
    y1 = max(yyVals) 
    x0 = min(xxVals)
    x1 = max(xxVals)
    return([x0,y0,x1,y1])



