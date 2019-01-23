import cv2
from skimage.filters import  threshold_otsu
from scipy.ndimage.morphology import binary_dilation, binary_erosion, binary_fill_holes
from utils.utils import *
import numpy as np
from scipy.ndimage import label
from utils.bounding_box import get_bounding_box
from utils.lines import  fill_lines
from skimage.filters import threshold_otsu, threshold_yen, threshold_li
from scipy.ndimage.filters import gaussian_filter
from utils.anisotropic_diffusion import *
import matplotlib.pyplot as plt 
from skimage.exposure import  equalize_hist
from scipy.ndimage.morphology import binary_closing
def curvature(img):
    xx, yy = np.gradient(img, edge_order=2)
    img = np.sqrt(xx**2 + yy**2)
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

def get_kmeans_img(img_dwn, nMeans=3):
    init=np.percentile(img_dwn, [0.1,0.5,0.99]).reshape(3,1)
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
        temp = binary_erosion(temp, iterations=5)
        temp = binary_dilation(temp, iterations=5)
        clustering[ temp == 1 ] = 1 #temp[clustering==n]

    return(clustering)


def binary_mask(im5):
    ymin, ymax, xmin, xmax, yi, xi = find_min_max(im5)

    im6 = np.zeros(im5.shape)   
    for y0, y1, i in zip(ymin, ymax, xi ) :
        if y0 != y1 :
            im6[y0:y1,i] = 1
    for x0, x1, i in zip(xmin, xmax, yi ) :
        if x0 != x1 :
            im6[i,x0:x1] += 1

    idx= im6 > 0 
    im6[idx ] = 1
    im6[~idx]=0

    im6 = binary_dilation(im6, iterations=3).astype(int)

    return im6


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

    numPixels_sort=np.sort(numPixels)
    max_l=cc_unique[numPixels.argmax()]
    #for c, n in zip(range(len(cc_unique)), numPixels):
    #    print(c,n)
    
    #if there is more than one labelled region
    if len(numPixels_sort) > 1 :
        ratio =   numPixels_sort[-2] / numPixels_sort[-1]
    else : 
        ratio=0
    print(ratio)a
    #for each label in the set of labels
    for l in cc_unique :
        #if there is overlap between the bounding box and the labelled region
        if True in (border_box == l) :
            #print(ratio, l, max_l)
            #If the current label is that for the largest region
            #and the next largest region is much smaller than the current region,
            #then ignore the overlap
            if  ratio < 0.05  and (l == max_l )  : continue
            #Otherwise set the labelled region to 0
            cc[cc == l] = 0
    return cc


from sklearn.cluster import spectral_clustering, DBSCAN, KMeans
import cv2.bgsegm 
from scipy import ndimage as ndi
from skimage import morphology
from skimage import filters

def find_bb_overlap(cc, cc_unique):
    temp=np.zeros(cc.shape)
    overlap=np.zeros(cc.shape)

    for i in cc_unique :
        temp *= 0
        temp[cc==i]=1
        ymin, ymax, xmin, xmax, yi, xi = find_min_max(temp)
        overlap[min(ymin) : max(ymax), min(xmin) : max(xmax)] = 1

    overlap, labels=label(overlap,structure=np.ones([3,3]))
    return overlap


def cropp_img(img, line, no_frame=False, low_contrast=False, plot=False):
    '''
    Use histogram thresholding and morphological operations to create a binary image that is equal to 1 for pixels which will be kept after cropping.

    Input : 
        img --  image to crop
    Output :
        im5 --  bounding box
        cc0  --  connected labeled regions
    '''
    #img =cv2.equalizeHist( imadjust(img)) #cv2.equalizeHist(im1)
    #img = imajust(img)
    if plot :
        plt.subplot(2,2,1)
        plt.imshow(img)

    img = gaussian_filter(img, sigma=1)
    if low_contrast : 
        img = equalize_hist(img)
    if plot :
        plt.subplot(2,2,2)
        plt.imshow(img)
    img_smoothed = img
    
    t=threshold_li(img)
    img_thr = np.zeros(img.shape)
    img_thr[ img < t ] = 1

    if no_frame : 
        img_thr[ line != 0 ] = 1
    else : 
        img_thr[ line != 0 ] = 0

    #img_thr = binary_dilation(binary_erosion(img_thr, iterations=5), iterations=5)
    cc, nlabels = label(img_thr, structure=np.ones([3,3]))

    if plot :
        plt.subplot(2,2,3)
        plt.imshow(img_thr)

    cc = remove_border_regions(cc)
    
    cc_unique = np.unique(cc)
    if len(cc_unique) >= 2 : cc_unique = np.delete(cc_unique, 0)
    
    cc2 = find_bb_overlap(cc, cc_unique)
    
    cc2_unique = np.unique(cc2)
    if len(cc2_unique) >= 2 : cc2_unique = np.delete(cc2_unique, 0)
    

    numPixels = np.array([np.sum(cc2 == l) for l in cc2_unique  ])

    #Keep only largest bounding box

    #Find index of largest region
    idx = numPixels.argmax()
    im4=np.zeros(img.shape);
    #Label largest region 1
    im4[ cc2 == cc2_unique[ idx ]  ] = 1;
    im4 = im4 * cc 
    im4[ im4 > 0 ] = 1
    
    bounding_box = binary_mask(im4)

    if plot :
        plt.subplot(2,2,4)
        plt.imshow(bounding_box)
        plt.show()

    return bounding_box, img_smoothed, img_thr, cc
    
