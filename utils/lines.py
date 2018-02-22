from skimage.transform import probabilistic_hough_line
from skimage.feature import canny
import numpy as np
from scipy.ndimage.morphology import binary_dilation, binary_erosion, binary_fill_holes
import scipy.signal
import matplotlib.pyplot as plt
from matplotlib import cm

from skimage.filters import threshold_otsu, threshold_yen, threshold_li

def remove_with_filter(img, fil) :
    temp = np.copy(img)
    img_fil =  scipy.signal.convolve2d(temp, fil, mode='same')

    thr_fil, = np.percentile(img_fil, [95]) #threshold_otsu(img_fil)

    mask=np.zeros(img.shape)
    mask[ img_fil > thr_fil] = 1
    mask = binary_dilation(mask, iterations=1).astype(int)

    img[ mask == 1] = np.median(img[mask==0])
    return img, mask

def fill_lines(img, n=5) :
    img = (img - np.mean(img))/np.std(img)
    mid = int((n-1)/2)
    fil0=np.ones((n,n))*-2
    fil0[ :, mid]=1
    fil0[mid, :]=1
    fil0 = fil0 / np.sum(np.abs(fil0))
    img, mask = remove_with_filter(img, fil0)
    #fil1=np.ones((n,n))*-2
    #fil1[range(n),range(n)]=fil1[range(n-1,-1,-1),range(n)]=1
    #fil1 = fil1 / np.sum(np.abs(fil1))
    #img = remove_with_filter(img, fil1)

    return img, mask 


'''def fill_lines(lines, img_dwn) :
    imgf= np.zeros(img_dwn.shape)
    for l in lines :
        x0 = l[0][0]
        y0 = l[0][1]
        x1 = l[1][0]
        y1=  l[1][1]
        m = (y1-y0)/(0.00001+x1-x0)
        b = y1 - m*x1
        X=np.linspace(x0,x1, img_dwn.shape[1]*2)
        Y=b+m*X
        for x,y in zip(X,Y): imgf[int(round(y)),int(round(np.round(x)))] = 1
        #p0, p1 = l
        #plt.plot((p0[0], p1[0]), (p0[1], p1[1]))
    imgf = binary_dilation(imgf, iterations=7).astype(int)
    img_dwn[ imgf == 1 ] = np.median(img_dwn[ imgf != 1 ])
    #plt.imshow( imgf )
    #plt.show()
    return img_dwn
    '''
