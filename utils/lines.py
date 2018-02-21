from skimage.transform import probabilistic_hough_line
from skimage.feature import canny
import numpy as np
from scipy.ndimage.morphology import binary_dilation, binary_erosion, binary_fill_holes
import matplotlib.pyplot as plt
from matplotlib import cm
def get_lines(img) :
    edges = canny(img, 1 )
    #plt.imshow( edges, cmap='gray' )
    
    lines = probabilistic_hough_line(edges,  line_length=max(img.shape)/5,  line_gap=4)
    #nLines =len(lines)
    #lineSizes=[]
    #for line in lines:
    #    p0, p1 = line
    #    plt.plot((p0[0], p1[0]), (p0[1], p1[1]))
    #    lineSizes.append(np.sqrt((p1[0] - p0[0])**2 + (p1[1] - p0[1]) **2))
    #if nLines == 0 : return 0
    #else : np.mean(lineSizes)
    #plt.show()
    return lines 


def fill_lines(lines, img_dwn) :
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
    imgf = binary_dilation(imgf, iterations=5).astype(int)
    img_dwn[ imgf == 1 ] = np.median(img_dwn[ imgf != 1 ])
    #plt.imshow( imgf )
    #plt.show()
    return img_dwn
