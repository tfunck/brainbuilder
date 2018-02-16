import numpy as np



def rgb2gray(rgb): return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def find_min_max(seg):

    fmin = lambda a : min(np.array(range(len(a)))[a == 1])
    fmax = lambda a : max(np.array(range(len(a)))[a == 1])
    xmax = [ fmax(seg[i,:]) if np.sum(seg[i,:]) != 0 else 0  for i in range(seg.shape[0]) ]
    xmin = [ fmin(seg[i,:]) if np.sum(seg[i,:]) != 0 else 0  for i in range(seg.shape[0]) ]

    ymax = [ fmax(seg[:,i])  if np.sum(seg[:,i]) != 0 else 0  for i in range(seg.shape[1]) ]
    ymin = [ fmin(seg[:,i])  if np.sum(seg[:,i]) != 0 else 0 for i in range(seg.shape[1]) ]


    #Flatten series along axis 1
    #sum_series_0 = np.sum(seg, axis=1)
    #Flatten series along axis 0
    #sum_series_1 = np.sum(seg, axis=0)
    #Create two arrays that number from 0 to length of the flattened series
    #av0=np.arange(len(sum_series_0))
    #av1=np.arange(len(sum_series_1))
    #Get the range values that are larger than 0 in flattened series
    #ar0 = av0[ sum_series_0 > 0]
    #ar1 = av1[ sum_series_1 > 0]
    #Calculate min/max
    #max0 = int(np.max(ar0))
    #min0 = int(np.min(ar0))
    #max1 = int(np.max(ar1))
    #min1 = int(np.min(ar1))
    return ymin,ymax,xmin,xmax

