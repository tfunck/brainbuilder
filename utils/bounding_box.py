import numpy as np
from utils.utils import *

from scipy.ndimage.morphology import binary_dilation, binary_erosion, binary_fill_holes
def bounding_box_extrema(seg):
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
    return min0, max0, min1, max1

def get_bounding_boxes(labels, nlabels):
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
    xx, yy = np.meshgrid(range(labels.shape[1]), range(labels.shape[0]))
    for l in range(1,nlabels) :
        label_sum =np.sum(labels[labels==l])

        #temp *= 0
        #temp[ labels == l ]=1
        #y0, y1, x0, x1 = bounding_box_extrema(temp)
        xxVals = (xx[labels == l])
        yyVals = (yy[labels == l])
        y0 = min(yyVals)
        y1 = max(yyVals) 
        x0 = min(xxVals)
        x1 = max(xxVals)


        start_check = x0 > d and y0 > d
        end_check = x1 < (dim1-d) and y1 < (dim0-d)
        area_temp = (x1-x0)*(y1-y0)
        #if method == "bounding_box" :  area_temp = (x1-x0)*(y1-y0)
        #elif method == "largest_region" : area_temp=np.sum(temp)
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

def get_largest_region(labels, nlabels):
    okay_list, maybe_list, ignore_list, probably_not_list, boxes, boxSum, area=get_bounding_boxes(labels,nlabels)
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


def get_bounding_box(labels, nlabels):
    okay_list, maybe_list, ignore_list, probably_not_list, boxes, boxSum, area=get_bounding_boxes(labels,nlabels)
    boxes, area = concat_bounding_boxes(okay_list, boxes, area)
    max_area, boxes = check_extra_boxes(okay_list, maybe_list, probably_not_list, area, boxes)
    bounding_box = np.zeros(labels.shape)
    bounding_box[ boxes[max_area] > 0 ] = 1
    bounding_box= binary_dilation(bounding_box, iterations=2).astype(bounding_box.dtype)
    return bounding_box, boxSum
