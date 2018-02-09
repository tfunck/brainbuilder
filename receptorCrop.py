import numpy as np
import scipy.misc
from sklearn.cluster import spectral_clustering, DBSCAN, KMeans
from glob import glob
from sklearn.feature_extraction import image
from sys import exit, argv
import os
import matplotlib.pyplot as plt
import imageio
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage import label
from scipy.ndimage.measurements import center_of_mass
from scipy.ndimage.morphology import binary_dilation, binary_erosion
import pandas as pd
import argparse

def rgb2gray(rgb): return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def downsample(img, subject_fn, step0=0.25, step1=0.25):
    #Calculate length of image based on assumption that pixels are 0.02 x 0.02 microns
    l0 = img.shape[0] * 0.2 
    l1 = img.shape[1] * 0.2
    #Calculate the length for the downsampled image
    dim0=int(np.ceil(l0 / step0))
    dim1=int(np.ceil(l1 / step1))
    #Calculate the standard deviation based on a FWHM (pixel step size) of downsampled image
    sd0 = step0 / 2.634 
    sd1 = step1 / 2.634 
    #Gaussian filter
    img_blr = gaussian_filter(img, sigma=[sd0, sd1])
    #Downsample
    img_dwn = scipy.misc.imresize(img_blr,size=(dim0, dim1),interp='cubic' )
    print("Downsampled:", subject_fn +'img_dwn.jpg')
    scipy.misc.imsave(subject_fn +'img_dwn.jpg', img_dwn)
    return(img_dwn)

def cluster(img_dwn, subject_fn, nMeans=2):
    #Use KMeans clustering with 2 classes
    #db = KMeans(nMeans).fit_predict(img_dwn.reshape(-1,1))
    db = KMeans(nMeans).fit_predict(img_dwn.reshape(-1,1))
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

    for n in range(1, nMeans) :
        temp = np.zeros(img_dwn.shape)
        temp[clustering == n ] =1 
        temp = binary_erosion(temp, iterations=1)
        temp = binary_dilation(temp, iterations=1)
        clustering[ clustering == n ] = temp[clustering==n]
    #Separate clusters into labels
    labels, nlabels = label(clustering, structure=np.ones([3,3]))
    print('Clustering:', subject_fn +'kmeans.jpg')
    print('Labels:', subject_fn +'labels.jpg')
    scipy.misc.imsave(subject_fn +'labels.jpg', labels)
    scipy.misc.imsave(subject_fn +'kmeans.jpg', clustering)
    return labels, clustering, nlabels

def get_bounding_boxes(labels, nlabels):
    boxes=np.zeros([nlabels] + list(labels.shape))
    boxSum = np.zeros(labels.shape)
    area=np.zeros(nlabels)
    ignore_list=[]
    dim0=labels.shape[0]
    dim1=labels.shape[1]
    #print(labels.shape)
    #print("start", dim0 * 0.1, dim1 * 0.1)
    #print("end", dim0 * 0.9,  dim1 * 0.9)
    for l in range(nlabels) :
        temp=np.zeros(labels.shape)
        temp[ labels == l ]=1
        y0, y1, x0, x1 = find_min_max(temp)
        intersect = len(set([y0, y1, x0, x1]).intersection([0]+list(labels.shape))) 
    
        start_check = x0 > dim1 * 0.01 and y0 > dim0 * 0.01
        end_check = x1 < dim1 * 0.99 and y1 < dim0 * 0.99
        area_temp =  (x1-x0)*(y1-y0)

        if area_temp > 2 and start_check and end_check : 
            boxes[l,y0:y1, x0:x1] = 1.
            boxSum += boxes[l]
            area[l] =area_temp
            #if area_temp == area.max() :
            #    print(area_temp, y0, y1, x0, x1)
        else:
            ignore_list.append(l)
        del temp
   
    for l in range(nlabels) :
        for l0 in range(l,nlabels) :
            if l0 in ignore_list : continue
            
            if l0 != l :
                overlap = np.sum(boxes[l] * boxes[l0] )
                if overlap != 0 :
                    boxes[l] = boxes[l] +boxes[l0]
                    boxes[l0] *= 0 #np.zeros(labels.shape)
                    area[l] += area[l0]
                    area[l0]=0
                    ignore_list.append(l0)
   
    max_area = area.argmax()
    bounding_box = np.zeros(labels.shape)
    bounding_box[ boxes[max_area] > 0 ] = 1
    bounding_box= binary_dilation(bounding_box, iterations=2).astype(bounding_box.dtype)
    #scipy.misc.imsave('test.jpg', boxSum)
    return bounding_box, boxSum



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

def crop(source_files, output_dir, nMeans, clobber=True) :
    for f in source_files :
        #Set output filename
        fsplit = os.path.splitext(os.path.basename(f))
        qc_dir = output_dir + os.sep + "qc" + os.sep 
        subject_output_dir = output_dir + os.sep + fsplit[0] + os.sep 
        subject_output_base = subject_output_dir+fsplit[0]+'_'
        fout = output_dir + os.sep + fsplit[0]+'_cropped'+fsplit[1]
        if "cropped" in f : continue 
        if not os.path.exists(fout) or clobber :
            #if fsplit[0] != "OL#hg#MR1s6#L#musc#6400#06" : continue 
            print("Input:", f)
            if not os.path.exists(subject_output_dir) : os.makedirs(subject_output_dir)
            if not os.path.exists(qc_dir) : os.makedirs(qc_dir)
            #Load image and convert from rgb to grayscale
            img = imageio.imread(f)
            if len(img.shape) > 2 : img = rgb2gray(img)

            #Downsampled the image to make processing faster
            img_dwn = downsample(img, subject_output_base, 1, 1)

            #Cluster the image with KMeans
            labels, clustering, nlabels = cluster(img_dwn, subject_output_base, nMeans)
            bounding_box_dwn, boxSum = get_bounding_boxes(labels, nlabels)

            bounding_box = scipy.misc.imresize(bounding_box_dwn,size=(img.shape[0],img.shape[1]),interp='nearest' )

            #Crop image
            #cropped = img[min0:max0,min1:max1]
            cropped = bounding_box * img
            scipy.misc.imsave(fout, cropped)
            
            #Quality Control
            boxSum_up = scipy.misc.imresize(boxSum,size=(img.shape[0],img.shape[1]),interp='nearest' )
            clustering_up = scipy.misc.imresize(clustering,size=(img.shape[0],img.shape[1]),interp='nearest' )
            qc=np.zeros([ img.shape[0]*2,  img.shape[1]*2 ])
            qc[0:img.shape[0],0:img.shape[1]]=img / img.max()
            qc[0:img.shape[0],img.shape[1]:(2*img.shape[1])]=clustering_up/clustering_up.max()

            qc[img.shape[0]:(img.shape[0]*2),0:(1*img.shape[1]) ]=boxSum_up/boxSum_up.max()
            qc[img.shape[0]:(img.shape[0]*2),img.shape[1]:(2*img.shape[1]) ]=cropped/cropped.max()
            scipy.misc.imsave(qc_dir+os.sep+fsplit[0]+'_cropped_qc.png', qc)

            print("Cropped:", fout,"\n")    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--source', dest='source_dir',  help='Directory with raw images')
    parser.add_argument('--output', dest='output_dir',  help='Directory name for outputs')
    parser.add_argument('--ext', dest='ext', default=".tif", help='File extension for input files (default=.tif)')
    parser.add_argument('--kmeans', dest='nMeans', type=int, default=2, help='Number of means for KMeans (default=2)')

    args = parser.parse_args()
    if not os.path.exists(args.output_dir) : 
        os.makedirs(args.output_dir)
    source_files = glob(args.source_dir+os.sep+"*"+args.ext)
    crop(source_files, args.output_dir, args.nMeans)

