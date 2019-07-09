import argparse
import numpy as np
import h5py
import os
import matplotlib.pyplot as plt
import scipy
import os
import imageio
import nibabel as nib
import gzip
import shutil
from skimage.filters import threshold_otsu, threshold_li
from scipy.ndimage.filters import gaussian_filter
from sklearn.cluster import KMeans
from scipy.ndimage import label
from scipy.ndimage.morphology import binary_dilation, binary_erosion, binary_closing
from scipy.signal import find_peaks
from sys import argv
#
# Purpose:
# Performs GM classification on neurotransmitter receptor autoradiography sections using K-Means.
# If K-means does not work, uses simple thresholding based on 90th percentile of values in image. 
# Reads in a 3D volume with aligned 2D autoradiographs and outputs a 3D GM classification volume.
#
def myKMeans(s):
    upper=np.max(s)
    mid = np.median(s[s>0])
    if np.isnan(mid) : return np.zeros(s.shape)
    init=np.array([0, mid, upper]).reshape(-1,1)
    cls = KMeans(3, init=init).fit_predict(s.reshape(-1,1)).reshape(s.shape)
    cls[ cls != 2 ] = 0
    cls[cls == 2] = 1
    return cls

def local_kmeans(img,step, stride, sd=10, hist=False, threshold=True):
    #plt.imshow(img); plt.show()
    if hist == True :
        img = equalize_hist(img)
    img = gaussian_filter(img, sd)
    out = np.zeros(img.shape)
    out[ img > threshold_li(img[img>0]) ] = 1
    #plt.imshow(out); plt.show()
    #n = np.zeros(img.shape)
    #for x in range(0, img.shape[0], stride):
    #    for z in range(0, img.shape[1], stride) :
    #        out[x:(x+step),z:(z+step)] += myKMeans(img[x:(x+step),z:(z+step)])
    #        n[x:(x+step),z:(z+step)] += 1.
    #out = out / n 
    #if threshold :
    #    out[ out < 0.5 ] = 0
    #    out[ out >= 0.5 ] = 1
    return out

def compress(ii, oo) :
    print("Gzip compression from ", ii, 'to', oo)
    with open(ii, 'rb') as f_in, gzip.open(oo, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
    
    if os.path.exists(oo) : 
        os.remove(ii)
    return 0

def denoise(s) :
    cc, nlabels = label(s, structure=np.ones([3,3,3]))
    cc_unique = np.unique(cc)
    numPixels = np.bincount(cc.reshape(-1,))

    numPixels = numPixels[1:]
    cc_unique = cc_unique[1:]
    
    idx = numPixels < (np.max(numPixels) * 0.01)
    labels_to_remove = cc_unique[ idx ]
    labels_to_keep = cc_unique[ ~ idx ]
    
    out = np.zeros(cc.shape)
    for i in labels_to_keep :
        out[ cc == i ] = 1
    
    return out

def myKMeansOriginal(s):
    s = np.max(s) - s
    upper=np.max(s)
    mid = np.median(s[s>0])
    #lower = np.min(s[s>0])
    init=np.array([0, mid, upper]).reshape(-1,1)
    cls = KMeans(3, init=init).fit_predict(s.reshape(-1,1)).reshape(s.shape)
    cls[ cls != 2 ] = 0 
    return cls

def safe_h5py_open(filename, mode='r+'):
    '''open hdf5 file, exit elegantly on failure'''
    try :
        f = h5py.File(filename, mode)
        return f
    except OSError :
        print('Error: Could not open', filename)
        exit(1)


def classifyReceptorSlices(in_fn, out_dir, out_fn, morph_iterations=5, clobber=False) :
    out_fn = out_dir +os.sep+out_fn
    if not os.path.exists(out_fn) or clobber :
        #
        # Check Inputs
        #
        if not os.path.exists(in_fn):
            print("Error: could not find ", in_fn)
            exit(1)

        if not os.path.exists(out_dir) :
            os.makedirs(out_dir)

        #
        # Read Input HDF5 Minc File and create output volume
        #
        vol1 = nib.load(in_fn)
        ar1 = vol1.get_data()

        data=np.zeros(ar1.shape)

        #
        # Perform K-means clustering on coronal sections in volume
        #
        valid_slices=[]
        invalid_slices=[]
        qc=[]
        for i in range(0, ar1.shape[1]) : #
            s0 = ar1[:, i, :]
            if np.max(s0) > 0 :
                #cls0=local_kmeans(s0, 100,25, sd=0) #myKMeansOriginal(s0)
                cls=local_kmeans(s0, 100,25, sd=1) #myKMeansOriginal(s0)
                #cls2=local_kmeans(s0, 100,25, sd=2) #myKMeansOriginal(s0)
                #cls3=local_kmeans(s0, 100,25, sd=3) #myKMeansOriginal(s0)
                #cls4=local_kmeans(s0, 100,25, sd=4) #myKMeansOriginal(s0)
                #cls= (cls0 + cls1 + cls2 ) / 3. #
                #cls_1=np.zeros(s0.shape)
                #cls_1[ s0 > threshold_li(s0[s0>0]) ] = 1
                #plt.subplot(3,1,1); plt.imshow(s0)
                #plt.subplot(3,1,2); plt.imshow(cls_1)
                #plt.subplot(3,1,3); plt.imshow(cls)
                #plt.show()
                #if (clobber or not os.path.exists(qc_dir+os.sep+'/tmp_'+str(i)+'.png')) :
                          #print( "1", cls2.shape)
                    #scipy.misc.imsave(qc_dir+'/tmp_'+str(i)+'.png', cls2)
                #else :
                #    cls2= imageio.imread(qc_dir+os.sep+'/tmp_'+str(i)+'.png') / 255
                #    #cls2 = cls2.T
                #    #print("2", cls2.shape)
                data[:,i,:] = cls
                valid_slices.append(i)
                #qc.append(np.sum(cls2))
            else :
                invalid_slices.append(i)

        del ar1
       
        #
        # Fill in missing slices using nearest neighbour interpolation
        #
        valid_slices = np.array(valid_slices) 
        print("num of invalid slices:", len(valid_slices))
        for i in invalid_slices :
            dif = np.absolute(valid_slices - i)
            ii=valid_slices[np.argmin(dif)]
            data[:,i,:] = data[:,ii,:]

        #Denoise data
        data=denoise(data)
        
        #
        # Save output volume
        #   
        print("Writing output to", out_fn)

        img_cls = nib.Nifti1Image(data, vol1.get_affine() )     
        img_cls.to_filename(out_fn)

        return 0

if __name__ == "__main__":
    #
    # Set Inputs
    #

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--input-file', dest='input_file',  help='Input MINC filename')
    parser.add_argument('--output-file', dest='output_file',  help='Output MINC filename')
    parser.add_argument('--output-dir', dest='output_dir',  help='Output MINC filename')
    parser.add_argument('--morph-iterations', dest='morph_iterations', type=int, default=5, help='Number of iterations to use for morphological erosion and then dilation across the Y axis')
    parser.add_argument('--clobber', dest='clobber', action='store_true', default=False, help='Clobber results')
    args = parser.parse_args()
    classifyReceptorSlices(args.input_file, args.output_dir, args.output_file, args.morph_iterations, args.clobber)



