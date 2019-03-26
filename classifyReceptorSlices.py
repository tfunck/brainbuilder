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
    qc_dir=out_dir+os.sep+"qc"
    out_fn = out_dir +os.sep+out_fn
    if not os.path.exists(out_fn) or clobber :
        #
        # Check Inputs
        #
        if not os.path.exists(in_fn):
            print("Error: could not find ", in_fn)
            exit(1)

        if not os.path.exists(qc_dir) :
            os.makedirs(qc_dir)

        #
        # Read Input HDF5 Minc File and create output volume
        #
        vol1 = nib.load(in_fn)
        ar1 = vol1.get_data()

        slab_ymin=-126

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
                cls0=myKMeansOriginal(s0)
                cls2=cls0 #

                #if (clobber or not os.path.exists(qc_dir+os.sep+'/tmp_'+str(i)+'.png')) :
                          #print( "1", cls2.shape)
                    #scipy.misc.imsave(qc_dir+'/tmp_'+str(i)+'.png', cls2)
                #else :
                #    cls2= imageio.imread(qc_dir+os.sep+'/tmp_'+str(i)+'.png') / 255
                #    #cls2 = cls2.T
                #    #print("2", cls2.shape)
                data[:,i,:] = cls2
                valid_slices.append(i)
                qc.append(np.sum(cls2))
            else :
                invalid_slices.append(i)

        #
        # Perform QC by identifying slices with large sums
        #
        minProminence=0.1*cls2.shape[0]*cls2.shape[1]
        qc_peaks, qc_peaks_dict=find_peaks(qc, prominence=minProminence)
        plt.plot(valid_slices, qc)
        plt.scatter(np.array(valid_slices)[qc_peaks], np.array(qc)[qc_peaks] )
        plt.savefig(qc_dir+os.sep+"qc_plot.png")

        #
        # Rerun bad K-Means slices with threshold at 90th percentile  
        #
        #for i in np.array(valid_slices)[qc_peaks]:
        #    #if not os.path.exists(qc_dir+'/tmp_'+str(i)+'_post-qc.png' ) or clobber :
        #    print("Updating Slice:", i)
        #    s0 = ar1[:, i, :]
        #    cls2=np.zeros_like(s0)
        #    cls2[ s0 >= np.percentile(s0, 80) ] = 1
        #    cls2 = binary_dilation(denoise(cls2), iterations=2).astype(int)
        #    scipy.misc.imsave(qc_dir+'/tmp_'+str(i)+'_post-qc.png', cls2)
        #    #else : 
        #    #    cls2 = imageio.imread(qc_dir+'/tmp_'+str(i)+'_post-qc.png') / 255
        #    data[:,i,:] = cls2
        #    valid_slices.remove(i)

        del ar1
        del cls2
       
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



