from sys import argv
import numpy as np
import h5py
import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.ndimage import label
from scipy.ndimage.morphology import binary_dilation, binary_erosion, binary_closing
import scipy
import os
import imageio
from pyminc.volumes.factory import *
from scipy.signal import find_peaks
import gzip
import shutil

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
    return 0

def denoise(s) :
    cc, nlabels = label(s, structure=np.ones([3,3]))
    cc_unique = np.unique(cc)
    cc_unique = cc_unique[ cc_unique != 0 ]
    numPixels = np.array([np.sum(cc == l) for l in cc_unique  ])

    for s, i in zip(numPixels, cc_unique) :
        if s < 0.01 * max(numPixels) : 
            cc[ cc == i ] = 0
    cc[ cc != 0 ] = 1
    return cc

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
    if os.path.splitext(out_fn)[1] == ".gz" :
        out_fn=os.path.splitext(out_fn)[0]

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
    vol1 = safe_h5py_open(in_fn, "r+")
    ar1=np.array(vol1['minc-2.0/']['image']['0']['image'])

    slab_ymin=-126
    img_cls = volumeFromDescription(out_fn, dimnames=("zspace","yspace","xspace"), sizes=ar1.shape, starts=(-90,slab_ymin,-72),steps=(0.2, 0.02, 0.2),volumeType="ushort")


    #
    # Perform K-means clustering on coronal sections in MINC volume
    #
    valid_slices=[]
    invalid_slices=[]
    qc=[]
    for i in range(0, ar1.shape[1]) : #
        s0 = ar1[:, i, :]
        if np.max(s0) > 0 :
            if (clobber or not os.path.exists(qc_dir+os.sep+'/tmp_'+str(i)+'.png')) :
                cls0=myKMeansOriginal(s0)
                cls2=denoise(cls0)
                scipy.misc.imsave(qc_dir+'/tmp_'+str(i)+'.png', cls2)
            else :
                cls2= imageio.imread(qc_dir+os.sep+'/tmp_'+str(i)+'.png') / 255
            img_cls.data[:,i,:] = cls2
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
    for i in np.array(valid_slices)[qc_peaks]:
        if not os.path.exists(qc_dir+'/tmp_'+str(i)+'_post-qc.png' ) or clobber :
            print("Updating Slice:", i)
            s0 = ar1[:, i, :]
            cls2=np.zeros_like(s0)
            cls2[ s0 >= np.percentile(s0, 90) ] = 1
            cls2 = binary_dilation(denoise(cls2), iterations=2).astype(int)
            scipy.misc.imsave(qc_dir+'/tmp_'+str(i)+'_post-qc.png', cls2)
        else : 
            cls2 = imageio.imread(qc_dir+'/tmp_'+str(i)+'_post-qc.png') / 255
        img_cls.data[:,i,:] = cls2
        valid_slices.remove(i)

    del ar1
    del cls2
    
    #
    # Fill in missing slices using nearest neighbour interpolation
    #
    valid_slices = np.array(valid_slices) 
    for i in invalid_slices :
        dif = np.absolute(valid_slices - i)
        ii=valid_slices[np.argmin(dif)]
        img_cls.data[:,i,:] = img_cls.data[:,ii,:]

    struct=np.zeros([3,3,3])
    struct[1, :, 1] = 1

    #
    # Perform morphological operations to denoise image
    #
    img_cls.data = binary_erosion(img_cls.data, structure=struct, iterations=morph_iterations )
    img_cls.data = binary_dilation(img_cls.data, structure=struct, iterations=morph_iterations )
    img_cls.data = binary_closing(img_cls.data, iterations=1 )
    
    #
    # Save output volume
    #   
    print("Writing output to", out_fn)
    img_cls.writeFile()
    img_cls.closeVolume()
          
    #
    # Compress output with gzip
    #
    compress(out_fn, out_fn+'.gz')

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



