import argparse
import numpy as np
import h5py
import os
import matplotlib.pyplot as plt
import scipy
import os
import ants
import imageio
import nibabel as nib
import gzip
import multiprocessing
import shutil
import re
from glob import glob
from re import sub
from utils.utils import *
from nibabel.processing import resample_to_output
from skimage.filters import threshold_otsu, threshold_li
from scipy.ndimage.filters import gaussian_filter
from sklearn.cluster import KMeans
from scipy.ndimage import label
from scipy.ndimage.morphology import binary_dilation, binary_erosion, binary_closing
from scipy.signal import find_peaks
from sys import argv
from joblib import Parallel, delayed
#
# Purpose:
# Performs GM classification on neurotransmitter receptor autoradiography sections using K-Means.
# If K-means does not work, uses simple thresholding based on 90th percentile of values in image. 
# Reads in a 3D volume with aligned 2D autoradiographs and outputs a 3D GM classification volume.
#

def downsample_2d(in_fn, resolution, out_fn, y=0):
    if not os.path.exists(out_fn):
        #load image
        img = nib.load(in_fn)

        #change start value for header
        img.affine[2,3] = -126. + 0.02 * float(y)

        #load volume 
        try :
            vol = img.get_fdata()
        except EOFError :
            print('Errror: ', in_fn)
            exit(1)

        #blur volume prior to downsampling
        vol = gaussian_filter(vol, float(resolution)/(0.02*2))
        
        #create a new Nifti1Image so that we can resample it with nibabel
        img = nib.Nifti1Image(vol,img.affine)
        
        # resample to new resolution
        img = resample_to_output(img, [float(resolution)]*2,order=5)
       
        # get image volume again
        vol = img.get_fdata()

        # reshape it to make sure it's 2d (resampling will make it 3d: [x,y,1])
        vol = vol.reshape(vol.shape[0], vol.shape[1])
        
        #create a new Nifti1Image that we can save
        img = nib.Nifti1Image(vol,img.affine)

        assert len(img.shape) == 2, 'Images should be 2D but have more than 2 dimensions {}'.format(img.shape)

        #write resample image
        img.to_filename(out_fn)

def resample_and_transform(output_dir, resolution_2d, resolution_3d, row):
    seg_fn = row['seg_fn']
    seg_rsl_fn = get_seg_fn(output_dir, row['volume_order'], resolution_2d, seg_fn, '_rsl')
    tfm_input_fn = seg_rsl_fn
    seg_rsl_tfm_fn = get_seg_fn(output_dir, row['volume_order'], resolution_3d, seg_fn, '_rsl_tfm')

    downsample_2d(seg_fn, resolution_2d, seg_rsl_fn, y=row['global_order'])

    if resolution_2d != resolution_3d :
        tfm_input_fn = get_seg_fn(output_dir, row['volume_order'], resolution_2d, seg_fn, '_rsl')
        downsample_2d(seg_fn, resolution_3d, tfm_input_fn, y=row['global_order'])

    if not os.path.exists(seg_rsl_tfm_fn) : 
        # get initial rigid transform
        tfm_fn = row['tfm']  
        if type(tfm_fn) == str :
            tfm = ants.read_transform(tfm_fn)
            shell(f'antsApplyTransforms -v 0 -d 2 -i {tfm_input_fn} -r {seg_rsl_fn} -t {tfm_fn} -o {seg_rsl_tfm_fn}')
        else :
            shutil.copy(tfm_input_fn, seg_rsl_tfm_fn)
            print('\tNo transform for',row['volume_order'])



def resample_transform_segmented_images(df,resolution_2d,resolution_3d, output_dir):

    os.makedirs(output_dir, exist_ok=True)
    num_cores = min(14, multiprocessing.cpu_count() )
    Parallel(n_jobs=num_cores)(delayed(resample_and_transform)(output_dir, resolution_2d, resolution_3d, row) for i, row in df.iterrows()) 



def threshold(img,sd=1):
    img = gaussian_filter(img, sd)
    out = np.zeros(img.shape)
    out[ img > threshold_li(img[img>0]) ] = 1
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

def safe_h5py_open(filename, mode='r+'):
    '''open hdf5 file, exit elegantly on failure'''
    try :
        f = h5py.File(filename, mode)
        return f
    except OSError :
        print('Error: Could not open', filename)
        exit(1)


def classifyReceptorSlices(df, in_fn, in_dir, out_dir, out_fn, morph_iterations=5, clobber=False, resolution=0.2) :
    if not os.path.exists(out_fn)  or clobber :
        #
        # Check Inputs
        #
        if not os.path.exists(in_fn):
            print("Error: could not find ", in_fn)
            exit(1)

        if not os.path.exists(out_dir) :
            os.makedirs(out_dir)

        # Remove full resolution transformed segmented images
        for fn in glob(in_dir+'full_res*nii.gz') : os.remove(fn)

        #
        # Read Input HDF5 Minc File and create output volume
        #
        vol1 = nib.load(in_fn)
        example_2d_list = glob(in_dir +'/*nii.gz') # os.path.basename(df['seg_fn'].iloc[0])
        assert len(example_2d_list) > 0 , 'Error: no files found in {}'.format(in_dir)

        example_2d_img = nib.load(example_2d_list[0])
        data=np.zeros([example_2d_img.shape[0], vol1.shape[1],  example_2d_img.shape[1]],dtype=np.float32)

        #
        # Perform K-means clustering on coronal sections in volume
        #
        valid_slices=[]
        qc=[]

        for i, row in df.iterrows() :
            s0 = int(row['volume_order'])
            fn = get_seg_fn(in_dir, row['volume_order'], resolution, row['seg_fn'], '_rsl_tfm')
            img_2d = nib.load(fn).get_fdata()
            #FIXME : Skipping frames that have been rotated
            if img_2d.shape != example_2d_img.shape :
                pass
                #print(row['crop_fn'])
            else :
                data[:,s0,:] = img_2d.reshape([img_2d.shape[0],img_2d.shape[1]]) 
            #print(img_2d.max())
            valid_slices.append(int(row['volume_order']))
      
        invalid_slices=[ i for i in range(1+int(df['volume_order'].max()) ) if not i in valid_slices ]

        #
        # Fill in missing slices using nearest neighbour interpolation
        #
        valid_slices = np.array(valid_slices) 
        print("num of invalid slices:", len(valid_slices))
        for i in invalid_slices :
            dif = np.argsort( np.absolute(valid_slices - i) )

            i0=valid_slices[dif[0]]
            i1=valid_slices[dif[1]]
            i2=valid_slices[dif[2]]
            i3=valid_slices[dif[3]]
            data[:,i,:] = data[:,i0,:]*0.4 + data[:,i1,:]*0.35  + data[:,i2,:]*0.25

        # Denoise data
        data[data<0.55] =0 
        data[data>=0.55] =1 
        structure = np.zeros([3,3,3])
        structure[1,:,1] = 1
        # Binary Erosion
        data = binary_erosion(data,iterations=1,structure=structure)
        # Binary Dilation
        data = binary_dilation(data, iterations=3, structure=structure).astype(np.int16)
        # Gaussian blurring
        data = gaussian_filter(data.astype(float), float(resolution)/2. ).astype(float)

        #
        # Save output volume
        #   
        xstart = float(vol1.affine[0][3])
        ystart = float(vol1.affine[1][3])
        zstart = float(vol1.affine[2][3])
        xstep =  float(vol1.affine[0][0])
        zstep =  float(vol1.affine[2][2])

        aff=np.array([  [resolution, 0, 0, xstart],
                        [0, 0.02, 0, ystart ],
                        [0, 0,  resolution, zstart], 
                        [0, 0, 0, 1]]).astype(float)
        img_cls = nib.Nifti1Image(data, aff )     
        print("Writing output to", out_fn)
        img_cls = resample_to_output(img_cls, [float(resolution)]*3, order=5)
        
        img_cls.to_filename(out_fn)

        return 0

if __name__ == "__main__":
    #
    # Set Inputs
    #

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--input-file','-i', dest='input_file',  help='Input MINC filename')
    parser.add_argument('--output-file','-o', dest='output_file',  help='Output MINC filename')
    parser.add_argument('--output-dir','-d', dest='output_dir',  help='Output MINC filename')
    parser.add_argument('--morph-iterations', dest='morph_iterations', type=int, default=5, help='Number of iterations to use for morphological erosion and then dilation across the Y axis')
    parser.add_argument('--clobber', dest='clobber', action='store_true', default=False, help='Clobber results')
    args = parser.parse_args()
    classifyReceptorSlices(args.input_file, args.output_dir, args.output_file, args.morph_iterations, args.clobber)



