import argparse
import numpy as np
import h5py
import os
import matplotlib.pyplot as plt
import scipy
import os
import ants
import imageio
import utils.ants_nibabel as nib
import nibabel as nibabel
import gzip
import multiprocessing
import shutil
import re
from glob import glob
from re import sub
from utils.utils import *
from nibabel.processing import resample_to_output
from skimage.filters import threshold_otsu, threshold_li
from scipy.ndimage.filters import gaussian_filter, gaussian_filter1d
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
        #img.affine[2,3] = -126. + 0.02 * float(y)
        img.affine[2,2] = 1
        img.affine[2,3] = 0
        shape = img.shape
        #load volume 
        try :
            vol = img.get_fdata()
        except EOFError :
            print('Errror: ', in_fn)
            exit(1)

        #blur volume prior to downsampling
        vol = gaussian_filter(vol, (float(resolution)/0.02)/np.pi)
        
        #create a new Nifti1Image so that we can resample it with nibabel
        img = nib.Nifti1Image(vol,img.affine)
        
        # resample to new resolution
        print('1',img.shape)
        #img = resample_to_output(img, [float(resolution)]*2,order=5)
        print('2',img.shape)
        
        resize(img.get_fdata())

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

    seg_rsl_fn = get_seg_fn(output_dir, row['slab_order'], resolution_2d, seg_fn, '_rsl')
    tfm_input_fn = seg_rsl_fn
    seg_rsl_tfm_fn = get_seg_fn(output_dir, row['slab_order'], resolution_3d, seg_fn, '_rsl_tfm')

    new_starts = [ None, -126. + 0.02 * row['global_order'], None ]
    
    if not os.path.exists(seg_rsl_fn) :
        prefilter_and_downsample(seg_fn, [resolution_2d]*2, seg_rsl_fn, new_starts=new_starts  )
    
    if resolution_2d != resolution_3d :
        tfm_input_fn = get_seg_fn(output_dir, row['slab_order'], resolution_2d, seg_fn, '_rsl')
        if not os.path.exists(tfm_input_fn):
            prefilter_and_downsample(seg_fn, [resolution_3d]*2, tfm_input_fn, new_starts = new_starts  )

    if not os.path.exists(seg_rsl_tfm_fn) : 
        # get initial rigid transform
        tfm_fn = row['tfm']  
        print('\n-->',tfm_fn,'\n')
        if type(tfm_fn) == str :
            tfm = ants.read_transform(tfm_fn)
            cmdline = f'antsApplyTransforms -n NearestNeighbor -v 1 -d 2 -i {tfm_input_fn} -r {seg_rsl_fn} -t {tfm_fn} -o {seg_rsl_tfm_fn}'
            print(cmdline)
            shell(cmdline)
        else :
            shutil.copy(tfm_input_fn, seg_rsl_tfm_fn)
            print('\tNo transform for', seg_rsl_fn)



def resample_transform_segmented_images(df,resolution_2d,resolution_3d, output_dir):

    os.makedirs(output_dir, exist_ok=True)
    num_cores = min(14, multiprocessing.cpu_count() )
    Parallel(n_jobs=num_cores)(delayed(resample_and_transform)(output_dir, resolution_2d, resolution_3d, row) for i, row in df.iterrows()) 




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
        vol1 = nib.load(in_fn)
        example_2d_list = glob(in_dir +'/*nii.gz') # os.path.basename(df['seg_fn'].iloc[0])
        assert len(example_2d_list) > 0 , 'Error: no files found in {}'.format(in_dir)
        example_2d_img = nib.load(example_2d_list[0])
        data = np.zeros([example_2d_img.shape[0], vol1.shape[1], example_2d_img.shape[1]],dtype=np.float32)

        valid_slices = []
        qc=[]

        for i, row in df.iterrows() :
            s0 = int(row['slab_order'])
            fn = get_seg_fn(in_dir, row['slab_order'], resolution, row['seg_fn'], '_rsl_tfm')
            img_2d = nib.load(fn).get_fdata()
            #FIXME : Skipping frames that have been rotated
            if img_2d.shape != example_2d_img.shape :
                pass
            else :
                data[:,s0,:] = img_2d.reshape([img_2d.shape[0],img_2d.shape[1]]) 
            valid_slices.append(int(row['slab_order']))
      
        invalid_slices = [ i for i in range(1+int(df['slab_order'].max()) ) if not i in valid_slices ]

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
            #nearest neighbough interpolation
            data[:,i,:] = data[:,i0,:]
            #use weighting from adjacent sections
            #data[:,i,:] = data[:,i0,:]*0.4 + data[:,i1,:]*0.35  + data[:,i2,:]*0.25

        # Denoise data
        #data[data<0.55] =0 
        #data[data>=0.55] =1 
        structure = np.zeros([3,3,3])
        structure[1,:,1] = 1
        
        # Gaussian blurring
        sd = (float(resolution)/0.02)/np.pi
        #effective resolution across y is really actually closer to 1mm not 0.02, so trying that instead 
        #sd = (float(resolution)/1)/np.pi
        #only smooth along y axis because x and z axes are already at lower resolution
        data = gaussian_filter1d(data.astype(float), sd, axis=1 ).astype(float)

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
        
        img_cls = nibabel.Nifti1Image(data, aff )     
        
        print("Writing output to", out_fn)
        #print(aff)
        img_cls = resample_to_output(img_cls, [float(resolution)]*3, order=5)
        #print(img_cls.affine)

        # flip the volume along the y-axis so that the image is in RAS coordinates because ANTs requires RAS
        vol = img_cls.get_fdata()
        #vol = np.flip(vol,axis=1)
        nib.Nifti1Image(vol, img_cls.affine).to_filename(out_fn)

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



