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
from utils.utils import get_section_intervals, recenter
from glob import glob
from re import sub
from utils.utils import *
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
        try:
            vol = img.get_fdata()
        except EOFError :
            print('Errror: ', in_fn)
            exit(1)

        #blur volume prior to downsampling
        vol = gaussian_filter(vol, (float(resolution)/0.02)/np.pi)
        
        #create a new Nifti1Image so that we can resample it with nibabel
        img = nib.Nifti1Image(vol,img.affine)
        
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

def resample_and_transform(output_dir, resolution_itr, resolution_2d, resolution_3d, row, recenter_image=False):
    seg_fn = row['seg_fn']
    
    tfm_ref_fn = output_dir+'/2d_reference_image.nii.gz'
    seg_rsl_fn = get_seg_fn(output_dir, int(row['slab_order']), resolution_2d, seg_fn, '_rsl')
    tfm_input_fn = seg_rsl_fn
    seg_rsl_tfm_fn = get_seg_fn(output_dir, int(row['slab_order']), resolution_3d, seg_fn, '_rsl_tfm')

    new_starts = [ None, -126. + 0.02 * row['global_order'], None ]

    
    if not os.path.exists(seg_rsl_fn) :
        prefilter_and_downsample(seg_fn, [resolution_2d]*2, seg_rsl_fn, new_starts=new_starts  )


    if resolution_itr != 0: 

        # if 2d files with recentered coordinates have already been created (i.e.,
        # if this is not the first iteration in the multiresolution hierarchy)
        # then use the new x and z start coordiantes
        if not os.path.exists(tfm_ref_fn) : 
            ref_img = nib.load(row['nl_2d_rsl'])
            xstart, zstart = ref_img.affine[[0,1],3]
            affine = np.array([ [resolution_3d, 0, 0, xstart],
                              [0,  resolution_3d, 0,zstart],
                              [0, 0,  0, 1],
                              [0, 0, 0, 1]]).astype(float)
            #nib.Nifti1Image(nib.load(seg_rsl_fn).get_fdata(), affine).to_filename(tfm_ref_fn)
            nib.Nifti1Image(nib.load(seg_rsl_fn).dataobj, affine).to_filename(tfm_ref_fn)
    else :
        tfm_ref_fn=seg_rsl_fn

    if resolution_2d != resolution_3d :
        tfm_input_fn = get_seg_fn(output_dir, int(row['slab_order']), resolution_3d, seg_fn, '_rsl')
        if not os.path.exists(tfm_input_fn):
            prefilter_and_downsample(seg_fn, [resolution_3d]*2, tfm_input_fn, new_starts = new_starts )
    if not os.path.exists(seg_rsl_tfm_fn) : 
        # get initial rigid transform
        tfm_fn = row['tfm']  
        
        if type(tfm_fn) == str :
            #print('\n-->',tfm_fn, os.path.exists(tfm_fn),'\n')
            #print('\n-->',tfm_input_fn, os.path.exists(tfm_input_fn),'\n')
            #print('\n-->',seg_rsl_fn, os.path.exists(seg_rsl_fn), '\n')

            cmdline = f'antsApplyTransforms -n NearestNeighbor -v 0 -d 2 -i {tfm_input_fn} -r {tfm_ref_fn} -t {tfm_fn} -o {seg_rsl_tfm_fn}'
            shell(cmdline)
        else :
            print('\tNo transform for', seg_rsl_fn)
            shutil.copy(tfm_input_fn, seg_rsl_tfm_fn)



def resample_transform_segmented_images(df,resolution_itr,resolution_2d,resolution_3d, output_dir):

    os.makedirs(output_dir, exist_ok=True)
    os_info = os.uname()

    if os_info[1] == 'imenb079':
        num_cores = 1 
    else :
        num_cores = min(14, multiprocessing.cpu_count() )
    
    Parallel(n_jobs=num_cores)(delayed(resample_and_transform)(output_dir, resolution_itr, resolution_2d, resolution_3d, row) for i, row in df.iterrows()) 

def interpolate_missing_sections(vol, dilate_volume=False) :
    if dilate_volume :
        vol_dil = binary_erosion(binary_dilation(vol,iterations=4),iterations=4).astype(int)
    else :
        vol_dil = vol
    intervals = get_section_intervals(vol_dil)
    
    print(intervals)
    for i in range(len(intervals)-1) :
        j=i+1
        x0,x1 = intervals[i]
        y0,y1 = intervals[j]
        x = np.mean(vol[:,x0:(x1+1),:], axis=1)
        y = np.mean(vol[:,y0:(y1+1),:], axis=1)
        vol[:,x0:x1,:]  = np.repeat(x.reshape(x.shape[0],1,x.shape[1]), x1-x0, axis=1)
        print(x0,x1,y0,y1)   
        for ii in range(x1+1,y0) :
            den = (y0-x1-1)
            assert den != 0, 'Error: 0 denominator when interpolating missing sections'
            d = (ii - x1)/den
            #d = np.rint(d)
            z = x * (1-d) + d * y
            vol[:,ii,:] = z
    
    return vol




def classifyReceptorSlices(df, in_fn, in_dir, out_dir, out_fn, morph_iterations=5, flip_axes=(), clobber=False, resolution=0.2, interpolation='nearest') :
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
        example_2d_list = glob(in_dir +'/*rsl_tfm.nii.gz') # os.path.basename(df['seg_fn'].iloc[0])
        assert len(example_2d_list) > 0 , 'Error: no files found in {}'.format(in_dir)
        example_2d_img = nib.load(example_2d_list[0])

        data = np.zeros([example_2d_img.shape[0], vol1.shape[1], example_2d_img.shape[1]],dtype=np.float32)

        #TODO this works well for macaque but less so for human
        if interpolation=='linear' :
            for i, row in df.iterrows() :
                s0 = int(row['slab_order'])
                fn = get_seg_fn(in_dir, int(row['slab_order']), resolution, row['seg_fn'], '_rsl_tfm')
                img_2d = nib.load(fn).get_fdata()
                #FIXME : Skipping frames that have been rotated
                data[:,s0,:] = img_2d 
            data = interpolate_missing_sections(data, dilate_volume=True)
        else :
            valid_slices = []
            for i, row in df.iterrows() :
                s0 = int(row['slab_order'])
                fn = get_seg_fn(in_dir, int(row['slab_order']), resolution, row['seg_fn'], '_rsl_tfm')
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
                #i2=valid_slices[dif[2]]
                #i3=valid_slices[dif[3]]
                #nearest neighbough interpolation
                data[:,i,:] = data[:,i0,:]

        #
        # Save output volume
        #   
        xstart = float(vol1.affine[0][3])
        ystart = float(vol1.affine[1][3])
        zstart = float(vol1.affine[2][3])

        aff=np.array([  [resolution, 0, 0, xstart],
                        [0, 0.02, 0, ystart ],
                        [0, 0,  resolution, zstart], 
                        [0, 0, 0, 1]]).astype(float)
        if flip_axes != () :
            data = np.flip(data,axis=flip_axes)    
        
        
        data, aff = recenter(data, aff)

        # Gaussian blurring
        sd = (float(resolution)/0.02)/np.pi
        #effective resolution across y is really actually closer to 1mm not 0.02, so trying that instead 
        #only smooth along y axis because x and z axes are already at lower resolution

        data = gaussian_filter1d(data.astype(float), sd, axis=1 ).astype(float)
        ydim =  np.ceil( (vol1.affine[1,1] * vol1.shape[1]) / resolution).astype(int)
        data = resize(data,[example_2d_img.shape[0], ydim, example_2d_img.shape[1]], order=5 )
        
        aff[[0,1,2],[0,1,2]]=resolution
        
        print("Writing output to", out_fn)
        
        img_out = nib.Nifti1Image(data, aff, direction=[[1.,0.,0.],[0.,1.,0.],[0.,0.,-1.]])
        img_out.to_filename(out_fn)
        
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



