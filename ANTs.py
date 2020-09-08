import os
import nibabel as nib
from shutil import copy
from os.path import basename
from os import makedirs
from nibabel.processing import resample_from_to
from utils.utils import shell,splitext
from scipy.ndimage import gaussian_filter
from skimage.filters import threshold_otsu
from re import sub
import numpy as np
import SimpleITK as sitk

def generate_mask(fn, out_fn, sigma=8) :
    if os.path.exists(out_fn) :
        return None
    img  = nib.load(fn)
    vol  = img.get_fdata()
    vol  = gaussian_filter(vol, sigma)

    if np.sum(vol) == 0 : return 1
    idx  = vol > threshold_otsu(vol)
    vol[ idx ]  = 1
    vol[ ~idx ] = 0
    nib.Nifti1Image(vol, img.affine).to_filename(out_fn)
    return 0


def ANTs( tfm_prefix, fixed_fn, moving_fn, moving_rsl_prefix, iterations, tolerance=1e-08, metrics=None, nbins=32, tfm_type=["Rigid","Affine","SyN"], rate=None, shrink_factors=None,smoothing_sigmas=None, radius=4, init_tfm=None,init_inverse=False,sampling_method='Regular', sampling=1, dim=3, verbose=0, clobber=0, exit_on_failure=0, fix_header=False, generate_masks=True, no_init_tfm=False, mask_dir=None, write_composite_transform=True, n_tries=5, init_tfm_direction='moving' ) :
   
    nLevels = len(iterations)
    if verbose :
        print('Moving:', moving_fn)
        print('Fixed:', fixed_fn)

    if rate == None :
        rate = [0.1]*nLevels
    if shrink_factors == None :
        shrink_factors=['4x2x1vox'] * nLevels
    if smoothing_sigmas == None :
        smoothing_sigmas = ['4.0x2.0x1.0vox'] * nLevels
    if metrics == None :
        metrics = ['Mattes'] * nLevels
   
    moving_mask_fn = None
    fixed_mask_fn = None
    if generate_masks : 

        if mask_dir != None :
            moving_mask_fn = mask_dir +os.sep+ basename(splitext(moving_fn)[0])+'_mask.nii.gz'
            fixed_mask_fn = mask_dir + os.sep + basename(splitext(fixed_fn)[0])+'_mask.nii.gz'
        else :
            moving_mask_fn = moving_rsl_prefix + '_moving_mask.nii.gz'
            fixed_mask_fn  = moving_rsl_prefix + '_fixed_mask.nii.gz'

        s = max( [ int(float(j)) for i in smoothing_sigmas  for j in sub('vox|mm', '', i).split('x')  ] )

        r=0
        if not os.path.exists( moving_mask_fn ) :
            r += generate_mask(moving_fn, moving_mask_fn , s)

        if not os.path.exists( fixed_mask_fn ) :
            r += generate_mask(fixed_fn, fixed_mask_fn, s )
           
        if r != 0 : 
            moving_mask_fn = moving_mask_fn = None 

    final_moving_rsl_fn = moving_rsl_prefix + '_level-' + str(nLevels-1) + '_' + metrics[-1] + '_'+ tfm_type[-1] + '.nii.gz'
    final_tfm_fn = tfm_prefix + 'level-' + str(nLevels-1) + '_' + metrics[-1] + '_'+ tfm_type[-1]+'_Composite.h5'
    #If image volume is empty, write identity matrix
    if np.sum(nib.load(fixed_fn).get_data()) == 0  or np.sum( nib.load(moving_fn).get_data() ) == 0 :
        print("Warning: atleast one of the image volume is empty")
        identity = sitk.Transform(3, sitk.sitkIdentity)
        sitk.WriteTransform(identity, final_tfm_fn)
        copy( moving_fn, final_moving_rsl_fn )
    
   
    for level in range(nLevels) :
        moving_rsl_level_prefix = moving_rsl_prefix + '_level-' + str(level) + '_' + metrics[level] + '_'+ tfm_type[level]
        tfm_level_prefix = tfm_prefix + 'level-' + str(level) + '_' + metrics[level] + '_'+ tfm_type[level]+'_' 
        
        moving_rsl_fn = moving_rsl_level_prefix + '.nii.gz'

        config_file = moving_rsl_level_prefix + '_parameters.txt'

        if not os.path.exists(moving_rsl_fn) or not os.path.exists(final_tfm_fn) or clobber > 0 :
            # Set tfm file name
            tfm_fn = tfm_level_prefix + 'Composite.h5'

            moving_rsl_fn_inverse = moving_rsl_level_prefix + '_inverse_' + tfm_type[level] + '.nii.gz'

            ### Inputs
            cmdline =  "antsRegistration --verbose "+str(verbose)
            cmdline += " --write-composite-transform 1 --float --collapse-output-transforms 0 --dimensionality "+str(dim) +" "
            if not no_init_tfm :
                if init_tfm == None : 
                    cmdline += f" --initial-{init_tfm_direction}-transform [ "+fixed_fn+", "+moving_fn+", 1 ] "
                else : 
                    if type(init_tfm) != list :
                        init_tfm=[init_tfm]

                    if init_inverse :
                        cmdline += f" --initial-{init_tfm_direction}-transform [" + ','.join(init_tfm) + ",1] "
                    else :
                        cmdline += f" --initial-{init_tfm_direction}-transform " + ','.join(init_tfm) + " "

            cmdline += " --initialize-transforms-per-stage 1 --interpolation Linear "
        
            #Set up smooth sigmas and shrink factors
            smooth_sigma = smoothing_sigmas[level]
            shrink_factor = shrink_factors[level] 
            
            # Add masks
            if moving_mask_fn != None and moving_mask_fn != None :
                cmdline += " --masks [ "+fixed_mask_fn + " , " +moving_mask_fn + " ] "

            ### Set tranform parameters for level
            cmdline += " --transform "+tfm_type[level]+"[ "+str(rate[level])+" ] " 
            cmdline += " --metric "+metrics[level]+"["+fixed_fn+", "+moving_fn+", 1,"
            if metrics[level] == "Mattes" :
                cmdline += " "+str(nbins)+", "
            else :
                cmdline += " "+str(radius)+", "
            cmdline += sampling_method +" , "+str(sampling)+" ] "
            cmdline += " --convergence [ "+iterations[level]+" , "+str(tolerance)+" , 20 ] "
            cmdline += " --smoothing-sigmas "+smooth_sigma+"vox --shrink-factors "+shrink_factor
            cmdline += " --use-estimate-learning-rate-once 1 --use-histogram-matching 0 "
       
            if write_composite_transform : cmdline += " --write-composite-transform 1"

            ### Outputs
            cmdline+=" --output [ "+tfm_level_prefix+" ,"+moving_rsl_fn+","+moving_rsl_fn_inverse+"] "
            
            if verbose == 1 : print(cmdline) 
            
            errorcode=0
            for attempt in range(n_tries) :
                try : 
                    #Run command line
                    stdout, stderr, errorcode = shell(cmdline, exit_on_failure=False)
                    print('Attempt:', attempt, errorcode)
                    if verbose == 1 :
                        print(stdout)
                        print(stderr)
                except RuntimeError :
                    errorcode = 1
                    if exit_on_failure == 1 :
                        return(1)
                if errorcode == 0 : break
            if errorcode != 0 : return(1)
            
            with open(config_file, 'w+') as f :
                f.write(cmdline)
            
            #update init_tfm
            init_tfm = [tfm_fn]
            init_inverse=False

            if fix_header and os.path.exists(moving_rsl_fn) : 
                #resample_from_to( nib.load(moving_rsl_fn), nib.load(fixed_fn)).to_filename(moving_rsl_fn)
                nib.Nifti1Image(  nib.load(moving_rsl_fn).get_data(), nib.load(fixed_fn).affine ).to_filename(moving_rsl_fn)
            
            if fix_header and os.path.exists(moving_rsl_fn_inverse) : 
                #resample_from_to( nib.load(moving_rsl_fn_inverse), nib.load(moving_fn)).to_filename(moving_rsl_fn_inverse)
                nib.Nifti1Image(  nib.load(moving_rsl_fn_inverse).get_data(), nib.load(moving_fn).affine ).to_filename(moving_rsl_fn_inverse)

    return final_tfm_fn, moving_rsl_fn
