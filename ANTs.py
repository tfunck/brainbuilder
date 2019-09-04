import os
import nibabel as nib
from nibabel.processing import resample_from_to
from utils.utils import shell

def ANTs(outDir, tfm_prefix, fixed_fn, moving_fn, moving_rsl_prefix, iterations, tolerance=1e-08, metric='GC', nbins=64, tfm_type=["Rigid","Affine","SyN"], rate=[0.05,0.05,0.05], base_shrink_factor=1, radius=64, init_tfm=None,init_inverse=False, sampling=1, dim=3, verbose=0, clobber=0, exit_on_failure=0, fix_header=False) :


    moving_rsl_fn = moving_rsl_prefix + '_' + tfm_type[-1] + 'nii.gz'
    #if tfm_type[-1] == 'SyN' :
    #    tfm_fn = tfm_prefix + tfm_type[-1] + 'warp.h5'
    #else : 
    #    tfm_fn = tfm_prefix + tfm_type[-1] + '0GenericAffine.mat'   

    registration_level = len(iterations)
    for level in range(registration_level) :
        tfm_fn = tfm_prefix + tfm_type[-1] + '_Composite.h5'
        moving_rsl_fn = moving_rsl_prefix + '_' + tfm_type[level] + '.nii.gz'

        if not os.path.exists(moving_rsl_fn) or not os.path.exists(tfm_fn) or clobber > 0 :
            # Set tfm file name
            tfm_level_prefix = tfm_prefix + tfm_type[level] + '_'
            tfm_fn = tfm_level_prefix + 'Composite.h5'

            moving_rsl_fn_inverse = moving_rsl_prefix + '_inverse_' + tfm_type[level] + '.nii.gz'


            ### Inputs
            cmdline =  "antsRegistration --verbose "+str(verbose)
            cmdline += " --write-composite-transform 1 --float --collapse-output-transforms 1 --dimensionality "+str(dim) +" "

            if init_tfm == None : 
                cmdline += " --initial-moving-transform [ "+fixed_fn+", "+moving_fn+", 1 ] "
            else : 
                if init_inverse :
                    cmdline += " --initial-moving-transform [" + init_tfm + ",1] "
                else :
                    cmdline += " --initial-moving-transform " + init_tfm + " "

            cmdline += " --initialize-transforms-per-stage 0 --interpolation Linear "
        
            #Set up smooth sigmas and shrink factors
            temp = [ str(i+base_shrink_factor) for i,x in enumerate(iterations[level].split('x'))]
            temp.reverse()
            temp_float = [ str(i)+'.0'  for i in temp]
            smooth_sigma = 'x'.join(temp_float)
            shrink_factor = 'x'.join(temp)

            ### Set tranform parameters for level
            cmdline += " --transform "+tfm_type[level]+"[ "+str(rate[level])+" ] " 
            cmdline += " --metric "+metric+"["+fixed_fn+", "+moving_fn+", 1,"
            if metric == "Mattes" :
                cmdline += " "+nbins+", "
            else :
                cmdline += " "+str(radius)+", "
            cmdline += " Regular, "+str(sampling)+" ] "
            cmdline += " --convergence [ "+iterations[level]+" , "+str(tolerance)+" , 10 ] "
            cmdline += " --smoothing-sigmas "+smooth_sigma+"vox --shrink-factors "+shrink_factor
            cmdline += " --use-estimate-learning-rate-once 1 --use-histogram-matching 0 "
        
            ### Outputs
            cmdline+=" --output [ "+tfm_level_prefix+" ,"+moving_rsl_fn+","+moving_rsl_fn_inverse+"] "
            
            if verbose == 1 : print(cmdline) 
            #exit(0) 
            try : 
                #Run command line
                shell(cmdline)
            except RuntimeError :
                if exit_on_failure == 0 :
                    exit(1)
                elif exit_on_failure == 1 :
                    return(1)

            #update init_tfm
            init_tfm = tfm_fn
            init_inverse=False

        if fix_header and os.path.exists(moving_rsl_fn) : 
            #resample_from_to( nib.load(moving_rsl_fn), nib.load(fixed_fn)).to_filename(moving_rsl_fn)
            nib.Nifti1Image(  nib.load(moving_rsl_fn).get_data(), nib.load(fixed_fn).affine ).to_filename(moving_rsl_fn)
        
        if fix_header and os.path.exists(moving_rsl_fn_inverse) : 
            #resample_from_to( nib.load(moving_rsl_fn_inverse), nib.load(moving_fn)).to_filename(moving_rsl_fn_inverse)
            nib.Nifti1Image(  nib.load(moving_rsl_fn_inverse).get_data(), nib.load(moving_fn).affine ).to_filename(moving_rsl_fn_inverse)

    return tfm_fn, moving_rsl_fn
