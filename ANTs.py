import os
from utils.utils import shell

def ANTs(outDir, tfm_prefix, fixed_fn, moving_fn, moving_rsl_fn, moving_rsl_fn_inverse, iterations, metric='GC', nbins=64, tfm_type=["Rigid","Affine","SyN"], rate=[0.05,0.05,0.05], base_shrink_factor=1, radius=64, init_tfm=None,init_inverse=False, sampling=1, verbose=0, clobber=0, exit_on_failure=False) :
    if tfm_type[-1] == 'SyN' :
        tfm_fn = tfm_prefix + 'warp.h5'
    else : 
        tfm_fn = tfm_prefix + '0GenericAffine.mat'

    if not os.path.exists(moving_rsl_fn) or clobber > 0 :
        
        ### Inputs
        cmdline =  "antsRegistration --verbose "+str(verbose)
        cmdline += " --write-composite-transform 1 --float --collapse-output-transforms 1 --dimensionality 3 "

        if init_tfm == None : 
            cmdline += " --initial-moving-transform [ "+fixed_fn+", "+moving_fn+", 1 ] "
        else : 
            if init_inverse :
                cmdline += " --initial-moving-transform [" + init_tfm + ",1] "
            else :
                cmdline += " --initial-moving-transform " + init_tfm + " "

        cmdline += " --initialize-transforms-per-stage 0 --interpolation Linear "
        
        registration_level = len(iterations)
        for level in range(registration_level) :
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
            cmdline += " --convergence [ "+iterations[level]+" , 1e-08, 10 ] "
            cmdline += " --smoothing-sigmas "+smooth_sigma+"vox --shrink-factors "+shrink_factor
            cmdline += " --use-estimate-learning-rate-once 1 --use-histogram-matching 0 "
        
        ### Outputs
        cmdline+=" --output [ "+tfm_prefix+" ,"+moving_rsl_fn+","+moving_rsl_fn_inverse+"] "
        print(cmdline) ; 
        #exit(0) 
        try : 
            #Run command line
            shell(cmdline)
        except RuntimeError :
            if exit_on_failure :
                exit(1)
            else :
                pass
    return tfm_fn
