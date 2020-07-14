from nibabel import processing
from glob import glob
import nibabel as nib
import numpy as np
import re


def sum_volume(fn_string, out_fn) :

    img_list = []
    mri = nib.load("civet_out/mr1/final/mr1_t1_tal.nii")
    mri_vol=mri.get_data()
    outVolume = np.zeros(mri_vol.shape)
    for i in range(1,7) :
        try :
            fn=glob(re.sub("<slab>", str(i), fn_string))[0]
        except IndexError :
            continue
    
        try :
            img = processing.resample_from_to( nib.load(fn), mri )
            img_list.append( img  )
        except FileNotFoundError : 
            pass

    for i in range(0,len(img_list)) :
        print(i,)
        vol =  img_list[i].get_data()
        #vol[ vol < 0 ] = 0
        #vol = vol  / np.std(vol)
        #vol /= np.max(vol)
        print(vol.shape, outVolume.shape)
        outVolume += vol

    brain_mask = nib.load('civet_out/mr1/mask/mr1_brain_mask.nii').get_data()
    outVolume = brain_mask * outVolume
    print(out_fn)
    nib.Nifti1Image( outVolume, img_list[0].affine ).to_filename(out_fn)

#sum_volume("output//MR1/R_slab_<slab>/final/flum_space-mni_500um.nii.gz","flum_space-mni_500um_slabs-1-6.nii.gz" )
#sum_volume("flum_slab-<slab>_space-mni_500um.nii.gz","flum_space-mni_500um_slabs-1-6.nii.gz" )
#sum_volume("musc_slab-<slab>_space-mni_500um.nii.gz","musc_space-mni_500um_slabs-1-6.nii.gz" )
sum_volume("sr95_slab-<slab>_space-mni_500um.nii.gz","sr95_space-mni_500um_slabs-1-6.nii.gz" )
#sum_volume("cgp5_slab-<slab>_space-mni_500um.nii.gz","cgp5_space-mni_500um_slabs-1-6.nii.gz" )
