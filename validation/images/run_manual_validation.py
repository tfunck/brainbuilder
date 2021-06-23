import nibabel as nib
import numpy as np
import ants
import skimage
import contextlib
import os
import re
from subprocess import call, Popen, PIPE, STDOUT
from nibabel.processing import resample_to_output

newlines = ['\n', '\r\n', '\r']
def unbuffered(proc, stream='stdout'):
    stream = getattr(proc, stream)
    with contextlib.closing(stream):
        while True:
            out = []
            last = stream.read(1)
            # Don't loop forever
            if last == '' and proc.poll() is not None:
                break
            while last not in newlines:
                # Don't loop forever
                if last == '' and proc.poll() is not None:
                    break
                out.append(last)
                last = stream.read(1)
            out = ''.join(out)
            print(out)
            yield out

def shell(cmd, verbose=False,exit_on_failure=True):
    '''Run command in shell and read STDOUT, STDERR and the error code'''
    stdout=""
    if verbose :
        print(cmd)


    process=Popen( cmd, shell=True, stdout=PIPE, stderr=STDOUT, universal_newlines=True, )

    for line in unbuffered(process):
        stdout = stdout + line + "\n"
        if verbose :
            print(line)

    errorcode = process.returncode
    stderr=stdout
    if errorcode != 0:
        print ("Error:")
        print ("Command:", cmd)
        print ("Stderr:", stdout)
        if exit_on_failure : exit(errorcode)
    return stdout, stderr, errorcode

def fix_label_img(in_fn, ref_fn, out_fn):

    img = nib.load(in_fn)
    ref_img = nib.load(ref_fn)
    affine = img.affine
    
    affine[:,3] = ref_img.affine[:,3]
    nib.Nifti1Image(np.mean(img.get_fdata(),axis=2), affine).to_filename(out_fn)

def create_mask_rec_space(label_3d_space_rec_fn, y, nl_2d_fn, label_2d_rsl_fn,label_2d_rsl_dwn_fn, nl_3d_tfm):
    nl_2d_img = nib.load(nl_2d_fn)
    nl_2d_vol = nl_2d_img.get_fdata()
    label_3d = np.zeros_like(nl_2d_vol)

    label_2d_rsl_img = nib.load(label_2d_rsl_fn)
    label_2d_rsl_vol = label_2d_rsl_img.get_fdata()

    label_2d_rsl_vol = skimage.transform.resize(label_2d_rsl_vol, (label_3d.shape[0], label_3d.shape[2]))
    label_3d[:,y,:] = label_2d_rsl_vol
   
    affine = nl_2d_img.affine
    affine[1,1] = affine[2,2]
    affine[1,3] = affine[2,3]
    affine[2,2] = 0.02
    affine[2,3] = 0
    print(affine)
    #nib.Nifti1Image(label_2d_rsl_vol, affine).to_filename(label_2d_rsl_dwn_fn)

    nib.Nifti1Image(label_3d, nl_2d_img.affine).to_filename(label_3d_space_rec_fn)
    #label_3d = skimage.morphology.binary_dilation(label_3d).astype(int)

    # save the volume 
    nib.Nifti1Image(label_3d, nl_2d_img.affine).to_filename(label_3d_space_rec_fn)

def process_image(autoradiograph_fn, y, ligand_volume_fn, nl_2d_fn, nl_3d_tfm, conversion_factor, clobber=False):

    base_filename = re.sub('.nii.gz','',os.path.basename(autoradiograph_fn))
    autoradiograph_density_fn = f'{base_filename}_density.nii.gz' 
    label_2d_origin_fn = f'{base_filename}_label1.nii.gz'
    label_2d_fn = f'{base_filename}_label1.1.nii.gz'
    label_2d_rsl_fn = f'{base_filename}_label1.1_rsl.nii.gz'
    label_2d_rsl_dwn_fn = f'{base_filename}_label1.1_rsl_dwn.nii.gz'

    reconstructed_tfm_to_nl2d_fn=f'{base_filename}_space-nl2d.nii.gz'
    reconstructed_tfm_to_nat_fn =f'{base_filename}_space-nat.nii.gz'
    
    label_3d_space_rec_fn = f'{base_filename}_label1_3d_space-nat.nii.gz'
    label_3d_space_mni_fn = re.sub('.nii.gz', '_space-nl2d.nii.gz', nl_2d_fn )
    tfm_fn = f'y-{y}.0_Composite.h5'
    inv_tfm_fn = f'y-{y}.0_InverseComposite.h5'
   
    if not os.path.exists('y-282.0_rsl_density.nii.gz') :
        img = nib.load('y-282.0_rsl.nii.gz')
        data = img.get_fdata()
        data *= conversion_factor
        nib.Nifti1Image(data, img.affine).to_filename('y-282.0_rsl_density.nii.gz')
    
    if not os.path.exists(autoradiograph_density_fn) or clobber :
        img = nib.load(autoradiograph_fn)
        data = img.get_fdata()
        data *= conversion_factor
        nib.Nifti1Image(data, img.affine).to_filename(autoradiograph_density_fn)

    # Fix the dimensions of the label image
    if not os.path.exists(label_2d_fn) or clobber :
        fix_label_img(label_2d_origin_fn, autoradiograph_fn, label_2d_fn)

    # Apply 2D transform to linearized autoradiograph
    if not os.path.exists(label_2d_rsl_fn) or clobber :
        shell(f'antsApplyTransforms -v 1 -d 2  -i {label_2d_fn} -r  test_bad.nii.gz -t {tfm_fn} -o {label_2d_rsl_dwn_fn}')

    # Put the 2D linearized autoradiograph into a 3D volume 
    #if not os.path.exists(label_3d_space_rec_fn) or clobber:
    #    create_mask_rec_space(label_3d_space_rec_fn, y, nl_2d_fn, label_2d_rsl_fn, label_2d_rsl_dwn_fn, nl_3d_tfm)

    # Transform 3D volume to MNI space
    if not os.path.exists(label_3d_space_mni_fn)  or clobber :
        #shell(f'antsApplyTransforms -v 1 -d 3  -i {label_3d_space_rec_fn} -r {ligand_volume_fn} -t {nl_3d_tfm} -o {label_3d_space_mni_fn}')
        shell(f'antsApplyTransforms -v 1 -d 3  -r {nl_2d_fn} -i {ligand_volume_fn} -t {nl_3d_tfm} -o {label_3d_space_mni_fn}')

    if not os.path.exists(reconstructed_tfm_to_nl2d_fn) or clobber :
        data = nib.load(label_3d_space_mni_fn).get_fdata()
        section = data[:,y,:]
        affine = nib.load(label_2d_rsl_dwn_fn).affine
        nib.Nifti1Image(section, affine).to_filename(reconstructed_tfm_to_nl2d_fn)

    if not os.path.exists(reconstructed_tfm_to_nat_fn) or clobber:
        shell(f'antsApplyTransforms -v 1 -d 2  -r {autoradiograph_fn} -i {reconstructed_tfm_to_nl2d_fn} -t {inv_tfm_fn} -o {reconstructed_tfm_to_nat_fn}')
    

    return label_2d_fn,  label_2d_rsl_dwn_fn, label_3d_space_rec_fn, label_3d_space_mni_fn

def calc_init_receptor_density(autoradiograph_fn, label_fn) :
    ### Calculate receptor densities 
    # Initial
    print('\t','lignand:',autoradiograph_fn)
    print('\t','label:',label_fn)
    orig_2d_vol = nib.load(autoradiograph_fn).get_fdata()

    label_vol = nib.load(label_fn).get_fdata()

    #print('Conversion Factor', conversion_factor)
    idx = label_vol > 0

    density = np.mean(orig_2d_vol[ idx ]) # * label_vol[ idx ])
    return density


def calc_3d_receptor_density(ligand_vol_fn, label_fn):
    ### Calculate receptor densities 
    # Final
    ligand_vol = nib.load(ligand_vol_fn).get_fdata()
    label_vol  = nib.load(label_fn).get_fdata()

    idx = (label_vol > 0.50) & (ligand_vol > 0)

    density = np.mean(ligand_vol[ idx ]) # * label_3d_space_mni[idx])

    return density


def get_conversion_factor():
    CMax=172630.6146
    SA=70 
    KD=0.8 
    L=1.64
    conversion_factor = (CMax/( 255 * SA )) * ((KD + L) / L)
    return conversion_factor

if __name__ == '__main__':
    clobber=False

    ligand_volume_fn='MR1_R_oxot_0.4mm.nii.gz'
    #nl_3d_tfm='rec_to_mri_SyN_Composite.h5'
    nl_3d_tfm='rec_to_mri_SyN_InverseComposite.h5'
    nl_2d_fn = 'MR1_R_3_nl_2d_0.4mm.nii.gz'
    file_list = [('RD#HG#MR1s3#R#oxot#5774#17#L#L.nii.gz', 282),
                ]
    for autoradiograph_fn, y in file_list : 
        print(autoradiograph_fn)
        conversion_factor = get_conversion_factor()
        label_2d_fn, label_2d_rsl_dwn_fn, label_3d_rec_fn, label_3d_fn = process_image(autoradiograph_fn, y, ligand_volume_fn, nl_2d_fn, nl_3d_tfm, conversion_factor, clobber=clobber)
        

        init_density  = calc_init_receptor_density(autoradiograph_fn, label_2d_fn) 
        init_density *= conversion_factor
        print('\tInitial Mean:', init_density)

        inter_density = calc_3d_receptor_density('y-282.0_rsl.nii.gz', label_2d_rsl_dwn_fn)
        inter_density *= conversion_factor
        print('\tIntermediate 2D Mean:', inter_density)

        inter_density = calc_3d_receptor_density(nl_2d_fn, label_3d_rec_fn)
        inter_density *= conversion_factor
        print('\tIntermediate 3D Mean:', inter_density)

        #final_density = calc_3d_receptor_density(ligand_volume_fn, label_3d_fn)
        def transform_section_to_native(autoradiograph_fn, label_3d_fn, y, conversion_factor):
            data = nib.load(label_3d_fn).get_fdata()
            section = data[:,y,:]
            
            img = nib.load(autoradiograph_fn)
            section = img.get_fdata() * conversion_factor

            err[section<1] = 0
            print(err.shape)
            print(img.affine)
            nib.Nifti1Image(err, img.affine ).to_filename('error.nii.gz')
            final_density = calc_3d_receptor_density(label_3d_fn, label_3d_rec_fn)
            print('\tFinal Mean:', final_density)


