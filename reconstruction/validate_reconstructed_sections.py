import nibabel as nib
import numpy as np
import ants
import skimage
import contextlib
import os
import re
import matplotlib.pyplot as plt
import pandas as pd
from glob import glob
from scipy.ndimage.filters import gaussian_filter
from subprocess import call, Popen, PIPE, STDOUT
from nibabel.processing import resample_to_output, resample_from_to

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


def create_mask_rec_space(label_3d_space_rec_fn, y, nl_2d_fn, label_2d_rsl_fn):
    print(nl_2d_fn)
    nl_2d_img = nib.load(nl_2d_fn)
    nl_2d_vol = nl_2d_img.get_fdata()
    label_3d = np.zeros_like(nl_2d_vol)
    print(label_3d.shape)

    label_2d_rsl_img = nib.load(label_2d_rsl_fn)
    label_2d_rsl_vol = label_2d_rsl_img.get_fdata()
    label_3d[:,y,:] = label_2d_rsl_vol
   
    nib.Nifti1Image(label_3d, nl_2d_img.affine).to_filename(label_3d_space_rec_fn)


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


def plot_target_section(im_0_fn,im_1_fn,y):
    sec_0 = nib.load(im_0_fn).get_fdata()[:,y,:]
    sec_1 = nib.load(im_1_fn).get_fdata()[:,y,:]

    plt.subplot(2,1,1)
    plt.imshow(sec_0)
    plt.subplot(2,1,2)
    plt.imshow(sec_1)
    plt.show()

def transform_section_to_native(autoradiograph_fn, ligand_section_nl2d_fn, nl_2d_inv_tfm, out_dir, basename, y, clobber=False):

    ligand_section_nat_fn = f'{out_dir}/{basename}_y-{y}_space-nat.nii.gz'

    if not os.path.exists(ligand_section_nl2d_fn) or clobber :
        cmd = f'antsApplyTransforms -v 1 -d 2 -i {ligand_section_nl2d_fn} -r {autoradiograph_fn} -t {nl_2d_inv_tfm} -o {ligand_section_nat_fn}' 
        shell(cmd)

    return ligand_section_nat_fn

def get_basename(fn):
    return re.sub('#L#L.nii.gz','',os.path.basename(fn))
        

def difference_sections(autoradiograph_fn, ligand_section_fn, conversion_factor, out_dir, slab, y, clobber=False):
   
    out_fn = f'{out_dir}/slab-{slab}_y-{y}.nii.gz'

    img0 = nib.load(autoradiograph_fn)
    img1 = nib.load(ligand_section_fn)

    vol0 = img0.get_fdata()
    vol1 = img1.get_fdata()

    idx = (vol0 > 0) & (vol1 >0)

    out = np.zeros_like(vol0)

    out[idx] = np.abs(vol0[idx] - vol1[idx]) / vol0[idx] 

    nib.Nifti1Image(out, img0.affine).to_filename(out_fn)

    return out_fn


def get_factor_and_section(basename, df_info):
    bool_ar = [True if basename in fn else False for fn in df_info['lin_fn'] ]

    assert np.sum(bool_ar) > 0, f'Error: {basename} not found in data frame'
    
    conversion_factor = df_info['conversion_factor'].loc[bool_ar].values[0]
    section = df_info['global_order'].loc[bool_ar].values[0]
    slab = df_info['slab'].loc[bool_ar].values[0]
    return section, slab, conversion_factor

def transform_reconstructed_volume(ligand_volume_fn, nl_2d_fn, nl_3d_inv_fn, clobber=False):
    '''
    About:
        Transform reconstructed volume for a particular ligand into nl2d space.

    Arguments:
        ligand_volume_fn :  string, filename of reconstructed ligand volume, mni space.
        nl_2d_fn :          string, filename of autoradiographs non-linearly aligned to donor MRI, nl2d space.
        nl_3d_inv_fn:       string, filename of ANTs transform (hdf5) from mni to nl2d label_3d_space_rec_fn
        clobber :           bool, overwrite exisitng files
    Returns:
        reconstructed_tfm_to_nl2d_fn:   string, filename of reconstructed ligand volume transformed to nl2d space and resampled to coordinates of nl_2d_fn
    '''

    ligand_volume_basename = os.path.splitext(ligand_volume_fn)[0]
    reconstructed_tfm_to_nl2d_fn = f'{ligand_volume_basename}_space-nl2d.nii.gz'

    if not os.path.exists(reconstructed_tfm_to_nl2d_fn) or clobber:
        cmd = f'antsApplyTransforms -v 1 -d 3  -r {nl_2d_fn} -i {ligand_volume_fn} -t {nl_3d_inv_tfm} -o {reconstructed_tfm_to_nl2d_fn}' 
        print(cmd)
        shell(cmd)

    return reconstructed_tfm_to_nl2d_fn


def extract_section_from_volume(reconstructed_tfm_to_nl2d_fn, reference_section_fn, out_dir, basename, y, clobber=False):
    '''

    '''

    ligand_section_fn = f'{out_dir}/{basename}_y-{y}_space-nl2d.nii.gz'

    if not os.path.exists(ligand_section_fn) or clobber :
        img = nib.load(reconstructed_tfm_to_nl2d_fn)

        img2d=nib.load(reference_section_fn) 

        vol = img.get_fdata()
        
        section = vol[:,int(y),:]

        nib.Nifti1Image(vol, img2d.affine).to_filename(ligand_section_fn)

    return ligand_section_fn

def validate_reconstructed_sections(resolution, n_depths, df, base_out_dir='/data/receptor/human/output_2/', clobber=False):

    df_info = pd.read_csv(f'{base_out_dir}/autoradiograph_info_volume_order.csv')

    #get list of nifti images
    #nii_list = [ fn for fn in  glob(f'{base_out_dir}/0_crop/*{ligand}*nii.gz') ]
    #basename_list = [ get_basename(fn) for fn in nii_list ]
    #info_list = [ get_factor_and_section(basename, df_info) for basename in basename_list ]

    #for autoradiograph_fn, basename, (y, slab, conversion_factor) in zip(nii_list, basename_list, info_list) : 
    for row_index, row in df.iterrows():
        y = row['global_order']
        basename = row['basename']
        conversion_factor = row['conversion_factor']
        slab = row['slab']
        autoradiograph_fn = row['crop_fn']

        out_dir = f'{base_out_dir}/MR1_R_{slab}/{resolution}/'
        nl_3d_inv_fn = f'{out_dir}/4_nonlinear_2d/rec_to_mri_SyN_InverseComposite.h5'
        nl_2d_fn = f'{out_dir}/4_nonlinear_2d/MR1_R_{slab}_nl_2d_{resolution}mm.nii.gz'
        reference_section_fn = f'{out_dir}/4_nonlinear_2d/tfm/y-{y}.0_fx.nii.gz'
        nl_2d_inv_tfm = f'{out_dir}/4_nonlinear_2d/tfm/y-{y}.0_InverseComposite.h5'

        # Transform reconstructed ligand volume to nl2d space produced in stage 4 of reconstruction
        reconstructed_tfm_to_nl2d_fn = transform_reconstructed_volume(ligand_volume_fn, nl_2d_fn, nl_3d_inv_fn, clobber=clobber)
        
        # Get a current section (y) from ligand volume in nl2d space
        ligand_section_nl2d_fn = extract_section_from_volume(reconstructed_tfm_to_nl2d_fn, reference_section_fn, out_dir, basename, y, clobber=clobber)

        # Transform reconstructed ligand section into native autoradiograph space
        ligand_section_nat_fn = transform_section_to_native(autoradiograph_fn, ligand_section_nl2d_fn, nl_2d_inv_tfm, out_dir, basename, y, clobber=clobber) 

        # Calculate the difference between raw autoradiograph and the reconstructed section
        difference_fn = difference_sections(autoradiograph_fn, ligand_section_nl2d_fn, conversion_factor, out_dir, slab, y, clobber=clobber)
        

