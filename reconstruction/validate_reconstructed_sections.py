import contextlib
import os
import re
import matplotlib.pyplot as plt
import pandas as pd
import nibabel as nib
import numpy as np
import nibabel
import tempfile
import h5py
from utils.utils import splitext
from glob import glob
from reconstruction.surface_interpolation import get_valid_coords
from nibabel.processing import resample_from_to
from skimage.transform import resize
from scipy.ndimage.filters import gaussian_filter 
from nibabel.processing import resample_to_output
from subprocess import call, Popen, PIPE, STDOUT
from utils.utils import prefilter_and_downsample

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

def shell(cmd, verbose=False, exit_on_failure=True):
    '''Run command in shell and read STDOUT, STDERR and the error code'''
    stdout = ""
    if verbose:
        print(cmd)


    process = Popen(cmd, shell=True, stdout=PIPE, stderr=STDOUT, universal_newlines=True, )

    for line in unbuffered(process):
        stdout = stdout + line + "\n"
        if verbose:
            print(line)

    errorcode = process.returncode
    stderr=stdout
    if errorcode != 0:
        print("Error:")
        print("Command:", cmd)
        print("Stderr:", stdout)
        if exit_on_failure: exit(errorcode)
    return stdout, stderr, errorcode


def create_mask_rec_space(label_3d_space_rec_fn, y, nl_2d_fn, label_2d_rsl_fn):
    nl_2d_img = nib.load(nl_2d_fn)
    nl_2d_vol = nl_2d_img.get_fdata()
    label_3d = np.zeros_like(nl_2d_vol)

    label_2d_rsl_img = nib.load(label_2d_rsl_fn)
    label_2d_rsl_vol = label_2d_rsl_img.get_fdata()
    label_3d[:,y,:] = label_2d_rsl_vol
   
    nib.Nifti1Image(label_3d, nl_2d_img.affine).to_filename(label_3d_space_rec_fn)


def calc_init_receptor_density(autoradiograph_fn, label_fn):
    ### Calculate receptor densities 
    # Initial
    print('\t', 'lignand:', autoradiograph_fn)
    print('\t', 'label:', label_fn)
    orig_2d_vol = nib.load(autoradiograph_fn).get_fdata()

    label_vol = nib.load(label_fn).get_fdata()

    #print('Conversion Factor', conversion_factor)
    idx = label_vol > 0

    density = np.mean(orig_2d_vol[idx]) # * label_vol[ idx ])
    return density

def load_section(im_fn, y=0):
    img = nib.load(im_fn)
    if np.all(np.array(img.shape) > 1 ) and len(img.shape)>2:
        sec = nib.load(im_fn).get_fdata()[:,int(y),:]
    else :
        sec = nib.load(im_fn).get_fdata()
    return sec

def plot_target_section(image_list,label_list, out_dir, basename, y):
    n=len(image_list) 
    f, ax = plt.subplots(1,n,figsize=(16,9))
    for i, (image_fn, label_fn) in enumerate(zip(image_list, label_list)) :
        print('image', image_fn)
        print('label', label_fn)
        sec_0 = load_section(image_fn,y)
        sec_1 = load_section(label_fn,y)
        im0 = ax[i].imshow(sec_0, cmap='gray')
        im1 = ax[i].imshow(sec_1,vmin=1, cmap='tab20', alpha=0.5)
        f.colorbar(im1, ax=ax[i])
        #plt.colorbar()
    
    plt.savefig(f'{out_dir}/{basename}_{y}.png')
    plt.clf()
    plt.cla()


def transform_section( input_fn, reference_fn, tfm_fn, out_dir, basename, y, clobber=False):

    out_fn = f'{out_dir}/{basename}_y-{y}_space-nat.nii.gz'

    if not os.path.exists(nlaligned_nat_fn) or clobber :
        cmd = f'antsApplyTransforms -v 0 -n NearestNeighbor -d 2 -i {input_fn} -r {reference_fn} -t {tfm_fn} -o {out_fn}' 
        shell(cmd, verbose=False)

    return nlaligned_nat_fn

def get_basename(fn):
    return re.sub('#L#L.nii.gz','',os.path.basename(fn))
        

def difference_sections(autoradiograph_fn, nlaligned_fn, conversion_factor, out_dir, basename, y, clobber=False):

    out0_fn = f'{out_dir}/{basename}_y-{y}_accuracy.nii.gz' 
    img0_rsl_fn = f'{out_dir}/{basename}_y-{y}_auto-lores.nii.gz'

    #out1_fn = f'{out_dir}/{basename}_y-{y}_hires.nii.gz'

    img0 = nib.load(autoradiograph_fn)
    vol0 = img0.get_fdata()

    img1 = nib.load(nlaligned_fn)
    vol1 = img1.get_fdata()

    vol0 *= conversion_factor

    vol0 = gaussian_filter(vol0, (img1.affine[0,0]/img0.affine[0,0])/np.pi )
    vol0 = resize(vol0, vol1.shape, anti_aliasing=True)
    nib.Nifti1Image(vol0, img1.affine).to_filename(img0_rsl_fn)

    error_mean_lo, error_std_lo, error_vol = calculate_accuracy(img0_rsl_fn, nlaligned_fn, out0_fn )   

    #resample_from_to(img1, img0, order=5).to_filename(img1_rsl_fn) 
    #error_mean_hi, error_std_hi = calculate_accuracy(autoradiograph_fn, img1_rsl_fn, out1_fn )   
    im_max = max(np.percentile(vol0,[95])[0], np.percentile(vol1,[95])[0] )
    im_min = min(np.percentile(vol0,[5])[0], np.percentile(vol1,[5])[0] )
    
    fig, ax = plt.subplots(1, 3, figsize=(18,6)) 
    
    ax[0].set_title('Autoradiograph')
    i0 = ax[0].imshow(vol0, vmin=im_min, vmax=im_max)
    #fig.colorbar(i0, ax=ax[0])
    
    ax[1].set_title('Reconstructed')
    i1 = ax[1].imshow(vol1, vmin=im_min, vmax=im_max)
    #fig.colorbar(i1, ax=ax[1])
    
    #ax[1][0].set_title('Overlap')
    #ax[1][0].imshow(vol0, cmap='gray', vmin=im_min, vmax=im_max)
    #ax[1][0].imshow(vol1, cmap='hot', alpha=0.5, vmin=im_min, vmax=im_max)

    ax[2].set_title(f'Error: {np.round(error_mean_lo,2)}')
    ax[2].imshow(vol0/np.max(vol0), cmap='gray')
    ax[2].imshow(error_vol/np.max(error_vol), cmap='hot', alpha=0.6)
    plt.savefig(f'{out_dir}/{basename}_y-{y}_qc.png')
    plt.clf()
    plt.cla()

    return error_mean_lo, error_std_lo


def calculate_accuracy(img0_fn, img1_fn, out_fn, clobber=False):
    img0 = nib.load(img0_fn)
    img1 = nib.load(img1_fn)
    
    vol0 = img0.get_fdata()
    vol1 = img1.get_fdata()

    idx = (vol0 > 1) & (vol1 > 1)

    vol0 = gaussian_filter(vol0, (1/0.6)/np.pi)
    vol1 = gaussian_filter(vol1, (1/0.6)/np.pi)


    out = np.zeros_like(vol0)

    out[idx] =  100  * np.abs(vol0[idx] - vol1[idx]) / vol0[idx]
    
    nib.Nifti1Image(out, img0.affine).to_filename(out_fn)

    return np.mean(out[idx]), np.std(out[idx]), out


def get_factor_and_section(basename, df_info):
    bool_ar = [True if basename in fn else False for fn in df_info['lin_fn'] ]

    assert np.sum(bool_ar) > 0, f'Error: {basename} not found in data frame'
    
    conversion_factor = df_info['conversion_factor'].loc[bool_ar].values[0]
    section = df_info['global_order'].loc[bool_ar].values[0]
    slab = df_info['slab'].loc[bool_ar].values[0]
    return section, slab, conversion_factor

def transform_volume( input_fn, reference_fn, tfm_fn, ndim, out_dir, suffix, clobber=False):
    '''
    About:
        Transform reconstructed volume for a particular ligand into nl2d space.

    Arguments:
        input_fn :  string, filename of reconstructed ligand volume, mni space.
        reference_fn :          string, filename of autoradiographs non-linearly aligned to donor MRI, nl2d space.
        tfm_fn:       string, filename of ANTs transform (hdf5) from mni to nl2d label_3d_space_rec_fn
        clobber :           bool, overwrite exisitng files
    Returns:
        out_fn :   string, filename of reconstructed ligand volume transformed to nl2d space and resampled to coordinates of nl_2d_fn
    '''

    out_basename = out_dir + '/' + os.path.basename(splitext(input_fn)[0])
    out_fn = f'{out_basename}_{suffix}.nii.gz'
    
    if not os.path.exists(out_fn) or clobber:
        cmd = f'antsApplyTransforms -v 1  -n NearestNeighbor -d {ndim}  -r {reference_fn} -i {input_fn} -t {tfm_fn} -o {out_fn}'
        shell(cmd)

    return out_fn


def extract_section_from_volume(reconstructed_tfm_to_nl2d_fn, reference_section_fn, out_dir, basename, y, clobber=False):
    '''

    '''

    nlaligned_fn = f'{out_dir}/{basename}_y-{y}.nii.gz'

    if not os.path.exists(nlaligned_fn) or clobber :
        img = nib.load(reconstructed_tfm_to_nl2d_fn)
        print(reconstructed_tfm_to_nl2d_fn)
        print('\tSection shape:',img.shape)

        img2d = nib.load(reference_section_fn) 

        vol = img.get_fdata()
        
        section = vol[:,int(y),:]

        nib.Nifti1Image(section, img2d.affine).to_filename(nlaligned_fn)

    return nlaligned_fn

def calculate_regional_averages(atlas_fn, ligand_fn, source, resolution, conversion_factor=1, extra_mask_fn=''):
    print('labels:',atlas_fn)
    print('ligand:',ligand_fn)

    atlas_img = nib.load(atlas_fn)
    ligand_img = nib.load(ligand_fn)

    atlas_volume = atlas_img.get_fdata().astype(int)
    reconstucted_volume = ligand_img.get_fdata()

    atlas_volume[reconstucted_volume < 1] = 0

    averages_out, labels_out, n_voxels = average_over_label(atlas_volume.reshape(-1,), reconstucted_volume.reshape(-1,), conversion_factor=conversion_factor )

    voxel_size = atlas_img.affine[0,0] * atlas_img.affine[1,1] * atlas_img.affine[2,2]

    volume = n_voxels * voxel_size
    if source == 'original' :
        assert np.min(volume) >= float(resolution), f'Error: label size ({np.min(volume)}) less than resolution size of {resolution} for labels\n {labels_out[volume < float(resolution) ]} \nin \n {atlas_fn}'

    df = pd.DataFrame({'source':[source]*len(labels_out), 'label':labels_out,'average':averages_out, 'volume':volume})
    return df

def create_index_volume(nl_2d_fn, y, qc_dir, basename, clobber=False) :

    output_fn = f'{qc_dir}/{basename}_index_{y}.nii.gz'

    img = nib.load(nl_2d_fn)
    vol = img.get_fdata()

    output = np.zeros_like(vol)

    output[:,int(y),:] = 1

    nib.Nifti1Image(output, img.affine).to_filename(output_fn)

    return output_fn




def calculate_mesh_regional_averages(base_out_dir, resolution, slab, ligand, atlas_rsl_fn, nl_2d_fn, y, n_depths, source, conversion_factor=1):

    img = nib.load(atlas_rsl_fn)
    vol = img.get_fdata()

    nl_2d_img = nib.load(nl_2d_fn)
    slab_y_start = nl_2d_img.affine[1,3]
    slab_y_step = nl_2d_img.affine[1,1]

    dt = 1.0/ float(n_depths)
    depth_list = np.arange(0, 1+dt/10, dt)
    depth_list = np.insert(depth_list,0, 0)

    steps = [ img.affine[i,i] for i in [0,1,2] ]
    starts = [ img.affine[i,3] for i in [0,1,2] ]
    print('steps', steps, 'starts', starts)
    yw = np.rint( ( (y * slab_y_step + slab_y_start) - starts[0] ) / steps[0] ).astype(int)

    section = vol[:,int(yw),:]

    vertex_values = np.array([])
    vertex_labels = np.array([])

    for depth in depth_list :
        mni_mesh_fn = f'{base_out_dir}/5_surf_interp/surfaces/surf_{resolution}mm_{depth}_rsl.h5'
        slab_mesh_fn = f'{base_out_dir}/5_surf_interp/surfaces/slab-{slab}_surf_{resolution}mm_{depth}_rsl.h5'
        vertex_values_fn = f'{base_out_dir}/5_surf_interp/MR1_R_{ligand}_{resolution}mm_profiles_{depth}_raw.csv'

        y0w = y * slab_y_step + slab_y_start 
        y1w = y * slab_y_step + steps[1] + slab_y_start 

        slab_coords_h5 = h5py.File(slab_mesh_fn,'r')
        slab_coords = slab_coords_h5['data'][:]

        mni_coords_h5 = h5py.File(mni_mesh_fn,'r')
        mni_coords = mni_coords_h5['data'][:]

        temp, valid_coords_idx = get_valid_coords( slab_coords, [y0w,y1w] )
        #assert np.sum(valid_coords_idx) > 0 , f'Error: no valid vertices between {y0w} and {y1w}'
        
        valid_coords_world = mni_coords[valid_coords_idx]

        assert valid_coords_idx.shape[0] == mni_coords.shape[0], 'Error: size of valid_coords_idx does not equal that of mni_coords'

        x = np.rint( (valid_coords_world[:,0] - starts[0])/steps[0] ).astype(int)
        z = np.rint( (valid_coords_world[:,2] - starts[2])/steps[2] ).astype(int)

        vertex_labels = np.append(vertex_labels, section[x,z])

        all_values = pd.read_csv(vertex_values_fn, header=None,index_col=None ).values

        new_values = all_values[valid_coords_idx]

        vertex_values = np.append(vertex_values, new_values.reshape(-1,))

    vertex_labels[vertex_values<1] = 0
 
    averages_out, labels_out, n_vertices = average_over_label(vertex_labels, vertex_values, conversion_factor=conversion_factor )

    df = pd.DataFrame({'source':[source]*len(labels_out), 'label':labels_out,'average':averages_out, 'n':n_vertices})
    return df


def average_over_label(labels, values, conversion_factor=1 ):
    averages_out = []
    labels_out = []
    n_out = [] 

    #print('averages')
    unique_labels = np.unique( labels )[1:]
    for label in unique_labels :
        idx = labels == label
        n = np.sum(idx)
        if n >= 5 : 
            averages_out.append(np.mean(values[idx]) * conversion_factor )
            labels_out.append(label)
            n_out.append(n)
    return np.array(averages_out), np.array(labels_out), np.array(n_out)


def combine_data_frames(df_list,y,slab,ligand):

    df_out = pd.concat(df_list)
    
    df_out['y'] = [y]*df_out.shape[0]
    df_out['slab'] = [slab]*df_out.shape[0]
    df_out['ligand'] = [ligand]*df_out.shape[0]

    #df_out = pd.pivot_table(df_out, index=['slab','ligand','label'], columns=['source'], values=['average','n'])
    #df_out.dropna(inplace=True)
    #df_out['error'] = np.abs(df_out['reconstructed']-df_out['original'])/df_out['original']
    return df_out 


def create_nl2d_cls_vol(df, nl_2d_fn, out_dir, qc_dir, brain, hemisphere, slab, ligand, resolution, clobber=False):
    qc_cls_nl2d_dir = f'{qc_dir}/pseudo_cls'
    nl2d_cls_fn=f'{qc_dir}/{brain}_{hemisphere}_{slab}_{ligand}_pseudo-cls_space-nl2d.nii.gz'
    os.makedirs(qc_cls_nl2d_dir,exist_ok=True)

    if not os.path.exists(nl2d_cls_fn):
        nl2d_img = nib.load(nl_2d_fn)
        nl2d_cls_vol=np.zeros(nl2d_img.shape)
        for row_index, row in df.iterrows():
            y=row['volume_order']
            nl2d_auto_rsl_fn = f'{out_dir}/4_nonlinear_2d/tfm/y-{y}_rsl.nii.gz'
            cls_nat_fn = row['pseudo_cls_fn']
            nl_2d_tfm = f'{out_dir}/4_nonlinear_2d/tfm/y-{y}_Composite.h5'

            y = row['volume_order']
            basename = row['base']
            cls_nl2d_fn = transform_volume(cls_nat_fn, nl2d_auto_rsl_fn, nl_2d_tfm, 2, qc_cls_nl2d_dir, f'y-{y}', clobber=clobber) 
            width = np.rint(float(resolution) / nl2d_img.affine[1,1]).astype(int)
            pad = width/2+1 if width % 2 == 0 else (width+1)/2
            y0 = max(int(y) - pad, 0)
            y1 = min(int(y) + pad, nl2d_img.shape[1])
            print(y0,y1,pad) 
            dim=[nl2d_img.shape[0], 1, nl2d_img.shape[2]]
            section = nib.load(cls_nl2d_fn).get_fdata().reshape(dim)
            nl2d_cls_vol[:, int(y0):int(y1),:] =np.repeat(section,y1-y0, axis=1) 
        nib.Nifti1Image(nl2d_cls_vol, nl2d_img.affine ).to_filename(nl2d_cls_fn)

    return nl2d_cls_fn

def validate_reconstructed_sections(resolution, slabs, n_depths, df, base_out_dir='/data/receptor/human/output_2/', clobber=False):
    
    qc_dir = f'{base_out_dir}/6_quality_control/'
    out_fn = f'{qc_dir}/reconstruction_validation.csv'

    
    ligand = np.unique(df['ligand'])[0]
    reconstructed_volume_fn = f'{base_out_dir}/5_surf_interp/MR1_R_{ligand}_{resolution}mm_space-mni.nii.gz'

    if not os.path.exists(out_fn) or clobber :
        df_list=[]

        os.makedirs(qc_dir, exist_ok=True)

        for (brain, hemisphere, slab, ligand), slab_df in df.groupby(['mri', 'hemisphere', 'slab', 'ligand']) :

            nl_2d_fn = f'{base_out_dir}/5_surf_interp/thickened_{slab}_{ligand}_{resolution}.nii.gz'
            out_dir = f'{base_out_dir}/MR1_R_{slab}/{resolution}mm/'
            cls_nl2d_fn = create_nl2d_cls_vol(slab_df, nl_2d_fn, out_dir, qc_dir, brain, hemisphere, slab, ligand, resolution)
            print('\tPseudo-classified volume (space=nl2d):', cls_nl2d_fn)
            nl_3d_fn = f'{out_dir}/3_align_slab_to_mri/rec_to_mri_SyN_Composite.h5'
            cls_mni_fn = transform_volume(cls_nl2d_fn, reconstructed_volume_fn, nl_3d_fn, 3, qc_dir, f'space-mni')
            print('\tPseudo-classified volume (space=mni):', cls_mni_fn)
            df_nl2d = calculate_regional_averages(cls_nl2d_fn, nl_2d_fn, 'nl2d', resolution)
            df_mni = calculate_regional_averages(cls_mni_fn, reconstructed_volume_fn, 'reconstructed', resolution)
            df_list += [ combine_data_frames( [df_nl2d], None, slab, ligand) ]
            df_list += [ combine_data_frames( [df_mni], None, slab, ligand) ]
        
        for row_index, row in df.iterrows():
            slab = row['slab']
            out_dir = f'{base_out_dir}/MR1_R_{slab}/{resolution}mm/'
            y = row['volume_order']
            conversion_factor = row['conversion_factor']
            autoradiograph_fn = row['crop_fn']
            ligand = row['ligand']
            cls_nat_fn =row['pseudo_cls_fn']

            # Calculate averages on cropped autoradiograph
            df_nat = calculate_regional_averages(cls_nat_fn, autoradiograph_fn, 'original', resolution, conversion_factor=conversion_factor)
            df_list += [ combine_data_frames([df_nat], y, slab, ligand) ]



        pd.concat(df_list).to_csv(out_fn)

