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
import seaborn as sns
from utils.utils import splitext
from glob import glob
from reconstruction.surface_interpolation import get_valid_coords
from nibabel.processing import resample_from_to
from skimage.transform import resize
from scipy.ndimage.filters import gaussian_filter 
from nibabel.processing import resample_to_output
from subprocess import call, Popen, PIPE, STDOUT
from utils.utils import prefilter_and_downsample, shell

def create_error_volumes(df, qc_dir):

    for (slab, source, ligand), sdf in df.groupby( ['slab', 'source', 'ligand'] ) : 
        space = 'mni' if source == 'reconstructed' else 'nl2d'
        out_fn = f'./MR1_R_{slab}_{ligand}_val-error_space-{space}.nii.gz'
        
        if not os.path.exists(out_fn) :
            atlas_rsl_str = f'{qc_dir}/MR1_R_{int(slab)}_{ligand}_pseudo-cls*space-{space}.nii.gz'
            atlas_rsl_list  = glob(atlas_rsl_str)
            if len(atlas_rsl_list) == 1 : atlas_rsl_fn = atlas_rsl_list[0]
            else : 
                print('Skipping', slab, source, ligand)
                continue
            print('\tLabel volume:', atlas_rsl_fn)
            img = nib.load(atlas_rsl_fn)
            atlas = img.get_fdata().astype(int)

            unique_labels = np.unique(atlas)
            out_vol = np.zeros(atlas.shape)
            for i, row in sdf.iterrows() :
                label = int(row['label'])
                idx =  atlas == int(label)
                label_sum = np.sum(idx)
                assert np.sum(idx) >  0, 'Error, label {label} not found in label volume'
                average = np.abs(100- row['accuracy'])
                out_vol[ idx ] = average
            assert np.sum(out_vol) > 0 , 'Error: empty output volume'

            print('\tError volume:', out_fn)
            nib.Nifti1Image(out_vol.astype(np.float32), img.affine).to_filename(out_fn)

def norm(df) :

    df_target = df.loc[ (df['source'] == 'nl2d') | (df['source'] == 'reconstructed') ]
    df_orig = df.loc[ df['source'] == 'original' ]

    df_out = pd.DataFrame({  'source':[], 
                         'slab':[],
                         'y':[],
                         'label':[],
                         'volume':[], 
                         'accuracy':[] })

    for i, row in df_target.iterrows() :
        label = row['label']
        
        orig_row = df_orig.loc[ df_orig['label'] == label ] 
        orig_average = orig_row['average'].values[0]
        volume=orig_row['volume'].values[0]
        ratio = 100. * row['average'] / orig_average 
        df_out=df_out.append(pd.DataFrame({ 'source':[row['source']], 
                   'label':[row['label']],
                   'slab':[row["slab"]],
                   'ligand':[row["ligand"]],
                   'y':[row["y"]],
                   'volume':[volume], 
                   #FIXME
                   'accuracy':[ratio] }) 
                   )
    return df_out


def plot(df, out_fn='validation_accuracy.png') :
    nl2d_i = df['source']=='nl2d'
    rec_i = df['source']=='reconstructed'

    y0 = [ df['accuracy'].loc[df['source']=='nl2d'].mean() ] * 2
    y1 = [ df['accuracy'].loc[df['source']=='reconstructed'].mean() ] * 2

    x0=[0, df['volume'].max() ]

    fig, axes = plt.subplots(1, 2, figsize=(16,10))
    #ax[.title('Accuracy after 2d Alignment')
    axes[0].set_title('Accuracy after 2D Alignement')
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].set_xlabel('ROI Volume (mm2)')
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)

    axes[1].set_title('Accuracy after 3D Reconstruction')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_xlabel('ROI Volume (mm2)')
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)


    sns.scatterplot(x='volume', y='accuracy', hue='slab', data=df.loc[nl2d_i], ax=axes[0], palette='tab10') 
    sns.scatterplot(x='volume', y='accuracy', hue='slab', data=df.loc[rec_i], ax=axes[1], palette='tab10') 
    plt.tight_layout()
    plt.savefig(out_fn)


def validation_visualization(qc_fn):
    print(qc_fn)
    qc_dir = os.path.dirname(qc_fn)
    df = pd.read_csv(qc_fn)
    df = norm(df)
    plot(df)
    create_error_volumes(df, qc_dir)

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


def calculate_regional_averages(atlas_fn, ligand_fn, source, resolution, qc_csv_fn, conversion_factor=1, extra_mask_fn=''):
    print(qc_csv_fn)
    if not os.path.exists(qc_csv_fn):
        print('labels:',atlas_fn)
        print('ligand:',ligand_fn)

        atlas_img = nib.load(atlas_fn)
        ligand_img = nib.load(ligand_fn)

        atlas_volume = atlas_img.get_fdata().astype(int)
        reconstucted_volume = ligand_img.get_fdata()

        atlas_volume[reconstucted_volume < 1] = 0

        averages_out, labels_out, n_voxels = average_over_label(atlas_volume.reshape(-1,), reconstucted_volume.reshape(-1,), conversion_factor=conversion_factor )

        print(labels_out)
        voxel_size = atlas_img.affine[0,0] * atlas_img.affine[1,1] * atlas_img.affine[2,2]

        volume = n_voxels * voxel_size
        if source == 'original' :
            assert np.min(volume) >= float(resolution), f'Error: label size ({np.min(volume)}) less than resolution size of {resolution} for labels\n {labels_out[volume < float(resolution) ]} \nin \n {atlas_fn}'
        
        df = pd.DataFrame({'source':[source]*len(labels_out), 'label':labels_out,'average':averages_out, 'volume':volume})
        
        print(df)
        df.to_csv(qc_csv_fn)
    else :
        df = pd.read_csv(qc_csv_fn)
    return df


def average_over_label(labels, values, conversion_factor=1 ):
    averages_out = []
    labels_out = []
    n_out = [] 

    unique_labels = np.unique( labels )[1:]
    for label in unique_labels :
        idx = labels == label
        n = np.sum(idx)
        print('mean and conversion_factor')
        print(np.mean(values[idx]), conversion_factor, np.mean(values[idx]) * conversion_factor)
        averages_out.append(np.mean(values[idx]) * conversion_factor )
        labels_out.append(label)
        n_out.append(n)
    return np.array(averages_out), np.array(labels_out), np.array(n_out)


def combine_data_frames(df_list,y,slab,ligand):

    df_out = pd.concat(df_list)
    
    df_out['y'] = [y]*df_out.shape[0]
    df_out['slab'] = [slab]*df_out.shape[0]
    df_out['ligand'] = [ligand]*df_out.shape[0]

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
            assert not pd.isnull(y), f'Error: null y index, {row}'
            nl2d_auto_rsl_fn = f'{out_dir}/4_nonlinear_2d/tfm/y-{y}_rsl.nii.gz'
            
            # Pseudo-classified image in native space
            cls_nat_fn = row['pseudo_cls_fn']
            # Transform to bring native space image into nl2d space
            nl_2d_tfm = f'{out_dir}/4_nonlinear_2d/tfm/y-{y}_Composite.h5'
            # y index value
            basename = row['base']

            # Transform the native pseudo classified image into nl2d space
            cls_nl2d_fn = transform_volume(cls_nat_fn, nl2d_auto_rsl_fn, nl_2d_tfm, 2, qc_cls_nl2d_dir, f'y-{y}', clobber=clobber) 
            
            # Calcuate padding around y. This is used to 'thicken' the nl2d pseudo-classified images
            width = np.rint(float(resolution) / nl2d_img.affine[1,1]).astype(int)
            pad = width/2+1 if width % 2 == 0 else (width+1)/2
            y0 = max(int(y) - pad, 0)
            y1 = min(int(y) + pad, nl2d_img.shape[1])
            
            # Define dimensions for transformed section
            dim=[nl2d_img.shape[0], 1, nl2d_img.shape[2]]
            
            # Save nl2d pseudo-classified sections
            section = nib.load(cls_nl2d_fn).get_fdata().reshape(dim)
            # Put transformed section into a volume
            nl2d_cls_vol[:, int(y0):int(y1),:] =np.repeat(section,y1-y0, axis=1) 

        # Save volume of thickened pseudo-classified sections in nl2d    
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
            nl_3d_fn = f'{out_dir}/3_align_slab_to_mri/rec_to_mri_SyN_Composite.h5'

            # Create a volume in nl2d space based on
            cls_nl2d_fn = create_nl2d_cls_vol(slab_df, nl_2d_fn, out_dir, qc_dir, brain, hemisphere, slab, ligand, resolution)
            print('\tPseudo-classified volume (space=nl2d):', cls_nl2d_fn)

            #Transform nl2d psuedo-classified image into mni space
            cls_mni_fn = transform_volume(cls_nl2d_fn, reconstructed_volume_fn, nl_3d_fn, 3, qc_dir, f'space-mni')
            print('\tPseudo-classified volume (space=mni):', cls_mni_fn)

            qc_csv_fn=f'{qc_dir}/{brain}_{hemisphere}_{slab}_{ligand}_space-nl2d.csv'
            df_nl2d = calculate_regional_averages(cls_nl2d_fn, nl_2d_fn, 'nl2d',
                                                  resolution, qc_csv_fn)
            
            qc_csv_fn=f'{qc_dir}/{brain}_{hemisphere}_{slab}_{ligand}_space-reconstructed.csv'
            df_mni = calculate_regional_averages(cls_mni_fn, reconstructed_volume_fn, 'reconstructed', 
                    resolution, qc_csv_fn) 
            df_list += [ combine_data_frames( [df_nl2d], None, slab, ligand) ]
            df_list += [ combine_data_frames( [df_mni], None, slab, ligand) ]
        
        for row_index, row in df.iterrows():
            slab = row['slab']
            brain = row['mri']
            hemisphere = row['hemisphere']
            slab = row['slab']
            ligand = row['ligand']
            y = row['volume_order']

            out_dir = f'{base_out_dir}/MR1_R_{slab}/{resolution}mm/'
            assert not pd.isnull(y), f'Error: null y index, {row}'
            conversion_factor = row['conversion_factor']
            autoradiograph_fn = row['crop_fn']
            ligand = row['ligand']
            cls_nat_fn = row['pseudo_cls_fn']
            # Calculate averages on cropped autoradiograph
            qc_csv_fn=f'{qc_dir}/{brain}_{hemisphere}_{slab}_{ligand}_{y}_space-original.csv'
            print('calculate regional averages')
            print('\ty=', y, 'conversion:', conversion_factor)
            df_nat = calculate_regional_averages(cls_nat_fn, autoradiograph_fn, 'original', 
                                                resolution,qc_csv_fn, conversion_factor=conversion_factor)
            df_list += [ combine_data_frames([df_nat], y, slab, ligand) ]

        pd.concat(df_list).to_csv(out_fn)
        validation_visualization(out_fn)
