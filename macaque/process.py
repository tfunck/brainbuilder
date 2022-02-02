import os
import numpy as np
import imageio
import matplotlib.pyplot as plt
import shutil
import multiprocessing
import re
import pandas as pd
import nibabel 
import utils.ants_nibabel as nib
from utils.utils import prefilter_and_downsample, resample_to_autoradiograph_sections
from reconstruction.align_slab_to_mri import *
from utils.utils import get_section_intervals
from reconstruction.nonlinear_2d_alignment import create_2d_sections, receptor_2d_alignment, concatenate_sections_to_volume
from reconstruction.receptor_segment import classifyReceptorSlices, interpolate_missing_sections, resample_transform_segmented_images
from skimage.transform import resize
from scipy.ndimage.filters import gaussian_filter
from utils.ANTs import ANTs
from joblib import Parallel, delayed
from scipy.ndimage import binary_dilation
from glob import glob
from preprocessing.preprocessing import fill_regions
from utils.utils import safe_imread, points2tfm 
from skimage.filters import threshold_otsu, threshold_li , threshold_sauvola, threshold_yen
from scipy.ndimage import label, center_of_mass

def crop_image(crop_dir, y, fn, crop_fn):
    img = imageio.imread(fn)
    if len(img.shape) == 3 : img = img[:,:,0]
    mask_fn = '{}/{}'.format(crop_dir,os.path.splitext(os.path.basename(fn))[0]+'_mask.nii.gz')
    qc_fn = '{}/qc_{}'.format(crop_dir,os.path.splitext(os.path.basename(fn))[0]+'_qc.png')

    affine=np.array([[0.045,0,0,0],[0,0.045,0,0],[0,0,0.02,y*1.02],[0,0,0,1]])
    ligand = os.path.basename(fn).split('#')[4]

    if y > 37 : img = np.fliplr(img)

    if ligand in ['oxot', 'epib', 'zm24', 'racl'] :
        nib.Nifti1Image(img.astype(np.float32), affine  ).to_filename(crop_fn)
        return 0

    img_blur = gaussian_filter(img.astype(float), 5)

    t = threshold_otsu(img_blur[100:,:])
   
    seg=np.zeros_like(img)
    seg[ img_blur > t ] = 1
   
    border=np.zeros_like(img)
    bs=25
    border[0:bs,:] = border[-bs:,:] = 1
    border[:,0:bs] = border[:,-bs:] = 1

    seg_labels, uniq_labels = label(seg)

    cropped_img = np.copy(img)
    for l in range(uniq_labels+1) :
        temp = np.zeros_like(img)
        temp[ seg_labels == l ] = 1
        temp *= (1+border)
        if np.max(temp) > 1 :
            print(l, np.max(temp))
            cropped_img[ seg_labels == l ] = 0 
    
    if np.sum(cropped_img) > 0 :
        cropped_img = cropped_img.astype(np.float32)
    else :
        cropped_img = img.astype(np.float32)
    
    nib.Nifti1Image( np.flip(cropped_img, axis=0), affine  ).to_filename(crop_fn)
    plt.cla()
    plt.clf()
    plt.subplot(1,3,1); 
    plt.imshow(img)
    plt.subplot(1,3,2); 
    plt.imshow(seg_labels)
    plt.subplot(1,3,3)
    plt.imshow(cropped_img)
    plt.savefig(qc_fn)

def add_autoradiographs_images(auto_files):
    df_list = []
    for fn in auto_files :
        fn_split = re.sub('.TIF', '', fn).split('#')
        brain, hemisphere, ligand, repeat = [fn_split[i] for i in [2,3,4,6]]
        
        binding = 'S' if not 'S' in repeat and not '00' in repeat and not 'UB' in repeat else 'UB'
        if binding == 'UB' : ligand += '_ub'
        
        repeat = re.sub('UB','',repeat)
        repeat = re.sub('00[a-z]','0',repeat)

        df_list.append(pd.DataFrame({'raw':[fn], 'brain':[brain], 'hemisphere':[hemisphere], 'ligand':[ligand], 'repeat':[int(repeat)],'binding':[binding] }))

    df = pd.concat(df_list)
     
    df = df.loc[df['binding'] != 'UB']
    
    return df

def add_histology_images(hist_files):

    df_list = []
    for fn in hist_files :
        fn_split = re.sub('.nii.gz', '', os.path.basename(fn)).split('_')
        print(fn_split);
        brain, repeat, hemisphere = [fn_split[i] for i in [0,1,4]]

        if brain == 'DP1' : brain = '11530'

        if hemisphere == 'right' : hemisphere = 'R'
        elif hemisphere == 'left' : hemisphere = 'L'
        
        ligand = 'cellbody'
        binding = 'hist' 
        
        df_list.append(pd.DataFrame({'raw':[fn], 'brain':[brain], 'hemisphere':[hemisphere], 'ligand':[ligand], 'repeat':[int(repeat)],'binding':[binding] }))

    df = pd.concat(df_list)
    df['repeat']=df['repeat'].astype(int) 
    df = df.sort_values(['repeat'])
    df['repeat'] = np.arange(8,df.shape[0]+8)

    return df


def create_section_dataframe(auto_dir, crop_dir, csv_fn ):
    
    # Define the order of ligands in "short" repeats
    short_repeat = ['ampa', 'kain', 'mk80', 'cellbody_a', 'musc', 'cgp5', 'flum', 'pire', 'hist_myelin', 'oxot', 'damp', 'epib', 'praz', 'uk14','cellbody_b', 'dpat', 'keta', 'sch2', 'racl', 'hist_myelin', 'dpmg', 'zm24']

    # Define the order of ligands in "long" repeats
    long_repeat = ['ampa', 'ampa_ub', 'kain', 'kain_ub', 'mk80', 'mk80_ub','cellbody_a', 'musc', 'musc_ub', 'cgp5', 'cgp5_ub','flum', 'flum_ub', 'pire', 'pire_ub', 'hist_myelin', 'oxot', 'oxot_ub', 'damp', 'damp_ub', 'epib', 'epib_ub', 'praz', 'praz_ub', 'uk14', 'uk14_ub', 'cellbody_b', 'dpat', 'dpat_ub', 'keta', 'keta_ub', 'sch2', 'sch2_ub', 'racl', 'racl_ub', 'hist_myelin', 'dpmg', 'dpmg_ub', 'zm24', 'zm24_ub']

    # create dictionaries for short and long repeats with integers associated with each ligand
    short_repeat_dict = { v:k for k,v in dict(enumerate(short_repeat)).items() }
    long_repeat_dict = { v:k for k,v in dict(enumerate(long_repeat)).items() }

    if not os.path.exists(csv_fn) or True :
    
        # load raw tif files
        auto_files = [ fn for fn in glob(f'{auto_dir}/**/*TIF') ] #if not 'UB' in fn and not '#00' in fn ]
        short_repeat_size = len(short_repeat_dict) + 2 # +2 because there are two cell body and two myelin stains 
        long_repeat_size = len(long_repeat_dict) + 2

        df = add_autoradiographs_images(auto_files)

        # get the number of sections in a repeat to determine if it is 'short' or 'long'
        df['repeat_count'] = [0]*df.shape[0]
        for repeat, repeat_df in df.groupby(['repeat']): 
            df['repeat_count'].loc[ df['repeat'] == repeat ] = repeat_df['ligand'].count()
        df['repeat_count'] = df['repeat_count'].apply(lambda x: long_repeat_size if x > 24 else short_repeat_size) 

        # the order in which repeats are acquired flips after repeats 37
        # 1, 2, ..., 37, 60, 59, ..., 38 --> 1, 2, ..., 37, 38, ... 59
        
        min_repeat = 37
        max_repeat =  df['repeat'].max()

        fix_repeat_order = lambda repeat: repeat if repeat <= 37 else max_repeat - repeat + min_repeat
        fix_section_order = lambda repeat, section, section_count: section if repeat <= 37 else section_count - section
       
        section = np.array([  long_repeat_dict[ligand] if repeat_count > 30 else short_repeat_dict[ligand] for ligand, repeat_count in zip(df['ligand'],df['repeat_count']) ])
     
        section = np.array([ fix_section_order(r,s,c) for r,s,c in zip(df['repeat'],section,df['repeat_count']) ])
        repeat = df['repeat'].apply( fix_repeat_order )
       
        # 'section' represents the order within a repeat based on whether it's a long or a short repeat
        # 'repeat' represents block of brain tissue where sections were acquired sequentially.
        df['order'] = (repeat-1) * df['repeat_count'] + (repeat-1)* np.rint(1.5/0.02) + section
        
        df['crop'] = [None] * df.shape[0]

        #Add downsample and crop, files
        df['crop'] = df['raw'].apply(lambda fn: '{}/{}'.format(crop_dir,os.path.splitext(os.path.basename(fn))[0]+'_crop.nii.gz') )
        df['aligned']=[False]*df.shape[0]
        df.sort_values(['order'],inplace=True)
       
        '''
        for hemisphere in ['left']:#,'right'] :
            hist_list = glob(f'{hist_dir}/*{hemisphere}*nii.gz') #if not 'UB' in fn and not '#00' in fn ]
            hist_list.sort()

            df_prehist = df.loc[ df['ligand'].apply(lambda x : x in ['mk80', 'uk14', 'mk80_ub', 'uk14_ub'] ) ]
            for i, (row_name,row) in enumerate(df_prehist.iterrows())  :
                new_row = row.copy()
                #brain, repeat, hemisphere = [fn_split[i] for i in [0,1,4]]
                new_row['order'] = row['order'] + 1
                new_row['repeat'] = row['repeat']
                new_row['ligand'] = 'hist_cellbody'
                new_row['crop'] = hist_list[i]
                new_row['raw'] = hist_list[i]
                new_row['hemisphere'] = hemisphere
                df = df.append(new_row)
        '''
        df.sort_values(['order'],inplace=True)
        df.to_csv(csv_fn)
        print(csv_fn)
    else : 
        df = pd.read_csv(csv_fn)

    df = df.loc[df['binding']=='S']
    df['order'] = df['order'].max() - df['order']
    df['slab_order'] = df['order']
    df['global_order']=df['order'] 
    return df


def qc_align(moving_row, fixed_row, init_dir, mv_fn, mv_rsl_fn, qc_fn) :
    title_string = f'mv ({moving_row["order"]},{moving_row["ligand"]}) --> fx ({fixed_row["order"]},{fixed_row["ligand"]})'
    plt.cla()
    plt.clf()
    plt.title(title_string)

    plt.subplot(2,1,1)
    plt.title('Fx Vs Mv')
    plt.imshow(nib.load(fixed_row['init']).get_fdata(),cmap='Greys')
    plt.imshow(nib.load(mv_fn).get_fdata(),alpha=0.5);
    
    plt.subplot(2,1,2)
    plt.title('Fx Vs Mv Rsl')
    print(fixed_row['init'], mv_rsl_fn )
    plt.imshow(nib.load(fixed_row['init']).get_fdata(),cmap='Greys')
    plt.imshow(nib.load(mv_rsl_fn).get_fdata(),alpha=0.5);

    plt.savefig(qc_fn)

def align_sections(df, i_list, init_dir, reference_ligand, direction) :

    # Check if we are doing the initial alignment for the reference ligand
    ligand_list = df['ligand'].unique()
    if reference_ligand in ligand_list and len(ligand_list) == 1:
        reference_align=True
    else :
        reference_align=False

    # Pair down i_list so that it only containes indices for reference ligands
    i_list = [ i for i in i_list if df["ligand"].iloc[i] == reference_ligand ] 

    file_to_align='init'

    #  1     2    3      4      5   
    # .4 -> .8 -> 1.2 -> 2.4 -> 4.8
    # Iterate over reference ligands
    for i in i_list[:-1] :
        fixed_row = df.iloc[i]
        df['aligned'].iloc[i] = True
        
        j = i + direction 
        #Iterate over thee sections between the current reference section until the next reference section
        while df['ligand'].iloc[j] != reference_ligand or reference_align:
            moving_row= df.iloc[j]
             
            outprefix = f'{init_dir}/{moving_row["order"]}_{moving_row["ligand"]}/'
            os.makedirs(outprefix,exist_ok=True)
            mv_rsl_fn = outprefix + "_level-0_Mattes_Rigid.nii.gz"
            qc_fn=f'{init_dir}/{moving_row["order"]}_{moving_row["ligand"]}.png'

            if not os.path.exists(qc_fn) or not os.path.exists(mv_rsl_fn) :
                # ANTs registration 
                print('\t\t\tAligning')
                ANTs(tfm_prefix=outprefix,
                    fixed_fn=fixed_row[file_to_align], moving_fn=moving_row[file_to_align],  moving_rsl_prefix=outprefix, 
                    metrics=['Mattes'], tfm_type=['Rigid'],
                    iterations=['1000x500x250x125'],  shrink_factors=['4x3x2x1'], 
                    smoothing_sigmas=['2.0x1.0x0.5x0'], 
                    init_tfm=None, no_init_tfm=False, dim=2, nbins=32,
                    sampling_method='Regular',sampling=1, verbose=1, generate_masks=False, clobber=1)
                print(f'register {fixed_row[file_to_align]} {mv_rsl_fn}')
                qc_align(moving_row, fixed_row, init_dir, moving_row[file_to_align], mv_rsl_fn, qc_fn)
            df['init'].iloc[j] = mv_rsl_fn
            df['aligned'].iloc[j] = True 
            df['tfm'].iloc[j] = outprefix + '_level-0_Mattes_Rigid_Composite.h5'

            j += direction
            if reference_align : break

    df['init_tfm'] = df['tfm']
    return df

def concat_section_to_volume(df, affine, volume_fn, file_var='crop'):
    if not os.path.exists(volume_fn) :
        example_fn = df[file_var].iloc[0]
        print(example_fn)
        xdim, zdim = np.rint(np.array(nib.load(example_fn).shape)).astype(int)
        
        order_min = df['order'].min()
        ydim=int(df['order'].max() - order_min + 1)
        volume=np.zeros([xdim,ydim,zdim])

        for index, (i, row) in enumerate(df.iterrows()) :
            fn = row[file_var]
            if os.path.exists(fn) :
                y = int(row['order'] - order_min)
                section = nib.load(fn).get_fdata()
                #section = np.flipud(section)
                #section = gaussian_filter(section, affine[0,0]/(2*0.02))
                #section = resize(section, [xdim,zdim], order=0)
                volume[:,y,:] = section
        
        print('\tWriting to', volume_fn)
        #volume
        nib.Nifti1Image(volume, affine).to_filename(volume_fn)


def align(df, init_dir):
    reference_ligand='flum'
    df['aligned']=[False]*df.shape[0]
    df['init']=df['crop']
    df['tfm']=[None]*df.shape[0]
    aligned_df_list=[]
    ligand_contrast_order = ['cellbody_a', 'cellbody_b', 'flum', 'mk80', 'musc', 'cgp5', 'ampa', 'kain', 'pire', 'damp', 'praz', 'uk14', 'keta', 'sch2', 'dpmg', 'dpat'] #,  'zm24', 'racl', 'oxot', 'epib']
    ligand_contrast_order = [ ligand for ligand in ligand_contrast_order if ligand in np.unique(df['ligand']) ]

    for i, ligand in enumerate(ligand_contrast_order) : 
        ligand_check_fn = f'{init_dir}/{ligand}.csv'
        idx = df['ligand'].apply(lambda x : x in [reference_ligand,ligand])
        df_ligand = df.loc[ idx ]
        mid_section = int(df_ligand.shape[0]/2)
        df.loc[idx] = align_sections(df_ligand, range(mid_section,df_ligand.shape[0]), init_dir, reference_ligand, 1)
        df.loc[idx] = align_sections(df_ligand, range(mid_section, 0, -1), init_dir, reference_ligand, -1)
        aligned_df_list.append( df_ligand.loc[df_ligand['ligand']==ligand])
    
    aligned_df = pd.concat(aligned_df_list)

    return aligned_df

def downsample(df, downsample_dir, resolution_2d):
    for i, (index, row) in enumerate(df.iterrows()):
        crop_fn = row['crop']
        crop_rsl_fn=f'{downsample_dir}/{os.path.basename(crop_fn)}'
        if not os.path.exists(crop_rsl_fn) :
            print(crop_fn)
            prefilter_and_downsample(crop_fn, [resolution_2d]*2, crop_rsl_fn)
        #print('Downsampled:', i, crop_rsl_fn)
        df['crop'].iloc[i] = crop_rsl_fn
        #print(df['crop'].iloc[i])
    return df

def segment(subject_id, df, seg_dir):
    df['seg_fn'] = [''] * df.shape[0]
    for i, (index, row) in enumerate(df.iterrows()):
        crop_fn = row['crop']
        fn = os.path.splitext(row['raw'])[0]+'_seg.nii.gz'
        seg_fn = f'{seg_dir}/{os.path.basename(fn)}'
        df['seg_fn'].iloc[i] = seg_fn
        if not os.path.exists(seg_fn) :
            img = nib.load(crop_fn)
            data = img.get_fdata()
            t = threshold_otsu( data[data>0] )
            data[ data < t ] = 0
            data[ data > 0 ] = 1
            nib.Nifti1Image(data, img.affine).to_filename(seg_fn)

    return df

from scipy.ndimage.morphology import binary_dilation, binary_erosion, binary_closing
def _get_section_intervals(vol, interp_dir):

    section_sums = np.sum(vol, axis=(0,2))
    valid_sections = section_sums > np.min(section_sums)
    plt.subplot(2,1,1); plt.plot(section_sums)
    plt.subplot(2,1,2); plt.plot(valid_sections); 
    plt.savefig(f'{interp_dir}/val_sections_{np.sum(valid_sections)}.png'); plt.clf(); plt.cla()
    labeled_sections, nlabels = label(valid_sections)
    assert nlabels >= 2, 'Error: there must be a gap between thickened sections. Use higher resolution volumes.'

    intervals = [ (np.where(labeled_sections==i)[0][0], np.where(labeled_sections==i)[0][-1]) for i in range(1, nlabels) ]
    assert len(intervals) > 0 , 'Error: no valid intervals found for volume.'  
    return intervals


    
def align_3d(rec_fn, template_fn, out_dir, subject_id, res, syn_only=False, init_tfm=''):
    current_template_fn = f'{out_dir}/'+os.path.basename(re.sub('.nii',f'_{res}mm.nii', template_fn))
    rec_interp_fn = f'{out_dir}/'+os.path.basename(re.sub('.nii',f'_{res}mm.nii', rec_fn))
   
    if not os.path.exists(rec_interp_fn) :
        img = nib.load(rec_fn)
        data = img.get_fdata()
        data = interpolate_missing_sections(data)
        nib.Nifti1Image(data, img.affine).to_filename(rec_interp_fn)

    if not os.path.exists(current_template_fn) :
        prefilter_and_downsample(template_fn, [res]*3, current_template_fn)
    # 0.4 1 
    # 0.8 2
    # 1.6 3
    # 3.2 4
    # 6.4 5

    if not syn_only : tfm_type='affine'
    else : tfm_type = 'SyN'

    prefix=f'{out_dir}/{subject_id}_affine_'
    tfm_fn=f'{prefix}Composite.h5'
    inv_fn=f'{prefix}InverseComposite.h5'
    out_fn=f'{out_dir}/{subject_id}_{tfm_type}.nii.gz'
    inv_fn=f'{out_dir}/{subject_id}_{tfm_type}_inverse.nii.gz'
    f_str='5x4x3x2'
    s_str='2.5x2x1.5x1'
    lin_itr_str='1000x500x250x125'
    
    if init_tfm != '' :
        init_str=f'--initial-moving-transform {init_tfm}'
    else :
        init_str=f'--initial-moving-transform [{current_template_fn},{rec_interp_fn},1]' 

    if not syn_only :
        ants_str = f'antsRegistration -v 1 -a 1 -d 3  {init_str}' 
        rigid_str = f'-t Rigid[.1] -c {lin_itr_str}  -m Mattes[{current_template_fn},{rec_interp_fn},1,30,Regular,1] -s {s_str} -f {f_str}' 
        similarity_str = f'-t Similarity[.1]  -m Mattes[{current_template_fn},{rec_interp_fn},1,20,Regular,1]  -s {s_str} -f {f_str}  -c {lin_itr_str}'  
        str = f'-t Affine[.1] -m Mattes[{current_template_fn},{rec_interp_fn},1,20,Regular,1]  -s {s_str} -f {f_str}  -c {lin_itr_str}'
        out_str = f'-o [{prefix},{out_fn},{inv_fn}] '
        cmd_str = f'{ants_str} {rigid_str} {similarity_str} {str} {out_str}'
    else :
        ants_str = f'antsRegistration -v 1 -a 1 -d 3  {init_str}' 
        syn_str = f'-t SyN[.1] -c {lin_itr_str}  -m CC[{current_template_fn},{rec_interp_fn},1,30,Regular,1] -s {s_str} -f {f_str}' 
        out_str = f'-o [{prefix},{out_fn},{inv_fn}] '
        cmd_str = f'{ants_str} {syn_str} {out_str}'
    
    if not os.path.exists(tfm_fn) :
        print(cmd_str)
        shell(cmd_str)

    return tfm_fn, inv_fn

def multires_align_3d(subject_id, out_dir, volume_interp_fn, template_fn, resolution_list, curr_res, init_affine_fn):
    out_tfm_fn=f'{out_dir}/{subject_id}_align_3d_{curr_res}mm_SyN_Composite.h5' 
    out_inv_fn=f'{out_dir}/{subject_id}_align_3d_{curr_res}mm_SyN_InverseComposite.h5' 
    out_fn=    f'{out_dir}/{subject_id}_align_3d_{curr_res}mm_SyN.nii.gz' 

    resolution_itr = resolution_list.index( curr_res)

    max_downsample_level = get_max_downsample_level(resolution_list, resolution_itr)

    f_str, s_str, lin_itr_str, nl_itr_str = get_alignment_schedule(max_downsample_level, resolution_list, resolution_itr, base_nl_itr=100)
    run_alignment(out_dir, out_tfm_fn, out_inv_fn, out_fn, template_fn, template_fn, volume_interp_fn, s_str, f_str, lin_itr_str, nl_itr_str, curr_res, manual_affine_fn=init_affine_fn )

    return out_tfm_fn, out_inv_fn

def align_2d(df, output_dir, rec_fn, template_rsl_fn, mv_dir, resolution, resolution_itr, use_syn=False):
    df['slab_order'] = df['order']

    df = receptor_2d_alignment( df, rec_fn, template_rsl_fn, mv_dir, output_dir, resolution, resolution_itr,use_syn=use_syn) 

    return df

def reconstruct_ligands(ligand_dir, subject_id, curr_res, aligned_df,template_fn, final_3d_fn, volume_align_2d_fn):
    img = nib.load(volume_align_2d_fn) 
    data = img.get_fdata()
    for ligand, ligand_df in aligned_df.groupby(['ligand']) : 
        if ligand != 'flum' : continue

        ligand_no_interp_fn=f'{ligand_dir}/{subject_id}_{ligand}_space-nat_no-interp_{curr_res}mm.nii.gz'
        ligand_nat_fn=f'{ligand_dir}/{subject_id}_{ligand}_space-nat_{curr_res}mm.nii.gz'
        ligand_stereo_fn=f'{ligand_dir}/{subject_id}_{ligand}_space-stereo_{curr_res}mm.nii.gz'
        ligand_vol=np.zeros(data.shape) 

        for i, row in ligand_df.loc[ ligand_df['ligand'] == ligand ].iterrows() : 
            y=int(row['slab_order'])
            ligand_vol[:,y,:] = data[:,y,:]

        nib.Nifti1Image(ligand_vol, img.affine).to_filename(ligand_no_interp_fn)

        if not os.path.exists(ligand_nat_fn) :
            print(f'Interpolate Missing Sections: {ligand}')
            ligand_vol = interpolate_missing_sections(ligand_vol)
            nib.Nifti1Image(ligand_vol, img.affine).to_filename(ligand_nat_fn)

        shell(f'antsApplyTransforms -i {ligand_nat_fn} -t {final_3d_fn} -r {template_fn} -o {ligand_stereo_fn}')

def reconstruct(subject_id, auto_dir, template_fn, points_fn, out_dir='macaque/output/', ligands_to_exclude=[]):
    subject_dir=f'{out_dir}/{subject_id}/' 
    crop_dir = f'{out_dir}/{subject_id}/crop/'
    init_dir = f'{out_dir}/{subject_id}/init_align/'
    downsample_dir = f'{out_dir}/{subject_id}/downsample'
    init_3d_dir = f'{out_dir}/{subject_id}/init_align_3d/'
    ligand_dir = f'{subject_dir}/ligand/'
    csv_fn = f'{out_dir}/{subject_id}/{subject_id}.csv'

    for dirname in [subject_dir, crop_dir, init_dir, downsample_dir, init_3d_dir, ligand_dir] : os.makedirs(dirname, exist_ok=True)

    lowres=0.4
    affine=np.array([[lowres,0,0,0],[0,0.02,0,0.0],[0,0,lowres,0],[0,0,0,1]])

    df = create_section_dataframe(auto_dir, crop_dir, csv_fn )

    # Output files
    affine_fn = f'{init_3d_dir}/{subject_id}_affine.mat'
    volume_fn = f'macaque/output/{subject_id}/{subject_id}_volume.nii.gz'
    volume_init_fn = f'macaque/output/{subject_id}/{subject_id}_init-align_volume.nii.gz'
    volume_seg_fn = f'{subject_dir}/{subject_id}_segment_volume.nii.gz'
    volume_seg_iso_fn = f'{subject_dir}/{subject_id}_segment_iso_volume.nii.gz'

    ### 1. Section Ordering
    print('1. Section Ordering')
    if ligands_to_exclude != [] :
        df = df.loc[df['ligand'].apply(lambda x : not x in ligands_to_exclude  ) ] 
     
    ### 2. Crop
    print('2. Cropping')
    crop_to_do = [ (y, raw_fn, crop_fn) for y, raw_fn, crop_fn in zip(df['repeat'], df['raw'], df['crop']) if not os.path.exists(crop_fn) ]
    num_cores = min(1, multiprocessing.cpu_count() )
    Parallel(n_jobs=num_cores)(delayed(crop_image)(crop_dir, y, fn, crop_fn) for y, fn, crop_fn in  crop_to_do) 
    df['crop_raw_fn'] = df['crop']

    ### 3. Segment
    df = segment(subject_id, df, crop_dir)

    ### 3. Resample images
    df = downsample(df, downsample_dir, lowres)

    ### 4. Align
    print('3. Init Alignment')
    if not os.path.exists(volume_init_fn) : concat_section_to_volume(df, affine, volume_fn, file_var='crop' )
    aligned_df = align(df, init_dir )
    concat_section_to_volume(aligned_df, affine, volume_init_fn, file_var='init' )
    
    points2tfm(points_fn, affine_fn, template_fn, volume_init_fn,  ndim=3, transform_type="Affine", invert=True, clobber=True)
    
    tfm = ants.read_transform(affine_fn)
    print(tfm.fixed_parameters)
    print(tfm.parameters)
    shell(f'antsApplyTransforms -d 3 -v 1 -i {volume_init_fn} -t [{affine_fn},0] -r {template_fn} -o temp.nii.gz')
    print('1 -->', np.max(nib.load('temp.nii.gz').get_fdata()))
    exit(0)
    shell(f'antsApplyTransforms -d 3 -v 1 -i {volume_init_fn} -t [{affine_fn},0] -r {template_fn} -o temp.nii.gz'); 
    print('2 -->', np.max( nib.load('temp.nii.gz').get_fdata()))

    print('3 -->', np.max( ants.apply_transforms(ants.image_read(template_fn), 
                                                ants.image_read(volume_init_fn),
                                                [affine_fn] ).numpy() ) )
    print('4 -->', np.max(ants.apply_transforms(ants.image_read(template_fn), 
                                                ants.image_read(volume_init_fn),
                                                [ affine_fn ], invert_list=[True] ).numpy() ) )

    print('5 -->', np.max(ants.apply_transforms(ants.image_read(volume_init_fn),
                                                ants.image_read(template_fn),
                                                [ affine_fn ], invert_list=[False] ).numpy() ) )
    print('6 -->', np.max(ants.apply_transforms(ants.image_read(volume_init_fn),
                                                ants.image_read(template_fn),
                                                [ affine_fn ], invert_list=[True] ).numpy() ) )
    print('Init 3D')
    init_3d_fn, init_3d_inv_fn = align_3d(volume_init_fn, template_fn, init_3d_dir, subject_id, lowres, init_tfm=affine_fn)
    exit(0)

    resolution_list=[4,3,2,1] #[8,6,4,3,2,1]
    for itr, curr_res in enumerate(resolution_list):
        # Define some variables
        resolution_2d = curr_res
        # in human reconstruction, the 3d resolution isn't necessarily the same as 2d because very high 3d alignment
        # resolution eats too much RAM. 
        resolution_3d = curr_res 

        # Define directories
        seg_dir = f'{out_dir}/{subject_id}/{curr_res}mm/segment/'
        seg_2d_dir = f'{seg_dir}/2d/'
        align_3d_dir=f'{out_dir}/{subject_id}/{curr_res}mm/align_3d/'
        align_2d_dir=f'{out_dir}/{subject_id}/{curr_res}mm/align_2d/'
        
        # Create directories
        multires_dirs = [seg_dir, seg_2d_dir, align_2d_dir, align_3d_dir]
        for dirname in multires_dirs : os.makedirs(dirname, exist_ok=True)
       
        # Define volume filenames
        volume_seg_fn = f'{seg_dir}/{subject_id}_segment_volume.nii.gz'
        volume_align_2d_fn = f'{align_2d_dir}/{subject_id}_align_2d_space-nat.nii.gz'
        current_template_fn = f'{align_2d_dir}/'+os.path.basename(re.sub('.nii',f'_{curr_res}mm.nii', template_fn))
        template_rec_space_fn = f'{align_2d_dir}/'+os.path.basename(re.sub('.nii',f'_{curr_res}mm_space-nat.nii', template_fn))

        ### 5. Segment
        resample_transform_segmented_images(aligned_df, resolution_2d, resolution_3d, seg_2d_dir)
        classifyReceptorSlices(aligned_df, volume_init_fn, seg_2d_dir, seg_dir, volume_seg_fn, resolution=resolution_3d)

        ### 6. 3D alignment
        if not os.path.exists(current_template_fn):
            prefilter_and_downsample(template_fn, [resolution_3d]*3, current_template_fn)

        tfm_3d_fn, tfm_3d_inv_fn = multires_align_3d(subject_id, align_3d_dir, volume_seg_fn, current_template_fn, resolution_list, curr_res, init_3d_fn)
        
        ### 7. 2d alignement
        if not os.path.exists(template_rec_space_fn) : 
            resample_to_autoradiograph_sections(subject_id, '', '', float(curr_res), current_template_fn, volume_seg_fn, tfm_3d_inv_fn, template_rec_space_fn)
        
        create_2d_sections(aligned_df, template_rec_space_fn, float(curr_res), align_2d_dir )
        
        aligned_df = align_2d(aligned_df, align_2d_dir, volume_init_fn, template_rec_space_fn, seg_2d_dir, resolution_2d, itr, use_syn=True)
        if not os.path.exists(volume_align_2d_fn): 
            aligned_df = concatenate_sections_to_volume( aligned_df, template_rec_space_fn, align_2d_dir, volume_align_2d_fn)


    ### 8. Perform a final alignment to the template
    final_3d_dir = f'{subject_dir}/final_3d_dir/'
    os.makedirs(final_3d_dir, exist_ok=True)
    interp_align_2d_fn=f'{final_3d_dir}/{subject_id}_interp_{curr_res}mm.nii.gz'
    interp_align_2d_template_fn=f'{final_3d_dir}/{subject_id}_interp_space-template_{curr_res}mm.nii.gz'

    img = nib.load(volume_align_2d_fn)
    data = img.get_fdata()
    nib.Nifti1Image( interpolate_missing_sections( data ), img.affine).to_filename(interp_align_2d_fn)

    final_3d_fn, final_3d_inv_fn = align_3d(interp_align_2d_fn, template_fn, final_3d_dir, subject_id, curr_res, syn_only=True, init_tfm = tfm_3d_fn)

    reconstruct_ligands(ligand_dir, subject_id, resolution_list[-1], aligned_df, template_fn, final_3d_fn, volume_align_2d_fn)


    print('Done')

if __name__ == '__main__':
    launch()
