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
import json
from reconstruction.volumetric_interpolation import volumetric_interpolation
from reconstruction.surface_interpolation import surface_interpolation
from utils.utils import prefilter_and_downsample, resample_to_autoradiograph_sections, shell
from reconstruction.align_slab_to_mri import get_alignment_schedule, run_alignment
from utils.utils import get_section_intervals
from reconstruction.nonlinear_2d_alignment import create_2d_sections, receptor_2d_alignment, concatenate_sections_to_volume
from reconstruction.crop import crop
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

def crop_image(crop_dir, y, ligand, fn, crop_fn, pixel_size):
    img = imageio.imread(fn)
    if len(img.shape) == 3 : img = img[:,:,0]
    mask_fn = '{}/{}'.format(crop_dir,os.path.splitext(os.path.basename(fn))[0]+'_mask.nii.gz')
    qc_fn = '{}/qc_{}'.format(crop_dir,os.path.splitext(os.path.basename(fn))[0]+'_qc.png')
    
    affine=np.array([[pixel_size,0,0,0],[0,pixel_size,0,0],[0,0,0.02,y*1.02],[0,0,0,1]])
    

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
    
    nib.Nifti1Image( np.flip(cropped_img, axis=0), affine).to_filename(crop_fn)
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

def add_histology_images(hist_files, ligand = 'cellbody'):

    df_list = []
    for fn in hist_files :
        ext='.nii.gz'
        ext='.png'
        fn_split = re.sub(ext, '', os.path.basename(fn)).split('#')
        #'RH11530#L#AG#25.png'
        
        brain, hemisphere, ligand_label, repeat = [fn_split[i] for i in [2,3,4,6]]
        
        if hemisphere == 'right' : hemisphere = 'R'
        elif hemisphere == 'left' : hemisphere = 'L'
        
        binding = 'hist' 
        
        df_list.append(pd.DataFrame({'raw':[fn], 'brain':[brain], 'hemisphere':[hemisphere], 'ligand':[ligand], 'repeat':[int(repeat)],'binding':[binding] }))

    df = pd.concat(df_list)
    df['repeat']=df['repeat'].astype(int) 
    df = df.sort_values(['repeat'])

    return df


def get_section_for_repeated(repeat_count, first_in_repeat, short_section, long_section):
    if repeat_count > 30 : #if repeat_count > 30 then this is a long repeat section
        if first_in_repeat :
            section = long_section[0] - 1 #7-1
            first_section_in_repeat = False
        else :
            section = long_section[1] - 1 #27-1
            first_in_repeat=True
    else : # if repeat_count <= 30 then this is a short repeat section
        if first_in_repeat :
            section = short_section[0] - 1 #4-1
            first_section_in_repeat = False
        else :
            section = short_section[1] #15-1
            first_in_repeat=True
    return section, first_in_repeat

def get_section_numbering(df,short_repeat_dict,long_repeat_dict, short_repeat, long_repeat): 
    first_cellbody_in_repeat=True
    first_myelin_in_repeat=True
    section_list = []

    cellbody_short_repeats = get_index_numbers('cellbody', short_repeat)
    cellbody_long_repeats = get_index_numbers('cellbody', long_repeat)
    
    myelin_short_repeats = get_index_numbers('myelin', short_repeat)
    myelin_long_repeats = get_index_numbers('myelin', long_repeat)

    for ligand, repeat_count in zip(df['ligand'],df['repeat_count']) :
        if ligand == 'cellbody' :
            section, first_cellbody_in_repeat = get_section_for_repeated(repeat_count, first_cellbody_in_repeat, cellbody_short_repeats, cellbody_long_repeats)
        if ligand == 'myelin' :
            section, first_myelin_in_repeat = get_section_for_repeated(repeat_count, first_myelin_in_repeat, myelin_short_repeats, myelin_long_repeats)

        #TODO add same as above for myelin stains
        else : # ligand is not cellbody, i.e. is an autoradiograph
            if repeat_count > 30 : #if repeat_count > 30 then this is a long repeat section
                section = long_repeat_dict[ligand]
            else :
                section = short_repeat_dict[ligand]

        section_list.append(section)

    return np.array(section_list)

def get_index_numbers(element, ll):
    return [index for index, value in enumerate(ll) if value == element]


def get_cmax(fn) :
    fn_list = glob(f'trans_tabs/*/{fn}*')

    try :
        fn = fn_list[0]
    except IndexError :
        print('Warning: could not find .grt ','trans_tabs/*/{}'.format(fn) )
        return -1
    
    with open(fn,'r') as F :
        for cbe in F.readlines() :
            if 'cmax' in cbe.lower() :
                return float(cbe.rstrip().split(' ')[-1])
            
    print('Error Cmax not found in ', fn)
    exit(1)

def add_conversion_factor(df):
    sakdl_fn_list=glob('section_numbers/*sakdl.txt')
    sakdl_df = pd.concat([pd.read_csv(fn) for fn in sakdl_fn_list ])
    sakdl_df['image'] = sakdl_df['image'].apply(lambda f: str(f).lower())
    
    df['conversion_factor']=[1]*df.shape[0]
    df.reset_index(inplace=True)

    for i, row in df.iterrows():
        raw = row['raw']
        base_filename = os.path.splitext(os.path.basename(raw))[0][0:-2]
        base_filename = base_filename.lower()
         
        info_row = sakdl_df.loc[sakdl_df['image'] == base_filename]
        if info_row.shape[0] == 0 :
            print('Warning: skipping', base_filename, 'could not find entry in sakdl file')
            continue 

        Sa = float(info_row['sa'].values[0])
        Kd = float(info_row['kd'].values[0])
        L = float(info_row['l'].values[0])
        cmax_fn = info_row['film'].values[0]
        Cmax = get_cmax(cmax_fn) 
        if Cmax == - 1 : 
            print('Warning: skipping', base_filename, '. Could not find grt file')
            continue #Cmax == -1 means that the .grt file containing cmax was not found

        conversion_factor = Cmax / (255 * Sa) * (Kd + L) / L
        df['conversion_factor'].iloc[i] = conversion_factor

    return df



def create_section_dataframe(auto_dir, crop_dir, csv_fn, template_fn ):
    
    # Define the order of ligands in "short" repeats
    # cell body
    # DEBUG: 11531 only has a single cellbody that comes at position 27, will have to remove later
    short_repeat = ['ampa', 'kain', 'mk80', 'cellbody', 'musc', 'cgp5', 'flum', 'pire', 'myelin', 'oxot', 'damp', 'epib', 'praz', 'uk14','cellbody', 'dpat', 'keta', 'sch2', 'racl', 'myelin', 'dpmg', 'zm24']
    #short_repeat = ['ampa', 'kain', 'mk80', 'cellbody_a', 'musc', 'cgp5', 'flum', 'pire', 'myelin', 'oxot', 'damp', 'epib', 'praz', 'uk14','cellbody_b', 'dpat', 'keta', 'sch2', 'racl', 'myelin', 'dpmg', 'zm24']

    # Define the order of ligands in "long" repeats
    long_repeat = ['ampa', 'ampa_ub', 'kain', 'kain_ub', 'mk80', 'mk80_ub', 'cellbody', 'musc', 'musc_ub', 'cgp5', 'cgp5_ub','flum', 'flum_ub', 'pire', 'pire_ub', 'myelin', 'oxot', 'oxot_ub', 'damp', 'damp_ub', 'epib', 'epib_ub', 'praz', 'praz_ub', 'uk14', 'uk14_ub', 'cellbody', 'dpat', 'dpat_ub', 'keta', 'keta_ub', 'sch2', 'sch2_ub', 'racl', 'racl_ub', 'myelin', 'dpmg', 'dpmg_ub', 'zm24', 'zm24_ub']
    #long_repeat = ['ampa', 'ampa_ub', 'kain', 'kain_ub', 'mk80', 'mk80_ub','cellbody_a', 'musc', 'musc_ub', 'cgp5', 'cgp5_ub','flum', 'flum_ub', 'pire', 'pire_ub', 'myelin', 'oxot', 'oxot_ub', 'damp', 'damp_ub', 'epib', 'epib_ub', 'praz', 'praz_ub', 'uk14', 'uk14_ub', 'cellbody_b', 'dpat', 'dpat_ub', 'keta', 'keta_ub', 'sch2', 'sch2_ub', 'racl', 'racl_ub', 'myelin', 'dpmg', 'dpmg_ub', 'zm24', 'zm24_ub']

    # create dictionaries for short and long repeats with integers associated with each ligand
    short_repeat_dict = { v:k for k,v in enumerate(short_repeat) }
    long_repeat_dict = { v:k for k,v in enumerate(long_repeat) }
    
    if not os.path.exists(csv_fn) or True :
   
        # load raw tif files
        auto_files = [ fn for fn in glob(f'{auto_dir}/img_lin_modified/*TIF') ]  #if not 'UB' in fn and not '#00' in fn ]
        hist_files = [ fn for fn in glob(f'{auto_dir}/img_histology/*AG*.png') ] 
        myelin_files = [ fn for fn in glob(f'{auto_dir}/img_histology/*MS*.png') ] 

        df_auto = add_autoradiographs_images(auto_files)
        df_hist = add_histology_images(hist_files)
        df_ms = add_histology_images(myelin_files,ligand='myelin')
        df = pd.concat([df_auto,df_hist,df_ms])
       
        repeat_gap_n = get_repeat_gap_size(template_fn, df.shape[0], df['repeat'].max(), 0.02)

        short_repeat_size = len(short_repeat)
        long_repeat_size = len(long_repeat) 

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

        section = get_section_numbering(df,short_repeat_dict,long_repeat_dict, short_repeat, long_repeat) 
        
        #section = np.array([  long_repeat_dict[ligand] if repeat_count > 30 else short_repeat_dict[ligand] for ligand, repeat_count in zip(df['ligand'],df['repeat_count']) ])
     
        section = np.array([ fix_section_order(r,s,c) for r,s,c in zip(df['repeat'],section,df['repeat_count']) ])

        repeat = df['repeat'].apply( fix_repeat_order )
       
        # 'section' represents the order within a repeat based on whether it's a long or a short repeat
        # 'repeat' represents block of brain tissue where sections were acquired sequentially.
       
        # repeat
        #   0       1        2
        # repeat count
        #   3       3        3 
        # section-1 
        # 0 1 2   0 1 2   0  1  2
        # order
        # 0,1,2,3,4,5,6,7,8, 9, 10 
        # 6 = 1*3 + 1*1 + 1 
        # 9 = 2*3 + 2*1 + 1
        print('Repeat gap', repeat_gap_n) 
        df['order'] = (repeat-1) * df['repeat_count'] + (repeat-1) * repeat_gap_n + (section-1)
        print('Order max', df['order'].max(), 'df.shape', df.shape[0] ) 
        df['crop'] = [None] * df.shape[0]

        #Add downsample and crop, files
        df['crop'] = df['raw'].apply(lambda fn: '{}/{}'.format(crop_dir,os.path.splitext(os.path.basename(fn))[0]+'_crop.nii.gz') )
        df['aligned']=[False]*df.shape[0]
        df.sort_values(['order'],inplace=True)
      
        print('Receptor volumen length (mm)', (float(df['order'].max()) - float(df['order'].min()))*0.02 )

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

        print('Writing to', csv_fn)

        df = df.loc[ (df['binding']=='S') | (df['binding']=='hist') ]
        df['order'] = df['order'].max() - df['order']
        df['order'] = df['order'] - df['order'].min()
        df['slab_order'] = df['order']
        df['global_order']=df['order'] 
        df['volume_order']=df['order'] 

        df['slab']=[1] * df.shape[0]
        df['slab'].loc[df['repeat'].astype(int) > 37 ] = 2

        df['rotate']=[0] * df.shape[0]

        df['seg_fn'] = [''] * df.shape[0]
        for i, (index, row) in enumerate(df.iterrows()):
            crop_fn = row['raw']
            fn = os.path.splitext(row['raw'])[0]+'_seg.nii.gz'
            seg_fn = f'{crop_dir}/{os.path.basename(fn)}'
            df['seg_fn'].iloc[i]=seg_fn

        df = add_conversion_factor(df)

        df.to_csv(csv_fn)
        print(csv_fn)
    else : 
        df = pd.read_csv(csv_fn)

    

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
    plt.imshow(nib.load(fixed_row['init']).get_fdata(),cmap='Greys')
    plt.imshow(nib.load(mv_rsl_fn).get_fdata(),alpha=0.5);

    plt.savefig(qc_fn)

def align_sections(df, i_list, init_dir, reference_ligand, direction, metric='Mattes') :

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
            mv_rsl_fn = outprefix + f'_level-0_{metric}_Rigid.nii.gz'
            qc_fn=f'{init_dir}/{moving_row["order"]}_{moving_row["ligand"]}.png'

            if not os.path.exists(qc_fn) or not os.path.exists(mv_rsl_fn) :
                # ANTs registration 
                print('\t\t\tAligning')
                ANTs(tfm_prefix=outprefix,
                    fixed_fn=fixed_row[file_to_align], moving_fn=moving_row[file_to_align],  moving_rsl_prefix=outprefix, 
                    metrics=[metric], tfm_type=['Rigid'],
                    iterations=['1000x500x250x125'],
                    shrink_factors=['4x3x2x1'], 
                    smoothing_sigmas=[8/np.pi, 4/np.pi, 2/np.pi, 0 ], #['2.0x1.0x0.5x0'], 
                    init_tfm=None, no_init_tfm=False, dim=2, nbins=32,
                    sampling_method='Regular',sampling=1, verbose=0, generate_masks=False, clobber=1)
                qc_align(moving_row, fixed_row, init_dir, moving_row[file_to_align], mv_rsl_fn, qc_fn)
            df['init'].iloc[j] = mv_rsl_fn
            df['aligned'].iloc[j] = True 
            df['tfm'].iloc[j] = outprefix + f'_level-0_{metric}_Rigid_Composite.h5'

            j += direction
            if reference_align : break

    df['init_tfm'] = df['tfm']
    return df

def concat_section_to_volume(df, affine, volume_fn, file_var='crop'):
    if not os.path.exists(volume_fn) :
        example_fn = df[file_var].iloc[0]
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
                plt.title(fn)
                #plt.imshow(section)
                #print(f'/tmp/{y}.png')
                #plt.savefig(f'/tmp/{y}.png')
                #plt.cla()
                #plt.clf()
        print('\tWriting to', volume_fn)
        print(volume.shape)
        #volum 
        nib.Nifti1Image(volume, affine).to_filename(volume_fn)


def align(df, init_dir, ligand_contrast_order, metric='Mattes'):
    reference_ligand=ligand_contrast_order[0]
    df['aligned']=[False]*df.shape[0]
    df['init']=df['crop']
    df['tfm']=[None]*df.shape[0]
    aligned_df_list=[]
    #ligand_contrast_order = ['flum', 'mk80', 'musc', 'cgp5', 'ampa', 'kain', 'pire', 'damp', 'praz', 'uk14', 'keta', 'sch2', 'dpmg', 'dpat', 'cellbody', 'myelin']
            #,  'zm24', 'racl', 'oxot', 'epib']
    ligand_contrast_order = [ ligand for ligand in ligand_contrast_order if ligand in np.unique(df['ligand']) ]
    df = df.loc[df['ligand'].apply(lambda x: x in ligand_contrast_order)]    
    for i, ligand in enumerate(ligand_contrast_order) : 
        ligand_check_fn = f'{init_dir}/{ligand}.csv'
        idx = df['ligand'].apply(lambda x : x in [reference_ligand,ligand])
        df_ligand = df.loc[ idx ]
        mid_section = int(df_ligand.shape[0]/2)
        df.loc[idx] = align_sections(df_ligand, range(mid_section,df_ligand.shape[0]), init_dir, reference_ligand, 1, metric=metric)
        df.loc[idx] = align_sections(df_ligand, range(mid_section, 0, -1), init_dir, reference_ligand, -1, metric=metric)
        aligned_df_list.append( df_ligand.loc[df_ligand['ligand']==ligand])
    
    aligned_df = pd.concat(aligned_df_list)

    return aligned_df

def downsample(df, downsample_dir, resolution_2d):
    for i, (index, row) in enumerate(df.iterrows()):
        crop_fn = row['crop']
        crop_rsl_fn=f'{downsample_dir}/{os.path.basename(crop_fn)}'
        if not os.path.exists(crop_rsl_fn) :
            prefilter_and_downsample(crop_fn, [resolution_2d]*2, crop_rsl_fn)

        df['crop'].iloc[i] = crop_rsl_fn
        #print(df['crop'].iloc[i])
    return df


from scipy.ndimage.morphology import binary_dilation, binary_erosion, binary_closing


    
def align_3d(rec_fn, template_fn, out_dir, subject_id, res, f_str='5x4x3x2', s_str='2.5x2x1.5x1', lin_itr_str='1000x500x250x125', syn_only=False, init_tfm='', metric='GC'):
    current_template_fn = f'{out_dir}/'+os.path.basename(re.sub('.nii',f'_{res}mm.nii', template_fn))
    rec_interp_fn = f'{out_dir}/'+os.path.basename(re.sub('.nii',f'_{res}mm.nii', rec_fn))
   
    if not os.path.exists(rec_interp_fn) :
        img = nib.load(rec_fn)
        data = img.get_fdata()
        data = interpolate_missing_sections(data)
        nib.Nifti1Image(data, img.affine).to_filename(rec_interp_fn)
        prefilter_and_downsample(rec_interp_fn, [res]*3, rec_interp_fn)

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
    out_inv_fn=f'{out_dir}/{subject_id}_{tfm_type}_inverse.nii.gz'  
    
    
    
    if init_tfm != '' :
        init_str=f'--initial-moving-transform {init_tfm}'
    else :
        init_str=f'--initial-moving-transform [{current_template_fn},{rec_interp_fn},1]' 
    
    if not syn_only :
        ants_str = f'antsRegistration -v 1 -a 1 -d 3  {init_str}' 
        rigid_str = f'-t Rigid[.1] -c {lin_itr_str}  -m {metric}[{current_template_fn},{rec_interp_fn},1,30,Regular,1] -s {s_str} -f {f_str}' 
        similarity_str = f'-t Similarity[.1]  -m {metric}[{current_template_fn},{rec_interp_fn},1,20,Regular,1]  -s {s_str} -f {f_str}  -c {lin_itr_str}'  
        affine_str = f'-t Affine[.1] -m {metric}[{current_template_fn},{rec_interp_fn},1,20,Regular,1]  -s {s_str} -f {f_str}  -c {lin_itr_str}'
        out_str = f'-o [{prefix},{out_fn},{out_inv_fn}] '
        cmd_str = f'{ants_str} {rigid_str} {similarity_str} {affine_str} {out_str}'
    else :
        ants_str = f'antsRegistration -v 1 -a 1 -d 3  {init_str}' 
        syn_str = f'-t SyN[.1] -c {lin_itr_str}  -m CC[{current_template_fn},{rec_interp_fn},1,30,Regular,1] -s {s_str} -f {f_str}' 
        out_str = f'-o [{prefix},{out_fn},{out_inv_fn}] '
        cmd_str = f'{ants_str} {syn_str} {out_str}'
    
    if not os.path.exists(tfm_fn) :
        print(cmd_str)
        shell(cmd_str)

    return tfm_fn, inv_fn

def multires_align_3d(subject_id, out_dir, volume_interp_fn, template_fn, resolution_list, curr_res, init_affine_fn='', metric='GC'):
    out_tfm_fn=f'{out_dir}/{subject_id}_align_3d_{curr_res}mm_SyN_Composite.h5' 
    out_tfm_inv_fn=f'{out_dir}/{subject_id}_align_3d_{curr_res}mm_SyN_InverseComposite.h5' 
    out_inv_fn=f'{out_dir}/{subject_id}_align_3d_{curr_res}mm_SyN_inverse.nii.gz' 
    out_fn=    f'{out_dir}/{subject_id}_align_3d_{curr_res}mm_SyN.nii.gz' 

    resolution_itr = resolution_list.index( curr_res)

    #max_downsample_level = get_max_downsample_level(resolution_list, resolution_itr)
    #= get_alignment_schedule(max_downsample_level, resolution_list, resolution_itr, base_nl_itr=100)

    max_downsample_level, f_str, s_str, lin_itr_str, nl_itr_str = get_alignment_schedule(resolution_list, resolution_itr,base_nl_itr = 100 )
    
    run_alignment(out_dir, out_tfm_fn, out_inv_fn, out_fn, template_fn, template_fn, volume_interp_fn, s_str, f_str, lin_itr_str, nl_itr_str, curr_res, manual_affine_fn=init_affine_fn, metric=metric )

    return out_tfm_fn, out_tfm_inv_fn

def align_2d(df, output_dir, rec_fn, template_rsl_fn, mv_dir, resolution, resolution_itr, resolution_list, file_to_align='seg_fn', use_syn=False):
    df['slab_order'] = df['order']

    df = receptor_2d_alignment( df, rec_fn, template_rsl_fn, mv_dir, output_dir, resolution, resolution_itr, resolution_list, file_to_align=file_to_align, use_syn=use_syn, verbose=False) 
    return df

def get_template_y_scale(template_fn):
    template_img = nib.load(template_fn)
    template_vol = template_img.get_fdata()
    ystep = template_img.affine[1,1]

    return get_volume_y_scale(template_vol, ystep)

def get_receptor_y_scale(file_list, axis=0, ystep=0.02163):
    y_list=[]
    for fn in file_list :
        im = imageio.imread(fn)
        #print('1.', im.shape)
        im = np.sum(im, axis=axis)
        #print('2.', im.shape)
        first, last = np.where(im>0)[0][[0,-1]]
        y_list.append([first,last])

    y_list = np.array(y_list)
    y_scale_mm = ystep * np.mean(y_list[:,1] - y_list[:,0])

    return y_scale_mm

def get_volume_y_scale(template_vol, ystep) :
    template_vol = np.sum(template_vol, axis=0)
    
    intervals = np.array([ (np.where(template_vol[y,:]>0)[0][0],  np.where(template_vol[y,:]>0)[0][-1]) for y in range(template_vol.shape[1])  if np.sum(template_vol[y,:]) > 0 ])
    
    diff = ystep*(intervals[:,1] - intervals[:,0])
    y_scale_mm = np.mean(diff)

    return y_scale_mm

def reconstruct_ligands(ligand_dir, subject_id, curr_res, aligned_df,template_fn, final_3d_fn, volume_align_2d_fn):
    img = nib.load(volume_align_2d_fn) 
    data = img.get_fdata()
    for ligand, ligand_df in aligned_df.groupby(['ligand']) : 
        if ligand != 'dpmg' : continue

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

def get_repeat_gap_size(template_fn, n_sections, n_repeats, ystep):

    img = nib.load(template_fn)
    vol = img.get_fdata()

    vol = np.sum(vol, axis=(0,2))

    start, end = np.where(vol>0)[0][[0,-1]]
    
    template_length = (end - start) * img.affine[1,1]

    section_length = ystep * n_sections
    print(section_length, ystep, n_sections)

    extra_template_length = (template_length - section_length)
    
    print('Template length', template_length, section_length, n_repeats)
    repeat_gap_mm = extra_template_length/(n_repeats + 1)
    repeat_gap_n = np.round(repeat_gap_mm / ystep).astype(int)

    return repeat_gap_n

def get_section_scale_factor(out_dir, template_fn, section_files):
    scale_csv=f'{out_dir}/scale.csv'
    if not os.path.exists(scale_csv):
        template_scale = get_template_y_scale(template_fn)
        section_scale = get_receptor_y_scale(section_files)
        section_scale_factor = template_scale / section_scale
        with open(scale_csv,'w') as F : F.write(f'{section_scale_factor}\n')  
    else :
        with open(scale_csv,'r') as F : 
            for line in F.readlines() :
                section_scale_factor=float(line)

    return section_scale_factor

def get_ligand_contrast_order(df):

    df_list=[]
    for i, ligand_df in df.groupby(['ligand']):
        
        for j, row in ligand_df.iterrows() :
            ar = nib.load(row['raw'].values).dataobj
            i_max = np.max(ar)
            i_min = np.min(ar)
            contrast = (i_max-i_min) / (i_max+i_min)

            df_list.append(  pd.DataFrame({'ligand':[ligand], 'contrast':[contrast]})  )

    df = pd.concat(df_list)
    df_mean = df.groupby(['ligand']).mean()
    print(df_mean)
    

def reconstruct(subject_id, auto_dir, template_fn, scale_factors_json_fn, out_dir, csv_fn, native_pixel_size=0.02163,brain = '11530', hemi='B', pytorch_model='', ligands_to_exclude=[], resolution_list=[4,3,2,1], lowres=0.4, flip_dict={}, rat=False ):
    mask_dir = f'{auto_dir}/mask_dir/'
    subject_dir=f'{out_dir}/{subject_id}/' 
    crop_dir = f'{out_dir}/{subject_id}/crop/'
    init_dir = f'{out_dir}/{subject_id}/init_align/'
    downsample_dir = f'{out_dir}/{subject_id}/downsample'
    init_3d_dir = f'{out_dir}/{subject_id}/init_align_3d/'
    ligand_dir = f'{subject_dir}/ligand/'

    srv_max_resolution_fn=template_fn #might need to be upsampled to maximum resolution
    surf_dir = f'{auto_dir}/surfaces/'
    
    for dirname in [subject_dir, crop_dir, init_dir, downsample_dir, init_3d_dir, ligand_dir] : os.makedirs(dirname, exist_ok=True)

    
    affine=np.array([[lowres,0,0,-90],[0,0.02,0,0.0],[0,0,lowres,-70],[0,0,0,1]])
    
    df = pd.read_csv(csv_fn) 

    # Output files
    affine_fn = f'{init_3d_dir}/{subject_id}_affine.mat'
    volume_fn = f'{out_dir}/{subject_id}/{subject_id}_volume.nii.gz'
    volume_init_fn = f'{out_dir}/{subject_id}/{subject_id}_init-align_volume.nii.gz'
    volume_seg_fn = f'{subject_dir}/{subject_id}_segment_volume.nii.gz'
    volume_seg_iso_fn = f'{subject_dir}/{subject_id}_segment_iso_volume.nii.gz'

    scale_factors_json = json.load(open(scale_factors_json_fn,'r'))[brain][hemi]

    ### 1. Section Ordering
    print('1. Section Ordering')
    if ligands_to_exclude != [] :
        df = df.loc[ df['ligand'].apply(lambda x : not x in ligands_to_exclude  ) ] 
 
    section_scale_factor = get_section_scale_factor(out_dir, template_fn, df['raw'].values)

    pixel_size = section_scale_factor * native_pixel_size  
    
    ### 2. Crop
    print('2. Cropping')
    #crop_to_do = [ (y, raw_fn, crop_fn, ligand) for y, raw_fn, crop_fn, ligand in zip(df['repeat'], df['raw'], df['crop'],df['ligand']) if not os.path.exists(crop_fn) ]
    num_cores = min(1, multiprocessing.cpu_count() )
    
   
    # Crop non-cellbody stains
    crop(crop_dir, mask_dir, df.loc[df['ligand'] != 'cellbody'], scale_factors_json_fn, resolution=[60,45], remote=False, pad=0, clobber=False, create_pseudo_cls=False, brain_str='brain', crop_str='crop', lin_str='raw', flip_axes_dict=flip_dict, pytorch_model=pytorch_model )
    
    # Crop cellbody stains
    crop(crop_dir, mask_dir, df.loc[ df['ligand'] == 'cellbody' ], scale_factors_json_fn, resolution=[91,91], remote=False, pad=0, clobber=False,create_pseudo_cls=False, brain_str='brain', crop_str='crop', lin_str='raw', flip_axes_dict=flip_dict, pytorch_model=pytorch_model)
    df['crop_raw_fn'] = df['crop']

    df = df.loc[ df['crop'].apply(lambda fn : os.path.exists(fn)) ]
    assert df.shape[0] > 0 , 'Error: empty data frame'

    ### 3. Resample images
    df = downsample(df, downsample_dir, lowres)

    ### 4. Align
    print('3. Init Alignment')
    concat_section_to_volume(df, affine, volume_fn, file_var='crop' )

    
    #TODO: ligand_contrast_order = get_ligand_contrast_order()
    ligand_contrast_order = ['flum', 'mk80', 'musc', 'cgp5', 'ampa', 'kain', 'pire', 'damp', 'praz', 'uk14', 'keta', 'sch2', 'dpmg', 'dpat', 'cellbody', 'myelin']
    if rat :
        ligand_contrast_order=['oxot'] #DEBUG just for rat!
    
    aligned_df = align(df, init_dir, ligand_contrast_order )

    concat_section_to_volume(aligned_df, affine, volume_init_fn, file_var='init' )

    #points2tfm(points_fn, affine_fn, template_fn, volume_init_fn,  ndim=3, transform_type="Affine", invert=True, clobber=True)
    
    #shell(f'antsApplyTransforms -d 3 -v 1 -i {volume_init_fn} -t [{affine_fn},0] -r {template_fn} -o temp.nii.gz')
    print('Init 3D', np.power(lowres,5))
    # .1 , .2, .4, .8, 1.6, 
    affine_fn, init_3d_inv_fn = align_3d(volume_init_fn, template_fn, init_3d_dir,
                                        subject_id, lowres*np.power(2,4), 
                                        f_str='5x4x3', s_str='2.5x2x1.5', lin_itr_str='2000x1000x1000', metric='Mattes') #, init_tfm=affine_fn)
    #[8,6,4,3,2,1]
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
        template_rec_space_fn = f'{align_2d_dir}/'+os.path.basename(re.sub('.nii',f'_{curr_res}mm_y-0.02_space-nat.nii', template_fn))
        template_iso_rec_space_fn = f'{align_2d_dir}/'+os.path.basename(re.sub('.nii',f'_{curr_res}mm_space-nat.nii', template_fn))

        ### 5. Segment
        resample_transform_segmented_images(aligned_df, itr, resolution_2d, resolution_3d, seg_2d_dir)
        print(volume_init_fn)
        classifyReceptorSlices(aligned_df, volume_init_fn, seg_2d_dir, seg_dir, volume_seg_fn, resolution_2d, resolution_3d, interpolation='linear', flip_axes=())

        ### 6. 3D alignment
        if not os.path.exists(current_template_fn):
            prefilter_and_downsample(template_fn, [resolution_3d]*3, current_template_fn)

        tfm_3d_fn, tfm_3d_inv_fn = multires_align_3d(subject_id, align_3d_dir, volume_seg_fn, current_template_fn, resolution_list, curr_res, affine_fn, metric='Mattes')
        
        ### 7. 2d alignement
        if not os.path.exists(template_rec_space_fn) : 
            resample_to_autoradiograph_sections(subject_id, '', '', float(curr_res), current_template_fn, volume_seg_fn, tfm_3d_inv_fn, template_iso_rec_space_fn, template_rec_space_fn)

        create_2d_sections(aligned_df, template_rec_space_fn, float(curr_res), align_2d_dir )
        aligned_df = align_2d(aligned_df, align_2d_dir, volume_init_fn, template_rec_space_fn, seg_2d_dir, resolution_2d, itr, resolution_list, use_syn=True)
         
        aligned_df = concatenate_sections_to_volume( aligned_df, template_rec_space_fn, align_2d_dir, volume_align_2d_fn)

    ### 8. Perform a final alignment to the template
    final_3d_dir = f'{subject_dir}/final_3d_dir/'
    os.makedirs(final_3d_dir, exist_ok=True)
    interp_align_2d_fn=f'{final_3d_dir}/{subject_id}_interp_{curr_res}mm.nii.gz'
    interp_align_2d_template_fn=f'{final_3d_dir}/{subject_id}_interp_space-template_{curr_res}mm.nii.gz'

    #reconstruct_ligands(ligand_dir, subject_id, resolution_list[-1], aligned_df, template_fn, tfm_3d_fn, interp_align_2d_fn)

    highest_resolution = resolution_list[-1]
    n_depths=10
    n_vertices = 0 
    
    class Arguments() :
        pass
    args=Arguments()
    args.surf_dir = surf_dir
    args.n_vertices = n_vertices
    args.slabs=['1']
    df_ligand=aligned_df
    df_ligand['slab']=['1'] * df_ligand.shape[0]
    to_reconstruct=np.unique(aligned_df['ligand'])#['cellbody','flum','myelin']
    df_ligand = df_ligand[ df_ligand['ligand'].apply(lambda x: x in to_reconstruct ) ]
    files={ brain:{hemi:{'1':{}} }}
    files[brain][hemi]['1'][highest_resolution] = {
            'nl_3d_tfm_fn' : tfm_3d_fn,
            'nl_3d_tfm_inv_fn' : tfm_3d_inv_fn,
            'nl_2d_vol_fn' : volume_align_2d_fn,
            'srv_space_rec_fn':template_rec_space_fn,
            'srv_iso_space_rec_fn':template_iso_rec_space_fn
        }
    
    gm_label= np.percentile(nib.load(template_rec_space_fn).dataobj, [95])[0]
    
    slab_dict={'1':files[brain][hemi]['1'][highest_resolution]}

    subcortex_mask_fn='templates/MEBRAINS_T1_WM_L.nii.gz'
    for ligand, cur_df_ligand in df_ligand.groupby(["ligand"]):


        surface_interpolation(cur_df_ligand, slab_dict, ligand_dir, brain, hemi, highest_resolution,  template_fn, args.slabs, files[brain][hemi], scale_factors_json, input_surf_dir=surf_dir, n_vertices=n_vertices, upsample_resolution=20, n_depths=n_depths, gm_label=gm_label)

        volumetric_interpolation(brain, hemi, highest_resolution, slab_dict, to_reconstruct, subcortex_mask_fn, ligand_dir,n_depths)
    print('Done')

if __name__ == '__main__':
    launch()
