import os
import numpy as np
import imageio
import matplotlib.pyplot as plt
import shutil
import multiprocessing
import re
import pandas as pd
import nibabel as nib
from skimage.transform import resize
from scipy.ndimage.filters import gaussian_filter
from utils.ANTs import ANTs
from joblib import Parallel, delayed
from scipy.ndimage import binary_dilation
from glob import glob
from preprocessing.preprocessing import fill_regions
from utils.utils import safe_imread
from skimage.filters import threshold_otsu, threshold_li , threshold_sauvola, threshold_yen
from scipy.ndimage import label, center_of_mass

def crop_image(crop_dir, y, fn, crop_fn):
    img = imageio.imread(fn)
    mask_fn = '{}/{}'.format(crop_dir,os.path.splitext(os.path.basename(fn))[0]+'_mask.nii.gz')
    qc_fn = '{}/qc_{}'.format(crop_dir,os.path.splitext(os.path.basename(fn))[0]+'_qc.png')

    affine=np.array([[0.02,0,0,0],[0,0.02,0,0],[0,0,0.02,y*0.02],[0,0,0,1]])
    ligand = os.path.basename(fn).split('#')[4]

    if y > 37 : img = np.fliplr(img)

    if ligand in ['oxot', 'epib', 'zm24', 'racl'] :
        nib.Nifti1Image(img.astype(np.float32), affine  ).to_filename(crop_fn)
        return 0

    img_blur = gaussian_filter(img.astype(float), 5)

    t = threshold_li(img_blur[100:,:])
   
    seg=np.zeros_like(img)
    seg[ img_blur > t ] = 1
   

    border=np.zeros_like(img)
    bs=25
    border[0:102,:] = border[-bs:,:] = 1
    border[:,0:bs] = border[:,-bs:] = 1

    seg_labels, uniq_labels = label(seg)

    cropped_img = np.copy(img)
    for l in range(uniq_labels+1) :
        temp = np.zeros_like(img)
        temp[ seg_labels == l ] = 1
        temp += border
        if np.max(temp) > 1 :
            cropped_img[ seg_labels == l ] = 0 

    #x , y = center_of_mass(seg)
    #seg_filled = fill_regions(seg)
    #seg_filled_labels, uniq_labels = label(seg_filled)
    #seg_filled_labels = seg_filled_labels.astype(int)
    #min_dist=np.product(img.shape)
    #min_label=-1

    #for l in np.unique(seg_filled_labels)[1:] :
    #    temp = np.zeros_like(img).astype(int)
    #    temp[seg_filled_labels == l] = 1
    #    assert np.sum(seg_filled_labels == l) > 0 , f'Error: {l} not in filled labels: {np.unique(seg_filled_labels)}'
    #    x0, y0 = center_of_mass(temp)
        
    #    dist = np.sqrt(np.power(x-x0,2) + np.power(y-y0,2))
    #    min_label = l if dist < min_dist else min_label
    #    min_dist = dist if dist < min_dist else min_dist

    #assert np.sum( seg_filled_labels ) > 0, 'Error : seg_filled_labels is empty'

    #mask = seg #np.zeros_like(img)
    #mask[ min_label == seg_filled_labels ] = 1
    #assert np.sum(mask) > 0 , 'Error: maskped image is empty'
    #mask = binary_dilation(mask,iterations=2).astype(int)


    if np.sum(cropped_img) > 0 :
        nib.Nifti1Image(cropped_img.astype(np.float32), affine  ).to_filename(crop_fn)
    else :
        nib.Nifti1Image(img.astype(np.float32), affine  ).to_filename(crop_fn)

    #nib.Nifti1Image(mask.astype(np.uint16), affine  ).to_filename(mask_fn)
    plt.cla()
    plt.clf()
    plt.subplot(1,3,1); 
    plt.imshow(img)
    plt.subplot(1,3,2); 
    plt.imshow(seg_labels)
    #plt.subplot(2,2,3); 
    #plt.imshow(seg_filled_labels)
    #plt.subplot(1,2,2)
    #plt.imshow(mask)
    plt.subplot(1,3,3)
    plt.imshow(cropped_img)
    plt.savefig(qc_fn)

def create_section_dataframe(raw_files, short_repeat_dict, long_repeat_dict, crop_dir, csv_fn ):
    if not os.path.exists(csv_fn) :
        short_repeat_size = len(short_repeat_dict) + 2 # +2 because there are two cell body and two myelin stains 
        long_repeat_size = len(long_repeat_dict) + 2

        df_list = []
        for fn in raw_files :
            fn_split = re.sub('.TIF', '', fn).split('#')
            brain, hemisphere, ligand, repeat = [fn_split[i] for i in [2,3,4,6]]
            
            binding = 'S' if not 'S' in repeat and not '00' in repeat and not 'UB' in repeat else 'UB'
            if binding == 'UB' : ligand += '_ub'
            
            repeat = re.sub('UB','',repeat)
            repeat = re.sub('00[a-z]','0',repeat)

            df_list.append(pd.DataFrame({'raw':[fn], 'brain':[brain], 'hemisphere':[hemisphere], 'ligand':[ligand], 'repeat':[int(repeat)],'binding':[binding] }))

        df = pd.concat(df_list)
         
        df = df.loc[df['binding'] != 'UB']


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
        df['order'] = (repeat-1) * df['repeat_count'] + (repeat-1)* np.rint(.750/0.02) + section
        #plt.scatter(df['order'].values, [1]*df['order'].shape[0] )
        #plt.show()
        
        df['crop'] = [None] * df.shape[0]

        #Add downsample and crop, files
        df['crop'] = df['raw'].apply(lambda fn: '{}/{}'.format(crop_dir,os.path.splitext(os.path.basename(fn))[0]+'_crop.nii.gz') )
        df['aligned']=[False]*df.shape[0]
        df.sort_values(['order'],inplace=True)

        hist_list = glob('macaque/tif_lowres_split/*left*.nii.gz')
        hist_list.sort()
        df_prehist = df.loc[ df['ligand'].apply(lambda x : x in ['mk80', 'uk14', 'mk80_ub', 'uk14_ub'] ) ]
        for i, (row_name,row) in enumerate(df_prehist.iterrows())  :
            new_row = row.copy()
            new_row['order'] = row['order'] + 1
            new_row['ligand'] = 'hist_cellbody'
            new_row['crop'] = hist_list[i]
            new_row['raw'] = hist_list[i]
            df = df.append(new_row)

        df.sort_values(['order'],inplace=True)
        df.to_csv(csv_fn)
    else : 
        df = pd.read_csv(csv_fn)
    return df


def qc_align(moving_row, fixed_row, init_dir, mv_rsl_fn, qc_fn) :
    title_string = f'mv ({moving_row["order"]},{moving_row["ligand"]}) --> fx ({fixed_row["order"]},{fixed_row["ligand"]})'
    plt.cla()
    plt.clf()
    plt.title(title_string)
    plt.imshow(nib.load(fixed_row['crop']).get_fdata(),cmap='Greys')
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
                    fixed_fn=fixed_row['crop'], moving_fn=moving_row['crop'],  moving_rsl_prefix=outprefix, 
                    metrics=['Mattes'], tfm_type=['Rigid'],
                    iterations=['10x1x1'],  shrink_factors=['12x8x6'], smoothing_sigmas=['6x4x3'], 
                    init_tfm=None, no_init_tfm=False, dim=2,
                    sampling_method='Random', sampling=0.5, verbose=0, generate_masks=False, clobber=False  )
           
                qc_align(moving_row, fixed_row, init_dir, mv_rsl_fn, qc_fn)

            df['crop'].iloc[j] = mv_rsl_fn
            df['aligned'].iloc[j] = True 

            j += direction
            if reference_align : break

    return df

def concat_section_to_volume(df, volume_fn):
    example_fn = df['crop'].iloc[0]
    xdim,zdim=np.rint(np.array(nib.load(example_fn).shape) / 10).astype(int)
    
    order_min = df['order'].min()
    ydim=int(df['order'].max() - order_min + 1)
    volume=np.zeros([xdim,ydim,zdim])
    lowres=0.4
    for i, row in df.iterrows() :
        print(row['crop'])
        fn = row['crop']
        
        if os.path.exists(fn) :
            y = int(row['order'] - order_min)
            section = nib.load(fn).get_fdata()
            section = np.flipud(section)
            section = gaussian_filter(section, lowres/(2*0.02))
            section = resize(section, [xdim,zdim], order=0)
            volume[:,y,:] = section
    
    affine=np.array([[lowres,0,0,0],[0,0.02,0,0.0],[0,0,lowres,0],[0,0,0,1]])
    print('\tWriting to', volume_fn)
    nib.Nifti1Image(volume, affine).to_filename(volume_fn)


def launch():
    input_dir = 'macaque/img_lin/'
    crop_dir = 'macaque/crop/'
    init_dir = 'macaque/init_align/'
    csv_fn = 'macaque/sections.csv'
    volume_fn = 'macaque/volume.nii.gz'
    volume_init_fn = 'macaque/init_volume.nii.gz'

    short_repeat = ['ampa', 'kain', 'mk80', 'hist_cellbody', 'musc', 'cgp5', 'flum', 'pire', 'hist_myelin', 'oxot', 'damp', 'epib', 'praz', 'uk14','hist_cellbody', 'dpat', 'keta', 'sch2', 'racl', 'hist_myelin', 'dpmg', 'zm24']
    long_repeat = ['ampa', 'ampa_ub', 'kain', 'kain_ub', 'mk80', 'mk80_ub','hist_cellbody', 'musc', 'musc_ub', 'cgp5', 'cgp5_ub','flum', 'flum_ub', 'pire', 'pire_ub', 'hist_myelin', 'oxot', 'oxot_ub', 'damp', 'damp_ub', 'epib', 'epib_ub', 'praz', 'praz_ub', 'uk14', 'uk14_ub', 'hist_cellbody', 'dpat', 'dpat_ub', 'keta', 'keta_ub', 'sch2', 'sch2_ub', 'racl', 'racl_ub', 'hist_myelin', 'dpmg', 'dpmg_ub', 'zm24', 'zm24_ub']


    short_repeat_dict = { v:k for k,v in dict(enumerate(short_repeat)).items() }
    long_repeat_dict = { v:k for k,v in dict(enumerate(long_repeat)).items() }


    os.makedirs(crop_dir,exist_ok=True)
    os.makedirs(init_dir,exist_ok=True)

    num_cores = min(1, multiprocessing.cpu_count() )

    raw_files = [ fn for fn in glob(f'{input_dir}/*TIF') ] #if not 'UB' in fn and not '#00' in fn ]

    ### 1. Section Ordering
    print('1. Section Ordering')
    df = create_section_dataframe(raw_files, short_repeat_dict, long_repeat_dict, crop_dir, csv_fn )
    df = df.loc[df['ligand'].apply(lambda x : not x in [ 'racl', 'oxot','epib','zm24']  ) ] 
    df = df.loc[df['binding']=='S']

    ### 2. Crop
    print('2. Cropping')
    crop_to_do = [ (y, raw_fn, crop_fn) for y, raw_fn, crop_fn in zip(df['repeat'], df['raw'], df['crop']) if not os.path.exists(crop_fn) ]
    Parallel(n_jobs=num_cores)(delayed(crop_image)(crop_dir, y, fn, crop_fn) for y, fn, crop_fn in  crop_to_do) 
    exit(0)
    
    ### 3. Align
    print('3. Aligning')
    reference_ligand='flum'
    df['aligned']=[False]*df.shape[0]


    if not os.path.exists(volume_init_fn) : concat_section_to_volume(df, volume_init_fn )


    aligned_df_list=[]
    ligand_contrast_order = ['flum', 'mk80', 'musc', 'cgp5', 'ampa', 'kain', 'pire', 'damp', 'praz', 'uk14', 'keta', 'sch2', 'dpmg', 'dpat', 'cellbody'] #,  'zm24', 'racl', 'oxot', 'epib']
    ligand_contrast_order = [ ligand for ligand in ligand_contrast_order if ligand in np.unique(df['ligand']) ]

    for i, ligand in enumerate(ligand_contrast_order) : 
        ligand_check_fn = f'{init_dir}/{ligand}.csv'
        idx = df['ligand'].apply(lambda x : x in [reference_ligand,ligand])
        #if not os.path.exists(ligand_check_fn) : 
        df_ligand = df.loc[ idx]
        mid_section = int(df_ligand.shape[0]/2)
        df.loc[idx] = align_sections(df_ligand, range(mid_section,df_ligand.shape[0]), init_dir, reference_ligand, 1)
        df.loc[idx] = align_sections(df_ligand, range(mid_section, 0, -1), init_dir, reference_ligand, -1)
        #df_ligand.to_csv(ligand_check_fn)
        #else :
        #    df_ligand = pd.read_csv(ligand_check_fn)
        #    try : df_ligand.drop(['Unnamed: 0'],inplace=True, axis=1)
        #    except KeyError : pass
        #    df_ligand = df_ligand.reindex()
        aligned_df_list.append( df_ligand.loc[df_ligand['ligand']==ligand])
    
    aligned_df = pd.concat(aligned_df_list)
    concat_section_to_volume(aligned_df, volume_fn )

if __name__ == '__main__':
    launch()
