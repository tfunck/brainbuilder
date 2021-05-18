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

    affine=np.array([[0.02,0,0,0],[0,0.02,0,y*0.02],[0,0,0.02,0],[0,0,0,1]])
    ligand = os.path.basename(fn).split('#')[4]

    if y > 37 : img = np.fliplr(img)

    if ligand in ['oxot', 'epib', 'zm24', 'racl'] :
        nib.Nifti1Image(img.astype(np.float32), affine  ).to_filename(crop_fn)
        return 0


    img_blur = gaussian_filter(img.astype(float), 0.2/0.02)
    t = threshold_otsu(img_blur[30:,:])
   

    seg=np.zeros_like(img)
    seg[ img_blur > t ] = 1
    

    x , y = center_of_mass(seg)
    seg_filled = fill_regions(seg)
    seg_filled_labels, uniq_labels = label(seg_filled)
    seg_filled_labels = seg_filled_labels.astype(int)
    min_dist=np.product(img.shape)
    min_label=-1

    for l in np.unique(seg_filled_labels)[1:] :
        temp = np.zeros_like(img).astype(int)
        temp[seg_filled_labels == l] = 1
        assert np.sum(seg_filled_labels == l) > 0 , f'Error: {l} not in filled labels: {np.unique(seg_filled_labels)}'
        x0, y0 = center_of_mass(temp)
        
        dist = np.sqrt(np.power(x-x0,2) + np.power(y-y0,2))
        min_label = l if dist < min_dist else min_label
        min_dist = dist if dist < min_dist else min_dist

    assert np.sum( seg_filled_labels ) > 0, 'Error : seg_filled_labels is empty'
    print(np.unique(seg_filled_labels), min_dist)

    mask = np.zeros_like(img)
    mask[ min_label == seg_filled_labels ] = 1
    assert np.sum(mask) > 0 , 'Error: maskped image is empty'
    mask = binary_dilation(mask,iterations=1).astype(int)


    cropped_img = mask*img
    if np.sum(cropped_img) > 0 :
        nib.Nifti1Image(cropped_img.astype(np.float32), affine  ).to_filename(crop_fn)
    else :
        nib.Nifti1Image(img.astype(np.float32), affine  ).to_filename(crop_fn)

    nib.Nifti1Image(mask.astype(np.uint16), affine  ).to_filename(mask_fn)
    plt.cla()
    plt.clf()
    plt.subplot(2,2,1); 
    plt.imshow(img)
    plt.subplot(2,2,2); 
    plt.imshow(seg)
    plt.subplot(2,2,3); 
    plt.imshow(seg_filled_labels)
    plt.subplot(2,2,4)
    plt.imshow(mask)
    plt.savefig(qc_fn)

def create_section_dataframe(raw_files, ligand_repeat, crop_dir, csv_fn, n_ligands=15):
    if not os.path.exists(csv_fn) :
        df_list = []
        for fn in raw_files :
            fn_split = re.sub('.TIF', '', fn).split('#')
            brain, hemisphere, ligand, section = [fn_split[i] for i in [2,3,4,6]]
            df_list.append(pd.DataFrame({'raw':[fn], 'brain':[brain], 'hemisphere':[hemisphere], 'ligand':[ligand], 'section':[int(section)] }))
        
        df = pd.concat(df_list)
        min_section=37
        max_section=df['section'].max()
        
        get_section_order = lambda section: section if section <= 37 else max_section - section + min_section
        get_repeat_order = lambda repeat, section: repeat if section <= 37 else n_ligands - repeat
       
        repeat = df['ligand'].apply(lambda x : ligand_repeat[x])
        repeat = [ get_repeat_order(r,s) for s, r in zip(df['section'],repeat) ]
        section = df['section'].apply( get_section_order )
        
        df['order'] = repeat + (section-1) * n_ligands 
         
        df['crop'] = [None] * df.shape[0]
        df['downsample'] = [None] * df.shape[0]

        #Add downsample and crop, files
        df['crop'] = df['raw'].apply(lambda fn: '{}/{}'.format(crop_dir,os.path.splitext(os.path.basename(fn))[0]+'_crop.nii.gz') )
        df['aligned']=[False]*df.shape[0]
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
                    iterations=['10x5x5'],  shrink_factors=['12x8x6'], smoothing_sigmas=['6x4x3'], 
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

    for i, row in df.iterrows() :
        print(row['crop'])
        fn = row['crop']
        
        if os.path.exists(fn) :
            y = int(row['order'] - order_min)
            section = nib.load(fn).get_fdata()
            section = np.fliplr(np.flipud(section))
            section = gaussian_filter(section, 0.2/(2*0.02))
            section = resize(section, [xdim,zdim])
            volume[:,y,:] = section
    
    affine=np.array([[0.2,0,0,0],[0,0.02,0,0.0],[0,0,0.2,0],[0,0,0,1]])
    print('\tWriting to', volume_fn)
    nib.Nifti1Image(volume, affine).to_filename(volume_fn)


def launch():
    input_dir = 'macaque/img_lin/'
    crop_dir = 'macaque/crop/'
    init_dir = 'macaque/init_align/'
    csv_fn = 'macaque/sections.csv'
    volume_fn = 'macaque/volume.nii.gz'

    ligand_order = ['ampa', 'kain', 'mk80', 'musc', 'cgp5', 'flum', 'pire', 'oxot', 'damp', 'epib', 'praz', 'uk14','dpat', 'keta', 'sch2', 'racl', 'dpmg', 'zm24']
    ligand_contrast_order = ['flum', 'mk80', 'musc', 'cgp5', 'ampa', 'kain', 'pire', 'damp', 'praz', 'uk14', 'keta', 'sch2', 'dpmg',   'dpat',  'zm24', 'racl', 'oxot', 'epib']
    ligand_repeat = { v:k for k,v in dict(enumerate(ligand_order)).items() }
    n_ligands = len(ligand_repeat.keys())
    os.makedirs(crop_dir,exist_ok=True)
    os.makedirs(init_dir,exist_ok=True)

    num_cores = min(1, multiprocessing.cpu_count() )

    raw_files = [ fn for fn in glob(f'{input_dir}/*TIF') if not 'UB' in fn and not '#00' in fn ]

    ### 1. Section Ordering
    print('1. Section Ordering')
    df = create_section_dataframe(raw_files, ligand_repeat, crop_dir, csv_fn, n_ligands=n_ligands)
    df = df.loc[df['ligand'].apply(lambda x : not x in [ 'racl', 'oxot','epib','zm24']  ) ] 
    
    ### 2. Crop
    print('2. Cropping')
    crop_to_do = [ (y, raw_fn, crop_fn) for y, raw_fn, crop_fn in zip(df['section'], df['raw'], df['crop']) if not os.path.exists(crop_fn) ]
    Parallel(n_jobs=num_cores)(delayed(crop_image)(crop_dir, y, fn, crop_fn) for y, fn, crop_fn in  crop_to_do) 
    
    ### 3. Align
    print('3. Aligning')
    reference_ligand='flum'
    df['aligned']=[False]*df.shape[0]

    aligned_df_list=[]
    for i, ligand in enumerate(df['ligand'].unique()) : 
        ligand_check_fn = f'{init_dir}/{ligand}.csv'
        idx = df['ligand'].apply(lambda x : x in [reference_ligand,ligand])
        print(ligand) 
        if not os.path.exists(ligand_check_fn) : 
            df_ligand = df.loc[ idx]
            mid_section = int(df_ligand.shape[0]/2)
            df_ligand = align_sections(df_ligand, range(mid_section,df_ligand.shape[0]), init_dir, reference_ligand, 1)
            df_ligand = align_sections(df_ligand, range(mid_section, 0, -1), init_dir, reference_ligand, -1)
            df_ligand.to_csv(ligand_check_fn)
        else :
            df_ligand = pd.read_csv(ligand_check_fn)
            try : df_ligand.drop(['Unnamed: 0', 'Unnamed 0.1'],inplace=True, axis=1)
            except KeyError : pass
        
        aligned_df_list.append( df_ligand.loc[df_ligand['ligand']==ligand])
    aligned_df = pd.concat(aligned_df_list)
    concat_section_to_volume(aligned_df, volume_fn )

if __name__ == '__main__':
    launch()
