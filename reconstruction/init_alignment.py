import utils.ants_nibabel as nib
import nibabel 
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import re
import imageio
import pandas as pd
import json
import shutil
import sys
import re
from utils.utils import shell, read_points, points2tfm
from glob import glob
from skimage.transform import resize
from utils.utils import set_csv, add_padding
from utils.ANTs import ANTs
from ants import write_transform, read_transform, registration, apply_transforms, image_mutual_information, from_numpy, image_similarity, compose_ants_transforms, get_center_of_mass, image_read
#matplotlib.use('TKAgg')

def load2d(fn):
    ar = nib.load(fn).get_fdata()
    ar = ar.reshape(ar.shape[0],ar.shape[1])
    return ar 

def align_neighbours_to_fixed(i, j_list, df, transforms, iteration, shrink_factor,smooth_sigma, output_dir, tfm_type, desc,target_ligand=None,clobber=False):
    #For neighbours

    # a single section is selected as fixed (ith section), then the subsequent sections are considred moving sections (jth section)
    # and are registered to the fixed section. 

    i_idx = df['slab_order']==i
    fixed_fn=df['crop_fn'].loc[i_idx].values[0]
    for j in j_list :
        j_idx = df['slab_order']==j 
        outprefix ="{}/init_transforms/{}_{}-0/".format(output_dir,  j, tfm_type) 

        moving_rsl_fn = outprefix + "_level-0_Mattes_{}.nii.gz".format( tfm_type)
        tfm_fn = outprefix + "_level-0_Mattes_{}_Composite.h5".format( tfm_type)
        concat_tfm_fn = outprefix + "level-0_Mattes_{}_Composite_Concatenated.h5".format( tfm_type)
        qc_fn = "{}/qc/{}_{}_{}_{}_{}_{}-0.png".format(output_dir, *desc, j,i, tfm_type)
        moving_fn=df['crop_fn'].loc[ j_idx].values[0]

        #calculate rigid transform from moving to fixed images
        if  not os.path.exists(tfm_fn) or not os.path.exists(qc_fn)  :
            print() 
            print('\tFixed:', i, fixed_fn)
            print('\tMoving:',j, moving_fn)
            print('\tTfm:',tfm_fn, os.path.exists(tfm_fn))
            print('\tQC:',qc_fn, os.path.exists(qc_fn))
            print('\tMoving RSL:', moving_rsl_fn)

            if target_ligand != None :
                if df['ligand'].loc[ j_idx ].values[0] != target_ligand : 
                    print('\tSkipping')
                    continue

            os.makedirs(outprefix, exist_ok=True)
            inv_tfm_fn = outprefix + "level-0_Mattes_{}_InverseComposite.h5".format( tfm_type)
            
            ANTs(tfm_prefix=outprefix,
                fixed_fn=fixed_fn, 
                moving_fn=moving_fn, 
                moving_rsl_prefix=outprefix+'tmp', 
                iterations=[iteration], 
                metrics=['Mattes'], 
                tfm_type=[tfm_type], 
                shrink_factors=[shrink_factor],
                smoothing_sigmas=[smooth_sigma], 
                init_tfm=None, no_init_tfm=False, 
                dim=2,
                sampling_method='Random', sampling=0.5, verbose=0, generate_masks=False, clobber=True  )
            
        #concatenate the transformation files that have been applied to the fixed image and the new transform
        #that is being applied to the moving image
        if not os.path.exists(concat_tfm_fn):
            transforms_str='-t {} '.format( ' -t '.join(  transforms[i] + [tfm_fn] )  )
            shell(f'antsApplyTransforms -v 0 -d 2 -i {moving_fn} -r {moving_fn} {transforms_str} -o Linear[{concat_tfm_fn}] ',True,True)

        #apply the concatenated transform to the moving image. this brings it into correct alignment with all of the
        #aligned images.
        if not os.path.exists(moving_rsl_fn):
            shell(f'antsApplyTransforms -v 0 -d 2 -i {moving_fn} -r {fixed_fn} -t {concat_tfm_fn} -o {moving_rsl_fn}',True,True)

        if not os.path.exists(qc_fn) : 
            create_qc_image( load2d(fixed_fn), 
                        load2d(moving_fn), 
                        load2d(moving_rsl_fn),
                        i, j,df['tier'].loc[i_idx], df['tier'].loc[j_idx], df['ligand'].loc[i_idx], df['ligand'].loc[j_idx], qc_fn)


        df['crop_fn_new'].loc[df['slab_order'] == j] = moving_rsl_fn
        df['init_tfm'].loc[df['slab_order'] == j] =  concat_tfm_fn
        df['init_fixed'].loc[df['slab_order'] == j] = fixed_fn
        transforms[ j ] = [concat_tfm_fn] 
    
    return df, transforms
                
def create_qc_image(fixed, moving,rsl, fixed_order, moving_order,tier_fixed,tier_moving,ligand_fixed,ligand_moving,qc_fn):
    plt.subplot(1,2,1)
    plt.title('fixed (gray): {} {} {}'.format(fixed_order, tier_fixed, ligand_fixed))
    plt.imshow(fixed, cmap=plt.cm.gray )
    plt.imshow(moving, alpha=0.35, cmap=plt.cm.hot)
    plt.subplot(1,2,2)
    plt.title('moving (hot): {} {} {}'.format(moving_order, tier_moving, ligand_moving))
    plt.imshow(fixed, cmap=plt.cm.gray)
    plt.imshow(rsl, alpha=0.35, cmap=plt.cm.hot)
    plt.tight_layout()
    plt.savefig( qc_fn )
    plt.clf()

def adjust_alignment(df,  y_idx, mid, transforms, step, output_dir,desc, shrink_factor,  smooth_sigma,iteration,tfm_type,target_ligand=None, target_tier=1, clobber=False):
    '''
    '''
    os.makedirs(output_dir + "/qc/",exist_ok=True)
    i=mid
    j=i
    n=len(y_idx)
    y_idx_tier1 = df['slab_order'].loc[df['tier'] == 1].values.astype(int)
    y_idx_tier1.sort()
    i_max = step if step < 0 else df['slab_order'].values.max() + 1

    # Iterate over the sections along y-axis
    for y in y_idx_tier1[mid::step] : 
        j_list=[]
        for j in range(int(y+step), int(i_max), int(step)) :
            if j in df['slab_order'].loc[ df['tier'] == target_tier ].values.astype(int) :
                j_list.append(int(j))

            if j in y_idx_tier1 :
                break

        df, transforms = align_neighbours_to_fixed(y, j_list, df, transforms, iteration, shrink_factor,smooth_sigma, output_dir,  tfm_type, desc,target_ligand=target_ligand,clobber=clobber)

    #df.to_csv('{}/df_{}-0.csv'.format(output_dir,tfm_type))
    return transforms,df


def apply_transforms_to_sections(df,transforms,output_dir, tfm_type, target_ligand=None, clobber=False) :
    print('Applying Transforms') 
    if not target_ligand == None :
        df = df.loc[ df['ligand'] == target_ligand  ]

    for i,(rowi, row) in enumerate(df.iterrows()) :
        outprefix ="{}/init_transforms/{}_{}-0/".format(output_dir, int(row['slab_order']), tfm_type ) 
        final_rsl_fn= outprefix + "final_level-0_Mattes_{}-0.nii.gz".format( tfm_type)

        if not os.path.exists(final_rsl_fn) or clobber  :
            fixed_tfm_list = transforms[ row['slab_order']]
            if len(fixed_tfm_list) == 0 : continue
            transforms_str=' -t {}'.format( ' -t '.join(fixed_tfm_list) ) 
   
            fixed_fn = row['original_crop_fn']

            shell(f'antsApplyTransforms -v 1 -d 2 -i {fixed_fn} -r {fixed_fn} {transforms_str} -o {final_rsl_fn}',True)
        df['crop_fn_new'].iloc[i] = final_rsl_fn
    return df


def combine_sections_to_vol(df,z_mm,direction,out_fn,target_tier=1):
    example_fn=df["crop_fn"].iloc[0]
    shape = nib.load(example_fn).shape
    affine = nib.load(example_fn).affine
    xstart = affine[0,3]
    zstart = affine[1,3]
    scale_factor = 10 
    xstep = affine[0,0] * scale_factor
    zstep = affine[1,1] * scale_factor

    xmax = int( shape[0] / 10 )
    zmax = int( shape[1] / 10 )
    order_max=df["slab_order"].astype(int).max()
    order_min=df["slab_order"].astype(int).min()  
    slab_ymax=int(order_max+1) #-order_min + 1

    vol = np.zeros( [xmax,slab_ymax,zmax])
    print(np.unique(df['slab_order']).shape[0], df.shape[0])
    df = df.sort_values('slab_order')
    for i, row in df.iterrows():
        if row['tier'] == target_tier :
            y = row['slab_order']
            ar = nib.load(row['crop_fn']).get_fdata()
            ar = ar.reshape(ar.shape[0],ar.shape[1])
            ar = resize(ar,[xmax,zmax])
            print(y, row['crop_fn']) 
            vol[:,int(y),:] = ar
            #vol[:,int(y),:] += int(y) 
            del ar
        
    print("\n\tWriting Volume",out_fn,"\n")
    slab_ymin=-126+df["global_order"].min()*0.02 
    ystep = 0.02 
    affine=np.array([[xstep, 0, 0, xstart],
                    [0, ystep, 0, slab_ymin],
                    [0, 0 , zstep, zstart],
                    [0, 0, 0, 1]])
    affine = np.round(affine,3)
    # flip the volume along the y-axis so that the image is in RAS coordinates because ANTs requires RAS
    #vol = np.flip(vol, axis=1)
    nib.Nifti1Image(vol, affine ).to_filename( out_fn )

def alignment_stage(brain,hemi,slab, df, vol_fn_str, output_dir, scale, transforms,  desc=(0,0,0), target_ligand=None, target_tier=1, ligand_n=0,  clobber=False):
    '''
    Perform alignment of autoradiographs within a slab. Alignment is calculated once from the middle section in the
    posterior direction and a second time from the middle section in the anterior direction.
    '''
    # Set parameters for rigid transform
    tfm_type = 'Rigid'
    shrink_factor = '12' #'12x10x8' #x4x2x1'
    smooth_sigma =  str((2**11)/np.pi) # '6x5x4' #x2x1x0' 
    iterations = '100' # '100x50x25' #x100x50x20'

    csv_fn = vol_fn_str.format(output_dir,*desc,target_ligand,ligand_n,tfm_type+'-'+str(0),'.csv')

    if not os.path.exists(csv_fn) or not os.path.exists(out_fn) or True :
        z_mm = scale[brain][hemi][str(slab)]["size"]
        direction = scale[brain][hemi][str(slab)]["direction"]
        
        df.sort_values(['slab_order'],inplace=True,ascending=False)
        
        y_idx = df['slab_order'].values 
        
        y_idx_tier1 = df['slab_order'].loc[ df['tier'].astype(int) == np.min(df['tier']) ].values
        mid = int(len(y_idx_tier1)/2) 

        df['crop_fn_new'] = df['crop_fn']
        df['init_tfm'] = [None] * df.shape[0]
        df['init_fixed']=[None] * df.shape[0]

        # perform alignment in forward direction from middle section
        transforms, df = adjust_alignment(df, y_idx, mid, transforms, -1, output_dir,desc, shrink_factor, smooth_sigma, iterations, tfm_type, target_ligand=target_ligand, target_tier=target_tier, clobber=clobber)
        # perform alignment in reverse direction from middle section
        transforms, df = adjust_alignment(df, y_idx, mid, transforms, 1, output_dir, desc, shrink_factor, smooth_sigma, iterations, tfm_type, target_ligand=target_ligand,target_tier=target_tier, clobber=clobber)
        # update the crop_fn so it has the new, resampled file names
        df['crop_fn']=df['crop_fn_new']

        out_fn = vol_fn_str.format(output_dir,*desc,target_ligand,ligand_n,tfm_type+'-'+str(0),'nii.gz')
        if  not os.path.exists(out_fn) or clobber  :
            vol = combine_sections_to_vol(df,z_mm, direction, out_fn, target_tier)
    #else :
    #    df = pd.read_csv(csv_fn)

    return df, transforms




def create_manual_2d_df(df, manual_2d_dir):

    manual_points_list = glob(f'{manual_2d_dir}/*points.txt')
    manual_2d_df= pd.DataFrame({ 'brain':[], 'hemisphere':[], 'slab':[], 'ligand':[], 'fixed_image':[], 'moving_image':[], 'fixed_index':[], 'moving_index':[], 'tfm':[] })

    brain = np.unique(df['mri'])[0]
    hemisphere = np.unique(df['hemisphere'])[0]
    slab = np.unique(df['slab'])[0]

    for manual_points_fn in manual_points_list :
        fixed_points, moving_points, fixed_fn, moving_fn = read_points(manual_points_fn)
        
        brain, hemisphere, slab, ligand, moving_index = os.path.basename(manual_points_fn).split('_')[0:5]
        print('fixed_fn',fixed_fn)
        idx = df['crop_fn'].apply(lambda crop_fn: fixed_fn in crop_fn )
        
        if np.sum(idx) > 0 :
            fixed_index = df['slab_order'].loc[idx  ]
            fixed_index = int(fixed_index)

            manual_affine_fn = os.path.splitext(manual_points_fn)[0] + '_affine.mat' 
        
            points2tfm( manual_points_fn, manual_affine_fn, fixed_points, moving_points, ndim=2 )
            
            assert os.path.exists(manual_affine_fn), f'Error : {manual_affine_fn} does not exist'

            row = pd.DataFrame({'brain':[brain], 'hemisphere':[hemisphere], 'slab':[slab], 'ligand':[ligand],
                                'fixed_image':[fixed_fn], 'moving_image':[moving_fn], 
                                'fixed_index':[int(float(fixed_index))],
                                'moving_index':[int(float(moving_index))], 'tfm':[manual_affine_fn] })

            manual_2d_df = manual_2d_df.append(row)
    
    return manual_2d_df

def concat_manual_tfm(df, manual_2d_df, manual_idx,y, row, original_crop_fn, final_tfm_fn  ):
    fixed_y = manual_2d_df['fixed_index'].loc[ manual_idx ].values[0].astype(int)
    print('target ligand:', manual_2d_df['ligand'].loc[ manual_idx ])

    manual_tfm_fn = manual_2d_df['tfm'].loc[ manual_idx ].values[0] 
   
    fixed_tfm_fn = df['init_tfm'].loc[ fixed_y == df['slab_order'].values.astype(int) ].values[0] 
    
    print('\tmoving', y,'fixed', fixed_y, final_tfm_fn)
    manual_tfm_ants = read_transform(manual_tfm_fn)
    print(fixed_tfm_fn)
    original_crop_fn = row['original_crop_fn'].values[0]

    if not pd.isnull(fixed_tfm_fn) :
        # Manual points applied after fixed transform
        print('combining manual and fixed tfm')
        fixed_tfm_ants = read_transform(fixed_tfm_fn)
        tfm_concat = compose_ants_transforms( [ manual_tfm_ants, 
                                                read_transform(fixed_tfm_fn) ] )
        #write_transform( tfm_concat , final_tfm_fn)
        #
        shell(f'antsApplyTransforms -v 1 -d 2 -r {original_crop_fn} -i {original_crop_fn}  -t {fixed_tfm_fn}  -t {manual_tfm_fn}  -o Linear[{final_tfm_fn}]')
    else :
        # Manual points but no fixed transform
        print('copy manual tfm to final')
        final_tfm_fn=os.path.splitext(final_tfm_fn)[0]+'.mat'
        shutil.copy(manual_tfm_fn, final_tfm_fn)

    shell(f'antsApplyTransforms -v 1 -d 2 -r {original_crop_fn} -i {original_crop_fn} -t {final_tfm_fn} -o {final_section_fn}')

def create_final_outputs(final_tfm_dir, df, manual_2d_df, step):
    y_idx_tier = df['slab_order'].values.astype(int)
    y_idx_tier.sort()
    
    mid = int(len(y_idx_tier)/2) 
    if step < 0 : mid -= 1
    i_max = step if step < 0 else df['slab_order'].values.max() + 1
    
    #                     c
    #            <--------------
    #                    b 
    #            <---------
    #    l  l    |  a l b1 l   l
    #    l  l -> | <- l <- l   l
    #    l  l    |    l    l   l
    #   s0  s1   s2   s3   s4  s5              
    #   a ( b1 (s2) )  --> s2
    #
    for i, y in enumerate(y_idx_tier[mid::step]) : 
        row = df.loc[ y == df['slab_order'] ]

        final_tfm_fn = "{}/{}_final_Rigid.h5".format(final_tfm_dir, int(row['slab_order'].values[0]) )
        final_section_fn = "{}/{}{}".format(final_tfm_dir, int(row['slab_order'].values[0]), os.path.basename(row['crop_fn'].values[0]) )

        idx = df['slab_order'].values == row['slab_order'].values
        if not os.path.exists(final_tfm_fn) or not os.path.exists(final_section_fn) :
            
            original_crop_fn = row['original_crop_fn']
            
            moving_index = row['slab_order'].values[0].astype(int)
            manual_idx = manual_2d_df['moving_index'].apply( lambda x : x == moving_index )
            print(y)
            #if np.sum(manual_idx) > 0 :
            #    concat_manual_tfm(df, manual_2d_df, manual_idx, row, y, original_crop_fn, final_tfm_fn  )
                
                #vol = apply_transforms(original_crop_fn,original_crop_fn, [final_tfm_fn])
                #nib.Nifti1Image(vol, nib.load(original_crop_fn).affine ).to_filename(final_section_fn)

            if  type(row['init_tfm'].values[0]) == str  :
                # standard rigid transformation for moving image
                shutil.copy(row['init_tfm'].values[0], final_tfm_fn)

                if not os.path.exists(final_section_fn) :
                    os.symlink(row['crop_fn'].values[0],final_section_fn)
            else :
                if not os.path.exists(final_section_fn) :
                    os.symlink(row['crop_fn'].values[0], final_section_fn)
                final_tfm_fn = np.nan
                
            df['crop_fn'].loc[idx] = final_section_fn 
            df['init_tfm'].loc[ idx ] = final_tfm_fn
    return df

def receptorRegister(brain,hemi,slab, init_align_fn, init_tfm_csv, output_dir, manual_2d_dir, df, scale_factors_json="scale_factors.json",  clobber=False):
    
    os.makedirs(output_dir,exist_ok=True)
    manual_2d_df = create_manual_2d_df(df, manual_2d_dir)

    df['original_crop_fn'] = df['crop_fn']

    ligand_intensity_order = ['dpmg','flum','cgp5','damp','praz','afdx','sr95','keta','musc','ly34','pire','uk14','sch2','mk80','dpat','kain','rx82','oxot','ampa','epib']

    with open(scale_factors_json) as f : scale=json.load(f)

    df['tier'] = 1
    
    slab_img_fn_str = '{}/brain-{}_hemi-{}_slab-{}_ligand_{}_{}_init_align_{}.{}'

    n_ligands = len(df['ligand'].unique())
    z_mm = scale[brain][hemi][str(slab)]["size"]
    direction = scale[brain][hemi][str(slab)]["direction"]

    ###########
    # Stage 1 #
    ###########
    print('\t\tAlignment: Stage 1')
    #Perform within ligand alignment
    output_dir_1 = output_dir + os.sep + 'stage_1'

    # Iterate data frames over each ligand
    df_ligand  = df.loc[ df['ligand'] == ligand_intensity_order[0] ] 

    #Init dict with initial transforms
    transforms_1={}
    for i in df_ligand['slab_order'] :  transforms_1[i]=[]

    df_ligand, transforms_1 = alignment_stage(brain,hemi,slab,df_ligand, slab_img_fn_str, output_dir_1, scale, transforms_1,  target_ligand='dpmg',ligand_n=0,target_tier=1, desc=(brain,hemi,slab), clobber=clobber)
   
    # update the master dataframe, df, with new dataframe for the ligand 
    print(df_ligand['init_tfm'].loc[df['ligand']=='dpmg'].values) 
    df.loc[ df['ligand'] == ligand_intensity_order[0]] = df_ligand

    ###########
    # Stage 2 #
    ###########
    #Align ligands to one another based on mean pixel intensity. Start with highest first because these have better contrast
    output_dir_2 = output_dir + os.sep + 'stage_2'
    concat_list=[ df_ligand.loc[df_ligand['ligand'] == ligand_intensity_order[0]]   ]
    print('Stage 2') 
    for i in range(1,n_ligands) :
        current_ligands = [ligand_intensity_order[0], ligand_intensity_order[i]]

        target_ligand = current_ligands[-1]
        idx =  df['ligand'].apply(lambda x : x in current_ligands)
        df_ligand = df.loc[ idx ] 
        df_ligand['tier'].loc[ df_ligand['ligand']==target_ligand ] = 2
        df_ligand['tier'].loc[ df_ligand['ligand']==ligand_intensity_order[0] ] = 1
        #Init dict with initial transforms
        transforms={}
        for i in df_ligand['slab_order'] :  transforms[i]=[]

        df_ligand, transforms = alignment_stage(brain, hemi, slab, df_ligand, slab_img_fn_str, output_dir_2, scale, transforms, target_ligand=target_ligand, ligand_n=i, target_tier=2,  desc=(brain,hemi,slab), clobber=clobber)

        concat_list.append( df_ligand.loc[ df_ligand['tier'] == 2 ] )
   
        #print(target_ligand)
        #print(df_ligand['init_tfm'].loc[ df['ligand'] == target_ligand ])

        # update the master dataframe, df, with new dataframe for the ligand 
        df.loc[ df['ligand'] == target_ligand ] = df_ligand.loc[ df['ligand'] == target_ligand ]

    stage_2_df = pd.concat(concat_list)
 
    ###########
    # Stage 3 #
    ###########
    '''
    output_dir_3 = output_dir + os.sep + 'stage_3'
    concat_list=[ df.loc[df['ligand'] == ligand_intensity_order[0]]   ]
   
    current_ligands = [ligand_intensity_order[0]]
    print('Stage 3') 
    for i in range(1,n_ligands) :

        current_ligands.append( ligand_intensity_order[i] )

        target_ligand = current_ligands[-1]
        idx =  df['ligand'].apply(lambda x : x in current_ligands)
        df_ligand = df.loc[ idx ] 
        
        for ref_ligand in current_ligands[0:-1] :
            print('\t\tRef Ligand:', ref_ligand)
            df_ligand['tier'].loc[df_ligand['ligand'] == ref_ligand ] = 1
        

        df_ligand['tier'].loc[df_ligand['ligand'] == target_ligand] = 2
        #Init dict with initial transforms
        transforms_3={}
        for i in df_ligand['slab_order'] :  transforms_3[i]=[]

        df_ligand, transforms_3 = alignment_stage(brain, hemi, slab, df_ligand, slab_img_fn_str, output_dir_3, scale, transforms_3, target_ligand=target_ligand, ligand_n=i, target_tier=2,  desc=(brain,hemi,slab), clobber=clobber)
        concat_list.append( df_ligand.loc[ df_ligand['tier'] == 2] )


    # create a new dataframe
    stage_3_df = pd.concat(concat_list)
    
    df = stage_3_df.copy()
    '''
    final_tfm_dir =  output_dir + os.sep + 'final_tfm'
    os.makedirs(final_tfm_dir, exist_ok=True)
    df = stage_2_df

    df = create_final_outputs(final_tfm_dir, df, manual_2d_df, 1)
    df = create_final_outputs(final_tfm_dir, df, manual_2d_df, -1)
   
    print(df['init_tfm'].loc[df['ligand']=='dpmg'].values); 
    print('Writing:',init_tfm_csv)
        
    df.to_csv(init_tfm_csv)

    df['tier']=1
    if not os.path.exists(init_align_fn) :
        vol = combine_sections_to_vol(df,z_mm,direction,init_align_fn )

from scipy.ndimage import center_of_mass
def apply_transforms_to_landmarks(landmark_df, slab_df, landmark_dir, init_3d_fn):
    
    img_3d = nib.load(init_3d_fn)
    starts_3d = img_3d.affine[[0,1,2],[3,3,3]]
    steps_3d = img_3d.affine[[0,1,2],[0,1,2]]

    for i , row in landmark_df.iterrows() :
        volume_order = row['volume_order']
        if volume_order in slab_df['volume_order'].values :
            landmark_crop_fn = row['crop']
            landmark_init_fn = re.sub('.nii', '_init.nii', landmark_crop_fn)
            landmark_df['init'].iloc[i] = landmark_init_fn

            if not os.path.exists(landmark_init_fn) :
                # init aligned images
                idx = slab_df['volume_order']==volume_order
                tfm_fn = slab_df['init_tfm'].loc[ idx ].values[0]
                init_fn = slab_df['crop_fn'].loc[idx].values[0] 
                shell(f'antsApplyTransforms -v 1 -d 2 -r {init_fn} -i {landmark_crop_fn} -t {tfm_fn} -o {landmark_init_fn}')
                #img = image_read(landmark_init_fn)
                img = nibabel.load(landmark_init_fn)
                print(img.affine)
                steps = img.affine[[0,1],[0,1]]
                starts = img.affine[[0,1],[3,3]]
                print(steps)
                print(starts)
                comv = center_of_mass(img.get_fdata())
                #com = get_center_of_mass(img)
                print('comv', comv[1]*steps[0]+starts[0])
                print('comv', comv[0]*-steps[1]-starts[1])
                #print(img.spacing)
                #print(img.origin)
                #x = com[0] 
                y = slab_df['slab_order'].loc[idx].values[0] * 0.02 + starts_3d[1] 
                print(y, slab_df['slab_order'].loc[idx].values[0] , 0.02 ,starts_3d[1])
                #z = com[1] 
                #print('x,y,z', x,y,z)




if __name__ == '__main__' :
    print(sys.argv)
    brain = sys.argv[1]
    hemi = sys.argv[2]
    slab = sys.argv[3] 
    init_align_fn = sys.argv[4]
    source_dir = sys.argv[5]
    output_dir = sys.argv[6]
    receptor_df_fn = sys.argv[7]
    
    if not os.path.exists(init_align_fn) :
        receptorRegister( brain, hemi, slab, init_align_fn, source_dir, output_dir, receptor_df_fn,   clobber=False)

