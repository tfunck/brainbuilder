import nibabel as nib
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
from utils.utils import shell
from glob import glob
from skimage.transform import resize
from utils.utils import set_csv, add_padding
from utils.ANTs import ANTs
from ants import write_transform, read_transform, registration, apply_transforms, image_mutual_information, from_numpy, image_similarity
#matplotlib.use('TKAgg')

def load2d(fn):
    ar = nib.load(fn).get_fdata()
    ar = ar.reshape(ar.shape[0],ar.shape[1])
    return ar 

def create_final_transform(df, transforms, fixed_fn, output_dir, clobber=False) :
    if not os.path.exists(output_dir) :
        os.makedirs(output_dir)

    df['init_tfm']=['']*df.shape[0]
    for key, values in transforms.items():
        final_tfm_fn = "{}/{}_final_Rigid.h5".format(output_dir, key)
        if len(transforms[key]) > 0 :
            df['init_tfm'].loc[df['volume_order'].astype(float)==float(key)] = final_tfm_fn
        else :
            print('\tskipping',key)
        
        if not os.path.exists(final_tfm_fn)  or clobber :
            output_str = f'-o Linear[{final_tfm_fn}]'
            if len(transforms[key]) > 0 :
                transforms_str='-t {} '.format( ' -t '.join( transforms[key])  )
                shell(f'antsApplyTransforms -v 0 -d 2 -i {fixed_fn} -r {fixed_fn} {transforms_str} {output_str}',True,True)


    return  df


def align_neighbours_to_fixed(i, j_list, df, transforms, iteration, shrink_factor,smooth_sigma, output_dir, tfm_type, desc,target_ligand=None,clobber=False):
    #For neighbours

    # a single section is selected as fixed (ith section), then the subsequent sections are considred moving sections (jth section)
    # and are registered to the fixed section. 

    i_idx = df['volume_order']==i
    fixed_fn=df['crop_fn'].loc[i_idx].values[0]
    for j in j_list :
        j_idx = df['volume_order']==j 
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
            print('\n\n\n\t-->', transforms[i], '\n\n\n')
            transforms_str='-t {} '.format( ' -t '.join(  transforms[i] + [tfm_fn] )  )
            print(transforms_str)
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


        df['crop_fn_new'].loc[df['volume_order'] == j] = moving_rsl_fn
        df['init_tfm'].loc[df['volume_order'] == j] =  concat_tfm_fn
        df['init_fixed'].loc[df['volume_order'] == j] = fixed_fn
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
    y_idx_tier1 = df['volume_order'].loc[df['tier'] == 1].values.astype(int)
    y_idx_tier2 = df['volume_order'].loc[df['tier'] == 2].values.astype(int)
    y_idx_tier1.sort()
    i_max = step if step < 0 else df['volume_order'].values.max() + 1

    # Iterate over the sections along y-axis
    for y in y_idx_tier1[mid::step] : 
        j_list=[]
        for j in range(int(y+step), int(i_max), int(step)) :
            if j in df['volume_order'].loc[ df['tier'] == target_tier ].values.astype(int) :
                j_list.append(int(j))
            if j in y_idx_tier1 :
                break

        df, transforms = align_neighbours_to_fixed(y, j_list, df, transforms, iteration, shrink_factor,smooth_sigma, output_dir,  tfm_type, desc,target_ligand=target_ligand,clobber=clobber)

    df.to_csv('{}/df_{}-0.csv'.format(output_dir,tfm_type))
    return transforms,df


def apply_transforms_to_sections(df,transforms,output_dir, tfm_type, target_ligand=None, clobber=False) :
    print('Applying Transforms') 
    if not target_ligand == None :
        df = df.loc[ df['ligand'] == target_ligand  ]

    for i,(rowi, row) in enumerate(df.iterrows()) :
        outprefix ="{}/init_transforms/{}_{}-0/".format(output_dir, int(row['volume_order']), tfm_type ) 
        final_rsl_fn= outprefix + "final_level-0_Mattes_{}-0.nii.gz".format( tfm_type)

        if not os.path.exists(final_rsl_fn) or clobber  :
            fixed_tfm_list = transforms[ row['volume_order']]
            if len(fixed_tfm_list) == 0 : continue
            transforms_str=' -t {}'.format( ' -t '.join(fixed_tfm_list) ) 
   
            fixed_fn = row['original_crop_fn']

            shell(f'antsApplyTransforms -v 1 -d 2 -i {fixed_fn} -r {fixed_fn} {transforms_str} -o {final_rsl_fn}',True)
        df['crop_fn_new'].iloc[i] = final_rsl_fn
    return df


def combine_sections_to_vol(df,z_mm,direction,out_fn,target_tier=1):
    example_fn=df["crop_fn"].iloc[0]
    print(example_fn)
    shape = nib.load(example_fn).shape
    xmax = int( shape[0] / 10 )
    zmax = int( shape[1] / 10 )
    order_max=df["volume_order"].astype(int).max()
    order_min=df["volume_order"].astype(int).min()  
    slab_ymax=int(order_max+1) #-order_min + 1

    vol = np.zeros( [xmax,slab_ymax,zmax])
    for i, row in df.iterrows():
        if row['tier'] == target_tier :
            y = row['volume_order']
            #print()
            #print(row)
            print(y, 'reading : ', row['crop_fn'])
            #print()
            ar = nib.load(row['crop_fn']).get_fdata()
            ar=ar.reshape(ar.shape[0],ar.shape[1])
            ar=resize(ar,[xmax,zmax])

            vol[:,int(y),:]= ar
        
    xstep = 0.02 * 10
    zstep = z_mm/4164. * 10

    print("\n\tWriting Volume",out_fn,"\n")
    slab_ymin=-126+df["global_order"].min()*0.02 
    print("slab ymin:", slab_ymin)
    ystep = 0.02 
    affine=np.array([[xstep, 0, 0, -90],
                    [0,  ystep, 0, slab_ymin],
                    [0, 0 ,zstep, -72],
                    [0, 0, 0, 1]])
    affine = np.round(affine,3)
    nib.Nifti1Image(vol, affine ).to_filename( out_fn )

def alignment_stage(brain,hemi,slab, df, vol_fn_str, output_dir, scale, transforms,  desc=(0,0,0), target_ligand=None, target_tier=1, ligand_n=0,  clobber=False):
    '''
    Perform alignment of autoradiographs within a slab. Alignment is calculated once from the middle section in the
    posterior direction and a second time from the middle section in the anterior direction.
    '''

    # Set parameters for rigid transform
    tfm_type = 'Rigid'
    shrink_factor = '12x10x8' #x4x2x1'
    smooth_sigma = '6x5x4' #x2x1x0' 
    iterations = '100x50x25' #x100x50x20'

    csv_fn = vol_fn_str.format(output_dir,*desc,target_ligand,ligand_n,tfm_type+'-'+str(0),'.csv')

    if not os.path.exists(csv_fn) or not os.path.exists(out_fn) :
        z_mm = scale[brain][hemi][str(slab)]["size"]
        direction = scale[brain][hemi][str(slab)]["direction"]
        
        df.sort_values(['volume_order'],inplace=True,ascending=False)
        
        y_idx = df['volume_order'].values 
        
        y_idx_tier1 = df['volume_order'].loc[ df['tier'].astype(int) == np.min(df['tier']) ].values
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
    else :
        df = pd.read_csv(csv_fn)

    return df, transforms
    
def receptorRegister(brain,hemi,slab, init_align_fn, init_tfm_csv, output_dir, df,  scale_factors_json="scale_factors.json",  clobber=False):
    
    os.makedirs(output_dir,exist_ok=True)

    df['original_crop_fn']=df['crop_fn']

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
    for i in df_ligand['volume_order'] :  transforms_1[i]=[]

    df_ligand, transforms_1 = alignment_stage(brain,hemi,slab,df_ligand, slab_img_fn_str, output_dir_1, scale, transforms_1,  target_ligand='dpmg',ligand_n=0,target_tier=1, desc=(brain,hemi,slab), clobber=clobber)
   
    # update the master dataframe, df, with new dataframe for the ligand 
    df.loc[ df['ligand'] == ligand_intensity_order[0]] = df_ligand

    ###########
    # Stage 2 #
    ###########
    #Align ligands to one another based on mean pixel intensity. Start with highest first because these have better contrast
    output_dir_2 = output_dir + os.sep + 'stage_2'
    concat_list=[ df.loc[df['ligand'] == ligand_intensity_order[0]]   ]
    print('Stage 2') 
    for i in range(1,n_ligands) :
        current_ligands = [ligand_intensity_order[0], ligand_intensity_order[i]]

        target_ligand = current_ligands[-1]
        idx =  df['ligand'].apply(lambda x : x in current_ligands)
        df_ligand = df.loc[ idx ] 
        df_ligand['tier'].loc[df_ligand['ligand']==target_ligand] = 2
        df_ligand['tier'].loc[df_ligand['ligand']==ligand_intensity_order[0]] = 1
        #Init dict with initial transforms
        transforms_2={}
        for i in df_ligand['volume_order'] :  transforms_2[i]=[]

        df_ligand, transforms_2 = alignment_stage(brain, hemi, slab, df_ligand, slab_img_fn_str, output_dir_2, scale, transforms_2, target_ligand=target_ligand, ligand_n=i, target_tier=2,  desc=(brain,hemi,slab), clobber=clobber)

        concat_list.append( df_ligand.loc[ df_ligand['tier'] == 2] )
    
        # update the master dataframe, df, with new dataframe for the ligand 
        df.loc[ df['ligand'] == target_ligand ] = df_ligand.loc[ df['ligand'] == target_ligand ]

    stage_2_df = pd.concat(concat_list)



    ###########
    # Stage 3 #
    ###########
    output_dir_3 = output_dir + os.sep + 'stage_3'
    concat_list=[ df.loc[df['ligand'] == ligand_intensity_order[0]]   ]
   
    print('Stage 3') 
    for i in range(1,n_ligands) :
        current_ligands = [ligand_intensity_order[0], ligand_intensity_order[i]]

        target_ligand = current_ligands[-1]
        idx =  df['ligand'].apply(lambda x : x in current_ligands)
        df_ligand = df.loc[ idx ] 
        df_ligand['tier'].loc[df_ligand['ligand'] == ligand_intensity_order[0]] = 1
        df_ligand['tier'].loc[df_ligand['ligand'] == target_ligand] = 2
        #Init dict with initial transforms
        transforms_3={}
        for i in df_ligand['volume_order'] :  transforms_3[i]=[]

        df_ligand, transforms_3 = alignment_stage(brain, hemi, slab, df_ligand, slab_img_fn_str, output_dir_3, scale, transforms_3, target_ligand=target_ligand, ligand_n=i, target_tier=2,  desc=(brain,hemi,slab), clobber=clobber)
        concat_list.append( df_ligand.loc[ df_ligand['tier'] == 2] )
        df['tier'].loc[ df['ligand'] == target_ligand ] = 1
    
    # create a new dataframe
    stage_3_df = pd.concat(concat_list)
    
    df = stage_3_df.copy()
    
    final_tfm_dir =  output_dir + os.sep + 'final_tfm'
    os.makedirs(final_tfm_dir, exist_ok=True)
    ### create final transforms that take raw autoradiographs from their raw alignement to the initial
    ### rigid alignment. this involves combining the concatenated transforms of stages 2 and 3
    for i, row in stage_3_df.iterrows() : 

        final_tfm_fn = "{}/{}_final_Rigid.h5".format(final_tfm_dir, row['volume_order'])

        idx = df['volume_order'] == row['volume_order']

        if not os.path.exists(final_tfm_fn) :
            print( row['ligand'], row['ligand'] != ligand_intensity_order[0] );

            if row['ligand'] != ligand_intensity_order[0] :
                stage_2_init_tfm = stage_2_df['init_tfm'].loc[ stage_2_df['volume_order'] == row['volume_order']].values[0] 
                stage_3_init_tfm = row['init_tfm']
                crop_fn = row['crop_fn']
                shell(f'antsApplyTransforms -v 0 -d 2 -i {crop_fn} -r {crop_fn} -t {stage_3_init_tfm} -t {stage_2_init_tfm} -o Linear[{final_tfm_fn}]',True,True)
                df['init_tfm'].loc[ idx ] = final_tfm_fn

            elif not np.isnan(row['init_tfm']) :
                shutil.copy(row['init_tfm'], final_tfm_fn)
            else :
                final_tfm_fn = np.nan
        
        df['init_tfm'].loc[ idx ] = final_tfm_fn
            
        print('\t--->',df['init_tfm'].loc[ idx ].values[0])

    print('Writing:',init_tfm_csv)
        
    df.to_csv(init_tfm_csv)

    df['tier']=1
    if not os.path.exists(init_align_fn) :
        vol = combine_sections_to_vol(df,z_mm,direction,init_align_fn )

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

