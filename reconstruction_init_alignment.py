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
from utils.utils import shell
from glob import glob
import re
import cv2
from utils.utils import set_csv, add_padding
from ANTs import ANTs
from ants import write_transform, read_transform, registration, apply_transforms, image_mutual_information, from_numpy, image_similarity
matplotlib.use('TKAgg')

def load2d(fn):
    ar = nib.load(fn).get_fdata()
    ar = ar.reshape(ar.shape[0],ar.shape[1])
    return ar 

def create_final_transform(df, tfm_json_list, fixed_fn, output_dir, clobber=False) :
    if not os.path.exists(output_dir) :
        os.makedirs(output_dir)

    transforms={}
    for fn in tfm_json_list :
        with open(fn,'r') as f :
            temp_transform = json.load(f)
        print('N Items:', len(temp_transform.keys()))
        for key, values in temp_transform.items() :
            if values == [] : continue
            try :
                transforms[key] += values
            except KeyError :
                transforms[key]=values

    df['init_tfm']=['']*df.shape[0]
    for key, values in transforms.items():
        final_tfm_fn = "{}/{}_final_Rigid.h5".format(output_dir, key)
        df['init_tfm'].loc[df['volume_order']==int(float(key))] = final_tfm_fn
        if not os.path.exists(final_tfm_fn) or clobber :
            output_str = f'-o Linear[{final_tfm_fn}]'
            transforms_str='-t {} '.format( ' -t '.join(transforms[key])  )
            shell(f'antsApplyTransforms -v 1 -d 2 -i {fixed_fn} -r {fixed_fn} {transforms_str} {output_str}',True,True)

    return  df


def align_neighbours_to_fixed(i, j_list,df,transforms, iteration, shrink_factor,smooth_sigma, output_dir, y_idx, tfm_type, epoch,desc,target_ligand=None,clobber=False):
    #For neighbours
    i_idx = df['volume_order']==i
    fixed_fn=df['filename_rsl'].loc[i_idx].values[0]
    for j in j_list :
        outprefix ="{}/init_transforms/{}_{}-{}/".format(output_dir,  j, tfm_type, epoch) 

        moving_rsl_fn= outprefix + "_level-0_Mattes_{}.nii.gz".format( tfm_type)
        tfm_fn = outprefix + "level-0_Mattes_{}_Composite.h5".format( tfm_type)
        qc_fn = "{}/qc/{}_{}_{}_{}_{}_{}-{}.png".format(output_dir, *desc, j,i, tfm_type, epoch)
        
        if clobber or not os.path.exists(moving_rsl_fn) or not os.path.exists(tfm_fn)  or not os.path.exists(qc_fn)  :
            j_idx = df['volume_order']==j 
            moving_fn=df['filename_rsl'].loc[ j_idx].values[0]
            print('\tMoving:',j, moving_fn)
            print('\tTfm:',tfm_fn, os.path.exists(tfm_fn))
            print('\tQC:',qc_fn, os.path.exists(qc_fn))
            if target_ligand != None :
                print(df['ligand'].loc[ j_idx ].values[0])
                if df['ligand'].loc[ j_idx ].values[0] != target_ligand : 
                    print('\tSkipping')
                    continue

            os.makedirs(outprefix, exist_ok=True)
            inv_tfm_fn = outprefix + "level-0_Mattes_{}_InverseComposite.h5".format( tfm_type)
            
            ANTs(tfm_prefix=outprefix,
                fixed_fn=fixed_fn, 
                moving_fn=moving_fn, 
                moving_rsl_prefix=outprefix, 
                iterations=[iteration], 
                metrics=['Mattes'], 
                tfm_type=[tfm_type], 
                shrink_factors=[shrink_factor],
                smoothing_sigmas=[smooth_sigma], 
                init_tfm=None, no_init_tfm=False, 
                dim=2,
                sampling_method='Random', sampling=0.5, verbose=0, generate_masks=False, clobber=True  )

            create_qc_image( load2d(fixed_fn), 
                        None,
                        load2d(moving_fn), 
                        load2d(moving_rsl_fn),
                        None,
                        i, j,df['tier'].loc[i_idx], df['tier'].loc[j_idx], df['ligand'].loc[i_idx], df['ligand'].loc[j_idx], qc_fn)

        fixed_tfm_list = transforms[i ]
        transforms[ j ] = [tfm_fn] + fixed_tfm_list 
    
    return df, transforms
                
def create_qc_image(fixed,fixed_rsl, moving,rsl,final_rsl, fixed_order, moving_order,tier_fixed,tier_moving,ligand_fixed,ligand_moving,qc_fn):
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

def adjust_alignment(df,  epoch, y_idx, mid, transforms, step, output_dir,desc, shrink_factor,  smooth_sigma,iteration,tfm_type,target_ligand=None, target_tier=1, clobber=False):
    if not os.path.exists(output_dir + "/qc/") : os.makedirs(output_dir + "/qc/")
    i=mid
    j=i
    n=len(y_idx)
    y_idx_tier1 = df['volume_order'].loc[df['tier'] == 1].values.astype(int)
    y_idx_tier2 = df['volume_order'].loc[df['tier'] == 2].values.astype(int)
    y_idx_tier1.sort()
    i_max = step if step < 0 else df['volume_order'].values.max() + 1

    os.makedirs(output_dir+"/qc",exist_ok=True)

    # Iterate over the sections along y-axis
    for y in y_idx_tier1[mid::step] : 
        j_list=[]
        for j in range(int(y+step), int(i_max), int(step)) :
            if j in df['volume_order'].loc[ df['tier'] == target_tier ].values.astype(int) :
                j_list.append(int(j))
            if j in y_idx_tier1 :
                break

        df, transforms = align_neighbours_to_fixed(y, j_list, df, transforms, iteration, shrink_factor,smooth_sigma, output_dir, y_idx, tfm_type, epoch,desc,target_ligand=target_ligand,clobber=clobber)

    df.to_csv('{}/df_{}-{}.csv'.format(output_dir,tfm_type,epoch))
    return transforms,df

def average_affine_transforms(df,transforms,output_dir, tfm_type , epoch, target_ligand, clobber):

    avg_parameters = np.array([0.,0.,0.])

    df_target_ligand = df.loc[ df['ligand'] == target_ligand  ]

    print('Calculating averaged affine')
    for i,(rowi, row) in enumerate( df_target_ligand.iterrows() ) :
        outprefix ="{}/init_transforms/{}_{}-{}/".format(output_dir, row['volume_order'], tfm_type, epoch) 
        print(row)
        print(transforms[ row['volume_order']])
        fixed_tfm_fn = transforms[ row['volume_order']][0]
        print(fixed_tfm_fn)
        tfm = read_transform(fixed_tfm_fn)
        print(avg_parameters)
        avg_parameters += tfm.parameters

    avg_parameters /= df_target_ligand.shape[0]
        
    print('Applying averaged affine')
    for i,(rowi, row) in enumerate( df_target_ligand.iterrows() ) :
        outprefix ="{}/init_transforms/{}_{}-{}/".format(output_dir, row['volume_order'], tfm_type, epoch) 
        fixed_tfm_fn = transforms[ row['volume_order']][0]
        avg_tfm_fn = outprefix+'/averaged_fwd_Rigid.h5'
        tfm = read_transform(fixed_tfm_fn)
        tfm.set_parameters(avg_parameters)
        write_transform( tfm, avg_tfm_fn)
        transforms[ row['volume_order']]['lin_avg']=avg_tfm_fn
    return transforms

def apply_transforms_to_sections(df,transforms,output_dir, tfm_type, epoch,target_ligand=None,stage=1, clobber=False) :
    print('Applying Transforms') 
    if not target_ligand == None :
        df = df.loc[ df['ligand'] == target_ligand  ]

    for i,(rowi, row) in enumerate(df.iterrows()) :
        outprefix ="{}/init_transforms/{}_{}-{}/".format(output_dir, int(row['volume_order']), tfm_type, epoch) 
        final_rsl_fn= outprefix + "final_level-0_Mattes_{}-{}.nii.gz".format( tfm_type,epoch)

        if not os.path.exists(final_rsl_fn) or clobber  :
            fixed_tfm_list = transforms[ row['volume_order']]
            if len(fixed_tfm_list) == 0 : continue
            transforms_str=' -t {}'.format( ' -t '.join(fixed_tfm_list) ) 
   

            fixed_fn = row['filename_rsl']

            print(final_rsl_fn)
            shell(f'antsApplyTransforms -v 1 -d 2 -i {fixed_fn} -r {fixed_fn} {transforms_str} -o {final_rsl_fn}',True)
        df['filename_rsl_new'].iloc[i] = final_rsl_fn
    return df


def combine_sections_to_vol(df,z_mm,direction,out_fn):
    example_fn=df["filename_rsl"].iloc[0]
    print(example_fn)
    shape = nib.load(example_fn).shape
    xmax = shape[0] 
    zmax = shape[1]
    order_max=df["volume_order"].astype(int).max()
    order_min=df["volume_order"].astype(int).min()  
    slab_ymax=int(order_max+1) #-order_min + 1

    vol = np.zeros( [xmax,slab_ymax,zmax])
    for i, row in df.iterrows():
        #if row['volume_order'] < 500 or row['volume_order'] > 700 :
        #    continue
        y = row['volume_order']
        ar = nib.load(row['filename_rsl']).get_fdata()
        ar=ar.reshape(ar.shape[0],ar.shape[1])

        if direction == "rostral_to_caudal":
            ar = np.flip(ar, 1)
        ar = np.flip(ar, 0)

        vol[:,int(y),:]= ar

    
    xstep= z_mm/4164. * 10
    zstep= z_mm/4164. * 10

    print("\n\tWriting Volume",out_fn,"\n")
    slab_ymin=-126+df["volume_order"].min()*0.02 
    ystep = 0.02 
    affine=np.array([[xstep, 0, 0, -72],
                    [0,  ystep, 0, slab_ymin],
                    [0, 0 ,zstep, -90],
                    [0, 0, 0, 1]])
    nib.Nifti1Image(vol, affine ).to_filename(out_fn )

def alignment_stage(brain,hemi,slab, df, vol_fn_str, output_dir, scale, tfm_json_list=[], desc=(0,0,0), stage=1,target_ligand=None, target_tier=1, ligand_n=0,  clobber=False):
    
    z_mm = scale[brain][hemi][str(slab)]["size"]
    direction = scale[brain][hemi][str(slab)]["direction"]
    
    df.sort_values(['volume_order'],inplace=True,ascending=False)
    
    y_idx = df['volume_order'].values 
    
    y_idx_tier1 = df['volume_order'].loc[ df['tier'].astype(int) == np.min(df['tier']) ].values
    mid = int(len(y_idx_tier1)/2) 

    #Init dict with initial transforms
    transforms={}
    for i in y_idx :  transforms[i]=[]

    df['original_filename_rsl']=df['filename_rsl']
    tfm_type = 'Rigid'
    shrink_factor = '4x2x1'
    smooth_sigma = '2x1x0'
    iterations = '250x100x20'
    epoch=0

    df['filename_rsl_new']=df['filename_rsl']

    transforms,df = adjust_alignment(df,  epoch, y_idx, mid, transforms, -1, output_dir,desc, shrink_factor, smooth_sigma, iterations, tfm_type, target_ligand=target_ligand, target_tier=target_tier, clobber=clobber)
    transforms,df = adjust_alignment(df,  epoch, y_idx, mid, transforms, 1, output_dir, desc, shrink_factor, smooth_sigma, iterations, tfm_type, target_ligand=target_ligand,target_tier=target_tier, clobber=clobber)
    
    df = apply_transforms_to_sections(df,transforms,output_dir, tfm_type, epoch, target_ligand=target_ligand,stage=stage)

    df['filename_rsl']=df['filename_rsl_new']

    out_fn = vol_fn_str.format(output_dir,*desc,target_ligand,ligand_n,tfm_type+'-'+str(epoch),'nii.gz')
    if  not os.path.exists(out_fn) or clobber  :
        vol = combine_sections_to_vol(df,z_mm, direction, out_fn)
            
    return df, tfm_json_list
    
def receptorRegister(brain,hemi,slab, init_align_fn,  output_dir, df,  scale_factors_json="scale_factors.json",  clobber=False):
    
    os.makedirs(output_dir,exist_ok=True)

    ligand_intensity_order = ['dpmg','flum','cgp5','damp','praz','afdx','sr95','keta','musc','ly34','pire','uk14','sch2','mk80','dpat','kain','rx82','oxot','ampa','epib']

    with open(scale_factors_json) as f : scale=json.load(f)

    df['tier'] = 1
    
    slab_img_fn_str = '{}/brain-{}_hemi-{}_slab-{}_ligand_{}_{}_init_align_{}.{}'

    n_ligands = len(df['ligand'].unique())

    ###########
    # Stage 1 #
    ###########
    #Perform within ligand alignment
    output_dir_1 = output_dir + os.sep + 'stage_1'
    concat_list=[]

    for target_ligand, df_ligand  in df.groupby(['ligand']):
        ligand_n = ligand_intensity_order.index(target_ligand)
        print('\t\tAlignment: Stage 1')
        df_ligand, tfm_json_list = alignment_stage(brain,hemi,slab,df_ligand, slab_img_fn_str, output_dir_1, scale, stage=1, target_ligand=target_ligand,ligand_n=ligand_n,target_tier=1, desc=(brain,hemi,slab), clobber=clobber)

        if df['filename_rsl'].isnull().sum() > 0 : 
            print('nan in filename_rsl, stage 1, ligand', target_ligand)
            exit(1)
        concat_list.append( df_ligand )
    
    df = pd.concat(concat_list)

    ###########
    # Stage 2 #
    ###########
    #Align ligands to one another based on mean pixel intensity. Start with highest first because these have better contrast.
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
        
        df_ligand, tfm_json_list = alignment_stage(brain,hemi,slab,df_ligand, slab_img_fn_str, output_dir_2, scale,stage=2,  target_ligand=target_ligand, ligand_n=i, tfm_json_list=tfm_json_list,target_tier=2,  desc=(brain,hemi,slab), clobber=clobber)

        concat_list.append( df_ligand )
    
    df = pd.concat(concat_list)
    df = create_final_transform(df, tfm_json_list, df['filename_rsl'].values[0], output_dir+'/final', clobber=clobber)
    df.to_csv(f'{output_dir}/final/{brain}_{hemi}_{slab}_final.csv')

    z_mm = scale[brain][hemi][str(slab)]["size"]
    direction = scale[brain][hemi][str(slab)]["direction"]
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
    n_epochs= int(sys.argv[8])
    
    if not os.path.exists(init_align_fn) :
        receptorRegister( brain, hemi, slab, init_align_fn, source_dir, output_dir, receptor_df_fn, n_epochs=n_epochs,  clobber=False)

