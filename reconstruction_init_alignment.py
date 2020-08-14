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
from utils.utils import set_csv, add_padding, setup_tiers
from ANTs import ANTs
from ants import write_transform, read_transform, registration, apply_transforms, image_mutual_information, from_numpy, image_similarity
matplotlib.use('TKAgg')

global generic_affine
generic_affine=np.array([[0.2, 0, 0, -72],
    [0,  0.02, 0, 0],
    [0, 0 , 0.2, -90],
    [0, 0, 0, 1]])

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
            if values['lin'] == [] : continue
            print('key')
            print(key)
            #print('values')
            #print(values)
            try :
                transforms[key] += values['lin']
            except KeyError :
                transforms[key]=values['lin']

    df['init_tfm']=['']*df.shape[0]
    for key, values in transforms.items():
        final_tfm_fn = "{}/{}_final_Rigid.h5".format(output_dir, key)
        df['init_tfm'].loc[df['volume_order']==int(key)] = final_tfm_fn
        if not os.path.exists(final_tfm_fn) or clobber :
            output_str = f'-o Linear[{final_tfm_fn}]'
            transforms_str='-t {} '.format( ' -t '.join(transforms[key])  )
            shell(f'antsApplyTransforms -v 1 -d 2 -i {fixed_fn} -r {fixed_fn} {transforms_str} {output_str}',True,True)

    return  df


def align_neighbours_to_fixed(i, j_list,df,transforms, iteration, shrink_factor,smooth_sigma, output_dir, y_idx, tfm_type, epoch,desc,target_ligand=None,clobber=False):
    #For neighbours
    i_idx = df['volume_order']==i
    fixed_fn=df['filename_rsl'].loc[i_idx].values[0]
    print('Fixed:', i,fixed_fn)
    for j in j_list :
        j_idx = df['volume_order']==j 
        moving_fn=df['filename_rsl'].loc[ j_idx].values[0]
        print('\tMoving:',j, moving_fn)
        if target_ligand != None :
            print(df['ligand'].loc[ j_idx ].values[0])
            if df['ligand'].loc[ j_idx ].values[0] != target_ligand : 
                print('\tSkipping')
                continue

        outprefix ="{}/init_transforms/{}_{}-{}/".format(output_dir,  j, tfm_type, epoch) 
        if not os.path.exists(outprefix): os.makedirs(outprefix)
        tfm_fn = outprefix + "level-0_Mattes_{}_Composite.h5".format( tfm_type)
        inv_tfm_fn = outprefix + "level-0_Mattes_{}_InverseComposite.h5".format( tfm_type)
        moving_rsl_fn= outprefix + "_level-0_Mattes_{}.nii.gz".format( tfm_type)
        
        qc_fn = "{}/qc/{}_{}_{}_{}_{}_{}-{}.png".format(output_dir, *desc, j,i, tfm_type, epoch)
        if clobber or not os.path.exists(moving_rsl_fn) or not os.path.exists(tfm_fn)  or not os.path.exists(qc_fn)  :

            ANTs(tfm_prefix=outprefix,
                fixed_fn=fixed_fn, 
                moving_fn=moving_fn, 
                moving_rsl_prefix=outprefix, 
                iterations=[iteration], 
                metrics=['Mattes'], 
                tfm_type=[tfm_type], 
                shrink_factors=[shrink_factor],
                smoothing_sigmas=[smooth_sigma], 
                init_tfm=None, no_init_tfm=False, #transforms[ y_idx[j]]['fwd'],
                dim=2,
                sampling_method='Random', sampling=0.5, verbose=0, generate_masks=False, clobber=True  )
            create_qc_image( load2d(fixed_fn), 
                        None,
                        load2d(moving_fn), 
                        load2d(moving_rsl_fn),
                        None,
                        i, j,df['tier'].loc[i_idx], df['tier'].loc[j_idx], df['ligand'].loc[i_idx], df['ligand'].loc[j_idx], qc_fn)

            #   F       M
            #   | <fwd  |
            #   | inv>  |
            #   |       |
        fixed_tfm_list = transforms[i ]['lin']
        if tfm_type == 'Rigid' :
            transforms[ j ]['lin'] = [tfm_fn] + fixed_tfm_list 
        else : 
            # if df['tier'].loc[j_idx] == 1 :
            transforms[ i ]['inv'].append(inv_tfm_fn)
            transforms[ j ]['fwd'].append(tfm_fn)  
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
    y_idx_tier1 = df['volume_order'].loc[df['tier'] == 1].values
    y_idx_tier2 = df['volume_order'].loc[df['tier'] == 2].values
    y_idx_tier1.sort()
    i_max = step if step < 0 else df['volume_order'].values.max() + 1
    if not os.path.exists(output_dir+"/qc"): os.makedirs(output_dir+"/qc")
    # Iterate over the sections along y-axis
    for y in y_idx_tier1[mid::step] : 
        j_list=[]
        for j in range(y+step, i_max, step) :
            if j in df['volume_order'].loc[ df['tier'] == target_tier ].values :
                j_list.append(j)
            if j in y_idx_tier1 :
                break
        df, transforms = align_neighbours_to_fixed(y, j_list,df, transforms,iteration, shrink_factor,smooth_sigma, output_dir, y_idx, tfm_type, epoch,desc,target_ligand=target_ligand,clobber=clobber)



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
        fixed_tfm_fn = transforms[ row['volume_order']]['lin'][0]
        print(fixed_tfm_fn)
        tfm = read_transform(fixed_tfm_fn)
        print(avg_parameters)
        avg_parameters += tfm.parameters

    avg_parameters /= df_target_ligand.shape[0]
        
    print('Applying averaged affine')
    for i,(rowi, row) in enumerate( df_target_ligand.iterrows() ) :
        outprefix ="{}/init_transforms/{}_{}-{}/".format(output_dir, row['volume_order'], tfm_type, epoch) 
        fixed_tfm_fn = transforms[ row['volume_order']]['lin'][0]
        avg_tfm_fn = outprefix+'/averaged_fwd_Rigid.h5'
        tfm = read_transform(fixed_tfm_fn)
        tfm.set_parameters(avg_parameters)
        write_transform( tfm, avg_tfm_fn)
        transforms[ row['volume_order']]['lin_avg']=avg_tfm_fn
    return transforms

def apply_transforms_to_sections(df,transforms,output_dir, tfm_type, epoch,target_ligand=None,stage=1, clobber=False) :
    print('Applying Transforms') 
    print('target ligand',target_ligand)
    if not target_ligand == None :
        df = df.loc[ df['ligand'] == target_ligand  ]

    for i,(rowi, row) in enumerate(df.iterrows()) :
        print(row)
        outprefix ="{}/init_transforms/{}_{}-{}/".format(output_dir, row['volume_order'], tfm_type, epoch) 
        if tfm_type == 'Rigid' :
            #if stage==1 or stage == 2 :
            fixed_tfm_list = transforms[ row['volume_order']]['lin']
            if len(fixed_tfm_list) == 0 : continue
            transforms_str=' -t {}'.format( ' -t '.join(fixed_tfm_list) ) 
            #else :
             #   avg_tfm_fn=fixed_tfm_list = transforms[ row['volume_order']]['lin_avg']
             #   transforms_str=' -t {}'.format( avg_tfm_fn ) 
        else :
            fwd_tfm_list = transforms[ row['volume_order']]['fwd']
            inv_tfm_list = transforms[ row['volume_order']]['inv']
            avg_tfm_fn = outprefix+'/averaged_fwd_inv_SyN.h5'
            
            #if len(fwd_tfm_list) != epoch :
            #    #print('Error: too many/few non-linear transformation files.')
            #    #print('fwd',row['volume_order'],transforms[ row['volume_order']]['fwd'] )  
            #    continue

            #if len(inv_tfm_list) != epoch :
            #    #print('Error: too many/few non-linear transformation files.')
            #    #print('inv',row['volume_order'],transforms[ row['volume_order']]['inv'] )  
            #    continue

            #if not os.path.exists(avg_tfm_fn) or clobber :

            try :
                fwd_tfm_fn = fwd_tfm_list[-1]
                inv_tfm_fn = inv_tfm_list[-1]
                fwd_tfm = read_transform(fwd_tfm_fn) 
                inv_tfm = read_transform(inv_tfm_fn) 
                fwd_tfm.set_parameters(fwd_tfm.parameters*0.5 + inv_tfm.parameters*0.5)
                write_transform(fwd_tfm, avg_tfm_fn) 
                transforms_str = ' -t {}'.format(avg_tfm_fn)

                #transforms_str = ' -t {}'.format(fwd_tfm_list[-1])
            except IndexError :
                continue

        fixed_fn = row['filename_rsl']
        final_rsl_fn= outprefix + "final_level-0_Mattes_{}-{}.nii.gz".format( tfm_type,epoch)

        print(final_rsl_fn)
        if not os.path.exists(final_rsl_fn) or clobber or True  :
            shell(f'antsApplyTransforms -v 0 -d 2 -i {fixed_fn} -r {fixed_fn}  {transforms_str} -o {final_rsl_fn}',True)
        df['filename_rsl_new'].iloc[i] = final_rsl_fn
    return df



def concatenate_transforms(transforms,tfm_type,epoch):
    #Calculate combined transformation
    concat_transforms={}
    for i, tfm_list in transforms.items() :
        concat_tfm_fn ="{}/init_transforms/epoch-{}-{}/{}/concat_tfm.nii.gz".format(output_dir, tfm_type,epoch, i) 
        if len(tfm_list) > 0 : 
            if not os.path.exists(concat_tfm_fn) or clobber :
                shell('AverageImages 2 {} 0 {}'.format(concat_tfm_fn,' '.join(tfm_list) ))
            concat_transforms[i]=concat_tfm_fn 
    return concat_transforms


def combine_sections_to_vol(df,z_mm,direction,out_fn):
    example_fn=df["filename_rsl"].iloc[0]
    print(example_fn)
    shape = nib.load(example_fn).shape
    xmax = shape[0] 
    zmax = shape[1]
    order_max=df["volume_order"].max()
    order_min=df["volume_order"].min()  
    slab_ymax=order_max+1 #-order_min + 1

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

        vol[:,y,:]= ar

    
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

def alignment_stage( df, vol_fn_str, output_dir,scale,parameters, tfm_json_list=[], n_epochs=3, desc=(0,0,0), stage=1,target_ligand=None, target_tier=1, ligand_n=0, write_each_iteration=False, clobber=False):
    
    z_mm = scale[brain][hemi][str(slab)]["size"]
    direction = scale[brain][hemi][str(slab)]["direction"]
    
    df.sort_values(['volume_order'],inplace=True,ascending=False)
    
    y_idx = df['volume_order'].values 
    
    y_idx_tier1 = df['volume_order'].loc[ df['tier'].astype(int) == np.min(df['tier']) ].values
    mid = int(len(y_idx_tier1)/2) 

    #Init dict with initial transforms
    transforms={}
    for i in y_idx :  transforms[i]={'lin':[],'fwd':[], 'inv':[] }

    df['original_filename_rsl']=df['filename_rsl']

    for epoch, (tfm_type, shrink_factor, smooth_sigma,iterations) in enumerate(parameters) :

        df_fn=vol_fn_str.format(output_dir,*desc,target_ligand,ligand_n, tfm_type+'-'+str(epoch), 'csv')
        json_fn=vol_fn_str.format(output_dir,*desc,target_ligand,ligand_n, tfm_type+'-'+str(epoch), 'json')

        #Add current json filename to list of transforms so that these transforms can be concatenated together later
        tfm_json_list.append(json_fn) 

        out_fn = vol_fn_str.format(output_dir,*desc,target_ligand,ligand_n,tfm_type+'-'+str(epoch),'nii.gz')
        if (not os.path.exists(df_fn) or not os.path.exists(json_fn)) or clobber :
            df['filename_rsl_new']=df['filename_rsl']
            print(tfm_type, shrink_factor, smooth_sigma,iterations, target_ligand)

            transforms,df = adjust_alignment(df,  epoch, y_idx, mid, transforms, -1, output_dir,desc, shrink_factor, smooth_sigma, iterations, tfm_type, target_ligand=target_ligand, target_tier=target_tier, clobber=clobber)
            transforms,df = adjust_alignment(df,  epoch, y_idx, mid, transforms, 1, output_dir, desc, shrink_factor, smooth_sigma, iterations, tfm_type, target_ligand=target_ligand,target_tier=target_tier, clobber=clobber)
            
            #if stage == 2 :
            #    transforms = average_affine_transforms(df,transforms,output_dir, tfm_type , epoch, target_ligand, clobber)
            df = apply_transforms_to_sections(df,transforms,output_dir, tfm_type, epoch, target_ligand=target_ligand,stage=stage)

            df['filename_rsl']=df['filename_rsl_new']

            df.to_csv(df_fn)
            
            # save lists of transformations to apply to each image to bring it into alignment 
            initial_transforms_mod = { str(k):v for k, v in transforms.items() }
            with open(json_fn, 'w+') as fp : 
                json.dump(initial_transforms_mod, fp )
        else :
            df = pd.read_csv(df_fn)
        if  not os.path.exists(out_fn) or clobber  :
            vol = combine_sections_to_vol(df,z_mm, direction, out_fn)
            


    return df, tfm_json_list
    
def add_y_position(df):
    df["volume_order"]=[0]*df.shape[0]
    df["volume_order"] = df["global_order"].max() - df["global_order"] 
    #order_max=df["volume_order"].max()
    #for i, row in df.iterrows() :
    #    df["volume_order"].loc[ row["order"] == df["order"]] = order_max - row["order"]
    return df

def receptorRegister(brain,hemi,slab, init_align_fn, source_dir, output_dir, receptor_df_fn, tiers_string=None, scale_factors_json="scale_factors.json", n_epochs=3, write_each_iteration=False, clobber=False):
    print('receptor register')
    if not os.path.exists(output_dir) :
        os.makedirs(output_dir)

    if tiers_string==None:
        #tiers_string = "flum,musc,sr95,cgp5,afdx,uk20,pire,praz,keta,dpmg;sch2,dpat,rx82,kain,ly34,damp,mk80,ampa,uk18;oxot;epib"
        tiers_string = 'afdx,cgp5,damp,dpmg,flum,keta,ly34,musc,pire,praz,sr95,uk14;ampa,dpat,epib,kain,mk80,oxot,rx82,sch2'
    ligand_intensity_order = ['dpmg','flum','cgp5','damp','praz','afdx','sr95','keta','musc','ly34','pire','uk14','sch2','mk80','dpat','kain','rx82','oxot','ampa','epib']

    with open(scale_factors_json) as f : scale=json.load(f)
    df = pd.read_csv(receptor_df_fn)
    df = df.loc[ (df.mri ==brain) & (df.slab==int(slab)) & (df.hemisphere==hemi)]
    #df["volume_order"] = df["global_order"]
    df = add_y_position(df)
    df.sort_values(["volume_order"], inplace=True)
    df['filename']=source_dir+'/'+df['lin_fn'].apply(lambda x : os.path.splitext(os.path.basename(x))[0]) +'.png'
    df['filename']=df['filename'].apply(lambda x : re.sub('#L.png','.png',x))
    df['filename_rsl'] = df['filename'].apply(lambda x: 'reconstruction_output/0_crop/'+os.path.splitext(os.path.basename(x))[0]+'#L.nii.gz')

    # Remove UB , "unspecific binding" from the df
    df = df.loc[df['repeat'] != 'UB']
    slab_img_fn_str = '{}/brain-{}_hemi-{}_slab-{}_ligand_{}_{}_init_align_{}.{}'

    df = setup_tiers(df, tiers_string)
    df.to_csv(output_dir+'/tiers.csv')
    #df = df.loc[df['ligand'].apply(lambda x : x in ligand_intensity_order[0:5])]
    n_ligands = len(df['ligand'].unique())
    ###########
    # Stage 1 #
    ###########
    #Perform within ligand alignment
    output_dir_1 = output_dir + os.sep + 'stage_1'
    concat_list=[]

    parameters=[('Rigid','4x2x1','2x1x0','250x100x20')]
    for target_ligand, df_ligand  in df.groupby(['ligand']):
        ligand_n = ligand_intensity_order.index(target_ligand)
        df_ligand, tfm_json_list = alignment_stage(df_ligand, slab_img_fn_str, output_dir_1, scale, stage=1, target_ligand=target_ligand,ligand_n=ligand_n, n_epochs=n_epochs, parameters=parameters, target_tier=1, write_each_iteration=write_each_iteration,desc=(brain,hemi,slab), clobber=clobber)

        if df['filename_rsl'].isnull().sum() > 0 : 
            print('nan in filename_rsl, stage 1, ligand', target_ligand)
            exit(1)
        concat_list.append( df_ligand )
    
    df = pd.concat(concat_list)

    if df['filename_rsl'].isnull().sum() > 0 : 
        print('nan in filename_rsl')
        exit(1)

    ###########
    # Stage 2 #
    ###########
    #Align ligands to one another based on mean pixel intensity. 
    #Start with highest first because these have better contrast.
    output_dir_2 = output_dir + os.sep + 'stage_2'
    concat_list=[ df.loc[df['ligand'] == ligand_intensity_order[0]]   ]
    print('Stage 2') 
    for i in range(1,n_ligands) :
        current_ligands = [ligand_intensity_order[0], ligand_intensity_order[i]]
        print('1',current_ligands)


        target_ligand = current_ligands[-1]
        idx =  df['ligand'].apply(lambda x : x in current_ligands)
        df_ligand = df.loc[ idx ] 
        df_ligand['tier'].loc[df_ligand['ligand']==target_ligand] = 2
        df_ligand['tier'].loc[df_ligand['ligand']==ligand_intensity_order[0]] = 1
        
        df_ligand, tfm_json_list = alignment_stage(df_ligand, slab_img_fn_str, output_dir_2, scale,stage=2, n_epochs=n_epochs, target_ligand=target_ligand, ligand_n=i, tfm_json_list=tfm_json_list,target_tier=2, parameters=parameters, write_each_iteration=write_each_iteration,desc=(brain,hemi,slab), clobber=clobber)

        concat_list.append( df_ligand )
    
    df = pd.concat(concat_list)
    df = create_final_transform(df, tfm_json_list, df['filename_rsl'].values[0], output_dir+'/final', clobber=clobber)
    df.to_csv(f'{output_dir}/final/{brain}_{hemi}_{slab}_final.csv')

    z_mm = scale[brain][hemi][str(slab)]["size"]
    direction = scale[brain][hemi][str(slab)]["direction"]
    vol = combine_sections_to_vol(df,z_mm,direction,init_align_fn )

    exit(0)
    ###########
    # Stage 3 #
    ###########

    output_dir_3 = output_dir + os.sep + 'stage_3'
    df['tier'] = 1
    parameters=[('SyN','4','2','25'), ('SyN','2','1','25'),  ('SyN','1','0','25') ]
    df = alignment_stage(init_align_fn, df, slab_img_fn_str, output_dir_3, scale, stage=3, target_ligand=None,ligand_n=ligand_n, n_epochs=n_epochs, target_tier=1,parameters=parameters, write_each_iteration=write_each_iteration,desc=(brain,hemi,slab), clobber=clobber)

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

