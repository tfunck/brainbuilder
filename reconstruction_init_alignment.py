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
from utils.utils import set_csv, add_padding, setup_tiers
from ANTs import ANTs
from ants import read_transform, registration, apply_transforms, image_mutual_information, from_numpy, image_similarity
matplotlib.use('TKAgg')

def load2d(fn):
    ar = nib.load(fn).get_fdata()
    ar = ar.reshape(ar.shape[0],ar.shape[1])
    return ar 

def find_next_slice( i , step, df):
    #Get the the tier of the current section
    current_tier = df["tier"].iloc[i]
    i2=i
    i2_list = []
    while i2 > 0 and i2 < df.shape[0] -1   :
        i2 = i2 + step
        i2_list.append(i2)
        if df["tier"].iloc[i2] <= current_tier :
            break
    return i2_list


def create_qc_image(fixed,fixed_rsl, moving,rsl,final_rsl, fixed_order, moving_order,tier_fixed,tier_moving,ligand_fixed,ligand_moving,qc_fn):
    plt.subplot(3,1,1)
    plt.title('fixed (gray): {} {} {}'.format(fixed_order, tier_fixed, ligand_fixed))
    plt.imshow(fixed, cmap=plt.cm.gray )
    plt.imshow(moving, alpha=0.35, cmap=plt.cm.hot)
    plt.subplot(3,1,2)
    plt.title('moving (hot): {} {} {}'.format(moving_order, tier_moving, ligand_moving))
    plt.imshow(fixed, cmap=plt.cm.gray)
    plt.imshow(rsl, alpha=0.35, cmap=plt.cm.hot)
    plt.subplot(3,1,3)
    plt.title('fixed_rsl vs moving rsl')
    plt.imshow(fixed_rsl, cmap=plt.cm.gray)
    plt.imshow(final_rsl, alpha=0.35, cmap=plt.cm.hot)
    plt.tight_layout()
    plt.savefig( qc_fn )
    plt.clf()

def adjust_alignment(df,  epoch, y_idx, mid, transforms, step, output_dir,desc, shrink_factor,  smooth_sigma,iteration,tfm_type,clobber=False):
    if not os.path.exists(output_dir + "/qc/") : os.makedirs(output_dir + "/qc/")
    i=mid
    mi_list=[]
    n=len(y_idx)

    affine=np.array([[0.2, 0, 0, -72],
        [0,  0.02, 0, 0],
        [0, 0 , 0.2, -90],
        [0, 0, 0, 1]])
    if not os.path.exists(output_dir+"/qc"): os.makedirs(output_dir+"/qc")
    # Iterate over the sections along y-axis
    while i > 0 and i < len(y_idx)-1  :
        fixed_fn=df['filename_rsl'].iloc[i]
        
        if y_idx[i] < 500 : break
        #Find the next neighbouring section//
        if i % 10 == 0 : print(100.* np.round(i/n,3),end='\r')
        j_list = find_next_slice(i, step, df)
        
        #For neighbours
        for j in j_list :
            moving_fn=df['filename_rsl'].iloc[j]
            outprefix ="{}/init_transforms/{}_{}-{}/".format(output_dir,  y_idx[j], tfm_type, epoch) 
            if not os.path.exists(outprefix): os.makedirs(outprefix)
            tfm_fn = outprefix + "level-0_Mattes_{}_Composite.h5".format( tfm_type)
            inv_tfm_fn = outprefix + "level-0_Mattes_{}_InverseComposite.h5".format( tfm_type)
            moving_rsl_fn= outprefix + "_level-0_Mattes_{}.nii.gz".format( tfm_type)
            
            qc_fn = "{}/qc/{}_{}_{}_{}_{}_{}-{}.png".format(output_dir, *desc, y_idx[j], y_idx[i], tfm_type, epoch)
            if clobber or not os.path.exists(moving_rsl_fn) or not os.path.exists(tfm_fn)  or not os.path.exists(qc_fn)  :
                print(df['filename_rsl'].iloc[i])
                print('Moving:', moving_fn)
                print('Fixed:', fixed_fn)
                ANTs( tfm_prefix=outprefix,
                    fixed_fn=fixed_fn, 
                    moving_fn=moving_fn, 
                    moving_rsl_prefix=outprefix, 
                    iterations=[iteration], 
                    metrics=['Mattes'], 
                    tfm_type=[tfm_type], 
                    shrink_factors=[shrink_factor],
                    smoothing_sigmas=[smooth_sigma], 
                    init_tfm=transforms[ y_idx[j]]['fwd'],
                    sampling_method='Regular', sampling=1, verbose=0, generate_masks=False, clobber=True  )
                
                
            fixed_tfm_list=transforms[ y_idx[i]]['fwd']
            final_tfm_fn = outprefix + "final_level-0_Mattes_{}_Composite.h5".format( tfm_type)
            if not os.path.exists(final_tfm_fn) or clobber :
                if fixed_tfm_list != [] :
                    fixed_tfm_fn = fixed_tfm_list[-1]
                    shell(f'antsApplyTransforms -d 3 -i {moving_fn} -r {fixed_fn} -t {tfm_fn} -t {fixed_tfm_fn} -o Linear[{final_tfm_fn}]',False,True)
                    
                else :
                    shell(f'antsApplyTransforms -d 3 -i {moving_fn} -r {fixed_fn} -t {tfm_fn}  -o Linear[{final_tfm_fn}]',False,True)

            final_moving_rsl_fn= outprefix + "final_level-0_Mattes_{}-{}.nii.gz".format( tfm_type,epoch)
            if not os.path.exists(final_moving_rsl_fn) or clobber  :
                shell(f'antsApplyTransforms -v 1 -d 3 -i {moving_fn} -r {fixed_fn}  -t {final_tfm_fn} -o {final_moving_rsl_fn}')
            
            if not os.path.exists(qc_fn) :
                create_qc_image( load2d(fixed_fn), 
                            load2d(df['filename_rsl_new'].iloc[i]), 
                            load2d(moving_fn), 
                            load2d(moving_rsl_fn),
                            load2d(final_moving_rsl_fn), 
                            y_idx[i], y_idx[j],df['tier'].iloc[i], df['tier'].iloc[j], df['ligand'].iloc[i], df['ligand'].iloc[j], qc_fn)
            #   F       M
            #   | <fwd  |
            #   | inv>  |
            #   |       |
            transforms[ y_idx[j]]['inv'].append(inv_tfm_fn)
            transforms[ y_idx[j]]['fwd'].append(final_tfm_fn)
            df['filename_rsl_new'].iloc[j] = final_moving_rsl_fn
        i = j_list[-1]
    df.to_csv('{}/df_{}-{}.csv'.format(output_dir,tfm_type,epoch))
    return transforms,df

def concatenate_transforms(transforms,tfm_type,epoch):
    #Calculate combined transformation
    concat_transforms={}
    for i, tfm_list in transforms.items() :
        concat_tfm_fn ="{}/init_transforms/epoch-{}-{}/{}/concat_tfm.nii.gz".format(output_dir, tfm_type,epoch, i) 
        print(i, tfm_list)
        if len(tfm_list) > 0 : 
            if not os.path.exists(concat_tfm_fn) or clobber :
                shell('AverageImages 2 {} 0 {}'.format(concat_tfm_fn,' '.join(tfm_list) ))
            concat_transforms[i]=concat_tfm_fn 
    return concat_transforms

def apply_transforms_to_sections(df,transforms,output_dir) :
    temp_dir='{}/temp'.format(output_dir)
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    for i, row in df.iterrows() :
        fn = df['filename_rsl']
        i = df['global_order']
        tfm_fn = transforms[i]
        #out_fn='{}/'output_dir
        out_fn = "{}/{}_{}_{}_{}-{}_{}.nii.gz".format(temp_dir, *desc, tfm_type, epoch,i)
        shell('AntsApplyTransforms -i {} -t {} -r {} -o {}'.format( fn, tfm_fn, fn, out_fn  ))
        df['filename_rsl']=out_fn

def combine_sections_to_vol(df,z_mm,out_fn):

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
        if row['volume_order'] < 500 or row['volume_order'] > 700 :
            continue
        y = row['volume_order']
        ar = nib.load(row['filename_rsl']).get_fdata()
        ar=ar.reshape(ar.shape[0],ar.shape[1])
        vol[:,y,:]= ar

    
    xstep= z_mm/4164. * 10
    zstep= z_mm/4164. * 10

    print("\tWriting Volume",out_fn)
    slab_ymin=-126+df["volume_order"].min()*0.02 
    ystep = 0.02 
    affine=np.array([[xstep, 0, 0, -72],
                    [0,  ystep, 0, slab_ymin],
                    [0, 0 ,zstep, -90],
                    [0, 0, 0, 1]])
    nib.Nifti1Image(vol, affine ).to_filename(out_fn )

def receptorAdjustAlignment(init_align_fn, df, vol_fn_str, output_dir,scale, n_epochs=3, desc=(0,0,0), write_each_iteration=False, clobber=False):
    
    z_mm = scale[brain][hemi][str(slab)]["size"]
    y_idx = df['volume_order'].values 
    mid = np.rint(y_idx.shape[0] / 2.).astype(int)
    #Init dict with initial transforms
    transforms={}
    for i in y_idx :  transforms[i]={'fwd':[], 'inv':[] }

    parameters=[
            ('Rigid','2','1','250'),
            ('Rigid','1','0','100'),
            ('SyN','4','2','100'),
            ('SyN','2','1','50'),
            ('SyN','1','0','25')
            ]
    df['original_filename_rsl']=df['filename_rsl']

    for epoch, (tfm_type, shrink_factor, smooth_sigma,iterations) in enumerate(parameters) :
        out_fn = vol_fn_str.format(output_dir,*desc,tfm_type+'-'+str(epoch))
        df['filename_rsl_new']=df['filename_rsl']
        #if os.path.exists(out_fn) and not clobber : continue
        transforms,df = adjust_alignment(df,  epoch, y_idx, mid, transforms, -1, output_dir,desc, shrink_factor, smooth_sigma, iterations, tfm_type, clobber=clobber)
        #transforms,df = adjust_alignment(df, vol, epoch, y_idx, mid, transforms, 1, output_dir,desc, shrink_factor, smooth_sigma, iterations, tfm_type, clobber=clobber)
        df['filename_rsl']=df['filename_rsl_new']
        #concat_transforms= concatenate_transforms(transforms, tfm_type, epoch)
        vol = combine_sections_to_vol(df,z_mm, out_fn)
            
    

    initial_transforms_mod = { str(k):v for k, v in transforms.items() }
    
    with open(output_dir+os.sep+'transforms.json', 'w+') as fp : 
        json.dump(initial_transforms_mod, fp )

    
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
        tiers_string = "flum,musc,sr95,cgp5,afdx,uk20,pire,praz,keta,dpmg;sch2,dpat,rx82,kain,ly34,damp,mk80,ampa,uk18;oxot;epib"
    
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

    slab_img_fn_str = '{}/brain-{}_hemi-{}_slab-{}_init_align_{}.nii.gz'
    slab_img_0_fn = slab_img_fn_str.format(output_dir, brain,hemi,slab,0)

    df = setup_tiers(df, tiers_string)
    df.to_csv(output_dir+'/tiers.csv')
    receptorAdjustAlignment(init_align_fn, df, slab_img_fn_str, output_dir, scale, n_epochs=n_epochs, write_each_iteration=write_each_iteration,desc=(brain,hemi,slab), clobber=clobber)


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
    
    receptorRegister( brain, hemi, slab, init_align_fn, source_dir, output_dir, receptor_df_fn, n_epochs=n_epochs,  clobber=False)

