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

def find_next_slice( i , step, df, target_tier=-1):
    #Get the the tier of the current section
    current_tier = df["tier"].iloc[i]
    i2=i
    i2_list = []
    while i2 > 0 and i2 < df.shape[0] -1   :
        i2 = i2 + step
        i2_list.append(i2)
        if df["tier"].iloc[i2] <= current_tier and target_tier == -1 :
            break
        elif target_tier != -1 and df['tier'].iloc[i2] == target_tier:
            break
    return i2_list

def create_final_transform(outprefix, moving_fn, fixed_fn, tfm_fn, fixed_tfm_list, tfm_type, clobber=False) :
    final_tfm_fn = outprefix + "final_level-0_Mattes_{}_Composite.h5".format( tfm_type)
    if not os.path.exists(final_tfm_fn) or clobber :
        print(final_tfm_fn)
        if tfm_type == 'Rigid' :
            output_str = f'-o Linear[{final_tfm_fn}]'
        else :
            output_str = f'-o [{final_tfm_fn},1]'

        if fixed_tfm_list != [] :
            fixed_tfm_fn = fixed_tfm_list[-1]
            transforms_str=f'-t {tfm_fn} -t {fixed_tfm_fn}'
        else :
            transforms_str=f'-t {tfm_fn} '

        shell(f'antsApplyTransforms -v 1 -d 3 -i {moving_fn} -r {fixed_fn} {transforms_str} {output_str}',True,True)
    return final_tfm_fn

def get_image_points(fn):
    ar = nib.load(fn).get_fdata() 
    ar[ ar < threshold_otsu(ar) ] = 0
    sum0 = np.sum(ar, axis=1)
    sum1 = np.sum(ar, axis=0)
    np.argmax( sum0 > 0)
    np.argmax( sum0 > 0)


from skimage.exposure import  equalize_hist
from skimage.filters import threshold_otsu, threshold_li
from scipy.ndimage.filters import gaussian_filter
def align_by_points(im1_fn,im2_fn,rsl_fn,tfm_fn,qc_fn,max_features=1000):

    im1 = nib.load(im1_fn).get_fdata()#.astype(np.uint8)
    im2 = nib.load(im2_fn).get_fdata()#.astype(np.uint8)
    
    im1 = equalize_hist(gaussian_filter(im1,4))*255
    im2 = equalize_hist(gaussian_filter(im2,4))*255

    im2 = im2.astype(np.uint8)
    im1 = im1.astype(np.uint8)
    # Convert images to grayscale
    if len(im1.shape) > 2 :
        im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    else :
        im1Gray = im1

    if len(im2.shape) > 2 :
        im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    else : 
        im2Gray = im2

    # Detect ORB features and compute descriptors.
    #orb = cv2.ORB_create(max_features)
    #keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
    #keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)
    
    # Detect SIFT features and compute descriptors.
    #opencv-contrib-python==3.4.2.16
    s = cv2.xfeatures2d.SURF_create()
    #surf = cv2.xfeatures2d.SURF_create()
    keypoints1, descriptors1 = s.detectAndCompute(im1Gray, None)
    keypoints2, descriptors2 = s.detectAndCompute(im2Gray, None)

    # Match features.
    #matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    #matches = matcher.match(descriptors1, descriptors2, None)

    # BFMatcher with default params
    bf = cv2.BFMatcher()
    #matches = bf.knnMatch(keypoints1,keypoints2,k=2)
    matches = bf.match(descriptors1,descriptors2)
    #print(matches)
    # Sort matches by score
    #matches.sort(key=lambda x: x.distance, reverse=False)
    matches = sorted(matches, key = lambda x:x.distance)

    # Remove not so good matches
    matches = matches[0:3]

    # Draw top matches
    imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
    cv2.imwrite(qc_fn, imMatches)

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # Find homography
    #h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
    #h = cv2.estimateRigidTransform(points1,points2,False)
    #np.savetxt(tfm_fn,h)
    #height, width = im1.shape
    #im1Reg = cv2.warpAffine(im1, h, (width, height))
    #nib.Nifti1Image(im1Reg, generic_affine).to_filename(rsl_fn)
    #return im1Reg, h

def align_neighbours_to_fixed(i, j_list,df,transforms, iteration, shrink_factor,smooth_sigma, output_dir, y_idx, tfm_type, epoch,desc,clobber=False):
    #For neighbours
    fixed_fn=df['filename_rsl'].iloc[i]
    for j in j_list :
        moving_fn=df['filename_rsl'].iloc[j]
        outprefix ="{}/init_transforms/{}_{}-{}/".format(output_dir,  y_idx[j], tfm_type, epoch) 
        if not os.path.exists(outprefix): os.makedirs(outprefix)
        tfm_fn = outprefix + "level-0_Mattes_{}_Composite.h5".format( tfm_type)
        inv_tfm_fn = outprefix + "level-0_Mattes_{}_InverseComposite.h5".format( tfm_type)
        moving_rsl_fn= outprefix + "_level-0_Mattes_{}.nii.gz".format( tfm_type)
        
        qc_fn = "{}/qc/{}_{}_{}_{}_{}_{}-{}.png".format(output_dir, *desc, y_idx[j], y_idx[i], tfm_type, epoch)
        if clobber or not os.path.exists(moving_rsl_fn) or not os.path.exists(tfm_fn)  or not os.path.exists(qc_fn)  :
           
            print('Moving:',y_idx[j], moving_fn)
            print('Fixed:', y_idx[i],fixed_fn)
            ANTs(tfm_prefix=outprefix,
                fixed_fn=fixed_fn, 
                moving_fn=moving_fn, 
                moving_rsl_prefix=outprefix, 
                iterations=[iteration], 
                metrics=['Mattes'], 
                tfm_type=[tfm_type], 
                shrink_factors=[shrink_factor],
                smoothing_sigmas=[smooth_sigma], 
                init_tfm=None, #transforms[ y_idx[j]]['fwd'],
                dim=2,
                sampling_method='Regular', sampling=1, verbose=0, generate_masks=False, clobber=True  )
        
            create_qc_image( load2d(fixed_fn), 
                        None,
                        load2d(moving_fn), 
                        load2d(moving_rsl_fn),
                        None,
                        y_idx[i], y_idx[j],df['tier'].iloc[i], df['tier'].iloc[j], df['ligand'].iloc[i], df['ligand'].iloc[j], qc_fn)

            #   F       M
            #   | <fwd  |
            #   | inv>  |
            #   |       |
        fixed_tfm_list = transforms[y_idx[i] ]['lin']
        if tfm_type == 'Rigid' :
            transforms[ y_idx[j] ]['lin'] += [tfm_fn] + fixed_tfm_list 
        else : 
            if df['tier'].iloc[j] == 1 :
                transforms[ y_idx[i] ]['inv'].append(inv_tfm_fn)
            transforms[ y_idx[j] ]['fwd'].append(tfm_fn)  

    return df, transforms
                
def create_qc_image(fixed,fixed_rsl, moving,rsl,final_rsl, fixed_order, moving_order,tier_fixed,tier_moving,ligand_fixed,ligand_moving,qc_fn):
    plt.subplot(3,1,1)
    plt.title('fixed (gray): {} {} {}'.format(fixed_order, tier_fixed, ligand_fixed))
    plt.imshow(fixed, cmap=plt.cm.gray )
    plt.imshow(moving, alpha=0.35, cmap=plt.cm.hot)
    plt.subplot(3,1,2)
    plt.title('moving (hot): {} {} {}'.format(moving_order, tier_moving, ligand_moving))
    plt.imshow(fixed, cmap=plt.cm.gray)
    plt.imshow(rsl, alpha=0.35, cmap=plt.cm.hot)
    if final_rsl != None and fixed_rsl != None :
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


    if not os.path.exists(output_dir+"/qc"): os.makedirs(output_dir+"/qc")
    # Iterate over the sections along y-axis
    while i > 0 and i < len(y_idx)-1  :
        
        #if y_idx[i] < 500 : break
        #Find the next neighbouring section//
        if i % 10 == 0 : print(100.* np.round(i/n,3),end='\r')
       
        #If using Rigid transformation or using SyN and on a ligand with best image contrast (i.e. tier=1)
        if tfm_type=='Rigid' or df['tier'].iloc[i] == 1 :
            j_list = find_next_slice(i, step, df)
            i_next = j_list[-1]
        else :
            j_list = find_next_slice(i, step, df, target_tier=1)
            i_next = j_list[0]
            j_list = [j_list[-1]]
                        
        df, transforms = align_neighbours_to_fixed(i, j_list,df, transforms,iteration, shrink_factor,smooth_sigma, output_dir, y_idx, tfm_type, epoch,desc,clobber=clobber)

        i = i_next 


    df.to_csv('{}/df_{}-{}.csv'.format(output_dir,tfm_type,epoch))
    return transforms,df

def apply_transforms_to_sections(df,transforms,output_dir, tfm_type, epoch, clobber=False) :

    for i,(rowi, row) in enumerate(df.iterrows()) :
        outprefix ="{}/init_transforms/{}_{}-{}/".format(output_dir, row['volume_order'], tfm_type, epoch) 
        if tfm_type == 'Rigid' :
            fixed_tfm_list = transforms[ row['volume_order']]['lin']
            if len(fixed_tfm_list) == 0 : continue
            transforms_str=' -t {}'.format( ' -t '.join(fixed_tfm_list) ) 
        else :
            fwd_tfm_list = transforms[ row['volume_order']]['fwd']
            inv_tfm_list = transforms[ row['volume_order']]['inv']
            avg_tfm_fn = outprefix+'/averaged_fwd_inv_SyN.h5'
            
            if len(fwd_tfm_list) != epoch :
                #print('Error: too many/few non-linear transformation files.')
                #print('fwd',row['volume_order'],transforms[ row['volume_order']]['fwd'] )  
                continue

            if len(inv_tfm_list) != epoch :
                #print('Error: too many/few non-linear transformation files.')
                #print('inv',row['volume_order'],transforms[ row['volume_order']]['inv'] )  
                continue

            if not os.path.exists(avg_tfm_fn) or clobber :
                fwd_tfm_fn = fwd_tfm_list[-1]
                inv_tfm_fn = inv_tfm_list[-1]
                fwd_tfm = read_transform(fwd_tfm_fn) 
                inv_tfm = read_transform(inv_tfm_fn) 
                fwd_tfm.set_parameters(fwd_tfm.parameters*0.5 + inv_tfm.parameters*0.5)
                write_transform(fwd_tfm, avg_tfm_fn) 
            transforms_str = ' -t {}'.format(avg_tfm_fn)

        fixed_fn = row['filename_rsl']
        final_rsl_fn= outprefix + "final_level-0_Mattes_{}-{}.nii.gz".format( tfm_type,epoch)

        if not os.path.exists(final_rsl_fn) or clobber or True  :
            shell(f'antsApplyTransforms -v 0 -d 2 -i {fixed_fn} -r {fixed_fn}  {transforms_str} -o {final_rsl_fn}',False)
            df['filename_rsl_new'].iloc[i] = final_rsl_fn



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
        #if row['volume_order'] < 500 or row['volume_order'] > 700 :
        #    continue
        y = row['volume_order']
        ar = nib.load(row['filename_rsl']).get_fdata()
        ar=ar.reshape(ar.shape[0],ar.shape[1])
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

def receptorAdjustAlignment(init_align_fn, df, vol_fn_str, output_dir,scale, n_epochs=3, desc=(0,0,0), write_each_iteration=False, clobber=False):
    
    z_mm = scale[brain][hemi][str(slab)]["size"]
    y_idx = df['volume_order'].values 
    mid = np.rint(y_idx.shape[0] / 2.).astype(int)
    #Init dict with initial transforms
    transforms={}
    for i in y_idx :  transforms[i]={'lin':[],'fwd':[], 'inv':[] }

    parameters=[
            ('Rigid','4x2x1','2x1x0','1000x500x250'),
            ('SyN','2','1','1000'),
            ('SyN','2','1','1000'),
            ('SyN','1','0','500')
            ]
    df['original_filename_rsl']=df['filename_rsl']

    for epoch, (tfm_type, shrink_factor, smooth_sigma,iterations) in enumerate(parameters) :
        out_fn = vol_fn_str.format(output_dir,*desc,tfm_type+'-'+str(epoch))
        df['filename_rsl_new']=df['filename_rsl']
        #if os.path.exists(out_fn) and not clobber : continue
        print(tfm_type, shrink_factor, smooth_sigma,iterations)
        transforms,df = adjust_alignment(df,  epoch, y_idx, mid, transforms, -1, output_dir,desc, shrink_factor, smooth_sigma, iterations, tfm_type, clobber=clobber)
        transforms,df = adjust_alignment(df,  epoch, y_idx, mid, transforms, 1, output_dir,desc, shrink_factor, smooth_sigma, iterations, tfm_type, clobber=clobber)
        apply_transforms_to_sections(df,transforms,output_dir, tfm_type, epoch)
        df['filename_rsl']=df['filename_rsl_new']
        #concat_transforms= concatenate_transforms(transforms, tfm_type, epoch)
        if not os.path.exists(out_fn) or clobber :
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
        #tiers_string = "flum,musc,sr95,cgp5,afdx,uk20,pire,praz,keta,dpmg;sch2,dpat,rx82,kain,ly34,damp,mk80,ampa,uk18;oxot;epib"
        tiers_string = 'afdx,cgp5,damp,dpmg,flum,keta,ly34,musc,pire,praz,sr95,uk14;ampa,dpat,epib,kain,mk80,oxot,rx82,sch2'
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

