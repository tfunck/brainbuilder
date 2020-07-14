import nibabel as nib
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import json
import shutil
import sys
from glob import glob
from utils.utils import set_csv, add_padding, setup_tiers
from ants import registration, apply_transforms, image_mutual_information, from_numpy, image_similarity


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


def create_qc_image(fixed,moving,rsl, fixed_order, moving_order, qc_fn):
    plt.subplot(2,1,1)
    plt.title('fixed: '+str(fixed_order))
    plt.imshow(fixed, cmap=plt.cm.gray )
    plt.imshow(moving, alpha=0.35, cmap=plt.cm.hot)
    plt.subplot(2,1,2)
    plt.title('moving: '+ str(moving_order))
    plt.imshow(fixed, cmap=plt.cm.gray)
    plt.imshow(rsl, alpha=0.35, cmap=plt.cm.hot)
    plt.tight_layout()
    plt.savefig( qc_fn )
    plt.clf()

def adjust_alignment(df, vol, epoch, y_idx, mid, initial_transforms, step, output_dir,clobber=False):
    if not os.path.exists(output_dir + "/qc/") : os.makedirs(output_dir + "/qc/")
    i=mid
    mi_list=[]

    # Iterate over the sections along y-axis
    while i > 0 and i < len(y_idx)-1  :
        #Find the next neighbouring section//

        j_list = find_next_slice(i, step, df)
        #Set the current section to <fixed>
        fixed = from_numpy(vol[:,y_idx[i],:].T, spacing=(0.2,0.2))
        #print("I :", i, "J List: ", j_list)
        
        #for neighbours
        for j in j_list :
            outprefix = output_dir + "/transforms/epoch-"+str(epoch)+"/"+str(y_idx[j])+'/' 
            if not os.path.exists(outprefix): os.makedirs(outprefix)
            if not os.path.exists(output_dir+"/qc"): os.makedirs(output_dir+"/qc")
            tfm_fn = outprefix + "0GenericAffine.mat"
            qc_fn = output_dir + "/qc/"+str(y_idx[j])+"_"+str(y_idx[i])+"_"+str(epoch)+".png"
            
            #Set the section which is being moved to the fixed image 
            moving= from_numpy(vol[:,y_idx[j],:].T, spacing=(0.2,0.2))

            if clobber or  not os.path.exists(tfm_fn) : # or not os.path.exists(qc_fn)  :
                out = registration(fixed, moving, initial_transforms=initial_transforms[y_idx[j]], outprefix=outprefix, type_of_transform='DenseRigid', aff_metric='mattes', verbose=False)
               
                rsl = out['warpedmovout'].numpy().T
                del out
            else :
                init =  [ tfm_fn ] 
                rsl = apply_transforms(moving, moving, init, interpolator='linear').numpy().T

            if not os.path.exists(qc_fn) or clobber :
                create_qc_image( vol[:,y_idx[i],:], vol[:,y_idx[j],:], rsl, y_idx[i], y_idx[j], qc_fn)

            rsl_ants = from_numpy(rsl, spacing=(0.2,0.2))
            #mi_list.append( image_similarity(fixed, rsl_ants, metric_type='MattesMutualInformation' , sampling_strategy='regular', sampling_percentage=0.8 ))
            mi_list.append(0)

            initial_transforms[y_idx[j]] = [tfm_fn] + initial_transforms[y_idx[j]] 
            vol[:,y_idx[j],:] = rsl
        del fixed
        del moving
        del rsl
        del rsl_ants

        i = j_list[-1]
    

    return mi_list, vol 

def receptorAdjustAlignment(df, vol_fn, output_dir, n_epochs=3, write_each_iteration=False, clobber=False):
    
    vol_nii = nib.load(vol_fn)
    vol = vol_nii.get_data()
    mi_fn = output_dir + os.sep + 'mi.csv'

    y_idx = df['volume_order'].values 
    mid = np.rint(y_idx.shape[0] / 2.).astype(int)
    epochs=np.arange(0,n_epochs).astype(int)+1
    mi_list=[]
    #Init dict with initial transforms
    initial_transforms={}
    for i in y_idx :  
        initial_transforms[i]=[]

    for epoch in epochs :
        print("Epoch:", epoch)
        mi = 0
        mi_list_0, vol = adjust_alignment(df, vol, epoch, y_idx, mid, initial_transforms, -1, output_dir,clobber=clobber)
        mi_list_1, vol = adjust_alignment(df, vol, epoch, y_idx, mid, initial_transforms,  1, output_dir,clobber=clobber)
        if write_each_iteration :
            nib.Nifti1Image(vol, vol_nii.affine ).to_filename(output_dir+os.sep+'vol_'+str(epoch)+'.nii.gz')
    
        mi_list.append(sum(mi_list_0+mi_list_1))
        plt.clf()
        plt.plot(range(1,len(mi_list)+1), mi_list)
        plt.scatter(range(1,len(mi_list)+1), mi_list, c='r')
        plt.savefig(output_dir+os.sep+'mi.png')

    shutil.copy(output_dir+os.sep+"vol_"+str(epochs[-1])+".nii.gz", output_dir+os.sep+"vol_final.nii.gz")
    print(output_dir+os.sep+'transforms.json')
    initial_transforms_mod = { str(k):v for k, v in initial_transforms.items() }
    
    with open(output_dir+os.sep+'transforms.json', 'w+') as fp : 
        json.dump(initial_transforms_mod, fp )

def init_volume(slab_img_fn,  df, scale, ext=".nii.gz", clobber=False, align=True):
    example_fn=df["filename"].values[0]
    print(example_fn)
    shape = nib.load(example_fn).get_shape()
    xmax = shape[0] 
    zmax = shape[1]
    order_max=df["order"].max()
    order_min=df["order"].min()  
    slab_ymax=order_max-order_min + 1

    z0=x0=0
    for i, row in df.iterrows() :
        shape =  nib.load(row["filename"]).get_data().shape
        x0 = max([x0,shape[0]])
        z0 = max([z0,shape[1]])

    vol = np.zeros( [x0,slab_ymax,z0])

    for i, row in df.iterrows() :
        vorder = row["volume_order"]
        _file = row["filename"] 

        ligand = row["ligand"]

        slab = row["slab"]
        hemi = row["hemisphere"]
        mr = row["mri"]
        direction = scale[mr][hemi][str(slab)]["direction"]
        temp =  nib.load(_file).get_data()
        temp = temp.reshape(*temp.shape[0:2])
        if direction == "rostral_to_caudal":
            temp = np.flip(temp, 0)
            temp = np.flip(temp, 1)
        elif direction == "caudal_to_rostral":
            temp = np.flip(temp, 0)
        temp = add_padding(temp, x0,z0) 

        vol[ : , vorder , : ] = temp


    #if direction == "caudal_to_rostral":
    #    vol = np.flip(vol, 1)
    
    print("\tWriting",slab_img_fn)
    slab_ymin=-126+df["order"].min()*0.02
    affine=np.array([[0.2, 0, 0, -72],
                    [0, 0.02, 0, slab_ymin],
                    [0,0,0.2, -90],
                    [0, 0, 0, 1]])
    img = nib.Nifti1Image(vol, affine)   
    img.to_filename(slab_img_fn)

    
def add_y_position(df):
    df["volume_order"]=[0]*df.shape[0]
    order_max=df["order"].max()
    for i, row in df.iterrows() :
        df["volume_order"].loc[ row["order"] == df["order"]] = order_max - row["order"]
    return df

def receptorRegister(brain,hemi,slab, init_align_fn, output_dir, receptor_df_fn, tiers_string=None, scale_factors_json="data/scale_factors.json", n_epochs=3, write_each_iteration=False, clobber=False):
    print('receptor register')
    if not os.path.exists(output_dir) :
        os.makedirs(output_dir)

    if tiers_string==None:
        tiers_string = "flum,musc,sr95,cgp5,afdx,uk20,pire,praz,keta,dpmg;sch2,dpat,rx82,kain,ly34,damp,mk80,ampa,uk18;oxot;epib"
    
    with open(scale_factors_json) as f : scale=json.load(f)
    df = pd.read_csv(receptor_df_fn)
    df = df.loc[ (df.brain==brain) & (df.slab==slab) & (df.hemi==hemi),:]
    df = add_y_position(df)
    df.sort_values(["volume_order"], inplace=True)

    slab_img_fn = output_dir + os.sep + 'vol_0.nii.gz'
    if not os.path.exists(slab_img_fn) or clobber :
        init_volume(slab_img_fn,  df, scale, ext=".nii.gz", clobber=clobber)

    df = setup_tiers(df, tiers_string)
    print(df)
    exit(0)
    receptorAdjustAlignment(df, slab_img_fn, output_dir, n_epochs=n_epochs, write_each_iteration=write_each_itertion, clobber=clobber)


if __name__ == '__main__' :
    print(sys.argv)
    brain = sys.argv[1]
    hemi = sys.argv[2]
    slab = sys.argv[3] 
    init_align_fn = sys.argv[4]
    output_dir = sys.argv[5]
    receptor_df_fn = sys.argv[6]
    
    receptorRegister( brain, hemi, slab, init_align_fn, output_dir, receptor_df_fn, n_epochs=4, write_each_iteration=True, clobber=True)

