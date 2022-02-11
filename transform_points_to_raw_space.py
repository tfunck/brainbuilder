import ants
import numpy as np
import nibabel as nib
import shutil
import os
import pandas as pd
import matplotlib.pyplot as plt
import imageio
from sys import argv
from utils.utils import read_points, shell


def get_tfm(slab, y, clobber=False):
    tfm_fn=f'/data/receptor/human/output_2/MR1_R_{slab}/1_init_align/stage_*/init_transforms/{y}_Rigid-0/level-0_Mattes_Rigid_Composite_Concatenated.h5'

    local_tfm_fn=f'manual_points/{y}_{os.path.basename(tfm_fn)}'
    #local_tfm_fn=f'manual_points/{y}_Mattes_Rigid_Composite.h5'

    if not os.path.exists(local_tfm_fn) or clobber :
        shell(f'jud {tfm_fn}')
        shutil.move(os.path.basename(tfm_fn), local_tfm_fn )

    return local_tfm_fn

def get_raw(raw_fn, local_raw_fn, clobber=False):

    if not os.path.exists(local_raw_fn) or clobber:
        shell(f'jud {raw_fn}')
        #shutil.move( os.path.basename(raw_fn) , local_raw_fn )
        imageio.imwrite( local_raw_fn, imageio.imread( os.path.basename(raw_fn)) ) 
        os.remove(os.path.basename(raw_fn))

    return local_raw_fn


def get_slab(brain, hemisphere, slab, clobber=False):
    init_fn=f'/data/receptor/human/output_2/{brain}_{hemisphere}_{slab}/1_init_align/{brain}_{hemisphere}_{slab}_init_align.nii.gz'
    local_init_fn = f'temp/{brain}_{hemisphere}_{slab}_init_align.nii.gz'
    
    if not os.path.exists(local_init_fn) or clobber :
        shell(f'jud {init_fn}')
        shutil.move(os.path.basename(init_fn), local_init_fn)

    return local_init_fn


def get_points(brain, hemisphere, slab, init_align_img, init_align_vol, rec_points_vox, mri_points, df, points_df, clobber=False):

    for i in range(rec_points_vox.shape[0]):
        
        x, y, z = rec_points_vox[i,:].astype(int)
        xw,yw,zw= mri_points[i,:]
        print(brain, hemisphere, slab, y)
        source_fn = f'manual_points/{brain}_{hemisphere}_{slab}_src_{i}_{y}.png'
        target_fn = f'manual_points/{brain}_{hemisphere}_{slab}_tgt_{i}_{y}.png'
        lin_fn = df['lin_fn'].loc[ y == df['volume_order'] ].values[0]

        idx = (points_df['brain']==brain) & (points_df['hemisphere']==hemisphere) & (points_df['slab']==slab) & (points_df['point']==i)

        if points_df.loc[ idx ].shape[0] == 0 or not os.path.exists(source_fn) or not os.path.exists(target_fn)   :
            print(brain, hemisphere, slab, i, y)

            assert y in df['volume_order'].values, f'Error: could not find {y} in {brain} {hemisphere} {slab}'

            get_raw(lin_fn, source_fn, clobber=clobber )

            if type(init_align_vol) != type(np.array([])) :
                init_align_vol = init_align_img.get_fdata()

            section = init_align_vol[:,y,:]

            plt.imshow(section)
            plt.scatter(z, x,c='r') 
            plt.savefig(target_fn)
            plt.cla(); plt.clf()

            row=pd.DataFrame({'lin':[lin_fn],'source':[source_fn],'target':[target_fn],'brain':[brain],'hemisphere':[hemisphere], 'slab':[slab],'point':[i],'x':[xw], 'y':[yw], 'z':[zw]})
            points_df = points_df.append(row)
        
    return points_df, init_align_vol


def create_points(brain, hemisphere, df, points_df, clobber=False):

    for slab in range(1,7) :
        slab_df=df.loc[df['slab']==slab]
        # get init align slab
        local_init_fn = get_slab(brain, hemisphere, slab, clobber=clobber)

        init_align_img = nib.load(local_init_fn)
        init_align_vol = None 

        points_fn = f'manual_points/3d/MR1_R_{slab}_points.txt'
        img = nib.load(local_init_fn)

        affine = img.affine

        rec_points, mri_points, rec_file, mri_file = read_points(points_fn)
        rec_points_vox = np.zeros_like(rec_points)
        print(local_init_fn) 
        for i in range(3) :
            rec_points_vox[:,i] = np.rint( (rec_points[:,i] - affine[i,3]) / affine[i,i] ).astype(int)

        points_df, init_align_vol = get_points(brain, hemisphere, slab, init_align_img,  init_align_vol, rec_points_vox, mri_points, slab_df, points_df)

    return points_df 

df_csv='autoradiograph_info_volume_order.csv'
df = pd.read_csv(df_csv)
hemisphere='R'
brain='MR1'
df = df.loc[ (df['hemisphere'] == hemisphere) &
        (df['mri'] == brain) ]



points_csv_fn='manual_points/points.csv'

if not os.path.exists(points_csv_fn):
    points_df = pd.DataFrame({'brain':[],'hemisphere':[],'slab':[],'point':[],'lin':[],'target':[],'x':[], 'y':[], 'z':[]})
else :
    points_df = pd.read_csv(points_csv_fn)

points_df = create_points(brain, hemisphere, df, points_df, clobber=False)

points_df.to_csv(points_csv_fn, index=False)

for i, row in points_df.iterrows() :
    uniq_values = np.unique(imageio.imread(row['source']))
    if len(uniq_values) != 2 :
        source_fn = row['source']
        target_fn = row['target']
        print(source_fn, target_fn)
        shell( f'\tgimp {source_fn} {target_fn}' )
    



