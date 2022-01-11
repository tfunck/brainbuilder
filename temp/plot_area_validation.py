import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import utils.ants_nibabel as nib
#import nibabel as nib
import os
from glob import glob

def create_error_volumes(df, qc_dir):


    for (slab, source, ligand), sdf in df.groupby( ['slab', 'source', 'ligand'] ) : 
        space = 'mni' if source == 'reconstructed' else 'nl2d'

        atlas_rsl_str = f'{qc_dir}/MR1_R_{int(slab)}_{ligand}_pseudo-cls*space-{space}.nii.gz'
        atlas_rsl_list  = glob(atlas_rsl_str)
        if len(atlas_rsl_list) == 1 : atlas_rsl_fn = atlas_rsl_list[0]
        else : 
            print('Skipping', slab, source, ligand)
            continue
        print('\tLabel volume:', atlas_rsl_fn)
        img = nib.load(atlas_rsl_fn)
        atlas = img.get_fdata().astype(int)

        unique_labels = np.unique(atlas)
        out_vol = np.zeros(atlas.shape)
        for i, row in sdf.iterrows() :
            label = int(row['label'])
            idx =  atlas == int(label)
            label_sum = np.sum(idx)
            assert np.sum(idx) >  0, 'Error, label {label} not found in label volume'
            average = np.abs(100- row['accuracy'])
            out_vol[ idx ] = average
        assert np.sum(out_vol) > 0 , 'Error: empty output volume'

        out_fn = f'./MR1_R_{slab}_{ligand}_val-error_space-{space}.nii.gz'
        print('\tError volume:', out_fn)
        nib.Nifti1Image(out_vol.astype(np.float32), img.affine).to_filename(out_fn)

def norm(df) :

    df_target = df.loc[ (df['source'] == 'nl2d') | (df['source'] == 'reconstructed') ]
    df_orig = df.loc[ df['source'] == 'original' ]

    df_out = pd.DataFrame({  'source':[], 
                         'slab':[],
                         'y':[],
                         'label':[],
                         'volume':[], 
                         'accuracy':[] })

    for i, row in df_target.iterrows() :
        label = row['label']
        orig_row = df_orig.loc[ df_orig['label'] == label ] 
        orig_average = orig_row['average']
        
        df_out=df_out.append(pd.DataFrame({ 'source':row['source'], 
                   'label':row['label'],
                   'slab':row["slab"],
                   'ligand':row["ligand"],
                   'y':row["y"],
                   'volume':orig_row['volume'], 
                   'accuracy':100 * row['average'] / orig_row['average'] }) 
                   )
    print(df_out)
    return df_out


def plot(df, out_fn='validation_accuracy.png') :
    nl2d_i = df['source']=='nl2d'
    rec_i = df['source']=='reconstructed'

    y0 = [ df['accuracy'].loc[df['source']=='nl2d'].mean() ] * 2
    y1 = [ df['accuracy'].loc[df['source']=='reconstructed'].mean() ] * 2

    x0=[0, df['volume'].max() ]

    fig, axes = plt.subplots(1, 2, figsize=(16,10))
    #ax[.title('Accuracy after 2d Alignment')
    axes[0].set_title('Accuracy after 2D Alignement')
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].set_xlabel('ROI Volume (mm2)')
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)

    axes[1].set_title('Accuracy after 3D Reconstruction')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_xlabel('ROI Volume (mm2)')
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)


    sns.scatterplot(x='volume', y='accuracy', hue='slab', data=df.loc[nl2d_i], ax=axes[0], palette='tab10') 
    sns.scatterplot(x='volume', y='accuracy', hue='slab', data=df.loc[rec_i], ax=axes[1], palette='tab10') 
    plt.tight_layout()
    plt.savefig(out_fn)


def validation_visualization(qc_fn):
    qc_dir = os.path.dirname(qc_fn)
    df = pd.read_csv(qc_fn)
    df = norm(df)
    plot(df)
    create_error_volumes(df, qc_dir)

if __name__ == '__main__' :
    validation_visualization('./reconstruction_validation.csv')


