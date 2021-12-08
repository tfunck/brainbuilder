import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import nibabel as nib



def create_error_volume(atlas_rsl_fn, df, out_dir='./'):
    img = nib.load(atlas_rsl_fn)
    atlas = img.get_fdata().astype(int)

    for source, sdf in df.groupby(['Processing Stage']) :
        print('\n\n',source,  source != 'Autoradiographs')
        if source != 'Autoradiographs' :
            out_vol = np.zeros_like(atlas)

            for i, row in sdf.iterrows() :
                label = row['Atlas']
                average =np.abs(100- row['Accuracy'])
                print(row)
                out_vol[ atlas == int(label) ] = average

            nib.Nifti1Image(out_vol.astype(np.float32), img.affine).to_filename(f'{out_dir}/{source}.nii.gz')



df = pd.read_csv('reconstruction_validation.csv')
df.drop_duplicates(inplace=True)    
df = df.loc[df['label'] != 203 ]
df_list=[]

for (label, ligand, source), df2 in  df.groupby(['label','ligand','source']):
    total = df2['n'].sum()
    weights = df2['n'] / total
    weighted_avg = np.sum(df2['average'] * weights)
    df_temp = pd.DataFrame({'label':[label], 'ligand':[ligand], 'source':[source], 'average':[weighted_avg]})
    df_list.append( df_temp )

df = pd.concat(df_list)

df = pd.pivot_table(df, index=['ligand','label'], columns=['source'], values=['average'] )
df.reset_index(inplace=True)
df.dropna(inplace=True)
col0 = df.columns.get_level_values(0)
col1 = df.columns.get_level_values(1)
columns = [ col1[i] if col1[i] != '' else col0[i] for i in range(len(df.columns)) ]
df.columns = columns

df['nl2d'] = df['nl2d'] / df['original'] * 100.
df['reconstructed'] = df['reconstructed'] / df['original'] * 100.
df['original'] = df['original'] / df['original'] * 100.
df = pd.melt(df, id_vars=['ligand','label'], value_vars=['nl2d','original','reconstructed'], var_name='Processing Stage', value_name='Accuracy')

df['label'] = df['label'].astype(int)
df = df.rename(columns={'label':'Atlas', 'nl2d':'Aligned','reconstructed':'Reconstructed' })
df['Processing Stage'].loc[ df['Processing Stage'] == 'original'  ] = 'Autoradiographs'
df['Processing Stage'].loc[ df['Processing Stage'] == 'nl2d'  ] = 'Aligned'
df['Processing Stage'].loc[ df['Processing Stage'] == 'reconstructed'  ]='Reconstructed'

print(df)
plt.cla()
plt.clf()

# Set up a grid to plot survival probability against several variables
sns.set(font_scale=1.5)
#g = sns.catplot(data=df, y="Accuracy", col='Atlas', col_wrap=5, height=15, kind='point',
#                  x="Processing Stage", order=['Autoradiographs', 'Aligned','Reconstructed'])
g = sns.catplot(data=df, x="n", y="Accuracy", col='Processing Stage', height=15, kind='point') #"Processing Stage", order=['Autoradiographs', 'Aligned','Reconstructed'])
g.set(ylim=(0, None))

 # Draw a seaborn pointplot onto each Axes
plt.savefig('validation_accuracy_n.png')


create_error_volume('dka.nii.gz', df)
#create_error_volume('JuBrain_Map_v30_seg_1.0mm.nii', df)



