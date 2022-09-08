import pandas as pd
import os
from re import sub


ligand_order = {'AG-Nissl':0, 'oxot':1, 'afdx':2}
def create_data_frame(default_csv_fn, in_dir, crop_dir, out_csv_fn):
    
    if not os.path.exists(out_csv_fn) or True :
        df = pd.read_csv(default_csv_fn)

        df_list=[]
        for i, row in df.iterrows():
            raw=row['raw']
            ligand=row['ligand']
            L = float(row['L'])
            Kd = float(row['Kd'])
            Sa = float(row['Sa'])
            Cmax=100 #DEBUG
            conversion_factor=  Cmax / (255 * Sa) * (Kd + L) / L
            
            print(raw)

            order, brain, _, _, _, _, _ = raw.split('#') 
            hemisphere='B'

            raw=f'{in_dir}/{raw}'
            seg_fn=f'{crop_dir}/'+sub('.tif','_seg.nii.gz',os.path.basename(raw))
            print(raw)
            print(ligand, ligand_order[ligand])
            order = int(order) 
            if os.path.exists(raw) :
                df_list.append(pd.DataFrame({'raw':[raw], 'seg_fn':[seg_fn],'brain':[brain], 'hemisphere':[hemisphere], 'ligand':[ligand], 'order':[int(order)] }))
            else :
                print('Warning: skipping', raw)
        df = pd.concat(df_list)
        df['crop']=df['raw'].apply(lambda fn : sub('tif','nii.gz', fn) )
        df['slab']=[1]*df.shape[0]
        df['order'] = df['order'] - df['order'].min()
        df['global_order']=df['order']
        df['slab_order']=df['order']
        df['volume_order']=df['order']
        

        os.makedirs(os.path.dirname(out_csv_fn), exist_ok=True)
        df.to_csv(out_csv_fn,index=False)



