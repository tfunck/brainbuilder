import os
import re
import pandas as pd
from sys import argv
from glob import glob
import numpy as np


def split_filename(fn):
    fn2b = re.sub(r"([0-9])s([0-9])", r"\1#\2", fn, flags=re.IGNORECASE)
    fn2c = re.split('#|\.|\/|\_',fn2b)
    if len(fn2c) < 4 : return [np.nan]*6 

    return fn2c[2:8]

if __name__ == "__main__":
    section_info_list=[]
    csv_list = glob("MR*_*_slab_*_section_numbers.csv")
    source_dir = argv[1]

    for csv in csv_list :
        df0=pd.read_csv(csv)
        section_info_list.append( pd.DataFrame({'base':df0['name'],'slab_order':df0['number']})  )

    section_info = pd.concat(section_info_list)
    base_split = np.array(list( section_info['base'].apply(split_filename).values ) )
    section_info['mri'] = base_split[:,0]
    section_info['slab'] = base_split[:,1]
    section_info['hemisphere'] = base_split[:,2]
    section_info['ligand'] = base_split[:,3]
    section_info['sheet'] = base_split[:,4]
    section_info['repeat'] = base_split[:,5]
    section_info = section_info.loc[ section_info['mri'] != 'nan' ]
    section_info['slab']=section_info['slab'].astype(int)
    section_info["global_order"] =section_info["slab_order"].astype(int)

    section_info_unique = np.sort(section_info["slab"].unique())
    for i in section_info_unique[1:] :
        prev_slab = i - 1
        prev_slab_max = section_info["global_order"].loc[ section_info["slab"] == prev_slab ].max() 
        section_info["global_order"].loc[ section_info["slab"] == i ] += prev_slab_max

    section_info['lin_fn'] = ['nan'] * section_info.shape[0]
    for i, (slab_i, row) in enumerate(section_info.iterrows()) :
        ll = source_dir+'/*/*/*'+row['base']+'*'
        fn = glob(ll)
        if len(fn) == 1 : section_info['lin_fn'].iloc[i] = fn[0]
    section_info = section_info.loc[ section_info['lin_fn'] != 'nan' ]

    section_info.to_csv('autoradiograph_info.csv')

