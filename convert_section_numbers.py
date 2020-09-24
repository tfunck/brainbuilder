import os
import re
import pandas as pd
from sys import argv
from glob import glob
import numpy as np

base_dir = "/data1/users/tfunck/receptor/" #MR3/img_lin/L_slab_5/R
def fix_lin_fn(fn):
    
    if not os.path.exists(fn) :
        l = glob('%s/**/*%s*'%(base_dir,os.path.basename(fn)),recursive=True)
        if len(l) != 0 :
            fn = l[0]
        else :
            fn = 'nan'
            

    return fn

def split_filename(fn):
    print(fn)
    fn = os.path.basename(fn)
    fn2b = re.sub(r"([0-9])s([0-9])", r"\1#\2", fn, flags=re.IGNORECASE)
    fn2c = re.split('#|\.|\/|\_',fn2b)
    if len(fn2c) < 4 : return [np.nan]*6 
    out = fn2c[2:8]
    out.append( '#'.join(fn2c[0:8]) )
    if len(out) != 7 : 
        out.append(np.nan)
    return out

if __name__ == "__main__":
    section_info_list=[]
    csv_list = glob("MR*_*_slab_*_section_numbers.csv")
    source_dir = argv[1]

    for i, csv in enumerate(csv_list) :
        df0=pd.read_csv(csv)
        tdf = pd.DataFrame({'lin_fn':df0['name'],'slab_order':df0['number']})
        section_info_list.append( tdf  )
    print(section_info_list)

    section_info = pd.concat(section_info_list)
    section_info['lin_fn'] = section_info['lin_fn'].apply(fix_lin_fn)
    section_info = section_info.loc[  section_info['lin_fn'] != 'nan' ]
    base_split = list( section_info['lin_fn'].apply(split_filename).values )#.reshape(section_info.shape[0],-1)
    print(section_info)
    print(base_split)
    for i, l in enumerate(base_split) :
        if len(l) < 7 :
            base_split[i] += [np.nan]*(7-len(l))
    base_split = np.array(base_split)
    section_info['mri'] = base_split[:,0]
    section_info['slab'] = base_split[:,1]
    section_info['hemisphere'] = base_split[:,2]
    section_info['ligand'] = base_split[:,3]
    section_info['sheet'] = base_split[:,4]
    section_info['repeat'] = base_split[:,5]
    section_info['base'] = base_split[:,6]
    section_info = section_info.loc[ section_info['mri'] != 'nan' ]
    section_info['slab']=section_info['slab'].astype(int)
    section_info["global_order"] =section_info["slab_order"].astype(int)

    section_info_unique = np.sort(section_info["slab"].unique())
    for (mr,hemisphere), tdf in section_info.groupby(['mri','hemisphere']) : 
        print(mr, hemisphere)
        for i in section_info_unique[1:] :
            prev_slab = i - 1
            idx = (tdf["slab"] == prev_slab) & (tdf['mri']==mr) & (tdf['hemisphere']==hemisphere)
            prev_slab_max = tdf["global_order"].loc[ idx ].max() 
            section_info["global_order"].loc[ (section_info["slab"] == prev_slab) & (section_info['mri']==mr) & (section_info['hemisphere']==hemisphere) ] += prev_slab_max

    #for i, (slab_i, row) in enumerate(section_info.iterrows()) :
    #    ll = source_dir+'/*/*/*'+row['base']+'*'
    #    fn = glob(ll)
    #    if len(fn) == 1 : section_info['lin_fn'].iloc[i] = fn[0]
    section_info = section_info.loc[ section_info['lin_fn'] != 'nan' ]
    section_info.to_csv('autoradiograph_info.csv')

