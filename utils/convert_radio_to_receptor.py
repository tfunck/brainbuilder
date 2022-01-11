
import utils.ants_nibabel as nib
#import nibabel as nib
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import os
from re import sub
from glob import glob

def get_cmax(fn) :
    fn_list = glob('trans_tabs/*/{}'.format(fn.lower())) + glob('trans_tabs/*/{}'.format(fn))
    try :
        fn = fn_list[0]
    except IndexError :
        print('Warning: could not find .grt ','trans_tabs/*/{}'.format(fn) )
        return -1
    with open(fn,'r') as F :
        for cbe in F.readlines() :
            if 'Cmax' in cbe :
                return float(cbe.rstrip().split(' ')[-1])
    print('Error Cmax not found in ', fn)
    exit(1)

info= pd.read_csv('section_numbers/experimental_info.csv')
auto_info = pd.read_csv('section_numbers/autoradiograph_info.csv')
info['Image']=info['Image'].apply(lambda x: x.lower())

auto_info['conversion_factor']=[0.] * auto_info.shape[0]

#for fn in glob('reconstruction_output/0_crop/*nii.gz') :
for i, auto_row in auto_info.iterrows():
    slab = auto_row['slab']
    hemi = auto_row['hemisphere']
    ligand = auto_row['ligand']
    sheet = auto_row['sheet']
    repeat = auto_row['repeat']
    image = sub('.TIF', '', sub('#L.TIF','',os.path.basename(auto_row['lin_fn'])  ))
    info_row = info.loc[ info['Image'] == image.lower() ]

    if np.sum( info['Image'] == image.lower() )  == 0 : 
        print(image)
        continue

    Sa = float(info_row['SA'].values[0])
    Kd = float(info_row['KD'].values[0])
    L = float(info_row['l'].values[0])
    cmax_fn = info_row['cmax_in_file'].values[0]
    Cmax = get_cmax(cmax_fn) 
    if Cmax == - 1 : continue #Cmax == -1 means that the .grt file containing cmax was not found

    conversion_factor = Cmax / (255 * Sa) * (Kd + L) / L
    print('\t', image, '->', conversion_factor)
    assert conversion_factor > 0 , f'Error : conversion factor < 0. {image}'

    #auto_info['conversion_factor'].loc[ (auto_info['hemisphere']==hemi) & (auto_info['ligand']==ligand) & (auto_info['sheet']==int(sheet)) & (auto_info['repeat'].astype(str)==str(repeat))  & (auto_info['slab']==int(slab)) ] = conversion_factor
    auto_info['conversion_factor'].iloc[i] = conversion_factor
    print(auto_info.iloc[i])
print(auto_info)
auto_info.to_csv('section_numbers/autoradiograph_info.csv')

