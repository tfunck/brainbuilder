import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from neuroCombat import neuroCombat,neuroCombatFromTraining

from skimage.filters import threshold_otsu, threshold_li
import pandas as pd
import numpy as np
import nibabel as nib
import os
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.interpolate import interp1d
from re import sub


def get_slab_based_functions(input_data,output_data,slabs):
    functions={}
    for slab in np.unique(slabs):
        functions[int(slab)] =  lowess_correction_function(input_data[:,slabs==slab].ravel(),
                                                 output_data[:,slabs==slab].ravel())
    return functions

def lowess_correction_function(input_centiles,output_centiles):
    lowess_tight = lowess(output_centiles, input_centiles, frac = .12)
    f_linear = interp1d(lowess_tight[:,0], 
                        y=lowess_tight[:,1], 
                        bounds_error=False, kind='linear', fill_value='extrapolate') 
    return f_linear


def extract_centiles_from_slab(df, slab_file_dict, n_centiles=10):

    # create centiles for 0 to 1 
    centiles = np.linspace(0,1,n_centiles+2)[1:-1]
    # data will contain centile distribution from the data
    data=np.zeros((n_centiles,df.shape[0]))
    slab_list = []
    ligand_list = []
    y_list = []
    si=0

    global_order_min = df['global_order'].min()
    for slab, slab_file in slab_file_dict.items() : 
        img = nib.load(slab_file)
        vol = img.get_fdata()
        slab_df = df.loc[ df['slab'].astype(int) == int(slab) ]
        for i, row in slab_df.iterrows() :
            y_global = np.float32(row['global_order'])
            y_volume = np.float32(row['volume_order']) 
            
            slab_list.append(slab)
            #ligand_list.append(str(row['ligand']))
            y_list.append(y_global)
            # save centile for section
            #section = nib.load(row['nl_2d_rsl']).get_fdata() * row['conversion_factor']
            section = vol[:,int(y_volume),:]
            #mask = nib.load(row['nl_2d_cls_rsl']).get_fdata()
            mask = np.zeros_like(section)
            mask[ section >= threshold_otsu(section)] = 1
            section_values = section[mask >= .9]
            data[:,si] = np.percentile(section_values,100*centiles)
            si += 1
    #combat_df = pd.DataFrame({'slab':slab_list, 'y':y_list, 'ligand':ligand_list})

    combat_df = pd.DataFrame({'slab':slab_list, 'y':y_list })
    combat_df['y'] = combat_df['y'].astype(float)

    return data, combat_df

def combat_apply(slab_file_dict, out_file_dict, interp_functions):
    
    for slab, slab_vol_file in slab_file_dict.items()  :
        function = interp_functions[int(slab)]
        out_file = out_file_dict[str(slab)]
        receptor_img = nib.load(slab_vol_file)
        receptor_vol = receptor_img.get_fdata()

        corrected_receptor_vol = function(receptor_vol)

        corrected_receptor_vol[ pd.isnull(corrected_receptor_vol) ] = 0

        nib.Nifti1Image(corrected_receptor_vol, receptor_img.affine).to_filename(out_file)




    return out_file_dict

def get_output_dict(slab_file_dict) :
    out_file_dict = {}

    for slab, slab_vol_file in slab_file_dict.items()  :
        out_file = sub('.nii','_harmonized.nii', slab_vol_file)
        out_file_dict[ str(slab) ] = out_file

    return out_file_dict


def create_qc_dataframe(data, covars, tag):
    slab_list = []
    y_list = []
    values_list = []
    tag_list = []

    for i, (index,row) in enumerate(covars.iterrows()) :
        values = list(data[:,i])
        print(len(values))
        values_list += values
        slab_list +=  [row['slab']]*len(values) 
        y_list += [row['y']]*len(values) 
        tag_list += [tag]*len(values) 
    
    df = pd.DataFrame({'slab':slab_list, 'y':y_list, 'norm':tag_list, 'values': values_list})
    return df

def combat_slab_normalization(df, slab_file_dict):

    out_file_dict = get_output_dict(slab_file_dict) 
    keys = list(slab_file_dict.keys())
    out_file_dict[ str( keys[0] ) ]

    #if all the output files already exist, then return early
    if True in [ not os.path.exists(fn) for fn in out_file_dict.values() ] : 
        data, covars = extract_centiles_from_slab(df, slab_file_dict)
        covars["y"] = 0.02 * covars["y"] 
        #veteran_data = neuroCombat(dat=data, covars=covars, batch_col='slab', continuous_cols=["y"])['data']
        veteran_data = neuroCombat(dat=data, covars=covars, batch_col='slab')['data']
       
        df_raw  =   create_qc_dataframe(data, covars, 'raw')
        df_combat = create_qc_dataframe(veteran_data, covars, 'combat')
        
        sns.catplot(x='slab', y='y', hue='slab', data=df_raw)
        plt.savefig('combat_qc_0.png')
        sns.catplot(x='slab', y='values', hue='slab', data=df_raw)
        plt.savefig('combat_qc_1.png')
        sns.catplot(x='slab', y='values', hue='slab', data=df_combat)
        plt.savefig('combat_qc_2.png')
        exit(0)

        interp_functions = get_slab_based_functions(data, veteran_data, covars['slab'].values)

        out_file_dict = combat_apply(slab_file_dict, out_file_dict, interp_functions)

    return out_file_dict




