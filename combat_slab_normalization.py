import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ptitprince as pt
%matplotlib inline
from neuroCombat import neuroCombat,neuroCombatFromTraining
import pandas as pd
import numpy as np
import nibabel as nb
from statsmodels.nonparametric.smoothers_lowess import lowess


def get_slab_based_functions(input_data,output_data,slabs):
    functions=[]
    for slab in np.unique(slabs):
        functions.append(lowess_correction_function(input_data[:,slabs==slab].ravel(),
                                                 output_data[:,slabs==slab].ravel()))
    return functions

def lowess_correction_function(input_centiles,output_centiles):
    lowess_tight = lowess(output_centiles, input_centiles, frac = .12)
    f_linear = interp1d(lowess_tight[:,0], 
                        y=lowess_tight[:,1], 
                        bounds_error=False, kind='linear', fill_value='extrapolate') 
    return f_linear


def extract_centiles_from_slab(df,n_centiles=10):

    # create centiles for 0 to 1 
    centiles = np.linspace(0,1,n_centiles+2)[1:-1]

    # data will contain centile distribution from the data
    data=np.zeros((n_centiles,df.shape[0]))
    slab_list = []
    ligand_list = []
    y_list = []
    si=0
    for (slab, ligand), slab_df in df.groupby(['slab', 'ligand']):
        slab_min = slab_df['global_order'].min()

        for i, row in slab_df.iterrows() :
            y = slab_min + df['global_order'] * 0.02

            slab_list.append(row['slab'])
            ligand_list.append(row['ligand'])
            y_list.append(y)

            # save centile for section
            section = nib.load(row['crop_fn']).get_fdata() * row['correction_factor']
            mask = nib.load(row['seg_fn']).get_fdata()
            data[:,si] = np.percentile(receptor_vol[mask==1],100*centiles)
            si += 1

    combat_df = {'slab':slab_list, 'y':y_list, 'ligand':ligand_list}

    return combat_df, data

def combat_training(data, covars, receptor_vol) :
    #train combat
    #sample n-samples randomly from cortex of each section
    # Specifying the batch (scanner variable) as well as a biological covariate to preserve:

    # To specify names of the variables that are categorical:
    continuous_cols = ['y']

    # To specify the name of the variable that encodes for the scanner/batch covariate:
    batch_col = 'slab'

    #Harmonization step:
    data_estimates = neuroCombat(dat=data,
        covars=covars,
        batch_col=batch_col,
        continuous_cols=continuous_cols)
    estimates = data_estimates['estimates']
    data_combat=data_estimates['data']
    
    return estimates, data_combat 


def combat_apply(slab_file_list, out_file_list, receptor_vol):

    file_list = zip(slab_file_list, out_file_list)

    for slab, (slab_vol_file, out_file) in enumerate(file_list)  :
        function = interp_functions[slab]

        receptor_img = nib.load(slab_vol_file)
        receptor_vol = receptor_img.get_fdata()

        corrected_receptor_vol = function(receptor_vol)

        nib.Nifti1Image(corrected_receptor_vol, receptor_img.affine).to_filename(out_file)


def combat_slab_normalization(df_file, slab_file_list, out_file_list, out_file):
    data, covars = extract_centiles_from_slab(receptor_df)

    interp_functions = get_slab_based_functions(data, veteran_data, covars['slab'])

    combat_apply(slab_file_list, out_file_list, interp_functions)




