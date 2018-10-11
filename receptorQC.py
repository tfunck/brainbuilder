from sklearn.metrics import normalized_mutual_info_score
from scipy.stats import kendalltau
import pandas as pd
import matplotlib.pyplot as plt
import os

def create_cmap(x):
    xuni = np.unique(x)
    xunienum = enumerate(xuni)
    d=dict(xunienum)
    e = { v:k for k,v in  d.items() }
    cmap = np.array([ e[i]  for i in x ])
    cmap = cmap / np.max(cmap)
    return cmap

def receptorQC( df_fn, output_dir, clobber=False ):
    df = df.iloc[  df.order > 0 ]
    df = pd.read_csv( df_fn) 
    df["nmi"] = [0] * df.shape[0]
    df["cc"]  = [0] * df.shape[0]

    if not os.path.exists(output_dir) :
        os.path.makedir(output_dir)

    output_df_fn = output_dir + os.sep + "receptor_slices_qc.csv"

    if not os.path.exists(output_df_fn) or clobber :
        for i in range(df.shape[0]) :
            #if you use index to acces the rows, then you will start at the end of the slab
            row=df.iloc[i,]
            j=i+1
            k=j+1
            if j >=  df.shape[0] : continue
            fixed=df.filenames.iloc[i,] 

            if df.rsl.iloc[j,] == "" : 
                moving=df.filenames.iloc[j,]
            else :
                moving=df.rsl.iloc[j,]

            fixed = imageio.imread(fixed_fn).reshape(-1,)
            rsl = imageio.imread(rsl_fn).reshape(-1,)

            # Get normalized mutual information between slices
            df["nmi"].iloc[ i, ] = normalized_mutual_info_score(fixed, rsl)

            # Get cross-correlation between slices
            df["cc"].iloc[ i, ] = kendalltau(masked_pet_data, masked_mri_data)[0]

            # Get center of gravity


        plt.subplot(2,1,1)
        plt.scatter(df.order, df.nmi)
        plt.subplot(2,1,2)
        plt.scatter(df.order, df.cc)
        plt.show()
        plt.savefig(output_df_fn)
