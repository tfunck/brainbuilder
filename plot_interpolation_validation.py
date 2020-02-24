import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import nibabel as nib
import numpy as np
from sys import argv

def process_auto_df(df, auto_fn) :
    auto_df = pd.read_csv(auto_fn)
    slab=1
    ligand='flum'
    auto_df = auto_df.loc[ (auto_df['slab'] == slab) & (auto_df['ligand'] == ligand) ]
    df['Type']=['Interpolated'] * df.shape[0]
    
    for i, row in auto_df.iterrows() :
        df['Type'].loc[ df['i'] == row['volume_order']  ] = 'Acquired'
 
    return df

def plot_validation(df, out_fn) :
    g = sns.lineplot(data=df, x="i", y="Mean",  palette="tab10") #, linewidth=2.5) 
    #g = sns.scatterplot(data=df.loc[df['Type']=='Acquired'], x="i", y="Mean",  palette="tab10", linewidth=2.5) 
    #plt.plot( df['i'], df['Mean'] )
    #plt.scatter( df['i'].loc[df['Type']=='Acquired'], df['Mean'].loc[df['Type']=='Acquired'] )
    g.set( xlabel='Coronal Section', ylabel='Interpolated / True' ) 
    #ylim=(0.75, 1.25),
    
    plt.savefig(out_fn, dpi=300)

def calc_error(error_fn):
    error_img = nib.load(error_fn) 
    error_vol = error_img.get_data()

    #cls_img = nib.load(cls_fn)
    #cls_vol = cls_img.get_data()
    m=[]
    s=[]
    ii=[]
    for i in range(error_vol.shape[1]) :
        #cls = cls_vol[ :, i, : ]
        err = error_vol[ :, i, : ]

        idx = err > 0.75 
        _m = np.mean( err[idx])
        _s = np.std( err[idx] )
        if not np.isnan(_m) and not np.isnan(_s) :
            m.append( _m )
            s.append( _s )
            ii.append(i)

    df = pd.DataFrame( {'i':ii, 'Mean':m, 'StdDev':s} )
    return df

if __name__ == "__main__" :
    df = calc_error(argv[1])
    df = process_auto_df(df, argv[2])
    print(df)
    plot_validation(df, argv[3])
