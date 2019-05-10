from sklearn.cluster import KMeans
from sys import argv
from utils.utils import splitext
from scipy.signal import decimate
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import os
import pandas as pd


def kmeans_vol(ii_fn,n , oo_fn):
    vol0 = nib.load(ii_fn)
    vol0_data = vol0.get_data()

    print("Segmenting using k-means")
    n_samples = int(np.product(vol0_data.shape)/1000)
    vol_0_dwn = np.random.choice(vol0_data.reshape(-1,), n_samples, replace=False)

    vol1_data = KMeans(n ).fit(vol_0_dwn.reshape(-1,1)).predict(vol0_data.reshape(-1,1)).reshape(vol0_data.shape)
    nib.Nifti1Image(vol1_data, vol0.affine).to_filename(oo_fn)

    vol2_data=np.zeros(vol0_data.shape)
    m=[]
    
    for i in np.unique(vol1_data) :
        km=np.mean(vol0_data[vol1_data == i])
        if km < 100 : km = 0.0 
        m.append(pd.DataFrame([[i, i, km  ]] ))
        vol2_data[vol1_data == i] = km
        #print(np.sum(vol1_data == i))
        print(i, km)
    del vol1_data
    df=pd.concat(m)
    df.to_csv(splitext(oo_fn)[0]+'.csv', sep=' ', index=False, header=False)

    oo2_fn=splitext(oo_fn)[0]+'_values.nii.gz'
    print("Writing volume.")
    vol2 = nib.Nifti1Image(vol2_data, vol0.affine)
    vol2.to_filename(oo2_fn)
