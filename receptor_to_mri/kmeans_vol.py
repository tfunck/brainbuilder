from pyminc.volumes.factory import *
from sklearn.cluster import KMeans
import numpy as np
from sys import argv
import os
import pandas as pd

ii_fn=argv[1]
n=int(argv[2])
oo_fn=argv[3]
print(ii_fn, n, oo_fn)



if os.path.exists(ii_fn):
    vol0 = volumeFromFile(ii_fn)

vol1=volumeLikeFile(ii_fn, oo_fn)
vol2=volumeLikeFile(ii_fn, os.path.splitext(oo_fn)[0]+'_values.mnc')


print("Segmenting using k-means")
#init = np.append(np.array([0]) , np.random.uniform(np.mean(vol0.data), np.max(vol1.data), n-1))

vol1.data = KMeans(n ).fit_predict(vol0.data.reshape(-1,1)).reshape(vol1.data.shape)
print(vol1.data.shape)
m=[]
for i in range(n) :
    km=np.mean(vol0.data[vol1.data == i])
    m.append(pd.DataFrame([[i, i, km  ]] ))
    vol2.data[vol1.data == i] = km
    print(np.sum(vol1.data == i))

df=pd.concat(m)
df.to_csv(os.path.splitext(oo_fn)[0]+'.txt', sep=' ', index=False, header=False)


print("Writing volume.")
vol0.closeVolume()
vol1.writeFile()
vol1.closeVolume()
vol2.writeFile()
vol2.closeVolume()
