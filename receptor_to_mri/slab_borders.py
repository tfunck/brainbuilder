from pyminc.volumes.factory import *
from sys import argv
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

slice_fn=argv[1]
scale_factors_fn=argv[2]
slab=int(argv[3])
ii=argv[4]
oo=argv[5]


vol0 = volumeFromFile(ii)
vol1 = volumeLikeFile(ii,oo)
y_step = vol0.separations[1]

srv_sum=np.sum(vol0.data, axis=(0,2))
l=np.arange(len(srv_sum))
srv_min=np.min(l[srv_sum > 1])
srv_max=np.max(l[srv_sum > 1])

df = pd.read_csv(slice_fn)
global_max = df["order"].max()
order_max =df["order"].loc[ (df["slab"] == int(slab)) ].max() 
order_min =df["order"].loc[ (df["slab"] == int(slab)) ].min()

scale_factors = json.load(open(scale_factors_fn, "r+"))
direction = scale_factors["MR1"]["R"][str(slab)]["direction"]

if direction == "rostral_to_caudal" :
    w0 = np.round( 0.95 * ((srv_max*y_step) - (order_max - 1) * 0.02) / y_step).astype(int)
    w1 = np.round( 1.05 * ((srv_max*y_step) - (order_min - 1) * 0.02) / y_step).astype(int)
elif direction == "caudal_to_rostral" :
    w0 = np.round(.95 * (srv_min*y_step + (global_max - order_max)*0.02) / y_step ).astype(int)  #np.round( 0.95 * ((srv_min*y_step) + (order_min - 1) * 0.02) / y_step).astype(int)
    w1 = np.round( 1.05 * (srv_min*y_step + (global_max - order_min)*0.02) / y_step ).astype(int)  
    #np.round( 1.05 * ((srv_min*y_step) + (order_max-1) * 0.02) / y_step).astype(int)
else :
    exit(1)
print(srv_min, srv_max)
print(w0, w1)
vol1.data[:, w0:w1, : ] = vol0.data[:, w0:w1, : ]
vol1.writeFile()
vol1.closeVolume()





