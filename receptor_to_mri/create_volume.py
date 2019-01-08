from pyminc.volumes.factory import *
from numpy import *
import sys
import re
import os
from scipy.interpolate import BSpline, splrep
import numpy as np

def interpolate_missing(img, y_list):
    test = np.sum(img, axis=(0,2))
    y_list = np.arange(img.shape[1])[ test > 0. ] 
    y_list_min = np.min( y_list )
    y_list_max = np.max( y_list )
    y_missing = np.array([ i for i in range(img.shape[1]) if not i in y_list and i > y_list_min and i < y_list_max  ])
    
    global_mean = [ np.mean(img[z,y_list,x]) for x in range(img.shape[2]) for z in range(img.shape[0]) ]
    global_max  = np.max(global_mean)
    
    for z in range(img.shape[0]) :
        for x in range(img.shape[2]) :
            series = img[z,y_list,x]
            series_mean = np.mean(series) 
            if series_mean > 0.25 * global_max:
                idx = series > 0.
                series_0 = series[ idx ]
                y_list_0 = y_list[ idx ]
                y_missing_0 = y_missing[ (y_missing > np.min(y_list_0)) & (y_missing < np.max(y_list_0)) ] 

                interp_f =  BSpline(*splrep(y_list_0, series_0, k=1))
                
                ylin = np.arange(np.min(y_list_0), np.max(y_list_0)+1) #np.sort(np.concatenate([y_list_0, y_missing_0]))
                series_interp = interp_f(ylin)

                #series_interp = gaussian_filter( series_interp, 3)
    
                img[z, ylin, x ] = series_interp

    return img

file_list = sys.argv[1]
n=len(sys.argv)
slice_list=[]
fn_list=[]

for fn in sys.argv[1:(n-2)] : 
    if fn == 'create_volume.py' : continue
    end=fn.split('_')[-1]
    s=int(re.sub('.mnc', '', end))
    fn_list.append(fn)
    slice_list.append(s)

if not os.path.exists(sys.argv[-2]) :
    vol = volumeLikeFile(sys.argv[-2], sys.argv[-1])
    for fn, s in zip(fn_list, slice_list) :
        mnc=volumeFromFile(fn)
        vol.data[:,s,:]=mnc.data
    vol.writeFile()
else :
    vol=volumeFromFile(sys.argv[-1])

out_fn=re.sub('.mnc', '_interp.mnc', sys.argv[-1])

print("Output file :", out_fn)


vol_int = volumeLikeFile(sys.argv[-2], out_fn)
vol_int.data = interpolate_missing(vol.data, slice_list)

vol_int.writeFile()
vol_int.closeVolume()
vol.closeVolume()
