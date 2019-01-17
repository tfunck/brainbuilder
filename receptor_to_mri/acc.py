from sys import argv
import numpy as np
import h5py
import os

def safe_h5py_open(filename, mode='r+'):
    '''open hdf5 file, exit elegantly on failure'''
    try :
        f = h5py.File(filename, mode)
        return f
    except OSError :
        print('Error: Could not open', filename)
        exit(1)

if not os.path.exists(argv[1]):
    print("Error: could not find ", argv[1])
    exit(1)
if not os.path.exists(argv[2]):
    print("Error: could not find ", argv[2])
    exit(1)

vol1 = safe_h5py_open(argv[1])
vol2 = safe_h5py_open(argv[2])
ar1=np.array(vol1['minc-2.0/']['image']['0']['image'])
ar2=np.array(vol2['minc-2.0/']['image']['0']['image'])
ar3=ar1+ar2
print( np.sum( ar3 > 1  ).astype(float) / np.sum(ar3 >= 1))


