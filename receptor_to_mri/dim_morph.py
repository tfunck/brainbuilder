from sys import argv
from pyminc.volumes.factory import volumeFromFile 
from scipy.ndimage.morphology import binary_dilation, binary_erosion
import numpy as np

#nD=argv[1]
#nE=argv[2]
#ii=argv[3]
#oo=argv[4]

struct = np.zeros([3,3,3],dtype=bool)
struct[:,1,1]=1


test = np.zeros([5,5,5],dtype=bool)

test[[0, 2,4], 1:4, 1:4] = 1

dilTest=binary_dilation(test, struct, 1)
print(test)
print(dilTest)



