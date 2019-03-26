from pyminc.volumes.factory import *
import nibabel as nib
from sys import argv
import os
import re
import numpy as np

like_vol=argv[1]

#Get y max
y_list=[]
for arg in argv[2:-1] :
    fn = os.path.basename(arg)
    fn1= fn.split('.')[0]
    y = int(re.sub('cls_', '', fn1))
    y_list.append(y)


data = nib.load(argv[2]).get_data()

xmax = data.shape[1]
ymax = max(y_list)+1
zmax = data.shape[0]

print(xmax,ymax,zmax)
out_data=np.zeros([xmax,ymax,zmax])

for arg in argv[2:-1] :
    fn = os.path.basename(arg)
    fn1= fn.split('.')[0]
    y = int(re.sub('cls_', '', fn1))
    data = nib.load(arg).get_data()
    out_data[:,y,:] = data.T

ref = nib.load(argv[1])
affine=ref.get_affine()
print(affine)
out_img=nib.Nifti1Image(out_data, affine)
out_img.to_filename(argv[-1])
print("Writing volume.")
