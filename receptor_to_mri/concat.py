from pyminc.volumes.factory import *
from sys import argv
import os
import re

like_vol=argv[1]
out_vol=volumeLikeFile(argv[1], argv[-1])

for arg in argv[2:-1] :
    fn = os.path.basename(arg)
    fn1= fn.split('.')[0]
    z = int(re.sub('cls_', '', fn1))
    temp = volumeFromFile(arg)
    out_vol.data[:,z,:] = temp.data
print("Writing volume.")
out_vol.writeFile()
out_vol.closeVolume()
