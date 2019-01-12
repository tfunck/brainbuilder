from pyminc.volumes.factory import *
from sys import argv
import numpy as np
l0=int(argv[1])
l1=int(argv[2])
fn=argv[3]
out_fn=argv[4]

vol = volumeFromFile(fn)
out = volumeLikeFile(fn, out_fn)

i0 = int((l0 - vol.starts[1]) / vol.separations[1])
i1 = int((l1 - vol.starts[1]) / vol.separations[1])
print(i0, i1)
out.data[:, i0:i1, :] =vol.data[:, i0:i1, : ]

out.writeFile()
out.closeVolume()

