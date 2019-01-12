import numpy as np
import sys
import re

def read_xfm_matrix(fn):
    f = open(fn)

    ar=[]
    read=False
    for l in f.readlines() :
        if read :
            l = re.sub("\n", "", re.sub(";", "", l))
            l =[ float(i)  for i in l.split(" ") if i != '' ]
            ar += l
        if "Linear_Transform" in l : read = True

    ar += [0,0,0,1]
    ar = np.array(ar).reshape(4,4)

    return ar

y=sys.argv[1]
fn=sys.argv[2]


pts = np.array([0., float(y), 0., 1.])
tfm = read_xfm_matrix(fn)
new_pts = np.matmul(tfm, pts).T

print(int(np.round(new_pts[1])))
