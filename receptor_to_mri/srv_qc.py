from pyminc.volumes.factory import volumeFromFile 
import matplotlib.pyplot as plt
from sys import argv
from os.path import exists

def set_dim(dimnames) :
    if dimnames[0] == "yspace" :
        return 1
    elif dimnames[0] == "zspace" :
        return 0
    else :
        print("File", arv[1]," must be zyx or yzx")
        exit(1)

def pexit(msg) :
    print(msg)
    exit(1)


if len(argv) != 5 : pexit("Error: Useage = srv_qc.py <2d MINC> <2d MINC> <2d MINC> <Output PNG>")
if not exists(argv[1]) : pexit("Error: Could not find file "+ str(argv[1]))
if not exists(argv[2]) : pexit("Error: Could not find file "+ str(argv[2]))
if not exists(argv[3]) : pexit("Error: Could not find file "+ str(argv[3]))

cls=volumeFromFile(argv[1])
srv=volumeFromFile(argv[2])
cls_rsl=volumeFromFile(argv[3])

d0 = set_dim(cls.dimnames)
d1 = set_dim(srv.dimnames)
d2 = set_dim(cls_rsl.dimnames)

ar_cls = cls.data.reshape(cls.sizes[d0], srv.sizes[2])
ar_srv = srv.data.reshape(srv.sizes[d1], srv.sizes[2])
ar_cls_rsl = cls_rsl.data.reshape(cls_rsl.sizes[d2], cls_rsl.sizes[2])

plt.subplot(2,2,1)
plt.imshow(ar_cls)

plt.subplot(2,2,2)
plt.imshow(ar_srv)

plt.subplot(2,2,3)
plt.imshow(0.5*ar_srv + 0.5*ar_cls )

plt.subplot(2,2,4)
plt.imshow(0.5*ar_srv + 0.5*ar_cls_rsl )

plt.savefig(argv[4])
