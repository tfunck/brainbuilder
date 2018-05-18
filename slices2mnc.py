from pyminc.volumes.factory import *
from numpy import *
from glob import glob
import argparse
import imageio
import os
import pandas as pd
from utils.utils import *

def slices2mnc(source_file, output_file, clobber=False) :

    img = imageio.imread(source_file)
    if len(img.shape) > 2 : img = np.mean(img,axis=2)
    zmax=img.shape[1]
    xmax=img.shape[0]
    ymax=1

    vol = volumeFromDescription(output_file, dimnames=("zspace","yspace","xspace"), sizes=(zmax,ymax,xmax), starts=(0,0,0), steps=(0.2,0.02,0.2), volumeType="ushort")
    vol.data[:,0,:]  = img.T 
    vol.writeFile()
    vol.closeVolume()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-i', dest='source_file', default=None,help='Directory with images to be converted to volume')
    parser.add_argument('-o', dest='output_file', default=None,help='Output MINC filename')

    args = parser.parse_args()
    if args.output_file == None : 
        args.output_file = os.path.splitext(args.source_file)[0] + '.mnc'

    slices2mnc(args.source_file, args.output_file)

