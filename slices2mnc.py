from pyminc.volumes.factory import *
from numpy import *
from glob import glob
import argparse
import imageio
import os
import pandas as pd
from utils.utils import *

def slices2mnc(source_files, receptor_fn,  output_dir, output_file, exclude_str="", include_str="", clobber=False) :
    output_file =output_dir + os.sep + output_file 
    if not os.path.exists(output_dir) :
        os.makedirs(output_dir)

    n=len(source_files)
    order_list=[]
    for fn in source_files :
        fn1 = os.path.basename(fn)
        order_list.append(int(fn1.split("_")[0]))

    ymax = np.max(order_list)+1
    zmax=xmax=0
    for i in range(len(source_files)) : 
        f=source_files[i]
        fn = output_dir + os.sep + os.path.basename(f)
        if not os.path.exists(fn) :
            img = imageio.imread(f)
            img = downsample(img, fn,  step=0.2)
        else : 
            img = imageio.imread(fn)
        z=img.shape[1]
        x=img.shape[0]
        if x > xmax : xmax=x 
        if z > zmax : zmax=z 
        source_files[i] = fn

    df=pd.DataFrame({"fn":source_files,"order":order_list})
    df.sort_values(by=["order"], inplace=True)
    j=0
    print(zmax,ymax,xmax)
    img = np.zeros((zmax,xmax))

    #if not os.path.exists(output_file) or clobber :
    # print("Creating ", output_file)
    vol = volumeFromDescription(output_file, dimnames=("zspace","yspace","xspace"), sizes=(zmax,ymax,xmax), starts=(-90,-126,-72), steps=(0.2,0.02,0.2), volumeType="ushort")

    for i, row in df.iterrows() : 
        f=row.fn
        order=int(row.order)
        out_split = os.path.splitext(f)[0]
        temp = imageio.imread(f)
        z=temp.shape[0]
        x=temp.shape[1]
        #img[ 0:, 0: ] = temp.T / np.max(temp)
        #vol.data[:,order,:] = img #.T
        vol.data[:,order,:]  = temp.T / np.max(temp)
        #vol.data[:,j,:] = img #.T
    vol.writeFile()
    vol.closeVolume()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--source', dest='source_dir', default=None,  help='Directory with images to be converted to volume')
    parser.add_argument('--output-file', dest='output_file',  help='Output MINC filename')
    parser.add_argument('--output-dir', dest='output_dir',  help='Output MINC filename')

    parser.add_argument('--receptor-csv', dest='receptor_fn',  help='Output MINC filename')
    parser.add_argument('--ext', dest='ext', default=".png", help='File extension for input files (default=.tif)')
    
    args = parser.parse_args()
    
    source_files = glob(args.source_dir+os.sep+"*"+args.ext)
    if source_files == [] :
        print("Error: could not find any files in:", args.source_dir)
        exit(1)
    slices2mnc(source_files, args.receptor_fn, args.output_dir, args.output_file)

