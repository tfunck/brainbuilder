from pyminc.volumes.factory import *
from numpy import *
from glob import glob
import argparse
import imageio
import os
from utils.utils import *

def slices2mnc(source_files, output_fn, include_str="") :

    df = set_csv(source_files, output_dir="", include_str=include_str)
    df.sort_values(["slab","n"], inplace=True)
    df.index=range(df.shape[0])
    print(df)
    n=len(source_files)

    for slab_name, slab in df.groupby(["slab"]) :
        slab.sort_values(["n"], inplace=True)
        transform_list=[]
        print("Slab:", slab_name)
        for i in range(slab.shape[0]) :
            f=slab.iloc[i,].filename
            img = imageio.imread(f)
            if len(img.shape) == 3 :img= np.mean(img, axis=2)
            if i == 0 :
                out_split = os.path.splitext(output_fn)
                temp_fn = out_split[0]+"_"+slab_name+out_split[1]
                vol = volumeFromDescription(temp_fn, dimnames=("zspace","yspace","xspace"), sizes=(img.shape[1],n,img.shape[0]), starts=(-90,-126,-72), steps=(0.02,0.02,0.02), volumeType="ushort")
            vol.data[:,i,:] = img.T
            del img
    
        vol.writeFile()
        vol.closeVolume()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--source', dest='source_dir', default=None,  help='Directory with images to be converted to volume')
    parser.add_argument('--output', dest='output_fn',  help='Output MINC filename')
    parser.add_argument('--ext', dest='ext', default=".png", help='File extension for input files (default=.tif)')
    
    args = parser.parse_args()
    
    source_files = glob(args.source_dir+os.sep+"*"+args.ext)
    if source_files == [] :
        print("Error: could not find any files in:", args.source_dir)
        exit(1)
    slices2mnc(source_files, args.output_fn)

