from glob import glob
from sys import argv, exit
from os.path import basename
import os
import pandas as pd
import cv2
from PIL import Image 
import numpy as np
from sitk_register import register
import scipy.misc
from utils.utils import *
import imageio

def set_csv(source_files, output_dir, out_fn="receptor.csv", clobber=True):
    if not os.path.exists(out_fn) or clobber:
        cols=["filename", "mr", "slab", "hemi", "tracer", "n"]
        df=pd.DataFrame(columns=cols)
        for f0 in source_files:
            f=basename(f0)
            f_split = basename(f).split("#")
            
            if "s" in f_split[2] : sep="s"
            else : sep="S"

            mr = f_split[2].split(sep)[0]
            slab = f_split[2].split(sep)[1]

            hemi = f_split[3]
            tracer = f_split[4]
            n = f_split[6].split("_")[0]
            #print(f0, mr, slab, hemi, tracer,n)
            df = df.append( pd.DataFrame([[f0, mr, slab, hemi, tracer,n]], columns=cols))
        df.sort_values(by=["slab"], inplace=True)
        df.to_csv(output_dir+os.sep+out_fn)
    else :
        df = pd.read_csv(output_dir+os.sep+out_fn)

    return(df)

def apply_function(df, output_dir, f):
    for slab_name, slab in df.groupby(["slab"]) :
        slab.sort_values(["n"], inplace=True)
        for i in range(slab.shape[0]) :
            i = int(i)
            if i == 0 : continue #skip first image
            f1=slab.iloc[i-1,].filename
            f2=slab.iloc[i,].filename
            f1_split = os.path.splitext(f1)
            output_img_fn=output_dir+os.sep+os.path.basename(f1_split[0])+'_reg.mha' #+f1_split[1]
            output_transform_fn=output_dir+os.sep+os.path.basename(f1_split[0])+'_reg.tfm'
            print(f1)
            print(f2)
            f1_dwn=os.path.splitext(f1)[0] + "_rsl.png"
            f2_dwn=os.path.splitext(f2)[0] + "_rsl.png"

            if not os.path.exists(f1_dwn) :
                scipy.misc.imsave( f1_dwn, downsample(imageio.imread(f1), step=0.1))

            if not os.path.exists(f2_dwn) :
                scipy.misc.imsave( f2_dwn, downsample(imageio.imread(f1), step=0.1))
            f(f1_dwn, f2_dwn, output_img_fn, output_transform_fn)

if __name__ == "__main__":

    source_dir=argv[1]
    output_dir=argv[2]
    ext=".tif"
    if not os.path.exists(output_dir) : 
        os.makedirs(output_dir)

    source_files=glob(source_dir+os.sep+"*"+ext)
    if source_files == []:
        print("Could not find files in ", source_dir)
        exit(1)
    df = set_csv(source_files, output_dir)
    apply_function(df,output_dir, register)

