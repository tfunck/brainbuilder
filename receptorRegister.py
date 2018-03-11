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
import argparse



def apply_function(df, output_dir, clobber=False):
    dwn_dir = output_dir + os.sep + "downsample"
    qc_dir = output_dir + os.sep + "qc"
    if not os.path.exists(dwn_dir) : os.makedirs(dwn_dir)
    if not os.path.exists(qc_dir) : os.makedirs(qc_dir)
    print(df)
    for slab_name, slab in df.groupby(["slab"]) :
        
        print("Slab",slab_name)
        if not slab_name == "1" : continue
        slab.sort_values(["order"], inplace=True)
        transform_list=[]
        for i in range(slab.shape[0]) :
            i = int(i)
            if i == 0 : continue #skip first image
            f1=slab.iloc[i-1,].filename
            f2=slab.iloc[i,].filename
            #if not f1=="crop/QF#HG#MR1s1#L#musc#4306#02_cropped.png" : continue
            f1_split = os.path.splitext(f1)
            f2_split = os.path.splitext(f2)
            output_img_fn=output_dir+os.sep+os.path.basename(f2_split[0])+'_reg.png' #+f1_split[1]
            output_qc_fn=qc_dir+os.sep+os.path.basename(f2_split[0])+'_reg_qc.png' #+f1_split[1]
            output_transform_fn=output_dir+os.sep+os.path.basename(f2_split[0])+'_reg.tfm'
            print(f1)
            print(f2)
            f1_dwn=dwn_dir+os.sep+os.path.basename(f1_split[0])+ "_rsl.png"
            f2_dwn=dwn_dir+os.sep+os.path.basename(f2_split[0])+ "_rsl.png"

            if not os.path.exists(f1_dwn) or clobber :
                scipy.misc.imsave( f1_dwn, downsample(imageio.imread(f1), step=0.1))

            if not os.path.exists(f2_dwn) or clobber :
                scipy.misc.imsave( f2_dwn, downsample(imageio.imread(f1), step=0.1))
            
            if not os.path.exists(output_transform_fn) or clobber :
                register(f1_dwn, f2_dwn, output_img_fn,output_qc_fn, output_transform_fn, transform_list)

            transform_list = [output_transform_fn] + transform_list
            



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--source', dest='source_dir', default=None,  help='Directory with raw images')
    parser.add_argument('--output', dest='output_dir',  help='Directory name for outputs')
    parser.add_argument('--slice-order', dest='slice_order_fn',  help='CSV file that contains ordering of slices')
    parser.add_argument('--include-only', dest='include_str', default='',  help='Comma separated list of tracers to include')
    parser.add_argument('--step', dest='downsample_step', default="1", type=int, help='File extension for input files (default=.tif)')

    parser.add_argument('--ext', dest='ext', default=".tif", help='File extension for input files (default=.tif)')
    parser.add_argument('--clobber', dest='clobber', action='store_true', default=False, help='Clobber results')

    args = parser.parse_args()
    if not os.path.exists(args.output_dir) : 
        os.makedirs(args.output_dir)

    source_files=glob(args.source_dir+os.sep+"*"+args.ext)
    if source_files == []:
        print("Could not find files in ",args.source_dir)
        exit(1)
    df = set_csv(source_files, args.output_dir, args.include_str, slice_order_fn=args.slice_order_fn)
    apply_function(df,args.output_dir, args.clobber)

