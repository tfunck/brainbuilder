from numpy import *
from utils.utils import *
from glob import glob
import nibabel as nib
import argparse
import imageio
import os, sys
import pandas as pd
import matplotlib.pyplot as plt
import h5py
import json


def init_volume(slab_fn,slab_img_fn, source_dir, zmax, ymax, xmax, df, scale, cur_slab, clobber=False, align=True):
    n=1
    if not os.path.exists( slab_img_fn ) or clobber :
        order_max=df["order"].max()
        order_min=df["order"].min()
        slab_ymax=order_max-order_min + 1
        vol = np.zeros( [xmax,slab_ymax,zmax])
        
        y=0
        for i, row in df.iterrows() :
            forder = row["order"]
            f=glob(source_dir+os.sep+str(forder)+"_*png")
            if len(f) == 0 :
                continue
            f=f[0]
            slab = row["slab"]
            hemi = row["hemisphere"]
            mr = row["mri"]
            direction = scale[mr][hemi][str(slab)]["direction"]
            temp =  imageio.imread(f)
            if align :
                #temp = temp.T
                if direction == "rostral_to_caudal":
                    temp = np.flip(temp, 0)
                temp = np.flip(temp, 1)
            else :
                temp = np.flip(temp, 0)
            vol[ : , order_max - forder , : ] =temp

            n += 1
    
        print("\tWriting",slab_img_fn)
        slab_ymin=-126+df["order"].min()*0.02
        affine=np.array([[0.2, 0, 0, -72],
                        [0, 0.02, 0, slab_ymin],
                        [0,0,0.2, -90],
                        [0, 0, 0, 1]])
        img = nib.Nifti1Image(vol, affine)   
        img.to_filename(slab_img_fn)
        del vol
        del img
        
def get_maxima(df, source_files, align):
    ymax = df["order"].values.max() + 1
    zmax=xmax=0
    sys.stdout.flush()
    for i in range(len(source_files)) :
        f=source_files[i]
        img = imageio.imread(f)
        if align :
            z=img.shape[1]
            x=img.shape[0]
        else :
            z=img.shape[0]
            x=img.shape[1]
        if x > xmax : xmax=x
        if z > zmax : zmax=z
        break
    return zmax, ymax, xmax


def slices2vol(receptor_fn, source_dir, output_dir,  slabs_to_run=[], clobber=False, align=True, ext='.png') :
    
    if not os.path.exists(output_dir) :
        os.makedirs(output_dir)
    
    source_files = glob(source_dir+os.sep+"*"+ext)
    if source_files == [] :
        print("Error: could not find any files in:", source_dir)
        exit(1)


    df=pd.read_csv(receptor_fn)
    df.sort_values(by=["order"], inplace=True)

    with open("scale_factors.json") as f : scale=json.load(f)
    df = df.loc[ df["order"] >= 0 ]
    zmax, ymax, xmax =get_maxima(df, source_files, align)

    for cur_slab in slabs_to_run :
        print('\tCurrent slab:', cur_slab)
        slab=df.loc[ df.slab == cur_slab ]
        slab_fn = output_dir+os.sep+"vol_"+str(cur_slab)+".hdf5"
        slab_img_fn= output_dir+os.sep+"vol_"+str(cur_slab)+".nii.gz"
        init_volume(slab_fn,slab_img_fn, source_dir, zmax, ymax, xmax, slab, scale, cur_slab, clobber)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--source', dest='source_dir', default=None,  help='Directory with images to be converted to volume')
    parser.add_argument('--output-dir', dest='output_dir',  help='Output MINC filename')
    parser.add_argument('--receptor-csv', dest='receptor_fn',  help='Output MINC filename')
    parser.add_argument('--ext', dest='ext', default=".png", help='File extension for input files (default=.tif)')
    parser.add_argument('--clobber', dest='clobber', action='store_true', default=False, help='Clobber results')
    parser.add_argument('--no-align', dest='align', action='store_false', default=True, help='Rotate / flip images when loading them')
    parser.add_argument('--slabs', dest='slabs_to_run', nargs='+', help='Slabs to process into volume')
    args = parser.parse_args()

    slices2vol( args.receptor_fn, args.source_dir, args.output_dir,  args.slabs_to_run, args.clobber, args.align)
