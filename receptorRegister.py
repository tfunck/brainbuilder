from glob import glob
from sys import argv, exit
from os.path import basename
import os
import pandas as pd
import cv2
from PIL import Image 
import numpy as np
#from sitk_register import register
from elastix_register import register
import scipy.misc
from utils.utils import *
import imageio
import argparse


def get_z_x_max(source_files, output_dir):
    out_fn=output_dir + os.sep + "z_x_max.txt"
    if not os.path.exists(out_fn) :
        zmax=xmax=0
        for i in range(len(source_files)) : 
            f=source_files[i]
            fn = output_dir + os.sep + os.path.basename(f)
            if not os.path.exists(fn) :
                img = imageio.imread(f)
            else : 
                img = imageio.imread(fn)
            z=img.shape[0]
            x=img.shape[1]
            if x > xmax : 
                xmax=x 
            if z > zmax : 
                zmax=z 
        maxima=np.array([zmax, xmax])
        np.savetxt(out_fn, maxima)
    else :
        maxima = np.loadtxt(out_fn)
    zmax=int(maxima[0]*1.3)
    xmax=int(maxima[1]*1.3)
    return zmax, xmax

def register_slices(fixed,moving, order, order_moving, output_dir, qc_dir, zmax, xmax, clobber=False):
    #print("Fixed", order, fixed)
    #print("Moving", order_moving, moving)
    
    transform_list=[] 
    #if not os.path.exists(output_transform_fn) or not os.path.exists(output_qc_fn) or clobber :
    if not os.path.exists(output_qc_fn) or clobber :
        register(fixed, moving, moving, output_img_fn, output_qc_fn, output_transform_fn, transform_list, zmax, xmax, order, order_moving)
    #transform_list = [output_transform_fn] + transform_list
    return output_img_fn

def slab_coregistration(slab, df_master, output_dir, qc_dir, zmax, xmax, start_tier, clobber=False):
    m = int(slab.shape[0]/2)
    bottom = list(range(m+1))
    bottom.reverse()
    top = list(range(m,slab.shape[0]))
    
    order_list_top = np.array( list(slab.order.values[top])  )
    order_list_bottom = np.array( list(slab.order.values[bottom])  )
    filenames_top = np.array( list(slab.filename.values[top]) )
    filenames_bottom = np.array( list(slab.filename.values[bottom]) )
    tier_list_top = np.array( list(slab.tier.values[top]) )
    tier_list_bottom = np.array( list(slab.tier.values[bottom])  )
   
    df_top = pd.DataFrame({"i": top, "tier":tier_list_top, "filenames":filenames_top, "order": order_list_top })
    df_bottom=pd.DataFrame({"i": bottom, "tier":tier_list_bottom, "filenames":filenames_bottom, "order": order_list_bottom })
    
    base_row = pd.DataFrame( df_top.loc[ df_top.tier == 1 ].iloc[0,] ).T
    base_row.index = [df_top.shape[0] ]
    df_top = df_top.append(base_row)
    df = pd.concat([df_top, df_bottom])
    df.index=range(df.shape[0])

    #df=pd.DataFrame({"i": top+bottom, "tier":tier_list, "filenames":filenames, "order": order_list })
    #print(df)
    df.to_csv("temp.csv")
    for tier in sorted(slab.tier.unique()) :
        #if tier == 3 : exit(0)
        df0 = df.loc[ df.tier <= tier]
        #df0.index = range(df0.shape[0])
        j=0
        transform_list=[]
        for i in range(df0.shape[0]) :
            #if you use index to acces the rows, then you will start at the end of the slab
            row=df0.iloc[i,]
            order_fixed = df0.iloc[i,].order
            j=i+1
            if j >=  df0.shape[0] : continue
            fixed=df0.filenames.iloc[i,]
            moving=df0.filenames.iloc[j,]
            if df0.index[i,] == m : continue
            if df0.tier.iloc[j,] == tier :
                order_moving = df0.iloc[j,].order
                
                moving_split = os.path.splitext(moving)
                output_img_fn=output_dir+os.sep+ str(order_moving) +"_" + os.path.basename(moving_split[0])+'_reg.png' 
                output_qc_fn=qc_dir+os.sep+ str(order_moving) +"_"+os.path.basename(moving_split[0])+'_reg_qc.png' 
                output_transform_fn=output_dir+os.sep+ str(order_moving) +"_"+os.path.basename(moving_split[0])+'_reg.tfm'
                if order_moving == 225 or order_fixed == 225 :
                    print( order_fixed, order_moving)
                    print("fixed",fixed)
                    print("moving",moving)
                    print("output qc", output_qc_fn)          
                    print(tier, start_tier)
                if (not os.path.exists(output_qc_fn) or clobber) and tier >= start_tier :
                    register(fixed, moving, moving, output_img_fn, output_qc_fn, output_transform_fn, transform_list, zmax, xmax, order_fixed, order_moving)
                df0["filenames"].iloc[j,] =  output_img_fn
                df["filenames"].loc[ df.filenames == moving] = output_img_fn
                df_master["filename"].loc[ df_master.filename == moving] = output_img_fn

    df_master.to_csv(output_dir+os.sep+"coregistered_receptor_slices.csv")


def apply_function(df, output_dir, zmax, xmax, start_tier, clobber=False):
    qc_dir = output_dir + os.sep + "qc"
    if not os.path.exists(qc_dir) : os.makedirs(qc_dir)
    df = df.loc[ df["order"] >= 0 ]
    for slab_name, slab in df.groupby(["slab"]) :
        slab.sort_values(["order"], inplace=True)
        transform_list=[]
        slab_coregistration(slab, df, output_dir, qc_dir, zmax, xmax, start_tier, clobber)

def setup_tiers(df, tiers_str):
    tiers_list = [ j.split(",") for j in tiers_str.split(";")]
    i=1
    df["tier"]=[0] * df.shape[0]
    if tiers_str != '' :
        for tier in tiers_list :
            for ligand in tier :
                df["tier"].loc[ df.ligand == ligand ] = i
            
            i += 1
        df=df.loc[df["tier"] != 0 ]
    return(df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--source', dest='source_dir', default=None,  help='Directory with raw images')
    parser.add_argument('--output', dest='output_dir',  help='Directory name for outputs')
    parser.add_argument('--slice-order', dest='slice_order_fn',  help='CSV file that contains ordering of slices')
    parser.add_argument('--include-only', dest='include_str', default='',  help='Comma separated list of tracers to include')
    parser.add_argument('--tiers', dest='tiers_str', default='',  help='Tier of slices in , and ; string (Example "a,b,c;d,e,f;g,h,i)')
    parser.add_argument('--start-tier', dest='start_tier', default=1, type=int,  help='Starting tier to be run (allows to clobber for specific tiers)')
    parser.add_argument('--exclude', dest='exclude_str', default='',  help='Comma separated list of tracers to exclude')
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
    df = set_csv(source_files, args.output_dir, args.include_str, args.exclude_str, slice_order_fn=args.slice_order_fn)
    df = setup_tiers(df, args.tiers_str)
    zmax, xmax = get_z_x_max(source_files, args.output_dir)
    apply_function(df,args.output_dir, zmax, xmax, args.start_tier, args.clobber)

