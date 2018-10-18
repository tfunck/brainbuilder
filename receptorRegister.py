from glob import glob
from sys import argv, exit
from os.path import basename
import os
import pandas as pd
import cv2
import json
import numpy as np
#from sitk_register import register
#from elastix_register import register, resample, get_param
from sitk_register import register, resample, display_images_with_alpha, myshow
import scipy.misc
from utils.utils import *
from utils.anisotropic_diffusion import anisodiff
import imageio
import argparse
from re import sub
from skimage.exposure import  equalize_hist
from skimage.transform import resize 
from utils.utils import imadjust
import matplotlib.pyplot as plt
from shutil import copy
from math import ceil


def get_z_x_max(source_files, output_dir):
    out_fn=output_dir + os.sep + "z_x_max.txt"
    zmax=xmax=0
    if not os.path.exists(out_fn) :
        for i in range(len(source_files)) : 
            f=source_files[i]
            fn = output_dir + os.sep + os.path.basename(f)
            if not os.path.exists(fn) :
                img = imageio.imread(f)
            else : 
                img = imageio.imread(fn)
            z=img.shape[0]
            x=img.shape[1]
            if z > zmax : zmax = z
            if x > xmax : xmax = x
            #stdout = shell("mincinfo -attvalue xspace:length -attvalue yspace:length -attvalue zspace:length "+f, False)[0] 
        maxima=np.array([zmax, xmax])
        np.savetxt(out_fn, maxima)
    else :
        maxima = np.loadtxt(out_fn)
    zmax=int(maxima[0]*1.2)
    xmax=int(maxima[1]*1.2)
    return zmax, xmax

def create_coregistration_df(slab, df_fn, clobber=True):

    m = int(slab.shape[0]/2)
    if not os.path.exists(df_fn) or clobber :
        bottom = list(range(m+1))
        bottom.reverse()
        top = list(range(m,slab.shape[0]))
        
        order_list_top = np.array( list(slab.order.values[top])  )
        order_list_bottom = np.array( list(slab.order.values[bottom])  )
        filenames_top = np.array( list(slab.filename.values[top]) )
        filenames_bottom = np.array( list(slab.filename.values[bottom]) )
        tier_list_top = np.array( list(slab.tier.values[top]) )
        tier_list_bottom = np.array( list(slab.tier.values[bottom])  )

        skip_list_top = np.array( [False] * len(order_list_top) )
        skip_list_bottom = np.array( [False] * len(order_list_bottom)  )
        skip_list_bottom[0] = True
        
        ligand_list_top = np.array( list(slab.ligand.values[top]) )
        ligand_list_bottom = np.array( list(slab.ligand.values[bottom])  )

        df_top = pd.DataFrame({"i": top, "tier":tier_list_top, "ligand":ligand_list_top, "filenames":filenames_top, "order": order_list_top, "skip":skip_list_top})
        df_bottom=pd.DataFrame({"i": bottom, "tier":tier_list_bottom, "ligand":ligand_list_bottom,"filenames":filenames_bottom, "order": order_list_bottom, "skip":skip_list_bottom })
        
        df = pd.concat([df_top, df_bottom])

        df["nmi"] = [0.] * df.shape[0]
        df.index=range(df.shape[0])
        df["rsl"] = df["filenames"]  #["empty"] * df.shape[0]

        df.to_csv(df_fn)
    else :
        df = pd.read_csv(df_fn)
    return df, m


def slice_coregistration(df,resolutions, max_iterations, max_length, output_dir, qc_dir, m, tier, transform_dict, clobber=False) :
    transform_dir = output_dir + os.sep + "transforms"
    resample_dir = output_dir + os.sep + "resample"
    if not os.path.exists(transform_dir) : os.makedirs(transform_dir)
    if not os.path.exists(resample_dir) : os.makedirs(resample_dir)
    params = []
    transform_fn_list = []

    for i in range(df.shape[0]) :
        #if you use index to acces the rows, then you will start at the end of the slab
        row=df.iloc[i,]
        
        j=i-1
        if  j < 0: continue
        
        order_fixed = df.iloc[j,].order
        fixed_tier = df.iloc[j,].tier
        order_moving = df.iloc[i,].order
        moving_tier = df.iloc[i,].tier
        fixed=df.filenames.iloc[j,]
        if row.skip : continue
        moving=df.rsl.iloc[i,]

        moving_split = os.path.splitext(moving)

        qc_fn=qc_dir+os.sep+str(order_moving)+"_"+os.path.basename(moving_split[0])+'_reg_qc.png' 
        transform_fn=transform_dir+os.sep+str(order_moving)+"_"+os.path.basename(moving_split[0])+'_reg.txt'
        rsl_fn=resample_dir+os.sep+str(order_moving)+"_"+os.path.basename(moving_split[0])+'_rsl.png' 
  
        if row["tier"] == tier :
            transform_dict[ order_moving  ] = [ transform_fn ] + transform_dict[ order_fixed ] 

            if not os.path.exists(qc_fn) or not os.path.exists(transform_fn) or clobber  :
                print("Fixed:\t", order_fixed,"\t", fixed)
                register(fixed, moving, transform_fn, resolutions, max_iterations, max_length )
                print("Moving:\t", order_moving,"\t", moving)
                print("RSL Transformation :", transform_dict[ order_moving][0])
                rsl = resample(moving, fixed, [transform_dict[ order_moving][0]] )
                display_images_with_alpha( 0.5, imageio.imread(fixed), imageio.imread(moving), rsl, qc_fn,  order_fixed, order_moving, fixed_tier, moving_tier)
        if not os.path.exists(rsl_fn) or clobber :    
            resample(moving, fixed,  transform_dict[ order_moving ], rsl_fn)
                
        df["rsl"].loc[ df.filenames == moving] = rsl_fn

    return df

#def slab_coregistration(slab, df_master, output_dir, qc_dir, start_tier, clobber=False):

def apply_slice_registration(df, output_dir, clobber=False):
    df = df.loc[ df["order"].astype(int) >= 0 ]
    df["rsl"] = df["filename"] #[""] * df.shape[0]

    qc_dir = output_dir + os.sep + "qc"
    if not os.path.exists(qc_dir) : os.makedirs(qc_dir)
    tiers = np.sort(np.unique(df["tier"].values))

    transform_dict = {}
    for o in df.order : transform_dict[ o ] = []

    for slab, df0 in df.groupby(["slab"]) :
        resolutions=[  5, 4,  3, 2, 1]
        base_iterations = max_iterations = 1000000
        max_length = 1
        #if slab != 2 : continue
        for t in tiers :
            #if t != 1 : continue
            print("Slab: ", slab, "Tier :", t)
            df1 = df0.loc[ df0["tier"] <= t ].copy()
            df2, m = create_coregistration_df(df1, output_dir+os.sep+str(slab)+"_coregistration_df.csv")
            
            
            df2.to_csv(output_dir+os.sep+"slab-"+str(slab) +"_tier-"+str(t)+"_df.csv") 

            df_rsl = slice_coregistration(df2,resolutions, max_iterations, max_length,output_dir,qc_dir,m, t, transform_dict, clobber=clobber)
            for f in df_rsl.filenames :
                df["rsl"].loc[ df.filename == f ] = df_rsl["rsl"].loc[ df_rsl.filenames == f ].values[0]
    return df




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

def pad_size(d):
    if d % 2 == 0 :
        pad = (int(d/2), int(d/2))
    else :
        pad = ( int((d-1)/2), int((d+1)/2))
    return(pad)
    

def add_padding(img, zmax, xmax):
    z=img.shape[0]
    x=img.shape[1]
    dz = zmax - z
    dx = xmax - x
    z_pad = pad_size(dz)
    x_pad = pad_size(dx)
    img_pad = np.pad(img, (z_pad,x_pad), 'minimum')
    return img_pad

def downsample_slices(source_files, output_dir):
    common_scale = 93 #FIXME Not sure this should be hardcoded to this
    common_step = 0.2
    

    with open("scale_factors.json") as f : scale=json.load(f)
    temp_fn_list = []
    for f in source_files :
        f2 = output_dir + os.sep +"preprocess"+ os.sep + os.path.basename(f)
        temp_fn = output_dir + os.sep +"preprocess"+os.sep+"downsample"+ os.sep + os.path.basename(f)
        temp_fn_list.append(temp_fn)

        dim0_list = dim1_list = []
        if not os.path.exists(temp_fn) :
            ### 1. Calculate scale factor
            mr_slab = [ x for x in f.split("#") if "MR" in x and os.sep not in x ][0]
            hemisphere = [ x for x in f.split("#") if x == "R" or x == "L" ][0]
            mr = re.split('s|S', mr_slab)[0]
            slab = re.split('s|S', mr_slab)[1]
            slab_size = scale[mr][hemisphere][slab]["size"]

            ### 2. Read Image
            img = imageio.imread(f)
            if len(img.shape) > 2 : img = np.mean(img, axis=2)
            
            ### 3. Downsample Image
            #new_step = 0.2
            #old_step = slab_scale_factor / 4164.

            #Calculate length of image based on assumption that pixels are 0.02 x 0.02 microns
            #l0 = img.shape[0] * 0.02 #FIXME : NOT SURE IF THIS IS 100% ACCURATE!
            #l1 = img.shape[1] * old_step  
            ##Calculate the length for the downsampled image
            #dim0=int(np.ceil(l0 / new_step))
            #dim1=int(np.ceil(l1 / new_step))

            #dim0_list.append(dim0)
            #dim1_list.append(dim1)
            ##Calculate the standard deviation based on a FWHM (pixel step size) of downsampled image
            step_0 = slab_size /  4164

            nvox_0 = ceil( img.shape[0] * step_0 / common_step )
            nvox_1 = ceil( img.shape[1] * step_0 / common_step )

            img_dwn = resize(img, output_shape=(nvox_0, nvox_1), anti_aliasing=True )
            #plt.subplot(2,1,1)
            #plt.imshow(img)
            #plt.subplot(2,1,2)
            #plt.imshow(img_dwn)
            #plt.show()

            imageio.imsave(temp_fn, img_dwn)
    return temp_fn_list

def hist_and_pad(temp_fn_list, output_dir, zmax, xmax):
    prep_files = []
    ######################################################
    # Load downsampled images, adjust histogram and pad  #
    ######################################################
    low_contrast_ligands=["oxot", "epib", "ampa", "uk14", "mk80" ]
    for f in temp_fn_list :
        f2 = output_dir + os.sep +"preprocess"+ os.sep + os.path.basename(f)
            
        if not os.path.exists(f2) :
            ### 1. Read Image
            img = imageio.imread(f)
            if len(img.shape) > 2 : img = np.mean(img, axis=2)

            #for ligand in low_contrast_ligands :
            #    if ligand in os.path.basename(f) :
            #        #img =  equalize_hist(img)
            #        img = anisodiff(img,niter=10,kappa=10,gamma=0.1)
            #        break
            
            img = imadjust(img)
            img = add_padding(img, zmax, xmax)
            imageio.imsave(f2, img)
            ii = imageio.imread(f2)
            prep_files.append(f2)
        else :
            prep_files.append(f2)
    return prep_files


from skimage.filters import try_all_threshold, threshold_mean

from scipy.ndimage.morphology import binary_dilation, binary_erosion, binary_fill_holes
def create_masks(source_files, output_dir):

    if not os.path.exists(output_dir+os.sep+"masks") :
        os.makedirs(output_dir+os.sep+"masks")

    for f in source_files : 
        f2 = output_dir + os.sep +"masks"+ os.sep + os.path.basename(f)
        if not os.path.exists(f2) :
            img = imageio.imread(f)
            if len(img.shape) > 2 : img = np.mean(img, axis=2)

            #fig, ax = try_all_threshold(img, figsize=(10, 8), verbose=False)
            #plt.show()
            t = threshold_mean(img)
            idx = t < img
            out = np.zeros(img.shape)
            out[idx] = 1
            #img = binary_dilation(binary_erosion(img, iterations=5), iterations=5).astype(int)
            #plt.imshow(img)
            #plt.show()
            imageio.imsave(f2, out)


def preprocess(source_files,  output_dir, step=0.2, ystep=0.02):
    if not os.path.exists(output_dir + os.sep +"preprocess"+os.sep+"downsample") :
        os.makedirs(output_dir + os.sep +"preprocess"+os.sep+"downsample")
    temp_fn_list = downsample_slices(source_files, output_dir)
    #create_masks(temp_fn_list, output_dir)
    zmax, xmax = get_z_x_max(temp_fn_list, output_dir)
    prep_files = hist_and_pad(temp_fn_list, output_dir, zmax, xmax)
    return prep_files

def concatenate_slices(df, best_ligands, percentiles, out_fn):
    order_0, order_1 = np.percentile(df["order"].loc[ df["ligand"].isin(best_ligands) ].values, percentiles, interpolation="nearest")
    rsl_fn = df["rsl"].loc[ (df["ligand"].isin(best_ligands)) & (df["order"] >= order_0)  & (df["order"] <= order_1) ].values
    ymax = len(rsl_fn)

    img = imageio.imread(rsl_fn[0])
    zmax = img.shape[0]
    xmax = img.shape[1]

    vol = np.zeros([zmax,ymax,xmax])
    for i, fn in enumerate(rsl_fn) : vol[:,i,:] = imageio.imread(fn)
    vol_blr = gaussian_filter(vol, 4)

    imageio.imsave(out_fn, vol_blr[:,int(ymax/2),:] )

    return int(ymax/2)


def slab_align(slab_coreg_pts, i, slab_coreg_qc_dir, transform_dir, clobber ):
    resolutions=[5, 4, 3,  2, 1]
    max_iterations = 650000
    max_length = 1
    
    moving_slab = slab_coreg_pts[ i ] 
    j = moving_slab["direction"] + i
    fixed_slab = slab_coreg_pts[ j] 
  
    fixed_side  = moving_slab["side"]["fixed"] 
    moving_side = moving_slab["side"]["moving"]

    fixed_fn = fixed_slab[ fixed_side  ]
    moving_fn =moving_slab[ moving_side]
    
    #if sign_0 < 0 :
    fixed_order = fixed_slab["order_"+fixed_side]
    moving_order= moving_slab["order_"+moving_side ]  

    transform_fn = transform_dir + os.sep + "tfm_"+str(i)+"-to-"+str(j)+".txt"
    if not os.path.exists(transform_fn) or clobber :
        qc_fn = slab_coreg_qc_dir + os.sep +"slab_"+ str(i) +"-to-" +str(j)+"_coreg.png"
        register(fixed_fn, moving_fn, transform_fn, resolutions, max_iterations, max_length )
        rsl = resample(moving_fn, fixed_fn, [transform_fn] )
        display_images_with_alpha( 0.5, imageio.imread(fixed_fn), imageio.imread(moving_fn), rsl, qc_fn, fixed_order, moving_order, j, i)

    
    return transform_fn


def slab_registration(i,m, slab_coreg_tfm_dir, slab_coreg_rsl_dir, slab_coreg_pts, slab_coreg_qc_dir, transform_list, df, clobber ):
    transform_fn = slab_align(slab_coreg_pts, i, slab_coreg_qc_dir, slab_coreg_tfm_dir, clobber )
    
    transform_list = [transform_fn] + transform_list
    for f in  df["rsl"].loc[ df["slab"] == i ] :
        rsl_fn = slab_coreg_rsl_dir + os.sep + os.path.basename(f)
        rsl_qc_fn = slab_coreg_qc_dir + os.sep + "qc_"+os.path.basename(f)
        if not os.path.exists(rsl_fn)  or clobber :
            rsl = resample(f, f, transform_list )
            imageio.imsave(rsl_fn, rsl)
            #plt.subplot(2,1,1)
            #plt.imshow(0.5* rsl + 0.5*imageio.imread(slab_coreg_pts[m]["left"]))
            #plt.subplot(2,1,2)
            #plt.imshow(0.5*rsl + 0.5*imageio.imread(slab_coreg_pts[m]["right"]))
            #plt.savefig(rsl_qc_fn)
    return transform_list

def apply_slab_registration(df, output_dir, clobber=False) :
    best_ligands = np.unique(df["ligand"].loc[ df["tier"] == int(1) ])
    slab_coreg_rsl_dir = output_dir + os.sep + "slab_coregistration/resample"
    slab_coreg_qc_dir = output_dir + os.sep + "slab_coregistration/qc"
    slab_coreg_tfm_dir = output_dir + os.sep + "slab_coregistration/transform"
    if not os.path.exists(slab_coreg_qc_dir)  : os.makedirs(slab_coreg_qc_dir)
    if not os.path.exists(slab_coreg_rsl_dir)  : os.makedirs(slab_coreg_rsl_dir)
    if not os.path.exists(slab_coreg_tfm_dir)  : os.makedirs(slab_coreg_tfm_dir)
    
    n_slabs = len(np.unique(df["slab"]))
  
    #define order of slabs
    m=np.round(n_slabs/2).astype(int)
    slab_coreg_pts = {}
    slab_order_0 = list(range(m-1,0,-1))
    slab_order_1 = list(range(m+1,n_slabs+1))
    n0 = len(slab_order_0)
    n1 = len(slab_order_1)
    slab_order = slab_order_0 + slab_order_1
    slab_order_full = [m] + slab_order_0 + slab_order_1
    fixed_list = [0] + [1] * n0  +  [-1] * n1
    side_list = [{"fixed": None, "movng": None }] + [{"fixed":"left","moving":"right"}] * n0 +  [{"fixed":"right","moving":"left"}] * n1


    #for slab, df0 in df.groupby(["slab"]) :
    for slab, direction, side in zip(slab_order_full, fixed_list, side_list) : 
        df0 = df.loc[ df["slab"] == slab ]
        right_fn=slab_coreg_qc_dir+os.sep+"slab-"+str(slab)+"_right.png"
        left_fn=slab_coreg_qc_dir+os.sep+"slab-"+str(slab)+"_left.png"
        order_right = concatenate_slices(df0, best_ligands, [75, 80], right_fn )
        if slab == 4 : percentiles = [ 20, 25 ]
        else : percentiles = [ 15, 20 ]
        order_left  = concatenate_slices(df0, best_ligands, percentiles, left_fn )
        slab_coreg_pts[slab] =  {"left":left_fn, "right":right_fn, "order_left":order_left, "order_right":order_right, "direction":direction, "side":side }
    
    transform_list_0 = transform_list_1 = []
    for i in slab_order :
        print("Slab", i)
        transform_list_0 = slab_registration(i,m, slab_coreg_tfm_dir, slab_coreg_rsl_dir, slab_coreg_pts, slab_coreg_qc_dir, transform_list_0, df, clobber )

    for f in  df["rsl"].loc[ df["slab"] == m ] :
        rsl_fn = slab_coreg_rsl_dir + os.sep + os.path.basename(f)
        copy(f, rsl_fn)

def receptorRegister(source_dir, output_dir, slice_order_fn, clobber, tiers_str, exclude_str="", ext=".png"):
    if not os.path.exists(output_dir) : 
        os.makedirs(output_dir)

    source_files=glob(source_dir+os.sep+ "**" + os.sep + "crop/*" + ext)
    source_files = preprocess(source_files, output_dir)
    df_rsl_fn = output_dir + os.sep + "df_rsl.csv"
    
    if source_files == []:
        print("Could not find files in ",source_dir)
        exit(1)
    
    df = set_csv(source_files,output_dir, "", exclude_str, slice_order_fn=slice_order_fn, clobber=clobber)
    df = setup_tiers(df, tiers_str)
    
    if not os.path.exists(df_rsl_fn) or clobber :
        df_rsl = apply_slice_registration(df,output_dir, clobber)
        df_rsl.to_csv(df_rsl_fn)
    else :
        df_rsl = pd.read_csv(df_rsl_fn)
    apply_slab_registration(df_rsl, output_dir)

    return 0



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--source', dest='source_dir', default=None,  help='Directory with raw images')
    parser.add_argument('--output', dest='output_dir',  help='Directory name for outputs')
    parser.add_argument('--slice-order', dest='slice_order_fn',  help='CSV file that contains ordering of slices')
    parser.add_argument('--tiers', dest='tiers_str', default='',  help='Tier of slices in , and ; string (Example "a,b,c;d,e,f;g,h,i)')
    parser.add_argument('--start-tier', dest='start_tier', default=1, type=int,  help='Starting tier to be run (allows to clobber for specific tiers)')
    parser.add_argument('--exclude', dest='exclude_str', default='',  help='Comma separated list of tracers to exclude')
    parser.add_argument('--step', dest='downsample_step', default="1", type=int, help='File extension for input files (default=.tif)')

    parser.add_argument('--ext', dest='ext', default=".png", help='File extension for input files (default=.tif)')
    parser.add_argument('--clobber', dest='clobber', action='store_true', default=False, help='Clobber results')

    args = parser.parse_args()

    receptorRegister(source_dir=args.source_dir, output_dir=args.output_dir, exclude_str=args.exclude_str, slice_order_fn=args.slice_order_fn, clobber=args.clobber, tiers_str=args.tiers_str , ext=args.ext )



