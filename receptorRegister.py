from glob import glob
from sys import argv, exit
from os.path import basename
import os
import pandas as pd
import cv2
import json
import numpy as np
#from sitk_register import register
from elastix_register import register
import scipy.misc
from utils.utils import *
from utils.anisotropic_diffusion import anisodiff
import imageio
import argparse
from re import sub
from skimage.exposure import  equalize_hist
from utils.utils import imadjust
import matplotlib.pyplot as plt

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

def create_coregistration_df(slab, df_fn):

    m = int(slab.shape[0]/2)

    if not os.path.exists(df_fn) :
        bottom = list(range(m))
        bottom.reverse()
        top = list(range(m,slab.shape[0]))
        
        order_list_top = np.array( list(slab.order.values[top])  )
        order_list_bottom = np.array( list(slab.order.values[bottom])  )
        filenames_top = np.array( list(slab.filename.values[top]) )
        filenames_bottom = np.array( list(slab.filename.values[bottom]) )
        tier_list_top = np.array( list(slab.tier.values[top]) )
        tier_list_bottom = np.array( list(slab.tier.values[bottom])  )

        masks_list_top = np.array( list(slab.masks.values[top]) )
        masks_list_bottom = np.array( list(slab.masks.values[bottom])  )

        ligand_list_top = np.array( list(slab.ligand.values[top]) )
        ligand_list_bottom = np.array( list(slab.ligand.values[bottom])  )


        df_top = pd.DataFrame({"i": top, "tier":tier_list_top, "ligand":ligand_list_top, "filenames":filenames_top, "order": order_list_top, "masks":masks_list_top})

        df_bottom=pd.DataFrame({"i": bottom, "tier":tier_list_bottom, "ligand":ligand_list_bottom,"filenames":filenames_bottom, "order": order_list_bottom, "masks":masks_list_bottom })
        
        #base_row = pd.DataFrame( df_top.loc[ df_top.tier == 1 ].iloc[0,] ).T
        #base_row.index = [df_top.shape[0] ]
        #df_top = df_top.append(base_row)
        df = pd.concat([df_top, df_bottom])

        df["nmi"] = [0.] * df.shape[0]
        df.index=range(df.shape[0])
        df["rsl"] = df["filenames"]  #["empty"] * df.shape[0]

        df.to_csv(df_fn)
    else :
        df = pd.read_csv(df_fn)

    return df, m

from sklearn.metrics import normalized_mutual_info_score

def nmi(fixed_fn, rsl_fn):
    fixed = imageio.imread(fixed_fn).reshape(-1,)
    rsl = imageio.imread(rsl_fn).reshape(-1,)
    n = normalized_mutual_info_score(fixed, rsl)
    #print("NMI: ", n)

    return n

def create_cmap(x):
    xuni = np.unique(x)
    xunienum = enumerate(xuni)
    d=dict(xunienum)
    e = { v:k for k,v in  d.items() }
    cmap = np.array([ e[i]  for i in x ])
    cmap = cmap / np.max(cmap)
    return cmap

def slice_nmi(df):
    for i in range(df.shape[0]) :
        #if you use index to acces the rows, then you will start at the end of the slab
        row=df.iloc[i,]
        j=i+1
        k=j+1
        if j >=  df.shape[0] : continue
        fixed=df.filenames.iloc[i,] 

        if df.rsl.iloc[j,] == "" : 
            moving=df.filenames.iloc[j,]
        else :
            moving=df.rsl.iloc[j,]
        df["nmi"].iloc[ i, ] = nmi(fixed, moving)
    return df

def slice_coregistration(df,resolutions, max_iterations, max_length, output_dir, qc_dir, m, tier, clobber=False) :

    for i in range(df.shape[0]) :
        #if you use index to acces the rows, then you will start at the end of the slab
        row=df.iloc[i,]
        
        if row["tier"] != tier : continue
        
        j=i-1
        if  j < 0: continue
        if df.index[i,] == m : continue
        order_fixed = df.iloc[j,].order
        order_moving = df.iloc[i,].order
        fixed=df.filenames.iloc[j,]
        fixed_mask = df.masks.iloc[j,]

        moving=df.rsl.iloc[i,]
        moving_mask = df.masks.iloc[i,]


        moving_split = os.path.splitext(moving)

        qc_fn=qc_dir+os.sep+str(order_moving)+"_"+os.path.basename(moving_split[0])+'_reg_qc.png' 
        transform_fn=output_dir+os.sep+str(order_moving)+"_"+os.path.basename(moving_split[0])+'_reg.tfm'
        rsl_fn=output_dir+os.sep+str(order_moving)+"_"+os.path.basename(moving_split[0])+'_rsl.png' 
        if not os.path.exists(qc_fn) or clobber  :
            print(row)
            print("Moving:\t", order_moving,"\t", moving)
            print("Fixed:\t", order_fixed,"\t", fixed)
            register(fixed, moving, fixed_mask, moving_mask, rsl_fn, qc_fn, transform_fn, order_fixed, order_moving, resolutions, max_iterations, max_length )
           # pass
        df["rsl"].loc[ df.filenames == moving] = rsl_fn

    return df

#def slab_coregistration(slab, df_master, output_dir, qc_dir, start_tier, clobber=False):

def apply_function(df, output_dir, clobber=False):
    qc_dir = output_dir + os.sep + "qc"
    if not os.path.exists(qc_dir) : os.makedirs(qc_dir)
    tiers = np.sort(np.unique(df["tier"].values))
    for slab, df0 in df.groupby(["slab"]) :
        df1 = df0.copy()
        df1, m = create_coregistration_df(df1, output_dir+os.sep+str(slab)+"_coregistration_df.csv")
        df1 = df1.loc[ df1["order"] >= 0 ]


        resolutions=["24 24",  "16 16", "8 8", "4 4", "1 1"]
        base_iterations = max_iterations = 500
        max_length = 50
        for t in tiers :
            tier = df1.loc[ df1["tier"] <= t ]
            cmap = create_cmap(df["ligand"].values)
            
            slice_coregistration(tier,resolutions, max_iterations, max_length,output_dir, qc_dir, m, t, clobber=clobber)
            
            max_iterations += base_iterations
            #df = slice_nmi(df)
            #plt.scatter(range(df.shape[0]), df["nmi"].values, c=cmap )
            #plt.savefig(qc_dir +os.sep+"qc_"+str(_pass)+"_iter-"+str(max_iterations)+"_length-"+str(max_length)+".png")
            #plt.clf()
        exit(0)



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
            slab_scale_factor = scale[mr][hemisphere][slab]

            ### 2. Read Image
            img = imageio.imread(f)
            if len(img.shape) > 2 : img = np.mean(img, axis=2)
            
            ### 3. Downsample Image
            new_step=0.2
            old_step = slab_scale_factor / 4164

            #Calculate length of image based on assumption that pixels are 0.02 x 0.02 microns
            l0 = img.shape[0] * old_step 
            l1 = img.shape[1] * 0.02 #FIXME : NOT SURE IF THIS IS 100% ACCURATE!

            #Calculate the length for the downsampled image
            dim0=int(np.ceil(l0 / new_step))
            dim1=int(np.ceil(l1 / new_step))

            dim0_list.append(dim0)
            dim1_list.append(dim1)
            #Calculate the standard deviation based on a FWHM (pixel step size) of downsampled image
            sd0 = new_step / 2.634 
            sd1 = new_step / 2.634 

            img_blr = gaussian_filter(img, sigma=[sd0, sd1])
            img_dwn = scipy.misc.imresize(img_blr,size=(dim0, dim1) )
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
    create_masks(temp_fn_list, output_dir)
    zmax, xmax = get_z_x_max(temp_fn_list, output_dir)
    prep_files = hist_and_pad(temp_fn_list, output_dir, zmax, xmax)
    return prep_files


def receptorRegister(source_dir, output_dir, slice_order_fn, clobber, tiers_str, exclude_str="", ext=".png"):
    if not os.path.exists(output_dir) : 
        os.makedirs(output_dir)

    source_files=glob(source_dir+os.sep+ "**" + os.sep + "crop/*" + ext)
    source_files = preprocess(source_files, output_dir)
    if source_files == []:
        print("Could not find files in ",source_dir)
        exit(1)
    df = set_csv(source_files,output_dir, "", exclude_str, slice_order_fn=slice_order_fn, clobber=clobber)
    df = setup_tiers(df, tiers_str)
    df["masks"] = df["filename"].apply( lambda x : re.sub("preprocess", "masks", x)) 
    apply_function(df,output_dir, clobber)

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



