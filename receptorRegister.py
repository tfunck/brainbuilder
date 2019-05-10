from glob import glob
from sys import argv, exit
from os.path import basename
from skimage.filters import try_all_threshold, threshold_mean
from scipy.ndimage.morphology import binary_dilation, binary_erosion, binary_fill_holes
from utils.utils import *
from utils.anisotropic_diffusion import anisodiff
from re import sub
from skimage.exposure import  equalize_hist
from skimage.transform import resize 
from utils.utils import imadjust, shell
from shutil import copy
from math import ceil
import os
import pandas as pd
import cv2
import json
import numpy as np
import imageio
import argparse
import SimpleITK as sitk
import scipy.misc
import matplotlib.pyplot as plt


def resample(moving_fn, transform_fn_list,  rsl_fn="" , ndim=2):
    movingImage = sitk.ReadImage(moving_fn)
    composite = sitk.Transform(ndim, sitk.sitkComposite )
    for fn in transform_fn_list :
        transform = sitk.ReadTransform(fn)
        composite.AddTransform(transform)
   
    interpolator = sitk.sitkCosineWindowedSinc
    rslImage = sitk.Resample(movingImage, composite, interpolator, 0.)
    rsl = np.copy(sitk.GetArrayViewFromImage(rslImage))

    if ndim == 2 :
        if rsl_fn != "" : imageio.imsave(rsl_fn, rsl)
    else :
        writer = sitk.ImageFileWriter()
        writer.SetFileName(rsl_fn)
        writer.Execute(rslImage)

    return rsl 

def display_images_with_alpha( alpha, fixed, moving, moving_resampled, fn, order_fixed, order_moving, fixed_tier, moving_tier, metric=0):
    fixed_npa = (fixed - fixed.min() ) / (fixed.max() - fixed.min())  #imageio.imread(fixed)
    moving_resampled_npa =(moving_resampled-moving_resampled.min()) / (moving_resampled.max() - moving_resampled.min())  #imageio.imread(moving_resampled)
    extent = 0, fixed_npa.shape[1], 0, fixed_npa.shape[0]
    
    plt.title( 'moving:'+str(moving_tier)+ ' fixed'+ str(order_moving))
    plt.imshow(fixed_npa, cmap=plt.cm.gray, interpolation='bilinear', extent=extent)
    plt.imshow(moving_resampled_npa, cmap=plt.cm.hot, alpha=0.35, interpolation='bilinear', extent=extent)
    plt.title('rsl moving vs fixed')
    plt.axis('off')


    plt.axis('off')
    plt.tight_layout()

    plt.savefig(fn, dpi=200,bbox_inches="tight" )
    plt.clf()
    return 0 

def get_z_x_max(source_files, output_dir, clobber=False):
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


def slice_coregistration(df, output_dir, qc_dir, m, tier, transform_dict, clobber=False) :
    transform_dir = output_dir + os.sep + "transforms"
    resample_dir = output_dir + os.sep + "resample"
    if not os.path.exists(transform_dir) : os.makedirs(transform_dir)
    if not os.path.exists(resample_dir) : os.makedirs(resample_dir)

    params = []
    transform_fn_list = []

    mass=[] 
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
        transform_base_fn=transform_dir+os.sep+str(order_moving)+"_"+os.path.basename(moving_split[0])+'_'
        transform_fn=transform_base_fn+'0GenericAffine.mat'
        rsl_fn=resample_dir+os.sep+str(order_moving)+"_"+os.path.basename(moving_split[0])+'_rsl.png' 

        if row["tier"] == tier :
            transform_dict[ order_moving  ] = [ transform_fn ] + transform_dict[ order_fixed ] 
            #transform_dict[ order_moving  ] =  transform_dict[ order_fixed ] + [ transform_fn ] 
            if not os.path.exists(qc_fn) or not os.path.exists(transform_fn) or clobber  :
                print("Fixed:\t", order_fixed,"\t", fixed)
                print('Moving', moving)
                
                moving_image_mask = mask(moving)
                fixed_image_mask = mask(fixed)

                cmdline="antsRegistration --verbose 1 --float --collapse-output-transforms 1 --dimensionality 2 --initial-moving-transform [ "+fixed+", "+moving+", 1 ] --interpolation  Linear --transform Rigid[ 0.1 ] --metric Mattes[ "+fixed+", "+moving+", 1, 64, Regular, 0.9 ] --convergence [ 1000x750x500x250, 1e-08, 10 ] --smoothing-sigmas 20.0x15.0x10.0x1.0vox --shrink-factors 8x4x2x1  --use-histogram-matching 1  --output [ "+transform_base_fn+" ] "
                shell(cmdline)

                print("Transform:", transform_dict[ order_moving][0])
                rsl = resample(moving,  [transform_dict[ order_moving][0]] )
                display_images_with_alpha( 0.5, imageio.imread(fixed), imageio.imread(moving), rsl, qc_fn,  order_fixed, order_moving, fixed_tier, moving_tier)
                print(transform_dict[order_moving])
                rsl = resample(moving,  [transform_dict[ order_moving][0]] )
                fixed_ar=imageio.imread(fixed)
                moving_ar=imageio.imread(moving)
                plt.subplot(2,2,1)
                plt.imshow(fixed_ar)
                plt.subplot(2,2,2)
                plt.imshow(moving_ar)
                plt.subplot(2,2,3)
                plt.imshow(moving_ar + rsl)
                plt.subplot(2,2,4)
                plt.imshow(fixed_ar + rsl)
                plt.show()

            rsl = resample(moving,  [transform_dict[ order_moving][0]] )
            if not os.path.exists(rsl_fn) or clobber :
                resample(moving, transform_dict[ order_moving ], rsl_fn)
                
        df["rsl"].loc[ df.filenames == moving] = rsl_fn
   
    
    return df

#def slab_coregistration(slab, df_master, output_dir, qc_dir, start_tier, clobber=False):

def apply_slice_registration(df, output_dir,slabs_to_run=[], clobber=False):
    df = df.loc[ df["order"].astype(int) >= 0 ]
    df["rsl"] = df["filename"] #[""] * df.shape[0]
    qc_dir = output_dir + os.sep + "qc"
    if not os.path.exists(qc_dir) : os.makedirs(qc_dir)
    tiers = np.sort(np.unique(df["tier"].values))
    
    transform_dict = {}
    for o in df.order : transform_dict[ o ] = []

    for slab, df0 in df.groupby(["slab"]) :
        if not slab in slabs_to_run and slabs_to_run != [] : continue
        for t in tiers :
            print("Slab: ", slab, "Tier :", t)
            df1 = df0.loc[ df0["tier"] <= t ].copy()
            df2, m = create_coregistration_df(df1, output_dir+os.sep+str(slab)+"_coregistration_df.csv")
            print(df2.iloc[m,:])
            print(m); exit(0) 
            df2.to_csv(output_dir+os.sep+"slab-"+str(slab) +"_tier-"+str(t)+"_df.csv") 

            df_rsl = slice_coregistration(df2,output_dir,qc_dir,m, t, transform_dict, clobber=clobber)
            for f in df_rsl.filenames :
                df["rsl"].loc[ df.filename == f ] = df_rsl["rsl"].loc[ df_rsl.filenames == f ].values[0]

    
    composite_transform_fn = output_dir+'/composite_transforms.json'
    json.dump(transform_dict, open(composite_transform_fn, 'w')) 
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



def downsample_slices(source_files, output_dir):
    common_step = 0.2
    

    with open("scale_factors.json") as f : scale=json.load(f)
    temp_fn_list = []
    for f in source_files :
        print(f)
        f2 = output_dir + os.sep +"preprocess"+ os.sep + os.path.basename(f)
        temp_fn = output_dir + os.sep +"preprocess"+os.sep+"downsample"+ os.sep + os.path.splitext(os.path.basename(f))[0] + '.png'
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
            
            ##Calculate the standard deviation based on a FWHM (pixel step size) of downsampled image
            step_0 = slab_size /  4164

            nvox_0 = ceil( img.shape[0] * step_0 / common_step )
            nvox_1 = ceil( img.shape[1] * step_0 / common_step )

            img_dwn = resize(img, output_shape=(nvox_0, nvox_1) )

            imageio.imsave(temp_fn, img_dwn)
    return temp_fn_list

def hist_and_pad(temp_fn_list, output_dir, zmax, xmax, clobber=False):
    prep_files = []
    ######################################################
    # Load downsampled images, adjust histogram and pad  #
    ######################################################
    low_contrast_ligands=["oxot", "epib", "ampa", "uk14", "mk80" ]
    for f in temp_fn_list :
        f2 = output_dir + os.sep +"preprocess"+ os.sep + os.path.basename(f)
            
        if not os.path.exists(f2) or clobber:
            ### 1. Read Image
            img = imageio.imread(f)
            if len(img.shape) > 2 : img = np.mean(img, axis=2)
            #if True in [ True for i in low_contrast_ligands if i in f ] : 
            #    idx = img > 0
            #    img[idx] = equalize_hist(img[idx])
            #img = add_padding(img, zmax, xmax)
            imageio.imsave(f2, img)
            ii = imageio.imread(f2)
            prep_files.append(f2)
        else :
            prep_files.append(f2)
    return prep_files


def mask(fn):
    out_fn = os.path.splitext(fn)[0]+"_mask.png"
    if not os.path.exists(out_fn) :
        img = plt.imread(fn)
        seg=KMeans(n_clusters=3, init=np.array([[0], [np.max(img)*0.1], [np.max(img)]]) ).fit(img.reshape(-1,1)).labels_.reshape(img.shape)
        seg[ seg != 2 ] = 0
        imageio.imsave(out_fn, seg)
    return out_fn

def preprocess(source_files,  output_dir, step=0.2, ystep=0.02,clobber=False):
    if not os.path.exists(output_dir + os.sep +"preprocess"+os.sep+"downsample") :
        os.makedirs(output_dir + os.sep +"preprocess"+os.sep+"downsample")
    temp_fn_list = downsample_slices(source_files, output_dir)

    zmax, xmax = get_z_x_max(temp_fn_list, output_dir, clobber)
    prep_files = hist_and_pad(temp_fn_list, output_dir, zmax, xmax, clobber)
    return prep_files


def receptorRegister(source_dir, output_dir, slice_order_fn, clobber, tiers_str, exclude_str="", slabs_to_run=[], ext=".png"):
    if not os.path.exists(output_dir) : 
        os.makedirs(output_dir)

    source_files=glob(source_dir+os.sep+ "*" + ext)
    source_files = preprocess(source_files, output_dir,clobber)
    df_rsl_fn = output_dir + os.sep + "df_rsl.csv"
    
    if source_files == []:
        print("Could not find files in ",source_dir)
        exit(1)
    
    df = set_csv(source_files,output_dir, "", exclude_str, slice_order_fn=slice_order_fn, clobber=clobber)
    df = setup_tiers(df, tiers_str)
    
    df_rsl = apply_slice_registration(df,output_dir,slabs_to_run, clobber)
    df_rsl.to_csv(df_rsl_fn)

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
    parser.add_argument('--slabs', dest='slabs_to_run', nargs='+', default=[], help='Slabs to process into volume')

    parser.add_argument('--ext', dest='ext', default=".png", help='File extension for input files (default=.tif)')
    parser.add_argument('--clobber', dest='clobber', action='store_true', default=False, help='Clobber results')

    args = parser.parse_args()

    receptorRegister(source_dir=args.source_dir, output_dir=args.output_dir, exclude_str=args.exclude_str, slice_order_fn=args.slice_order_fn, clobber=args.clobber, tiers_str=args.tiers_str , slabs_to_run=args.slabs_to_run, ext=args.ext )



