from utils import utils
from pyminc.volumes.factory import *
import numpy as np
import re
import imageio
import scipy.misc
from scipy.interpolate import Rbf, RegularGridInterpolator
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from glob import glob
import os
from sitk_register import * 
import json


def segment_rec(df, out_dir, clobber=False):
    gm_dir = out_dir +os.sep + "gm"
    if not os.path.exists(gm_dir) or clobber :
        os.makedirs(gm_dir)

    for i, row in df.iterrows():
        slab = row.slab
        order = row['order']
        out_fn=gm_dir+os.sep+str(order)+"_slab-" + str(slab)+"_gm.png"
        if not os.path.exists(out_fn) or clobber : 
            img = imageio.imread(row.filename) 
            img = utils.gm_segment(img)
            imageio.imwrite(out_fn, img)
        df.filename.iloc[i,] = out_fn

    return(df)


def read_xfm_matrix(fn): 
    f = open(fn)

    ar=[]
    read=False
    for l in f.readlines() :
        if read : 
            l = re.sub("\n", "", re.sub(";", "", l))
            l =[ float(i)  for i in l.split(" ") if i != '' ] 
            ar += l
        if "Linear_Transform" in l : read = True
    
    ar += [0,0,0,1]
    ar = np.array(ar).reshape(4,4)
    
    return ar

def check_files(df,output_dir, clobber) :
    check=0
    for slab_i, row in df.iterrows() :
        slab = row.slab
        order = row['order']
        out_fn=output_dir+os.sep+str(order)+"_mri_rsl_slab-" + str(slab)+".png"
        if not os.path.exists(out_fn) :
            check=1
            break
    if check == 1 or clobber : 
        return False

    return True

def realign_receptor_images(df, output_dir, clobber=False):
    with open("scale_factors.json") as f : scale=json.load(f)
    rec_rsl_dir = output_dir +os.sep+"receptor"
    if not os.path.exists(rec_rsl_dir) : os.makedirs(rec_rsl_dir)
    out_list = []

    for i, row in df.iterrows() :
        filename = row["filename"]
        order = row["order"]
        out_fn = rec_rsl_dir + os.sep+ os.path.basename(filename)
        if not os.path.exists(out_fn) or clobber : 
            slab = row["slab"]
            hemi = row["hemisphere"]
            mr = row["mri"]
            direction = scale[mr][hemi][str(slab)]["direction"]
            print("Realigning from:", filename)
            print("Realigning to:", out_fn)
            print()
            img =imageio.imread(filename)
            img = img.T
            if direction == "caudal_to_rostral":
                img = np.flip( img, 0)
            img = np.flip( img, 1)
            scipy.misc.imsave(out_fn, img)
        #out_list.append( out_fn )
        df["filename"].iloc[i] = out_fn
    return df
        


    

def reslice_mri(df0, mri_string, output_dir, clobber=False):
    if not os.path.exists(output_dir) : os.makedirs(output_dir)
    
    if check_files(df0, output_dir, clobber): return 0
    
    #1. load slice
    #img = imageio.imread(df0.filename[0])
    #img = img.T
    #img = np.flip(img, 1)

    total_slices = 7402
    slice_thickness = 0.02

    #1.3
    #vol = volumeFromFile("receptor_to_mri/input/receptor/MR1_R_slab_3_receptor.mnc.gz")
    vol = volumeFromFile("MR1/volume/vol_cls_1.mnc")
    zstep=vol.separations[0] #0.2   #
    ystep=vol.separations[1] #0.02  #
    xstep=vol.separations[2] #0.2   #

    zlength=vol.sizes[0] #750 #
    ylength=vol.sizes[1] #741 #
    xlength=vol.sizes[2] #548 #

    zmin=vol.starts[0] #-90 #
    ymin=vol.starts[1] #-126 #
    xmin=vol.starts[2] #-72 #
    del vol

    rec_xspace = np.linspace( xmin, xmin + xstep*xlength, xstep*xlength  / xstep ) #slice_thickness )
    rec_zspace = np.linspace( zmin, zmin + zstep*zlength, zstep*zlength  / zstep ) #slice_thickness )
    xx, zz = np.meshgrid(rec_xspace, rec_zspace)
    xx=xx.reshape(-1,)
    zz=zz.reshape(-1,)
    n=zz.shape[0]
    pts = np.array([xx,[0]*n,zz,[1]*n])

    for slab_i, df in df0.groupby("slab"):
        slab = df.slab.values[0]
        #1.1 load mri & create grid for interpolation
        try :
            temp_fn = re.sub('X', str(slab), mri_string)
            mri_fn=glob(temp_fn)[0]
        except IndexError :
            print("Error: could not find file for", mri_string+str(slab)+"*mnc*")
            exit(1)

        if not os.path.exists(mri_fn) :
            print("Error: could not find file for", mri_string+str(slab)+"*mnc*" )
            exit(1)
        print(mri_fn)
        mri = volumeFromFile(mri_fn)

        print("MRI dimensions:")
        print(mri.starts)
        print(mri.separations)
        print(mri.sizes[0], mri.sizes[1], mri.sizes[2])
        xspace=np.linspace(mri.starts[2],mri.starts[2]+mri.separations[2]*mri.sizes[2],mri.sizes[2])
        yspace=np.linspace(mri.starts[1],mri.starts[1]+mri.separations[1]*mri.sizes[1],mri.sizes[1])
        zspace=np.linspace(mri.starts[0],mri.starts[0]+mri.separations[0]*mri.sizes[0],mri.sizes[0])
        print(len(zspace), len(yspace), len(xspace)) 

        interpolator = RegularGridInterpolator((zspace, yspace, xspace), mri.data, bounds_error=False, fill_value=0 )
        
        order_max = 1444 #df['order'].max()
        order_min = df['order'].min()
        for i, row in df.iterrows() :
            order = row['order']
            print(order, order_max)
            out_fn=output_dir+os.sep+str(order)+"_mri_rsl_slab-" + str(slab)+".png"
            if  os.path.exists(out_fn) and not clobber : continue
            
            #3. define 3 points in receptor world space for plane of slice
            print("Order:", order)
            print("Voxel:",order_max , order , order_min , order_max - (order - order_min) )

            y =  ymin + slice_thickness * (order_max - (order - order_min)) #( total_slices - order)
            pts[1,:] = y
            print(pts.shape) 
            print("rec y=",y)
            #Load mri to receptor xfm file and invert it
            #tag_xfm_fn="receptor_to_mri/output/srv/srv_space-rec-"+str(slab)+".tfm" 
            #tfm_mri_to_rec = read_xfm_matrix(tag_xfm_fn)
            #tfm_rec_to_mri = np.linalg.inv(tfm_mri_to_rec)
            
            #4. multiply points by transform matrix for slab
            #new_pts = np.matmul(tfm_rec_to_mri , pts).T
            #new_pts = new_pts[:, 0:3]
            #print("mri y: ", new_pts[0,:])
            #5 interpolate mri based on oblique plane
            #mri_plane = interpolator(new_pts[:,[2,1,0]])
            #mri_plane = interpolator(pts[:,[2,1,0]])
            mri_plane = interpolator(pts[0:3].T)
            print(mri_plane.shape)
            mri_plane = mri_plane.reshape(len(rec_zspace),len(rec_xspace))
            plt.subplot(2,1,1)
            plt.imshow(imageio.imread(row.filename))
            plt.subplot(2,1,2)
            plt.imshow(mri_plane)
            plt.show()
            print("Writing", out_fn)
            scipy.misc.imsave(out_fn , np.flip( mri_plane, 0) )
            exit(0)
    return 0



def receptor_to_mri_alignment(df, output_dir, clobber) :
    mri_dir = output_dir + os.sep + "mri"
    tfm_dir = output_dir + os.sep + "transforms"
    qc_dir = output_dir + os.sep + "qc"
    rsl_dir = output_dir + os.sep + "rsl"
    if not os.path.exists(tfm_dir) : os.makedirs(tfm_dir)
    if not os.path.exists(rsl_dir) : os.makedirs(rsl_dir)
    if not os.path.exists(qc_dir) : os.makedirs(qc_dir)
    metric_list=[]
    iterations_list=[]
    resolutions=[  1 ] #[  5, 4,  3, 2, 1
    problematic_list=[530]
    base_iterations = max_iterations = 100000000
    for i, row in df.iterrows() :
        order = row['order']
        slab = row.slab
        #if not slab in [1,2,3] : continue 
        #if not order in problematic_list : continue
        #if not ( slab == 3 and order >= 3668 ) : continue
        #if slab == 3  : continue
        #if order > 3200 or order < 3000 : continue
        #if not order in problematic_list : continue
        rec_fn = row.filename #fixed
        mri_fn = mri_dir+os.sep+str(order)+"_mri_rsl_slab-" + str(slab)+".png"
        transform_fn = tfm_dir+os.sep + str(order)+"_tfm_slab-" + str(slab)+".txt"
        qc_fn=qc_dir+os.sep+str(order)+"_tfm_slab-" + str(slab)+".png"
        rsl_fn=rsl_dir+os.sep+str(order)+"_tfm_slab-" + str(slab)+".png"
        if not order % 10 == 0 or order < 500  : continue 
        #if not order in problematic_list : continue
        rsl=None
        if  not os.path.exists(transform_fn) or not os.path.exists(qc_fn) or clobber  :
            print(not os.path.exists(transform_fn))
            print(not os.path.exists(qc_fn))
            print(clobber)
            print("MRI:", mri_fn)
            print("QC: ", qc_fn)
            metric, iterations = register(rec_fn,mri_fn,transform_fn,resolutions,max_iterations,transform_type="AffineTransform",numberOfHistogramBins=256, stepLength=0.0005, valueTolerance=1e-10, stepTolerance=1e-10, mask=False, invert_transform=True)
            metric_list.append(metric)
            iterations_list.append(iterations)
            
            rsl = resample(rec_fn, [transform_fn] )
            display_images_with_alpha(0.5,imageio.imread(rec_fn),imageio.imread(mri_fn),rsl,qc_fn,order, order, 0, 0, metric=metric)

        if not os.path.exists(rsl_fn) or clobber :
            print("Receptor:", rec_fn)
            print("RSL: ", rsl_fn)
            rsl = resample(rec_fn, [transform_fn] )
            imageio.imsave(rsl_fn, rsl)
    print("Metric Mean:", np.mean(metric_list), "Metric Std Dev:", np.std(metric_list))
    print("Iterations Mean:", np.mean(iterations_list), "Metric Std Dev:", np.std(iterations_list))


def receptor_mri_2d(source_dir="MR1/coregistration/slab_coregistration/resample/", mri_string="receptor_to_mri/output/srv/mri1_gm_srv_space-slab-X.mnc", out_dir="temp"):

    source_files=glob(source_dir+"*png")
    print("Set csv")
    df = utils.set_csv(source_files, out_dir, clobber=False, df_with_order=True )
    print("Realign receptor images")
    df = realign_receptor_images(df, "temp", clobber=False)
    
    df.to_csv(out_dir+os.sep+"receptor_slices.csv", index=False)
    print("Reslice MRI")
    reslice_mri(df, mri_string, out_dir+"/mri", clobber=False)
    exit(0) 
    print("GM Segmentation")
    df = segment_rec(df, out_dir)
    print("Align to MRI")
    receptor_to_mri_alignment(df, out_dir, clobber=False)
