from utils import utils
from pyminc.volumes.factory import *
import numpy as np
import re
import imageio
import scipy.misc
from scipy.interpolate import Rbf, RegularGridInterpolator
import matplotlib.pyplot as plt
from glob import glob
import os
from sitk_register import * 
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
        order = int(row.order)
        out_fn=output_dir+os.sep+str(order)+"_mri_rsl_slab-" + str(slab)+".png"
        if not os.path.exists(out_fn) :
            check=1
            break
    if check == 1 or clobber : 
        return False

    return True

def realign_receptor_images(source_files, output_dir, clobber=False):
    rec_rsl_dir = output_dir +os.sep+"receptor"
    if not os.path.exists(rec_rsl_dir) : os.makedirs(rec_rsl_dir)
    out_list = []

    for filename in source_files :
        out_fn = rec_rsl_dir + os.sep+ os.path.basename(filename)
        if not os.path.exists(out_fn) or clobber :
            print("Realigning:", out_fn)
            img = np.flip( imageio.imread(filename).T, 1)
            scipy.misc.imsave(out_fn, img)
        out_list.append( out_fn )
    return out_list
        


    

def reslice_mri(df0, mri_string, output_dir, clobber=False):
    if not os.path.exists(output_dir) : os.makedirs(output_dir)
    
    if check_files(df0, output_dir, clobber): return 0
    
    #1. load slice
    img = imageio.imread(source_files[0])
    img = img.T
    img = np.flip(img, 1)

    total_slices = 7402
    slice_thickness = 0.02

    #1.3
    #rec = volumeFromFile("receptor_to_mri/input/receptor/MR1_R_slab_3_receptor.mnc.gz")
    zstep=0.2 #vol.separations[0]
    ystep=0.2 #vol.separations[1]
    xstep=0.2 #vol.separations[2]

    zlength=750 #vol.sizes[0]
    ylength=741 #vol.sizes[1]
    xlength=548 #vol.sizes[2]

    zmin=-90 #vol.starts[0]
    ymin=-126 #vol.starts[1]
    xmin=-72 #vol.starts[2]

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
        mri_fn=glob(mri_string+str(slab)+"*mnc*")[0]
        if not os.path.exists(mri_fn) :
            print("Error: could not find file for", mri_string+str(slab)+"*mnc*" )
            exit(1)
        mri = volumeFromFile(mri_fn)

        xspace=np.linspace(mri.starts[2],mri.starts[2]+mri.separations[2]*mri.sizes[2],mri.separations[2]*mri.sizes[2])
        yspace=np.linspace(mri.starts[1],mri.starts[1]+mri.separations[1]*mri.sizes[1],mri.separations[1]*mri.sizes[1])
        zspace=np.linspace(mri.starts[0],mri.starts[0]+mri.separations[0]*mri.sizes[0],mri.separations[0]*mri.sizes[0])
        interpolator = RegularGridInterpolator((zspace, yspace, xspace), mri.data, bounds_error=False, fill_value=0 )
        
        for i, row in df.iterrows() :
            order = int(row.order)
            out_fn=output_dir+os.sep+str(order)+"_mri_rsl_slab-" + str(slab)+".png"
            if  os.path.exists(out_fn) and not clobber : continue
            
            #3. define 3 points in receptor world space for plane of slice
            y =  ymin + slice_thickness * ( total_slices - order)
            pts[1,:] = y
            
            #Load mri to receptor xfm file and invert it
            tag_xfm_fn="receptor_to_mri/output/transforms/tag_lin_"+str(slab)+".xfm"
            tfm_mri_to_rec = read_xfm_matrix(tag_xfm_fn)
            tfm_rec_to_mri = np.linalg.inv(tfm_mri_to_rec)
            
            #4. multiply points by transform matrix for slab
            new_pts = np.matmul(tfm_rec_to_mri , pts).T
            new_pts = new_pts[:, 0:3]

            #5 interpolate mri based on oblique plane
            mri_plane = interpolator(new_pts[:,[2,1,0]])
            mri_plane = mri_plane.reshape(len(rec_zspace),len(rec_xspace))

            print("Writing", out_fn)
            scipy.misc.imsave(out_fn , np.flip( mri_plane, 0) )
    return 0

def receptor_to_mri_alignment(df, output_dir, clobber) :
    mri_dir = output_dir + os.sep + "mri"
    tfm_dir = output_dir + os.sep + "transforms"
    qc_dir = output_dir + os.sep + "qc"
    if not os.path.exists(tfm_dir) : os.makedirs(tfm_dir)
    if not os.path.exists(qc_dir) : os.makedirs(qc_dir)

    resolutions=[  5, 4,  3, 2, 1]
    base_iterations = max_iterations = 1000000
    max_length = 1
    for i, row in df.iterrows() :
        order = int(row.order)
        slab = row.slab
        if row.slab != 3 or order < 2800 : continue
        rec_fn = row.filename #fixed
        mri_fn = mri_dir+os.sep+str(order)+"_mri_rsl_slab-" + str(slab)+".png"
        transform_fn = tfm_dir+os.sep + str(order)+"_tfm_slab-" + str(slab)+".txt"
        qc_fn=qc_dir+os.sep+str(order)+"_tfm_slab-" + str(slab)+".png"
        print("Receptor:", rec_fn)
        print("MRI:", mri_fn)
        
        #receptor_show(rec_fn, mri_fn, str(order) )

        if not os.path.exists(qc_fn) or not os.path.exists(transform_fn) or clobber  :
            register(rec_fn,mri_fn,transform_fn,resolutions,max_iterations,max_length,transform_type="AffineTransform")
        
            rsl = resample(mri_fn, rec_fn, [transform_fn] )
        
            display_images_with_alpha(0.5,imageio.imread(rec_fn),imageio.imread(mri_fn),rsl,qc_fn,order, order, 0, 0)



mri_string="receptor_to_mri/input/mri/mr1_mri_space-slab-"
clobber=False
source_files=glob("MR1/coregistration/slab_coregistration/resample/*png")
source_files = realign_receptor_images(source_files, "temp", clobber=False)
df = utils.set_csv(source_files, "temp", clobber=clobber, df_with_order=clobber )
reslice_mri(df, mri_string, "temp/mri" ,clobber=clobber)
clobber=False
receptor_to_mri_alignment(df, "temp", clobber)
