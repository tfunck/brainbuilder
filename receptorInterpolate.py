import pandas as pd
import numpy as np
import nibabel as nib
import sys
import os
import matplotlib.pyplot as plt
from imageio import imwrite
from ants import registration, image_read, apply_transforms


def r(img):
    imwrite("/tmp/tmp.png", img)
    return image_read(  "/tmp/tmp.png" )

def get_ligand_slices(fn, ligand, slab):
    print("\t\tFind coronal voxel values for ",ligand,"in slab",int(slab))
    df = pd.read_csv(fn)
    if not ligand in df["ligand"].values :
        print("Error: could not find ligand ", ligand, "in ", fn)
        exit(1)
    order_max =df["order"].loc[ (df["slab"] == int(slab)) ].max() 
    order = df["order"].loc[ (df["ligand"] == ligand) & (df["slab"] == int(slab)) ]
    slice_location = np.flip(np.array(order_max - order ).astype(int), axis=0  ) 
    return slice_location


def alignLigandToSRV(slab , ligand, srv_fn, cls_fn, output_dir, slice_location, clobber=False):
    print("\t\tNonlinear alignment of coronal sections for ligand", ligand, "for slab", int(slab))
    srv=None
    cls=None
    tfm={}
    for _y0 in slice_location : 

        prefix = output_dir + os.sep + ''.join(["slab-",str(slab),"_ligand-",ligand,"_y-",str(_y0), "_"])
        if not os.path.exists(prefix+"1Warp.nii.gz") or clobber :

            if srv == None :
                srv_img = nib.load(srv_fn)
                srv = srv_img.get_data()
            if cls == None :
                cls = nib.load(cls_fn).get_data()

            srv_slice =r( srv[ :, int(_y0), : ] )
            cls_slice =r( cls[ :, int(_y0), : ] )
            reg = registration(fixed=srv_slice, moving=cls_slice, type_of_transform="SyN", outprefix=prefix, syn_metric="mattes"  )
            tfm[_y0]=reg['fwdtransforms']
        else :
            tfm[_y0]=[prefix+'1Warp.nii.gz', prefix+'0GenericAffine.mat' ]

    return tfm

def get_img(srv_y, rec_y, srv_s, tfm, affine,  slab, ligand, output_dir, fill_dir, _y, s, clobber=False) :
    prefix =''.join([output_dir,"/slab-",str(slab),"_ligand-",ligand,"_y-",str(_y), "_to_",str(s),"_"])
    if not os.path.exists(prefix+"0GenericAffine.mat") or clobber : 
        reg=registration(fixed=srv_s, moving=srv_y, type_of_transform='SyN', outprefix=prefix)
    else :
        reg={'fwdtransforms': [prefix+'1Warp.nii.gz', prefix+'0GenericAffine.mat' ] }

    img_fn = fill_dir + os.sep + str(_y)+"_to_"+ str(s) + ".nii.gz"

    affine[1,3] = s

    if not os.path.exists(img_fn) or clobber or True :
        img=apply_transforms(srv_s, rec_y,  reg['fwdtransforms']+ tfm[_y], interpolator='linear' ).numpy()
        #img=apply_transforms(srv_s, rec_y,  reg['fwdtransforms'], interpolator='linear' ).numpy()
        img = img.T
        nib.Nifti1Image(img , affine  ).to_filename( img_fn)
    else :
        img = nib.load(img_fn).get_data()


    return img

def interpolateMissingSlices(slice_location, tfm, slab, ligand, rec_fn, srv_fn, output_dir, out_fn, clobber=False ):
    print("\t\tInterpolating missing slices for ligand ", ligand, "in slab", int(slab))
    fill_dir=output_dir + os.sep + "fill/"
    if not os.path.exists(fill_dir) :
        os.makedirs(fill_dir)

    rec = nib.load(rec_fn).get_data()
    srv_img = nib.load(srv_fn)
    srv = srv_img.get_data()
    outVolume = np.zeros(srv.shape)

    for _y0, _y1 in zip(slice_location[0:-1], slice_location[1:]) :
        print("\t\t\t",_y0,_y1)
        _start = 1 + int(_y0)
        
        srv_y0 = r(srv[ :, int(_y0), : ])
        srv_y1 = r(srv[ :, int(_y1), : ])
        rec_y0 = r(rec[ :, int( _y0), : ])
        rec_y1 = r(rec[ :, int( _y1), : ])

        imgList=[]
        seq_list = np.arange(_start,_y1).astype(int)
        outVolume[:,_y0,:] = apply_transforms(srv_y0, rec_y0, tfm[_y0], interpolator='linear' ).numpy().T

        for s in seq_list :
            srv_s = r(srv[:,s,:])
            img0 = get_img(srv_y0, rec_y0, srv_s, tfm, srv_img.affine, slab, ligand, output_dir, fill_dir, _y0, s, clobber=False) 
            img1 = get_img(srv_y1, rec_y1, srv_s, tfm, srv_img.affine, slab, ligand, output_dir, fill_dir, _y1, s, clobber=False) 
            x=(s - float(_y0)) / (float(_y1) - float(_y0) ) 
            outVolume[:,s,:] = img0 * (1-x) + img1 * x

    nib.Nifti1Image(outVolume, srv_img.affine).to_filename(out_fn)

def receptorInterpolate( slab, rec_fn, srv_fn, cls_fn, output_dir, ligand, slice_info_fn, clobber=False ) :

    out_fn = output_dir+os.sep+ligand+".nii.gz"
    if os.path.exists(out_fn ) and not clobber : return 0

    if not os.path.exists(output_dir) :
        os.makedirs(output_dir)
    
    # 1. Get locations of slices for particular ligand in specified slab
    slice_location =  get_ligand_slices( slice_info_fn, ligand, slab )  
    
    # 2. Find affine transform from GM classified slice to MRI-derived SRV image
    tfm = alignLigandToSRV(slab ,ligand, srv_fn, cls_fn, output_dir, slice_location, clobber)
    
    # 3. Interpolate slices between acquired autoradiograph slices for a particular ligand
    interpolateMissingSlices(slice_location, tfm,  slab, ligand, rec_fn, srv_fn, out_fn, output_dir) 

