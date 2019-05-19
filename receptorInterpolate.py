import numpy as np
import nibabel as nib
import sys
import os
import json
import re
import matplotlib.pyplot as plt
import pandas as pd
from ants import registration, image_read, apply_transforms, from_numpy

def receptorSliceIndicator(rec_df_fn, ligand, receptor_volume_fn, offset, rec_slice_fn) :
    img = nib.load(receptor_volume_fn)
    vol = np.zeros( img.get_data().shape )

    df = pd.read_csv(rec_df_fn)
    df = df.loc[ df["ligand"] == ligand ]

    for i, row in df.iterrows() :
        _y0 = row["volume_order"]
        vol[:,(_y0-offset):(_y0+offset),:]=2
    
    nib.Nifti1Image(vol, img.affine).to_filename(rec_slice_fn)
    return 0



def alignLigandToSRV(df, slab , ligand, srv_fn, cls_fn, output_dir,  clobber=False):
    print("\t\tNonlinear alignment of coronal sections for ligand", ligand, "for slab", int(slab))
    srv=None
    cls=None
    tfm={}
    tfm_dir=output_dir + os.sep + 'tfm'
    if not os.path.exists(tfm_dir) : os.makedirs(tfm_dir)

    #if srv == None :
    srv_img = nib.load(srv_fn)
    srv = srv_img.get_data()

    #if cls == None :
    cls = nib.load(cls_fn).get_data()
    
    for i, row in df.iterrows() :
        _y0 = row["volume_order"]
        tfm[str(_y0)]={str(_y0):[]}

        prefix_string= ''.join(["slab-",str(slab),"_ligand-",ligand,"_y-",str(_y0)])
        prefix=tfm_dir+os.sep + ''.join(["slab-",str(slab),"_ligand-",ligand,"_y-",str(_y0), "_"]) + os.sep + prefix_string + os.sep 
        if not os.path.exists(prefix) : 
            os.makedirs(prefix)
        if not os.path.exists(prefix+"1Warp.nii.gz") or clobber :
            srv_slice = from_numpy( srv[ :, int(_y0), : ] )
            cls_slice = from_numpy( cls[ :, int(_y0), : ] )
            reg = registration(fixed=srv_slice, moving=cls_slice, type_of_transform="SyN", outprefix=prefix, syn_metric="mattes"  )
            tfm[str(_y0)][str(_y0)] = reg['fwdtransforms'] + tfm[str(_y0)][str(_y0)]
        else :
            tfm[str(_y0)][str(_y0)] = [prefix+'1Warp.nii.gz', prefix+'0GenericAffine.mat' ] + tfm[str(_y0)][str(_y0)]

    return tfm

def get_2d_srv_tfm(srv_y,  srv_s,  slab, ligand, output_dir,  _y, s, tfm, clobber=False) :
    tfm_dir=output_dir + os.sep + 'tfm'
    if not os.path.exists(tfm_dir) : os.makedirs(tfm_dir)
    prefix =''.join([tfm_dir,"/slab-",str(slab),"_ligand-",ligand,"_y-",str(_y), "_to_",str(s),"/"])
    if not os.path.exists(prefix) :
        os.makedirs(prefix)
    if not os.path.exists(prefix+"0GenericAffine.mat") or clobber : 
        reg=registration(fixed=srv_s, moving=srv_y, type_of_transform='SyN', outprefix=prefix)['fwdtransforms']
    else :
        reg= [prefix+'1Warp.nii.gz', prefix+'0GenericAffine.mat' ]
    tfm[str(_y)][str(s)] = reg + tfm[str(_y)][str(_y)] 
    return tfm

def alignSRVtoSelf(df, tfm, slab, ligand, srv_fn, output_dir, out_fn, clobber=False ):
    print("\t\t Align SRV to itself to fill in missing autoradiograph slices ", ligand, "in slab", int(slab))

    #Transformation dict will be saved to json file. If this json exists
    #then load it instead of recalculating all of the intra-SRV transforms

    srv_img = nib.load(srv_fn)
    srv = srv_img.get_data()

    # _y0 and _y1 reprsent spatial positions along coronal axis for slices
    # that were aquired for a specific ligand in a specific slab.
    # the value s is used to iterate between (_y0, _y1)
    slice_location = df["volume_order"].values
    for _y0, _y1 in zip(slice_location[0:-1], slice_location[1:]) :
        #print("\t\t\t",_y0,_y1)
        _start = 1 + int(_y0)
        
        srv_y0 = from_numpy(srv[ :, int(_y0), : ])
        srv_y1 = from_numpy(srv[ :, int(_y1), : ])

        imgList=[]
        seq_list = np.arange(_start,_y1).astype(int)

        # at each s position, a non-linear transformation from _y0 --> s
        # and _y1 --> s is calculated.
        for s in seq_list :
            srv_s =from_numpy(srv[:,s,:])
            tfm = get_2d_srv_tfm(srv_y0, srv_s, slab, ligand, output_dir,  _y0, s, tfm, clobber=clobber) 
            tfm = get_2d_srv_tfm(srv_y1, srv_s, slab, ligand, output_dir,  _y1, s, tfm, clobber=clobber) 
    
        #Save the updated tfm dict with all the transformation files to a .json file    

    return tfm

def interpolateMissingSlices(df, tfm, slab, ligand, rec_fn, srv_fn,  output_dir, out_fn, clobber=False ):
    print("\t\tInterpolating missing slices for ligand ", ligand, "in slab", int(slab))
    fill_dir=output_dir + os.sep + "fill/"
    if not os.path.exists(fill_dir) :
        os.makedirs(fill_dir)

    rec_vol = nib.load(rec_fn)
    rec = rec_vol.get_data() 

    srv_img =nib.load(srv_fn) 
    srv = srv_img.get_data()

    affine = np.copy(srv_img.affine)

    slice_location = df["volume_order"].values
    outVolume = np.zeros((500,np.max(slice_location)+1, 500))
    val = nib.load("MR1/R_slab_1/coregistration/vol_0.nii.gz").get_data()
    for _y0, _y1 in zip(slice_location[0:-1], slice_location[1:]) :
        print("\t\t\t",_y0,_y1)
        _start = 1 + int(_y0)
        rec_y0 = from_numpy(rec[ :, int( _y0), : ])
        rec_y1 = from_numpy(rec[ :, int( _y1), : ])
        srv_y0 = from_numpy(srv[ :, int(_y0), : ])
        srv_y1 = from_numpy(srv[ :, int(_y1), : ])

        imgList=[]
        seq_list = np.arange(_start,_y1).astype(int)
        print(tfm[str(_y0)][str(_y0)])
        outVolume[:,_y0,:] = apply_transforms(rec_y0, rec_y0, tfm[str(_y0)][str(_y0)], interpolator='linear' ).numpy()


        #plt.imshow(srv[ :, int(_y0), : ])
        #plt.imshow(outVolume[:,_y0,:],alpha=0.35,cmap=plt.cm.spectral)
        #plt.show()
        #plt.clf()
        if _y1 == slice_location[-1] :
            outVolume[:,_y1,:] = apply_transforms(rec_y1, rec_y1, tfm[str(_y1)][str(_y1)], interpolator='linear' ).numpy()
        for s in seq_list :
            interp_img_fn= fill_dir + os.sep + ''.join(["slab-",str(slab),"_ligand-",ligand,"_interp_y-",str(_y0), ".nii.gz"])
            if not os.path.exists(interp_img_fn) or clobber :
                srv_s = from_numpy(srv[:,s,:])
                #print(tfm[str(_y0)][str(s)][-1])
                #print(tfm[str(_y1)][str(s)][-1])
                #print(1444 - _y0, 1444 - _y1 )
                #img0 = apply_transforms(srv_y0, rec_y0, [tfm[str(_y0)][str(s)][-1]], interpolator='linear').numpy().T
                #print(tfm[str(_y0)][str(s)])
                img0 = apply_transforms(rec_y0, rec_y0, tfm[str(_y0)][str(s)], interpolator='linear').numpy()
                img1 = apply_transforms(rec_y0, rec_y1, tfm[str(_y1)][str(s)], interpolator='linear').numpy()
    
                x=(s - float(_y0)) / (float(_y1) - float(_y0) ) 
                interp_img = img0 * (1-x) + img1 * x
                affine[1,3]=s
                #if x == 0.08 or x == 0.5 or x==0.74 and False :
                #    plt.subplot(3,1,1)
                #    plt.imshow(img0)
                #    plt.subplot(3,1,2)
                #    plt.imshow(img1)
                #    plt.subplot(3,1,3)
                #    plt.imshow(srv[ :, int(s), : ])
                #    plt.imshow(interp_img,alpha=0.35,cmap=plt.cm.hot)
                #    plt.show()
                #    plt.clf()
                #print(s,'-->', x)
                nib.Nifti1Image( interp_img, affine ).to_filename(interp_img_fn)
            else :
                interp_img = nib.load(interp_img_fn).get_data()
            
            outVolume[:,s,:] = interp_img
        #plt.imshow(outVolume[:,_y0,:])
    print("Writing to",out_fn)
    nib.Nifti1Image(outVolume, srv_img.affine).to_filename(out_fn)

def receptorInterpolate( slab, rec_fn, srv_fn, cls_fn, output_dir, ligand, slice_info_fn, composite_transform_json,  clobber=False ) :
    print(rec_fn)
    out_fn = output_dir+os.sep+ligand+".nii.gz"
    #if os.path.exists(out_fn ) and not clobber : return 0
    
    if not os.path.exists(output_dir) :
        os.makedirs(output_dir)
    
    #Create name for json file that stores transformations to map raw/linearized 2D autoradiographs
    #into alignment with super-resolution GM representation extracted from MRI
    tfm_dict_fn = output_dir + os.sep + ligand +"_slab-"+str(slab)+"_tfm.json" 
    
    # 1. Get locations of slices for particular ligand in specified slab

    df = pd.read_csv(slice_info_fn)
    df = df.loc[ df["ligand"] == ligand ]
    #tfm = json.load(open(composite_transform_json,'r'))
    #tfm = { k:{k:v}  for k,v in tfm.items() }

    # 2. Find non-linear transform from autoradiograph GM classified slice to MRI-derived SRV image
    tfm = alignLigandToSRV(df, slab, ligand, srv_fn, cls_fn, output_dir,  clobber)
    
    # 3. Find transformations for slices of MRI-derived super-resolution GM mask at locations where autoradiographs were acquired to slices where no autoradiograph section were acquired.
    tfm = alignSRVtoSelf(df, tfm, slab, ligand, srv_fn, output_dir, out_fn, clobber=clobber )
    #if not  os.path.exists(tfm_dict_fn) or clobber :
      
        #json.dump(tfm, open(tfm_dict_fn, 'w'))
    #else : 
    #    tfm = json.load(open(tfm_dict_fn, 'r'))
    
    #clobber=True
    # 4. Interpolate slices between acquired autoradiograph slices for a particular ligand
    interpolateMissingSlices(df, tfm,  slab, ligand, rec_fn, srv_fn, output_dir, out_fn, clobber=clobber ) 

