import pandas as pd
import numpy as np
import nibabel as nib
import sys
import os
import json
import re
import matplotlib.pyplot as plt
from imageio import imwrite, imread
from ants import registration, image_read, apply_transforms

def get_ligand_slices(fn, ligand, slab, coreg_dir, lin_dir, ext='#L.TIF'):
    print("\t\tFind coronal voxel values for ",ligand,"in slab",int(slab))
    df = pd.read_csv(fn)
    if not ligand in df["ligand"].values :
        print("Error: could not find ligand ", ligand, "in ", fn)
        exit(1)
    order_max =df["order"].loc[ (df["slab"] == int(slab)) ].max() 
    order = df["order"].loc[ (df["ligand"] == ligand) & (df["slab"] == int(slab)) ]
    
    cropped_filenames = df["filename"].loc[ (df["ligand"] == ligand) & (df["slab"] == int(slab)) ].values

    modify_fn = lambda f,x :   os.path.basename(f).split('_')[0] +x

    slice_location = []
    source_files = []
    tfm_files = []
    tfm={}
    for o, f in zip(order, cropped_filenames) :
        source_files.append(lin_dir + os.sep+ modify_fn(f, '#L.TIF'))
        o2=int(order_max-o)
        tfm[str(o2)]= {str(o2) : [coreg_dir + os.sep+str(o)+'_'+ modify_fn(f, '_cropped_0GenericAffine.mat') ]}
        slice_location.append( o2 )
        #slice_location[int(order_max-o)] = modify_fn(f) 
    #slice_location = np.flip(np.array(order_max - order ).astype(int), axis=0  ) 
    return np.flip(slice_location, axis=0), np.flip(source_files, axis=0), tfm


def alignLigandToSRV(slab , ligand, srv_fn, cls_fn, output_dir, slice_location, tfm, clobber=False):
    print("\t\tNonlinear alignment of coronal sections for ligand", ligand, "for slab", int(slab))
    srv=None
    cls=None

    tfm_dir=output_dir + os.sep + 'tfm'
    if not os.path.exists(tfm_dir) : os.makedirs(tfm_dir)

    for _y0 in slice_location : 

        prefix = tfm_dir + os.sep + ''.join(["slab-",str(slab),"_ligand-",ligand,"_y-",str(_y0), "_"])
        if not os.path.exists(prefix+"1Warp.nii.gz") or clobber :

            if srv == None :
                srv_img = nib.load(srv_fn)
                srv = srv_img.get_data()
            if cls == None :
                cls = nib.load(cls_fn).get_data()

            srv_slice =r( srv[ :, int(_y0), : ] )
            cls_slice =r( cls[ :, int(_y0), : ] )
            reg = registration(fixed=srv_slice, moving=cls_slice, type_of_transform="SyN", outprefix=prefix, syn_metric="mattes"  )
            tfm[str(_y0)][str(_y0)] += reg['fwdtransforms']
        else :
            tfm[str(_y0)][str(_y0)] +=  [prefix+'1Warp.nii.gz', prefix+'0GenericAffine.mat' ] 

    return tfm

def get_2d_srv_tfm(srv_y,  srv_s,  slab, ligand, output_dir,  _y, s, tfm, clobber=False) :
    tfm_dir=output_dir + os.sep + 'tfm'
    if not os.path.exists(tfm_dir) : os.makedirs(tfm_dir)
    prefix =''.join([tfm_dir,"/slab-",str(slab),"_ligand-",ligand,"_y-",str(_y), "_to_",str(s),"_"])
    if not os.path.exists(prefix+"0GenericAffine.mat") or clobber : 
        reg=registration(fixed=srv_s, moving=srv_y, type_of_transform='SyN', outprefix=prefix)['fwdtransforms']
    else :
        reg= [prefix+'1Warp.nii.gz', prefix+'0GenericAffine.mat' ]
    tfm[str(_y)][str(s)] = tfm[str(_y)][str(_y)] + reg

    return tfm

def alignSRVtoSelf(slice_location, tfm, slab, ligand, srv_fn, output_dir, out_fn, clobber=False ):
    print("\t\tInterpolating missing slices for ligand ", ligand, "in slab", int(slab))
    

    #Transformation dict will be saved to json file. If this json exists
    #then load it instead of recalculating all of the intra-SRV transforms

    srv_img = nib.load(srv_fn)
    srv = srv_img.get_data()

    # _y0 and _y1 reprsent spatial positions along coronal axis for slices
    # that were aquired for a specific ligand in a specific slab.
    # the value s is used to iterate between (_y0, _y1)

    for _y0, _y1 in zip(slice_location[0:-1], slice_location[1:]) :
        print("\t\t\t",_y0,_y1)
        _start = 1 + int(_y0)
        
        srv_y0 = r(srv[ :, int(_y0), : ])
        srv_y1 = r(srv[ :, int(_y1), : ])

        imgList=[]
        seq_list = np.arange(_start,_y1).astype(int)

        # at each s position, a non-linear transformation from _y0 --> s
        # and _y1 --> s is calculated.
        for s in seq_list :
            srv_s = r(srv[:,s,:])
            tfm = get_2d_srv_tfm(srv_y0, srv_s, slab, ligand, output_dir,  _y0, s, tfm, clobber=False) 

            tfm = get_2d_srv_tfm(srv_y1, srv_s, slab, ligand, output_dir,  _y1, s, tfm, clobber=False) 
    
        #Save the updated tfm dict with all the transformation files to a .json file    


    return tfm

def r(img):
    imwrite("/tmp/tmp.png", img)
    return image_read(  "/tmp/tmp.png" )

def interpolateMissingSlices(slice_location,source_files, tfm, slab, ligand, srv_fn,  output_dir, out_fn, clobber=False ):
    print("\t\tInterpolating missing slices for ligand ", ligand, "in slab", int(slab))
    fill_dir=output_dir + os.sep + "fill/"
    if not os.path.exists(fill_dir) :
        os.makedirs(fill_dir)

    srv_img =nib.load(srv_fn) 
    srv = srv_img.get_data()
    affine = np.copy(srv_img.affine)
    outVolume = np.zeros(srv.shape)

    for _y0, _y1, src0, src1 in zip(slice_location[0:-1], slice_location[1:], source_files[0:-1], source_files[1:]) :
        print("\t\t\t",_y0,_y1)
        _start = 1 + int(_y0)
        rec_y0 = r(imread(src0))  #r(rec[ :, int( _y0), : ])
        rec_y1 = r(imread(src1))  #r(rec[ :, int( _y1), : ])
        srv_y0 = r(srv[ :, int(_y0), : ])
        srv_y1 = r(srv[ :, int(_y1), : ])

        imgList=[]
        seq_list = np.arange(_start,_y1).astype(int)
        outVolume[:,_y0,:] = apply_transforms(srv_y0, rec_y0, tfm[str(_y0)][str(_y0)], interpolator='linear' ).numpy().T
        for s in seq_list :
            interp_img_fn= fill_dir + os.sep + ''.join(["slab-",str(slab),"_ligand-",ligand,"_interp_y-",str(_y0), ".nii.gz"])
            if not os.path.exists(interp_img_fn) or clobber or True :
                srv_s = r(srv[:,s,:])
                #print(tfm[str(_y0)][str(s)])
                img0 = apply_transforms(srv_y0, rec_y0, tfm[str(_y0)][str(s)], interpolator='linear').numpy().T
                img1 = apply_transforms(srv_y0, rec_y1, tfm[str(_y1)][str(s)], interpolator='linear').numpy().T
                x=(s - float(_y0)) / (float(_y1) - float(_y0) ) 
                interp_img = img0 * (1-x) + img1 * x
                affine[1,3]=s
               

                #plt.subplot(3,2,1)
                #plt.imshow(imread(src0))
                #plt.subplot(3,2,2)
                #plt.imshow(imread(src1))
                #plt.subplot(3,2,3)
                #plt.imshow(img0)
                #plt.subplot(3,2,4)
                #plt.imshow(img1)
                #plt.subplot(3,2,5)
                #plt.imshow(interp_img)
                #plt.show()
                
                nib.Nifti1Image( interp_img, affine ).to_filename(interp_img_fn)
            else :
                interp_img = nib.load(interp_img_fn).get_data()
            
            outVolume[:,s,:] = interp_img

    nib.Nifti1Image(outVolume, srv_img.affine).to_filename(out_fn)

def receptorInterpolate( slab, rec_fn, srv_fn, cls_fn, output_dir, ligand, slice_info_fn, coreg_dir, lin_dir, clobber=False ) :

    out_fn = output_dir+os.sep+ligand+".nii.gz"
    #if os.path.exists(out_fn ) and not clobber : return 0
    
    if not os.path.exists(output_dir) :
        os.makedirs(output_dir)


    tfm_dict_fn = output_dir + os.sep + ligand +"_slab-"+str(slab)+"_tfm.json" 
    
    slice_location, source_files, tfm = get_ligand_slices( slice_info_fn, ligand, slab, coreg_dir, lin_dir )  
    if not  os.path.exists(tfm_dict_fn) or clobber :
        # 1. Get locations of slices for particular ligand in specified slab

        # 2. Find affine transform from GM classified slice to MRI-derived SRV image
        tfm = alignLigandToSRV(slab, ligand, srv_fn, cls_fn, output_dir, slice_location, tfm, clobber)
       
        # 3. Find inter-slice transformation
        tfm = alignSRVtoSelf(slice_location, tfm, slab, ligand, srv_fn, output_dir, out_fn, clobber=False )
        
        json.dump(tfm, open(tfm_dict_fn, 'w'))
    else : 
        tfm = json.load(open(tfm_dict_fn, 'r'))
    
    # 4. Interpolate slices between acquired autoradiograph slices for a particular ligand
    interpolateMissingSlices(slice_location, source_files, tfm,  slab, ligand, srv_fn, output_dir, out_fn ) 

