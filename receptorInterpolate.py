import numpy as np
import nibabel as nib
import pandas as pd
import sys
import os
import json
import re
import matplotlib.pyplot as plt
import pandas as pd
import time
from ants import registration, image_read, apply_transforms, from_numpy
from ANTs import ANTs
from utils.utils import splitext


def validate_section(i, out_fn, rec,srv,interp_img,tfm, posterior_slice_location, offset, s, _y0, _y1, rec_y0, rec_y1, pd_list, interpolator):
    validation_section = int(posterior_slice_location[i+int(offset/2)])
    x=round( (s - float(_y0)) / (float(_y1) - float(_y0) ), 1)
    r = from_numpy(rec[:, validation_section,:])
    gm = srv[ :, validation_section, : ]
    ground_truth = apply_transforms(r, r, tfm[str(validation_section)][str(validation_section)], interpolator=interpolator).numpy() *  gm
    n = np.sum(gm == 1)
    error_img = np.zeros(ground_truth.shape)
    #error_img[gm==1] = np.sqrt( np.power( interp_img[gm == 1] - ground_truth[gm == 1],2 ))
    error_img[gm==1] = interp_img[gm == 1] / ground_truth[gm == 1]
    error = np.sum(error_img)  / n
    pd_list.append(pd.DataFrame({"i":[validation_section], "tfm_interpolator":[interpolator], "error":[error]}))
    #errorVolume[:,s,:] = error_img

    validation_fn = splitext(out_fn)[0] + '_validation_'+str(s)+'_'+str(interpolator)+'.png'
    print(validation_fn)
    posterior_img = apply_transforms(rec_y0, rec_y0, tfm[str(_y0)][str(_y0)], interpolator=interpolator ).numpy()
    anterior_img = apply_transforms(rec_y1, rec_y1, tfm[str(_y1)][str(_y1)], interpolator=interpolator ).numpy()
    '''
    plt.clf()
    plt.subplot(2,1,1)
    plt.imshow(rec[ :, int(_y1), : ], cmap='hot')
    plt.imshow(anterior_img, alpha=0.3, cmap='winter')
    plt.subplot(2,1,2)
    plt.imshow(srv[ :, int(_y1), : ])
    plt.imshow(anterior_img/np.max(anterior_img), alpha=0.4, cmap='hot')
    plt.show()
    plt.clf()
    '''
    '''
    plt.clf()
    fig, ax = plt.subplots(3,3)
    fig.set_size_inches(18.5, 10.5, forward=True)
    unit="fmol/mg protein"
    
    im = ax[0,0].imshow(rec[ :, int(_y0), : ], cmap='spectral', vmin=0, vmax=30); 
    ax[0,0].set_title("Posterior (original)")
    cbar = fig.colorbar(im, extend='both', spacing='proportional', shrink=0.9, ax=ax[0,0])
    cbar.set_label(unit, color="black")
    
    im = ax[1,0].imshow(ground_truth, cmap='spectral', vmin=0, vmax=30); 
    ax[1,0].set_title("Ground Truth")
    cbar = fig.colorbar(im, extend='both', spacing='proportional', shrink=0.9, ax=ax[1,0])
    cbar.set_label(unit, color="black")

    im = ax[2,0].imshow(rec[ :, int(_y1), : ], cmap='spectral', vmin=0, vmax=30); 
    ax[2,0].set_title("Anterior (original)")
    cbar = fig.colorbar(im, extend='both', spacing='proportional', shrink=0.9, ax=ax[2,0])
    cbar.set_label(unit, color="black")

    im = ax[0,1].imshow(posterior_img, cmap='spectral', vmin=0, vmax=30); 
    ax[0,1].set_title("Posterior Autoradiograph " + str(1-x))
    cbar = fig.colorbar(im, extend='both', spacing='proportional', shrink=0.9, ax=ax[0,1])
    cbar.set_label(unit, color="black")
    
    im = ax[1,1].imshow(interp_img, cmap='spectral', vmin=0, vmax=30); 
    ax[1,1].set_title("Interpolated Image")
    cbar = fig.colorbar(im, extend='both', spacing='proportional', shrink=0.9, ax=ax[1,1])
    cbar.set_label(unit, color="black")
    
    im = ax[2,1].imshow(anterior_img, cmap='spectral', vmin=0, vmax=30); 
    ax[2,1].set_title("Posterior Autoradiograph " + str(x))
    cbar = fig.colorbar(im, extend='both', spacing='proportional', shrink=0.9, ax=ax[2,1])
    cbar.set_label(unit, color="black")
    
    im = ax[1,2].imshow(error_img, cmap='spectral', vmin=0, vmax=25); 
    ax[1,2].set_title("Error")
    cbar = fig.colorbar(im, extend='both', spacing='proportional', shrink=0.9, ax=ax[1,2])
    cbar.set_label(unit, color="black")
    
    [axi.set_axis_off() for axi in ax.ravel()]
    plt.savefig(validation_fn, dpi=300, facecolor='white')
    plt.clf()
    '''

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

def section2Nii(vol, y, affine0, filename, clobber=False):
    affine = np.copy(affine0)
    if not os.path.exists(filename) or clobber :
        affine[1,3] = affine[1,3] + y*affine[1,1]
        vol = vol.reshape(vol.shape[0], vol.shape[1],1)
        affine = affine[ [0,2,1,3], : ]
        affine[1,1] = affine[1,2]
        affine[2,2]=affine[2,1]
        affine[2,1]=affine[1,2]=0
        nib.Nifti1Image(vol, affine).to_filename(filename)
        

def alignLigandToSRV(df, slab , ligand, srv_fn, cls_fn, output_dir,  tfm_type_2d="SyNAggro", clobber=False):
    print("\t\tNonlinear alignment of coronal sections for ligand",ligand,"for slab",int(slab))
    srv=None
    cls=None
    tfm={}
    tfm_dir=output_dir + os.sep + 'tfm'
    if not os.path.exists(tfm_dir) : os.makedirs(tfm_dir)

    srv_img = nib.load(srv_fn)
    srv = srv_img.get_data()
    
    cls_img = nib.load(cls_fn)
    cls = cls_img.get_data()
    
    for i, row in df.iterrows() :
        _y0 = row["volume_order"]
        tfm[str(_y0)]={str(_y0):[]}

        prefix=tfm_dir+os.sep + ''.join(["cls_",str(_y0)]) + os.sep  

        fixed_fn = srv_slice_fn =  tfm_dir + os.sep +'srv_'+str(_y0)+'.nii.gz'
        moving_fn = cls_slice_fn = prefix+'cls_'+str(_y0)+'.nii.gz'
        moving_rsl_fn  =  prefix+'cls_rsl_'+str(_y0)+'.nii.gz'
        moving_rsl_fn_inverse  =  prefix+'srv_'+str(_y0)+'.nii.gz'
        tfm_fn  =  prefix+"Composite.h5"
        tfm_fn_inverse  =  prefix+"InverseComposite.h5"

        if not os.path.exists(prefix) :  os.makedirs(prefix)

        if not os.path.exists(moving_rsl_fn) or clobber :

            print("\t\t\t",_y0)
            section2Nii( srv[ :, int(_y0), : ], _y0, srv_img.affine, srv_slice_fn, clobber)  
            section2Nii( cls[ :, int(_y0), : ], _y0, cls_img.affine, cls_slice_fn, clobber)
            
            #reg = registration(fixed=srv_slice, moving=cls_slice, type_of_transform=tfm_type_2d,reg_iterations=(500,250,125), outprefix=prefix, syn_metric="mattes"  )
            ANTs(prefix, prefix, fixed_fn, moving_fn, moving_rsl_fn, moving_rsl_fn_inverse, iterations=['1500x1000x500x250x200', '500x250x100', '3000x200x1500x750x500x250'], tolerance=1e-09, base_shrink_factor=2, radius=64, metric="GC", dim=2, verbose=0, clobber=clobber,  exit_on_failure=1, fix_header=True)
        #'500x250x100', '500x500x300','1000x500x500'
        if not os.path.exists(tfm_fn) : 
            print("Error: could not find transformation file", tfm_fn)
        else :
            tfm[str(_y0)][str(_y0)] = [tfm_fn]
        #else :
        #    tfm[str(_y0)][str(_y0)] = [prefix+'1Warp.nii.gz', prefix+'0GenericAffine.mat' ] + tfm[str(_y0)][str(_y0)]

    return tfm

def get_2d_srv_tfm(srv,  slab, ligand, output_dir,  _y, s, tfm, affine, clobber=False) :
    tfm_dir=output_dir + os.sep + 'tfm'
    prefix=tfm_dir+os.sep + ''.join(["srv_y-",str(_y), "_to_",str(s)]) + os.sep  
    fixed_fn =  tfm_dir + os.sep +'srv_'+str(s)+'.nii.gz'
    moving_fn = tfm_dir + os.sep +'srv_'+str(_y)+'.nii.gz'
    moving_rsl_fn  =  prefix+'srv_rsl_'+str(_y)+'_to_'+str(s)+'.nii.gz'
    moving_rsl_fn_inverse  =  prefix+'srv_rsl_'+str(s)+'_to_'+str(_y)+'.nii.gz'

    tfm_fn  =  prefix+"Composite.h5"
    tfm_fn_inverse  =  prefix+"InverseComposite.h5"

    if not os.path.exists(tfm_dir) : os.makedirs(tfm_dir)

    if not os.path.exists(prefix) :
        os.makedirs(prefix)
    if not os.path.exists(tfm_fn) or clobber : 
        #reg=registration(fixed=srv_s, moving=srv_y, type_of_transform=tfm_type_2d,reg_iterations=(500,250,125), outprefix=prefix)['fwdtransforms']

        section2Nii( srv[ :, int(_y), : ], _y, affine, moving_fn, clobber)  
        section2Nii( srv[ :, int(s), : ], s, affine, fixed_fn, clobber)
        
        prev_tfm=s-1
        
        try  :
            init_tfm  = tfm[str(_y)][str(prev_tfm)][0]
        except KeyError :
            init_tfm = None

        #print( _y, s , prev_tfm, tfm[str(_y)][str(prev_tfm)] )
        start = time.time()
        ANTs(prefix, prefix, fixed_fn, moving_fn, moving_rsl_fn, moving_rsl_fn_inverse, iterations=['100x50'], tfm_type=['SyN'], tolerance=1e-07,base_shrink_factor=2, radius=32, metric="GC", dim=2, verbose=0, clobber=clobber, init_tfm=init_tfm,  exit_on_failure=1, fix_header=True)
        end = time.time()

        #   '200x150x100','500x250x100'
        print("\t\t\t",_y,'-->',s,  round(end - start,3) ) 

    if os.path.exists(tfm_fn) :
        tfm[str(_y)][str(s)] = [tfm_fn] + tfm[str(_y)][str(_y)] 

    return tfm

def alignSRVtoSelf(df, tfm, slab, ligand, srv_fn, output_dir, tfm_type_2d='SyNAggro',  clobber=False, validation=False ):
    print("\t\t Align SRV to itself to fill in missing autoradiograph slices ", ligand, "in slab", int(slab))
    #Transformation dict will be saved to json file. If this json exists
    #then load it instead of recalculating all of the intra-SRV transforms
    srv_img = nib.load(srv_fn)
    srv = srv_img.get_data()

    offset=1
    if validation :
        offset=2
    # _y0 and _y1 reprsent spatial positions along coronal axis for slices
    # that were aquired for a specific ligand in a specific slab.
    # the value s is used to iterate between (_y0, _y1)
    slice_location = df["volume_order"].values
    posterior_slice_location=slice_location[0:(len(slice_location)-offset)]
    anterior_slice_location =slice_location[offset:]

    for i, (_y0, _y1) in enumerate(zip(posterior_slice_location, anterior_slice_location)) :
        print("\t\t",_y0,_y1)
        _start = 1 + int(_y0)

        if not validation :
            seq_list = np.arange(_start,_y1).astype(int)
        else :
            #if we are at the end of the stack of images, then exit because we cannot perform validation on this slice
            if _y1 == slice_location[-1] :
                break
            print('-->',_y0, posterior_slice_location[i+1], _y1)
            seq_list = np.array([ posterior_slice_location[i+1] ]) 

        # at each s position, a non-linear transformation from _y0 --> s
        # and _y1 --> s is calculated.
        for s in seq_list :
            tfm = get_2d_srv_tfm( srv, slab, ligand, output_dir,  _y0, s, tfm, srv_img.affine, clobber=clobber) 
            tfm = get_2d_srv_tfm( srv, slab, ligand, output_dir,  _y1, s, tfm, srv_img.affine, clobber=clobber) 
    
        #Save the updated tfm dict with all the transformation files to a .json file    

    return tfm

def interpolateMissingSlices(df, tfm, slab, ligand, rec_fn, srv_fn,  output_dir, out_fn, interpolator='linear', clobber=False, validation=False, val_df=None ):
    print("\t\tInterpolating missing slices for ligand ", ligand, "in slab", int(slab))
    fill_dir=output_dir + os.sep + "fill/"
    if not os.path.exists(fill_dir) :
        os.makedirs(fill_dir)
    rec_vol = nib.load(rec_fn)
    rec = rec_vol.get_data() 

    srv_img =nib.load(srv_fn) 
    srv = srv_img.get_data()

    affine = np.copy(srv_img.affine)

    #Offset defines space between spaces
    offset=1
    if validation :
        offset=2

    slice_location = df["volume_order"].values
    if not validation : 
        outVolume = np.zeros((rec.shape[0], np.max(slice_location)+1, rec.shape[2]))

    posterior_slice_location=slice_location[0:(len(slice_location)-offset)]
    anterior_slice_location =slice_location[offset:]
    pd_list=[]
    
    n = rec.shape[0] * rec.shape[1] * rec.shape[2]

    for i, (_y0, _y1) in enumerate(zip(posterior_slice_location, anterior_slice_location )) :
        print("\t\t\t",_y0,_y1)
        _start = 1 + int(_y0)
        rec_y0 = from_numpy(rec[ :, int(_y0), : ])
        rec_y1 = from_numpy(rec[ :, int(_y1), : ])
        srv_y0 = from_numpy(srv[ :, int(_y0), : ])
        srv_y1 = from_numpy(srv[ :, int(_y1), : ])

        imgList=[]
        if not validation :
            seq_list = np.arange(_start,_y1).astype(int)
        else :
            #if we are at the end of the stack of images, then exit because we cannot perform validation on this slice
            if _y1 == slice_location[-1] :
                break
            seq_list = np.array([ posterior_slice_location[i+1] ]) 

        if not validation :
            outVolume[:,_y0,:] = apply_transforms(rec_y0, rec_y0, tfm[str(_y0)][str(_y0)], interpolator=interpolator ).numpy()

        if _y1 == slice_location[-1] and not validation :
            outVolume[:,_y1,:] = apply_transforms(rec_y1, rec_y1, tfm[str(_y1)][str(_y1)], interpolator=interpolator ).numpy()
        continue

        for s in seq_list :
            interp_img_fn= fill_dir + os.sep + ''.join(["slab-",str(slab),"_ligand-",ligand,"_interp_y-",str(s), ".nii.gz"])

            try :
                tfm[str(_y0)][str(s)]
                tfm[str(_y1)][str(s)]
                #print( tfm[str(_y0)][str(s)]) 
                #print( tfm[str(_y1)][str(s)])
            except KeyError : 
                continue
            #print(interp_img_fn,os.path.exists(interp_img_fn) )
            if not os.path.exists(interp_img_fn) or clobber or validation :
                srv_s = from_numpy(srv[:,s,:])
                #print(interpolator, _y0, str(s), _y1)
                img0  = apply_transforms(rec_y0, rec_y0, tfm[str(_y0)][str(s)], interpolator=interpolator).numpy()
                img1  = apply_transforms(rec_y0, rec_y1, tfm[str(_y1)][str(s)], interpolator=interpolator).numpy()
                x=(s - float(_y0)) / (float(_y1) - float(_y0) ) 
                interp_img = img0 * (1-x) + img1 * x
                affine[1,3]=s
                #print(interp_img.shape, outVolume.shape)
                nib.Nifti1Image( interp_img, affine ).to_filename(interp_img_fn)
            else :
                interp_img = nib.load(interp_img_fn).get_data()
            
            if validation  :
                validate_section(i, out_fn, rec,srv,interp_img,tfm, posterior_slice_location, offset, s, _y0, _y1, rec_y0, rec_y1, pd_list, interpolator)
            else :
                print("\t\t\tAssigning:",s)
                outVolume[:,s,:] = interp_img

    if not validation :
        print("Writing to",out_fn)
        nib.Nifti1Image(outVolume, srv_img.affine).to_filename(out_fn)
    else :
        validation_fn = splitext(out_fn)[0] + '_validation.csv'
        val_df = pd.concat([ val_df ] + pd_list)
        val_df.to_csv(validation_fn)

    return val_df


def receptorInterpolate( slab, out_fn, rec_fn, srv_fn, cls_fn, output_dir, ligand, slice_info_fn, composite_transform_json, tfm_type_2d='SyNAggro', clobber=False , validation=False) :
    #clobber=True

    if not os.path.exists(output_dir) :
        os.makedirs(output_dir)
    
    #Create name for json file that stores transformations to map raw/linearized 2D autoradiographs
    #into alignment with super-resolution GM representation extracted from MRI
    tfm_dict_fn = output_dir + os.sep + ligand +"_slab-"+str(slab)+"_tfm.json" 
    ligand_out_fn = output_dir + os.sep + ligand +"_slab-"+str(slab)+"_init.nii.gz" 
    
    # 1. Get locations of slices for particular ligand in specified slab

    df = pd.read_csv(slice_info_fn)
    if ligand != 'all':
        df = df.loc[ df["ligand"] == ligand ]

    #write_init_volume(df, rec_fn)

    # 2. Find non-linear transform from autoradiograph GM classified slice to MRI-derived SRV image
    tfm = alignLigandToSRV(df, slab, ligand, srv_fn, cls_fn, output_dir,  tfm_type_2d=tfm_type_2d, clobber=clobber)
     
    # 3. Find transformations for slices of MRI-derived super-resolution GM mask at locations where autoradiographs were acquired to slices where no autoradiograph section were acquired.
    #tfm = alignSRVtoSelf(df, tfm, slab, ligand, srv_fn, output_dir, tfm_type_2d=tfm_type_2d, clobber=clobber, validation=validation )
    
    # 4. Interpolate slices between acquired autoradiograph slices for a particular ligand
    interpolator_list = ["gaussian"]
    val_df=pd.DataFrame([])
    #if validation :
        #interpolator_list += ["gaussian", "bSpline", "welchWindowedSinc", "hammingWindowedSinc", "lanczosWindowedSinc"]

    for interpolator in interpolator_list :
        val_df = interpolateMissingSlices(df, tfm,  slab, ligand, rec_fn, srv_fn, output_dir, out_fn, interpolator=interpolator, clobber=clobber, validation=validation, val_df=val_df ) 

