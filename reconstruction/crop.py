import matplotlib
import shutil
import argparse
import os
import json
import h5py as h5
import numpy as np
import pandas as pd
import numpy as np
import imageio
import gc 
import matplotlib.pyplot as plt
import h5py as h5
import utils.ants_nibabel as nib
#import nibabel as nib
import pandas as pd
import ants
import multiprocessing
import SimpleITK as sitk
from scipy.ndimage import rotate
from utils.utils import safe_imread, downsample, shell
from scipy.ndimage.filters import gaussian_filter
from skimage.filters import threshold_otsu, threshold_li
#from utils.threshold_li import threshold_li
from glob import glob
from skimage import exposure
from nibabel.processing import *
from sklearn.cluster import KMeans
from re import sub
from skimage.transform import resize
from joblib import Parallel, delayed
from skimage.measure import label
from sklearn.mixture import GaussianMixture
from scipy.stats import entropy, kurtosis, skew
from scipy.interpolate import griddata
from skimage.segmentation import slic

#matplotlib.use("TkAgg")
np.set_printoptions(suppress=True)


def classify_section(crop, seg, max_roi=5):
    n_roi = np.random.randint(1,max_roi)
    
    im_cls = slic(crop,n_segments=n_roi, mask=seg)
    
    return im_cls

def pseudo_classify_autoradiograph(autoradiograph_fn, mask_fn, out_fn, y, slab, resolution):

    #load autoradiograph
    img = nib.load(autoradiograph_fn)
    vol = img.get_fdata()
    
    original_shape = vol.shape
    new_shape = np.rint( np.array(vol.shape)/10 )
    voxel_size = np.product(img.affine[[0,1,2],[0,1,2]])

    #load mask
    mask_img = nib.load(mask_fn)
    mask_vol = mask_img.get_fdata()
    mask_vol_rsl = resize(mask_vol, new_shape, order=0)
    vol_rsl = resize(vol, new_shape, order=5)
    out_rsl = classify_section(vol_rsl, mask_vol_rsl)


    out_unique = np.unique(out_rsl)[1:]
    #out_label_sizes = np.bincount(out_rsl.reshape(-1,))[1:].astype(float)
    #out_label_sizes = out_label_sizes[ out_label_sizes > 0 ]
    #out_label_sizes *= voxel_size


    for l in out_unique :
        index = np.core.defchararray.add(str(slab)+str(y), str(l)).astype(int)
        out_rsl[ out_rsl==l ] = index
        
    print(np.max(out_rsl))
        #out[ out > 0 ] = index

    if np.sum(out_rsl) == 0 :
        print('\tSum Out == 0')
        out_rsl = mask_vol_rsl

    out = resize(out_rsl.astype(float), original_shape, order=0) 
   
    plt.subplot(2,1,1)
    plt.imshow(out_rsl);
    plt.subplot(2,1,2)
    plt.imshow(out)
    plt.savefig('/tmp/test.png')
    
    print(np.max(out))
    if np.sum(out) == 0 : 
        print('Error empty pseudo cls'); exit(1)
    out = np.ceil(out).astype(np.uint32)
    if np.sum(out) == 0 : 
        print('Error empty pseudo cls after conversion to int'); exit(1)
    
    # save classified image as nifti
    print('Writing', out_fn)
    nib.Nifti1Image(out, img.affine).to_filename(out_fn)


def get_base(fn) :
    fn = os.path.splitext(os.path.basename(fn))[0] 
    return fn

def gen_affine(row, scale, dims, global_order_min, xstep_from_size=False, brain_str='mri'):
    brain = row[brain_str]
    hemi = row['hemisphere']
    slab = row['slab']

    direction = scale[brain][hemi][str(slab)]["direction"]

    z_mm = scale[brain][hemi][str(slab)]["size"]

    zstep = z_mm / dims[1] #4164. # pixel size in z-axis is variable depending on microscope magnification

    if xstep_from_size : 
        xstep = z_mm / dims[0] # pixel size in z-axis is variable depending on microscope magnification
    else :
        xstep = 0.02
    print('\tDigitized step size:', xstep, zstep)
    #commented becaause this doesn't seem to make sense
    
    affine=np.array([[xstep, 0, 0, -90],
                     [0,  zstep, 0, -72],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])
    return affine


def threshold(img,sd=1):
    img = gaussian_filter(img, sd).astype(np.float64)
    out = np.zeros(img.shape)
    values = img[img>0]
    idx = img > threshold_li(values)
    out[ idx ] = 1
    return out


def process_image(img, row, scale, pad, affine,brain_str='mri', mask_fn='', flip_axes_dict={}): 
    if mask_fn != '':
        mask_lores = imageio.imread(mask_fn)
        mask = resize(mask_lores, [img.shape[0],img.shape[1]],order=0).astype(int)
        # apply mask to image
        img[ mask < np.max(mask)*0.5 ] = 0

    brain = row[brain_str]
    hemi = row['hemisphere']
    slab = row['slab']

    p = int(pad/2)
    direction = scale[brain][hemi][str(slab)]["direction"]

    print(img.shape)
    img = img.reshape(img.shape[0], img.shape[1])

    for  dict_direction, flip_axes in flip_axes_dict.items() :
        if direction == dict_direction :
            print(direction, dict_direction, flip_axes)
            #plt.subplot(1,2,1); plt.imshow(img)
            #plt.title(f'{row["order"]}, {row["slab"]}')
            img = np.flip(img, axis=flip_axes)
            #plt.subplot(1,2,2); plt.imshow(img)
            #temp_fn = f'/tmp/{row["repeat"]}.png'
            #plt.savefig(temp_fn)
            #plt.clf()
            #plt.cla()

    # deleteme if flip_axes_dict works
    #if direction == "caudal_to_rostral": 
    #    img = np.flip( img, 1 )
    #img = np.flip(img, 0)

    ## pad image in case it needs to be rotated
    # all images are padded regardless of whether they are rotated because that way they all 
    # have the same image dimensions
    img_pad = np.zeros([img.shape[0]+pad,img.shape[1]+pad])

    img_pad[p:p+img.shape[0],p:p+img.shape[1]] =img
    try :
        section_rotation =  row['rotate']
        if section_rotation != 0 : img_pad = rotate(img_pad, section_rotation, reshape=False)
    except IndexError : 
        pass

    return img_pad

def find_landmark_files(landmark_dir, brain, hemisphere, slab, volume_order) :
    fn=f'{landmark_dir}/{brain}_{hemisphere}_{slab}_src_*_{int(volume_order)}.png'
    print(fn)
    landmark_files = glob(fn)
    if len(landmark_files) != 0 : print(landmark_files)
    return landmark_files 

def crop_parallel(row, mask_dir, scale, global_order_min, brain_str='mri', crop_str='crop_fn', lin_str='lin_fn', pytorch_model='', pad = 1000, clobber=True, flip_axes_dict={} ):
    fn = row[lin_str]
    base = os.path.splitext( os.path.basename(fn) )[0]

    crop_fn = row[crop_str] #'{}/{}.nii.gz'.format(out_dir , base)
    
    brain = row[brain_str]
    hemi = row['hemisphere']
    slab = row['slab']
    volume_order = row['volume_order']

    if not os.path.exists(crop_fn) or clobber : 
        print('\tinput_fn', fn)
        print('\t crop_fn', crop_fn) 
        # identify mask image filename
        mask_fn=glob(f'{mask_dir}/{base}*.png')
        
        if len(mask_fn) > 0 : 
            mask_fn = mask_fn[0]
        else : 
            print('Skipping', fn,f'{mask_dir}/{base}*.png' )
            mask_fn = f'{mask_dir}/{base}.png'
            
        print('\t\tMask fn:', mask_fn)
        
        # load mask image 
        img = imageio.imread(fn)
        if len(img.shape) > 2 : img = img[:,:,0] #np.max(img,axis=2)
        print(img.shape)
        print( len(np.unique(img)))
        affine = gen_affine(row, scale, img.shape, global_order_min, xstep_from_size=True, brain_str=brain_str)
        
        img = process_image(img, row, scale, pad, affine, brain_str=brain_str, mask_fn=mask_fn, flip_axes_dict=flip_axes_dict)
        print( len(np.unique(img)))

        #origin = list(affine[ [0,1],[3,3] ])
        #spacing = list( affine[ [0,1],[0,1] ])
        #ants_image = ants.from_numpy(img, origin=origin, spacing=spacing)
        #ants.image_write(ants_image, crop_fn)
        if np.max(img) == np.min(img) :
            print('Failed to create\n\t',crop_fn)
            exit(1)
        nib.Nifti1Image(img, affine ).to_filename(crop_fn)

    '''
    seg_fn = row['seg_fn']
    temp_crop_fn=crop_fn
    if not os.path.exists(seg_fn) or clobber :
        print('\t seg_fn', seg_fn) 
        print(pytorch_model)
        if pytorch_model == '': 
            img = nib.load(crop_fn)
            ar = img.get_fdata()
            ar = threshold(ar)
            nib.Nifti1Image(ar, img.affine ).to_filename(seg_fn)
        else :

            if 'epib' in crop_fn or 'oxot' in base : 
                img = nib.load(crop_fn)
                vol = img.get_fdata()
                vol = (100 + ((vol - np.mean(vol))/np.std(vol))*20 )
                print('\t epib --->', np.mean(vol), np.std(vol))
                plt.imshow(vol); plt.savefig(f'/tmp/{base}.png')
                temp_crop_fn=f'/tmp/{base}.nii.gz'
                nib.Nifti1Image(vol, img.affine).to_filename(temp_crop_fn)

            #shell(f'python3 caps/Pytorch-UNet/predict.py -m {pytorch_model} -i {temp_crop_fn} -om {seg_fn} -oe /tmp/tmp.nii.gz -s 0.1')

            shell(f'nnUNet_predict -i test_nnunet/ -o test_nnunet/predictions -t 501 -m 2d')
            img = nib.load(seg_fn)
            vol = img.get_fdata()
            vol[vol != 1] = 0
            nib.Nifti1Image(vol, img.affine).to_filename(seg_fn)

    '''
    #vol0 = nib.load(temp_crop_fn).get_fdata()
    #vol1 = nib.load(seg_fn).get_fdata()
    #perc0 = np.percentile(vol0,[0,98])
    #perc1 = np.percentile(vol1,[0,100])
    #plt.figure(figsize=(6,9))
    #plt.subplot(3,1,1); plt.imshow( vol0, vmin=perc0[0], vmax=perc0[1] ); plt.colorbar()
    #plt.subplot(3,1,2); plt.hist( vol0 );
    #plt.subplot(3,1,3); plt.imshow( vol1, vmin=perc1[0], vmax=perc1[1] )
    #plt.tight_layout()
    #qc_fn = sub('nii.gz','png', crop_fn)
    #plt.savefig(qc_fn)
    #plt.cla()
    #plt.clf()
    #print('\tQC:', qc_fn)

def process_landmark_images(df, landmark_in_dir, landmark_out_dir,  scale_factors_json, pad=1000):
    global_order_min = df["global_order"].min()
    # identify landmark file
    landmark_df = pd.DataFrame({'volume_order':[], 'tiff':[], 'crop':[], 'init':[] })

    with open(scale_factors_json) as f : scale=json.load(f)

    landmark_csv_fn = f'{landmark_out_dir}/landmarks.csv'

    if not os.path.exists(landmark_csv_fn) :
        for (brain, hemisphere, slab), temp_df in df.groupby(['mri','hemisphere','slab']) :
            for i, row in temp_df.iterrows():
                volume_order = row['volume_order']
                landmark_files = find_landmark_files(landmark_in_dir, brain, hemisphere, slab, volume_order)
                if len(landmark_files) != 0 :
                    for landmark_fn in landmark_files:
                        landmark_nii_fn=f'{landmark_out_dir}/{os.path.splitext(os.path.basename(landmark_fn))[0]}.nii.gz'
                        if not os.path.exists(landmark_nii_fn) :
                            affine = gen_affine(row, scale, global_order_min)
                            img = imageio.imread(landmark_fn)
                            if len(img.shape) == 3: img = np.max(img, axis=2)
                            landmark_ar = process_image(img, row, scale, pad, affine)
                            nib.Nifti1Image(landmark_ar, affine ).to_filename(landmark_nii_fn)
                
                        landmark_row = pd.DataFrame({'volume_order':[volume_order], 'tiff':[landmark_fn], 'crop':[landmark_nii_fn], 'init':[''] })
                        landmark_df = landmark_df.append(landmark_row)

        landmark_df.to_csv(landmark_csv_fn)

    landmark_df = pd.read_csv(landmark_csv_fn) 

    return landmark_df


def convert_2d_array_to_nifti(f: str, output_filename: str, res=[20,20], spacing=(999, 1, 1),
                              transform=None, is_seg: bool = False) -> None:
    """
    Converts numpy into a series of niftis.
    The image can have an arbitrary number of input channels which will be exported separately (_0000.nii.gz,
    _0001.nii.gz, etc for images and only .nii.gz for seg).
    Spacing can be ignored most of the time.
    !!!2D images are often natural images which do not have a voxel spacing that could be used for resampling. These images
    must be resampled by you prior to converting them to nifti!!!
    Datasets converted with this utility can only be used with the 2d U-Net configuration of nnU-Net
    If Transform is not None it will be applied to the image after loading.
    Segmentations will be converted to np.uint32!
    :param is_seg:
    :param transform:
    :param input_array:
    :param output_filename_truncated: do not use a file ending for this one! Example: output_name='./converted/image1'. This
    function will add the suffix (_0000) and file ending (.nii.gz) for you.
    :param spacing:
    :return:
    """
    if not os.path.exists(output_filename ) :
        print('\tFrom:',f)
        img=np.array(nib.load(f).get_fdata())

        img = resize(img, np.round(np.array(img.shape)*np.array(res)/200).astype(int),order=3)
        print(img.shape)

        if transform is not None:
            img = transform(img)
        if len(img.shape) == 2:  # 2d image with no color channels
            img = img[None, None]  # add dimensions
        else:
            assert len(img.shape) == 3, "image should be 3d with color channel last but has shape %s" % str(img.shape)
            # we assume that the color channel is the last dimension. Transpose it to be in first
            img = img.transpose((2, 0, 1))
            # add third dimension
            img = img[:, None]
        # image is now (c, x, x, z) where x=1 since it's 2d

        img=img.astype(np.uint32)
        
        if is_seg:
            assert img.shape[0] == 1, 'segmentations can only have one color channel, not sure what happened here'

        for j, i in enumerate(img):

            if is_seg :
                i = i.astype(np.uint32)

            itk_img = sitk.GetImageFromArray(i)
            itk_img.SetSpacing(list(spacing)[::-1])
            if not is_seg:
                #sitk.WriteImage(itk_img, output_filename_truncated + "_%04.0d.nii.gz" % j)
                sitk.WriteImage(itk_img, output_filename)
            else:
                print('\t2.', output_filename_truncated + ".nii.gz")
                #sitk.WriteImage(itk_img, output_filename_truncated + ".nii.gz")
        print('Wrote:',output_filename)

def convert_from_nnunet(fn, crop_fn, seg_fn, crop_dir, scale):
    crop_img = nib.load(crop_fn)
    ar = nib.load(fn).get_fdata()
   
    
    if np.sum(ar==1) == 0 :
        print('\nWarning: Found a section that nnUNet failed to segment!\n')
        print(crop_fn)
        ar = threshold(nib.load(crop_fn).dataobj)
    else :
        
        if np.sum(ar) == 0 :
            print('Error: empty segmented image with nnunet')
            exit(0)
        ar[ (ar == 3) | (ar==4) ] = 1
        
        gm=ar == 1
        wm=ar == 2
        
        ar *= 0
        ar[gm] = 1
        #ar[wm] = 1
        
        ar = ar.reshape([ar.shape[0],ar.shape[1]])
        ar = ar.T
        ar = resize(ar, crop_img.shape, order=0 )

    print('\tWriting Seg fn', seg_fn)
    nib.Nifti1Image(ar, crop_img.affine).to_filename(seg_fn)


def crop(crop_dir, mask_dir, df, scale_factors_json, resolution, pytorch_model='', remote=False, pad=1000, clobber=False, brain_str='mri', crop_str='crop_fn', lin_str='lin_fn', res=[20,20], flip_axes_dict={}, create_pseudo_cls=True):
    '''take raw linearized images and crop them'''

    os_info = os.uname()

    if os_info[1] == 'imenb079':
        num_cores = 4 
    else :
        num_cores = min(14, multiprocessing.cpu_count() )

    global_order_min = df["global_order"].min()
    
    with open(scale_factors_json) as f : scale=json.load(f)
    
    file_check = lambda x : not os.path.exists(x)
    crop_check = df[crop_str].apply( file_check ).values
    #seg_check =  df['seg_fn'].apply( file_check ).values

    if create_pseudo_cls :
        cls_check =  df['pseudo_cls_fn'].apply( file_check ).values
    else :
        cls_check= np.zeros_like(crop_check)

    print(crop_check)
    #missing_files = crop_check + seg_check + cls_check
    missing_files = crop_check + cls_check
    if np.sum( missing_files ) > 0 : 
        pass
    else : 
        return 0

    df_to_process = df.loc[ crop_check ]  

    Parallel(n_jobs=num_cores)(delayed(crop_parallel)(row, mask_dir, scale, global_order_min, pytorch_model=pytorch_model, pad=pad, brain_str=brain_str, crop_str=crop_str, lin_str=lin_str, flip_axes_dict=flip_axes_dict) for i, row in  df_to_process.iterrows()) 

    if pytorch_model != '' :

        nnunet_in_dir=f'{crop_dir}/nnunet/'
        nnunet_out_dir=f'{crop_dir}/nnunet_out/'
        os.makedirs(nnunet_in_dir, exist_ok=True)
        os.makedirs(nnunet_out_dir, exist_ok=True)

        #Parallel(n_jobs=num_cores)(delayed(convert_for_nnunet)(row, nnunet_in_dir) for i, row in  df.iterrows()) 

        to_do = []
        for f in df[crop_str].values:
            fname=os.path.split(f)[1].split('.')[0]
            output_filename_truncated = os.path.join(nnunet_in_dir,fname)
            output_filename = output_filename_truncated + "_0000.nii.gz"
            if not os.path.exists(output_filename) :
                to_do.append([f, output_filename]) 
        Parallel(n_jobs=num_cores)(delayed(convert_2d_array_to_nifti)(ii_fn,oo_fn,res=res) for ii_fn, oo_fn in to_do) 

        #shell(f'nnUNet_predict -i {nnunet_in_dir} -o {nnunet_out_dir} -t 502')
        to_do = []
        for i, row in df.iterrows():
            crop_fn = row[crop_str] 
            seg_fn = row['seg_fn'] 
            print(f'{nnunet_out_dir}/{os.path.basename(crop_fn)}')
            fn = glob(f'{nnunet_out_dir}/{os.path.basename(crop_fn)}')[0]
            if not os.path.exists(seg_fn) : 
                to_do.append((fn,crop_fn,seg_fn))
        print('\tConvert Files from nnUNet nifti files')
        Parallel(n_jobs=14)(delayed(convert_from_nnunet)(fn, crop_fn, seg_fn, crop_dir,scale) for fn, crop_fn, seg_fn in to_do) 
    
    if create_pseudo_cls :
        to_do=[]
        for i, row in df.iterrows():

            pseudo_cls_fn = row['pseudo_cls_fn']
            crop_fn = row[crop_str]
            seg_fn = row['seg_fn']
            slab_order = int(row['slab_order'])
            slab = int(row['slab'])
            if not os.path.exists(pseudo_cls_fn) : to_do.append([crop_fn, seg_fn,pseudo_cls_fn,slab_order,slab])
        
        Parallel(n_jobs=14)(delayed(pseudo_classify_autoradiograph)(crop_fn, seg_fn, pseudo_cls_fn, slab_order,slab,resolution) for  crop_fn, seg_fn, pseudo_cls_fn, slab_order, slab in to_do) 
    return 0


