from scipy.ndimage import gaussian_filter, center_of_mass
import pandas as pd
import numpy as np
import json
import utils.ants_nibabel as nib
import nibabel
import ants
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
import shutil
import re
import h5py as h5 
from utils.utils import simple_ants_apply_tfm
from glob import glob
from utils.ANTs import ANTs
from utils.utils import shell, splitext, points2tfm, read_points, AntsParams
from nibabel.processing import resample_from_to
from scipy.ndimage.filters import gaussian_filter1d
from scipy.io import loadmat
from numpy.linalg import det
from scipy.interpolate import interp1d
import matplotlib
#matplotlib.use('TKAgg')


def w2v(c, step, start):
    return np.round( (c-start)/step ).astype(int)
def v2w(i, step, start) :
    return start + i * step

def find_vol_min_max(vol) :
    '''
    Finds the min and max spatial coordinate of the srv image
    '''
    profile = np.max(vol, axis=(0,2))
    if np.sum(profile) == 0 :
        print("Error : empty srv file")
        exit(1)
    srvMin = np.argwhere(profile>=.01)[0][0]
    srvMax =  np.argwhere(profile>=.01)[-1][0]
    return srvMin, srvMax

def get_slab_width(slabs, df, ystep, ystart, unaligned_slab_list):
    #slab_minima={}
    #slab_maxima={}
    slab_width={}
    total_slab_width=0

    for slab_i in slabs :
        slab_vmin = df.loc[ df['slab']==int(slab_i) ]['global_order'].min()
        slab_vmax = df.loc[ df['slab']==int(slab_i) ]['global_order'].max()

        slab_min = v2w(slab_vmin, ystep, ystart)
        slab_max = v2w(slab_vmax, ystep, ystart)


        slab_minima[slab_i] = slab_min
        slab_maxima[slab_i] = slab_max
        slab_width[slab_i]  = (slab_vmax-slab_vmin)*ystep
        
        if slab_i in unaligned_slab_list :
            total_slab_width += slab_width[slab_i]
    
    return slab_width, total_slab_width



def get_slab_start_end(srv_rsl_fn, manual_points_fn,  orientation='lpi', verbose=False):
    img = nibabel.load(srv_rsl_fn)
    ystart = img.affine[1,3]
    ystep = img.affine[1,1]

    rec_points, mni_points, fn1, fn2 = read_points( manual_points_fn )
    y0w = np.min(mni_points[:,1])
    y1w = np.max(mni_points[:,1])

    y0 = (y0w - ystart)/ystep 
    y1 = (y1w - ystart)/ystep 

    y0_temp = min(y0,y1)
    y1_temp = max(y0,y1)
    #y0 = np.floor(y0_temp).astype(int)
    #y1 = np.ceil(y1_temp).astype(int)
    #DEBUG
    y0 = np.round(y0_temp).astype(int)
    y1 = np.round(y1_temp).astype(int)

    assert y0 > 0 , f'Error: y0 is negative: {y0}'
    assert y1 > 0 , f'Error: y1 is negatove: {y1}'
    if verbose : print(y0w,y1w,y0,y1) 
    return y0w, y1w, y0, y1

def pad_volume(vol, max_factor, affine, min_voxel_size=29, direction=[1,1,1]):
    xdim, ydim, zdim = vol.shape

    def padded_dim(dim, max_factor, min_voxel_size) :
        # min_voxel_size < dim / 2 ** (max_factor-1) 
        # min_voxel_size * 2 ** (max_factor-1) < dim
        downsampled_dim = np.ceil(dim / 2 ** (max_factor-1))
        if downsampled_dim < min_voxel_size :
            return np.ceil( (min_voxel_size-downsampled_dim)/2).astype(int)
        else :
            return 0

    x_pad = padded_dim(xdim, max_factor, min_voxel_size) 
    y_pad = padded_dim(ydim, max_factor, min_voxel_size)
    z_pad = padded_dim(zdim, max_factor, min_voxel_size)
    
    vol_padded = np.pad(vol, ((x_pad, x_pad), (y_pad, y_pad),(z_pad, z_pad))) 
    affine[0,3] -= x_pad * abs(affine[0,0]) * direction[0]
    affine[1,3] -= y_pad * abs(affine[1,1]) * direction[1]
    affine[2,3] -= z_pad * abs(affine[2,2]) * direction[2]

    return vol_padded, affine

def get_srv_info(srv_rsl_fn ) : 
    srv_img = nib.load(srv_rsl_fn)
    srv_vol = srv_img.get_fdata()
    ymax = srv_vol.shape[1]


    srv_ystep = abs(srv_img.affine[1,1] )
    srv_ystart = srv_img.affine[1,3] 
    srv_min, srv_max = list(map(lambda x: v2w(x,srv_ystep,srv_ystart), find_vol_min_max(srv_vol) ))
    srv_width = srv_max - srv_min

    return srv_width, srv_min, srv_max, srv_ystep, srv_ystart  

def pad_seg_vol(seg_rsl_fn,max_downsample_level):
    seg_img = nib.load(seg_rsl_fn)
    seg_vol = seg_img.get_fdata() 

    ants_img = ants.image_read(seg_rsl_fn)
    direction = ants_img.direction
    orientation=ants_img.orientation
    com0=ants.get_center_of_mass(ants_img)

    pad_seg_vol, pad_affine = pad_volume(seg_vol, max_downsample_level, seg_img.affine, direction=direction[[0,1,2],[0,1,2]])

    seg_rsl_pad_fn = re.sub('.nii','_padded.nii', seg_rsl_fn)
   
    
    nib.Nifti1Image(pad_seg_vol, pad_affine, direction=direction, dtype=np.uint8).to_filename(seg_rsl_pad_fn )

    com1=ants.get_center_of_mass(ants.image_read(seg_rsl_pad_fn))

    com_error = np.sqrt(np.sum(np.power(np.array(com0)-np.array(com1),2)))
    
    assert com_error < 0.1, f'Error: change in ceter of mass after padding {com0}, {com1}'
    return seg_rsl_pad_fn



        

def get_alignment_schedule(resolution_list, resolution, resolution_cutoff_for_cc = 0.3, base_nl_itr = 200, base_lin_itr = 500):
    #cur_res/res = 2^(f-1) --> f = 1+ log2(cur_res/res)
    #min_nl_itr = len( [ resolution for resolution in resolution_list[0:resolution_itr] if  float(resolution) <= .1 ] ) # I gues this is to limit nl alignment above 0.1mm
    base_cc_itr = np.rint(base_nl_itr/2)


    resolution_list = [ float(r) for r in resolution_list ]

    #CC APPEARS TO BE VERY IMPORTANT, especially for temporal lobe
    
    cc_resolution_list = [ r for r in resolution_list if float(r) > resolution_cutoff_for_cc ]
    print(cc_resolution_list)

    linParams = AntsParams(resolution_list, resolution, base_lin_itr) 
    print('lin')
    print(linParams.itr_str)
    print(linParams.f_str)
    print(linParams.s_str)

    #resolution_list_lo = [ f for r in resolution_listdd ]
    max_GC_resolution = 1.
    GC_resolution = max(resolution, max_GC_resolution)

    #nlParams = AntsParams(resolution_list, resolution, base_nl_itr, max_resolution=1.0)  
    nlParams = AntsParams(resolution_list, resolution, base_nl_itr)  

    nlParamsHi = AntsParams(resolution_list, resolution, base_nl_itr, start_resolution=0.999)  
    nlParamsHi = None
    
    print('nl')
    print(nlParams.itr_str)
    print(nlParams.s_str)
    print(nlParams.f_str)

    if nlParamsHi != None :
        print('NL Params Hi Resolution')
        print(nlParamsHi.itr_str)
        print(nlParamsHi.s_str)
        print(nlParamsHi.f_str)
    
    cc_resolution = max(min(cc_resolution_list), resolution)
    ccParams = AntsParams(cc_resolution_list, cc_resolution, base_cc_itr)  
    
    print('cc')
    print(ccParams.itr_str)
    print(ccParams.f_str)
    print(ccParams.s_str)
    
    max_downsample_level = linParams.max_downsample_factor

    return max_downsample_level, linParams, nlParams, nlParamsHi, ccParams 

def write_srv_slab(brain, hemi, slab, srv_rsl_fn, out_dir, y0w, y0, y1, resolution, srv_ystep, srv_ystart, max_downsample_level, clobber=False):

    srv_slab_fn=f'{out_dir}/{brain}_{hemi}_{slab}_{resolution}mm_srv_{y0}_{y1}.nii.gz' 
    if not os.path.exists(srv_slab_fn) or clobber :
        # write srv slab if it does not exist
        print(f'\t\tWriting srv slab for file\n\n{srv_rsl_fn}')
        srv_img = nib.load(srv_rsl_fn)
        direction=np.array(srv_img.direction)
        srv_vol = srv_img.get_fdata()
        aff = srv_img.affine 
        
        real_aff = nibabel.load(srv_rsl_fn).affine
        srv_ystep = real_aff[1,1]
        srv_ystart = real_aff[1,3]
        aff[1,3] = direction[1,1] * y0w

        srv_slab = srv_vol[:,y0:y1,:]
        
        pad_srv_slab, pad_aff = pad_volume(srv_slab, max_downsample_level, aff, direction=direction[[0,1,2],[0,1,2]] )

        #pad_aff[1,3] = direction[1,1] * pad_aff[1,3]
        #pad_srv_slab = (pad_srv_slab - np.min(pad_srv_slab)) / () 
        nib.Nifti1Image(pad_srv_slab.astype(np.float16), pad_aff, direction=direction, dtype=np.uint8).to_filename(srv_slab_fn)
    
    return srv_slab_fn


def gen_mask(fn, clobber=False) :
    out_fn = re.sub('.nii', '_mask.nii', fn)
    
    if not os.path.exists(out_fn) or clobber :
        from scipy.ndimage import binary_dilation
        img = nib.load(fn)
        vol = img.get_fdata()
        vol[ vol > 0.00001 ] = 1
        vol[ vol < 1 ] = 0
        
        average_resolution = np.mean( img.affine[[0,1,2],[0,1,2]] )
        iterations = np.ceil( 3 / average_resolution).astype(int)

        vol = binary_dilation(vol, iterations=iterations).astype(np.uint8)

        nib.Nifti1Image(vol, img.affine, direction=img.direction, dtype=np.uint8).to_filename(out_fn)
    
    return out_fn

def run_alignment(out_dir, out_tfm_fn, out_inv_fn, out_fn, srv_rsl_fn, srv_slab_fn, seg_rsl_fn, linParams, nlParams, nlParamsHi, ccParams, resolution, manual_affine_fn, metric='GC',nbins=10, use_init_tfm=False, use_masks=True, sampling=0.9, clobber=False ):
    prefix=re.sub('_SyN_Composite.h5','',out_tfm_fn)
    prefix_init = prefix+'_init_'
    prefix_rigid = prefix+'_Rigid_'
    prefix_similarity = prefix+'_Similarity_'
    prefix_affine = prefix+'_Affine_'
    prefix_manual = prefix+'_Manual_'
    
    prefix_syn = f'{prefix}_SyN_Mattes_'
    syn_out_fn = f'{prefix_syn}volume.nii.gz'
    syn_inv_fn = f'{prefix_syn}volume_inverse.nii.gz'

    temp_out_fn='/tmp/tmp.nii.gz'
    
    affine_out_fn = f'{prefix_affine}volume.nii.gz'
    affine_inv_fn = f'{prefix_affine}volume_inverse.nii.gz'
    manual_out_fn = f'{prefix_manual}Composite.nii.gz'


    seg_mask_fn = gen_mask(seg_rsl_fn,clobber=True)
    srv_mask_fn = gen_mask(srv_rsl_fn,clobber=True)

    srv_tgt_fn=srv_slab_fn
    step=0.1
    #calculate SyN

    nl_metric=metric
    #if float(resolution) <= 0.3 : 
    #if float(resolution) <= 0.5 : 
    #    nl_metric = 'Mattes'

    use_cc = True
    if float(resolution) >= 1.0 : use_cc = True
    

    syn_rate='0.1'
   
    base=f'antsRegistration -v 1 -a 1 -d 3 '
    if use_masks :
        base=f'{base} --masks [{srv_mask_fn},{seg_mask_fn}] '

    # set initial transform
    # calculate rigid registration
    skip_manual=True
    if not os.path.exists( manual_affine_fn ) or skip_manual :
        ### Create init tfm to adjust for brains of very differnt sizes
        if use_init_tfm and not os.path.exists(f'{prefix_init}Composite.h5'):
            s_str_0 = linParams.s_list[0] +'x'
            f_str_0 = linParams.f_list[0] 
            shell(f'{base}  --initial-moving-transform [{srv_slab_fn},{seg_rsl_fn},1]   -t Similarity[{step}]  -m {metric}[{srv_slab_fn},{seg_rsl_fn},1,{nbins},Random,{sampling}]  -s {s_str_0} -f {f_str_0}  -c 1000   -t Affine[{step}]  -m {metric}[{srv_slab_fn},{seg_rsl_fn},1,{nbins},Random,{sampling}]  -s {s_str_0} -f {f_str_0}  -c 1000  -o [{prefix_init},{prefix_init}volume.nii.gz,{prefix_init}volume_inverse.nii.gz]  ', verbose=True)
            init_str = f' --initial-moving-transform {prefix_init}Composite.h5 '
        else :
            init_str = f' --initial-moving-transform [{srv_slab_fn},{seg_rsl_fn},1] '

        # calculate rigid registration
        if not os.path.exists(f'{prefix_rigid}Composite.h5'):
            shell(f'{base}  {init_str}  -t Rigid[{step}]  -m {metric}[{srv_slab_fn},{seg_rsl_fn},1,{nbins},Random,{sampling}]  -s {linParams.s_str} -f {linParams.f_str}  -c {linParams.itr_str}  -o [{prefix_rigid},{prefix_rigid}volume.nii.gz,{prefix_rigid}volume_inverse.nii.gz] ', verbose=True)
        # calculate similarity registration

        if not os.path.exists(f'{prefix_similarity}Composite.h5'):
            shell(f'{base}  --initial-moving-transform  {prefix_rigid}Composite.h5 -t Similarity[{step}]  -m {metric}[{srv_slab_fn},{seg_rsl_fn},1,{nbins},Random,{sampling}]  -s {linParams.s_str} -f {linParams.f_str}  -c {linParams.itr_str}  -o [{prefix_similarity},{prefix_similarity}volume.nii.gz,{prefix_similarity}volume_inverse.nii.gz] ', verbose=True)
        affine_init = f'--initial-moving-transform {prefix_similarity}Composite.h5'
    else :
        print('\tApply manual transformation')
        shell(f'antsApplyTransforms -v 1 -d 3 -i {seg_rsl_fn} -r {srv_tgt_fn} -t [{manual_affine_fn},0] -o {manual_out_fn}', verbose=True)
        affine_init = f'--initial-moving-transform [{manual_affine_fn},0]'
    
    #calculate affine registration
    if not os.path.exists(f'{prefix_affine}Composite.h5'):
        shell(f'{base}  {affine_init} -t Affine[{step}] -m {metric}[{srv_tgt_fn},{seg_rsl_fn},1,{nbins},Random,{sampling}]  -s {linParams.s_str} -f {linParams.f_str}  -c {linParams.itr_str}  -o [{prefix_affine},{affine_out_fn},{affine_inv_fn}] ', verbose=True)
     
    prefix_syn = f'{prefix}_SyN_'
    syn_out_fn = f'{prefix_syn}volume.nii.gz'
    syn_inv_fn = f'{prefix_syn}volume_inverse.nii.gz'
    if not os.path.exists(f'{prefix_syn}Composite.h5'): 
        syn_init = f'{prefix_affine}Composite.h5'

        #nl_base += f' --masks [{srv_mask_fn},{seg_mask_fn}]'
            
        nl_base=f'{base}  --initial-moving-transform {prefix_affine}Composite.h5 -o [{prefix_syn},{syn_out_fn},{syn_inv_fn}] '
        
        nl_base += f' -t SyN[{syn_rate}] -m {nl_metric}[{srv_rsl_fn},{seg_rsl_fn},1,{nbins},Random,{sampling}]  -s {nlParams.s_str} -f {nlParams.f_str} -c {nlParams.itr_str} ' 

        #if type(nlParamsHi) != None :
        #    nl_base += f' -t SyN[{syn_rate}] -m Mattes[{srv_rsl_fn},{seg_rsl_fn},1,{nbins},Random,{sampling}]  -s {nlParamsHi.s_str} -f {nlParamsHi.f_str} -c {nlParamsHi.itr_str} ' 

        nl_base += f' -t SyN[{syn_rate}] -m CC[{srv_rsl_fn},{seg_rsl_fn},1,3,Random,{sampling}]  -s {ccParams.s_str} -f {ccParams.f_str} -c {ccParams.itr_str}'
        shell(nl_base, verbose=True)

    simple_ants_apply_tfm(seg_rsl_fn, srv_rsl_fn, prefix_syn+'Composite.h5', out_fn)
    simple_ants_apply_tfm(srv_rsl_fn, seg_rsl_fn, prefix_syn+'InverseComposite.h5', out_inv_fn)
    
def get_max_downsample_level(resolution_list, resolution_itr):
    cur_resolution = float(resolution_list[resolution_itr])
    max_resolution = float(resolution_list[0])
    max_downsample_factor = np.floor( max_resolution / cur_resolution ).astype(int)

    # log2(max_downsample_factor) + 1 = L
    max_downsample_level = np.int( np.log2(max_downsample_factor) + 1 )

    return max_downsample_level

def get_manual_tfm(resolution_itr,manual_alignment_points, seg_rsl_fn, srv_rsl_fn):

    if resolution_itr == 0 :
        assert os.path.exists( manual_alignment_points ), f'Need to manually create points to initialize registration between:\n\t1) {seg_rsl_fn}\n\t\2 {srv_rsl_fn}\n\tSave as:\n{manual_alignment_points}'
        #shell('')
    else :
        manual_tfm_fn=None

    return manual_tfm_fn

def align_slab_to_mri(brain, hemi, slab, seg_rsl_fn, srv_rsl_fn, out_dir, df, slabs, out_tfm_fn, out_tfm_inv_fn, out_fn, out_inv_fn,  resolution, resolution_list, slab_direction, manual_points_fn, manual_affine_fn=None, use_masks=False, clobber=False, verbose=True ) :
    print('\tRunning, resolution:', resolution )
    slab=int(slab)

    # Load super-resolution GM volume extracted from donor MRI.
    srv_width, srv_min, srv_max, srv_ystep, srv_ystart =  get_srv_info( srv_rsl_fn ) 

    # get iteration schedules for the linear and non-linear portion of the ants alignment
    # get maximum number steps by which the srv image will be downsampled by 
    # with ants, each step means dividing the image size by 2^(level-1)
    max_downsample_level, linParams, nlParams, nlParamsHi, ccParams = get_alignment_schedule(resolution_list, resolution )

    # pad the segmented volume so that it can be downsampled by the 
    # ammount of times specified by max_downsample_level
    seg_pad_fn = pad_seg_vol(seg_rsl_fn, max_downsample_level)

    if os.path.exists(manual_points_fn) and not os.path.exists( manual_affine_fn ) or True :
        print('\tCreating affine transform from manually defined points')
        points2tfm( manual_points_fn, manual_affine_fn, srv_rsl_fn, seg_rsl_fn)

    img = nib.load(srv_rsl_fn)
    ydim = img.shape[1] 
    orientation = ants.image_read(srv_rsl_fn).orientation

    # get the start and end values of the slab in srv voxel coordinates
    y0w, y1w, y0, y1 = get_slab_start_end(srv_rsl_fn, manual_points_fn, verbose=True)


    
    # extract slab from srv and write it
    srv_slab_fn = write_srv_slab(brain, hemi, slab, srv_rsl_fn, out_dir, y0w, y0, y1, resolution, srv_ystep, srv_ystart, max_downsample_level)
    print(srv_slab_fn) 
    print(y0, y1)
    # run ants alignment between segmented volume (from autoradiographs) to slab extracte
    run_alignment(out_dir, out_tfm_fn, out_inv_fn, out_fn, srv_rsl_fn, srv_slab_fn, seg_pad_fn, linParams, nlParams, nlParamsHi, ccParams, resolution, manual_affine_fn, use_masks=use_masks ) 
    return 0

'''
#this was for aligning really big slabs, block by block
def write_block(fn, start, end, out_fn) :
    img = nib.load(fn)
    vol = img.get_fdata()
    block = vol[start[0]:end[0],start[1]:end[1],start[2]:end[2]]
    nib.Nifti1Image(block, affine).to_filename(out_fn)
    return block
def hi_res_align(moving_fn, fixed_fn, resolution, tfm_dir, init_tfm, prefix, out_fn, out_inv_fn):
    kernel_dim = [10/resolution, 10/resolution, 10/resolution] 
    step = [ int(kernel_dim[0]/3) ] * 3
    out_tfm_fn = f'{prefix}Composite.h5'
    img = nib.load(moving_fn)
    image_dim = img.shape

    f = h5.File('/tmp/deformation_field.h5', 'w')
    dfield = f.create_dataset('field',(np.product(image_dim.shape[0])*3,),dtype=np.float64)

    for x in range(0,image_dim.shape[0],step[0]) :
        for y in range(0,image_dim.shape[1],step[1]) :
            for z in range(0,image_dim.shape[2],step[2]) :
                #
                start=[x,y,z]
                end=[x+step[0],y+step[1],z+step[2]]

                block_fn = f'{out_dir}/block_{x}-{end[0]}_{y}-{end[1]}_{z}-end[2]'
                # extract block from moving image, save to tmp directory 
                write_block(moving_fn, start, end, '/tmp/moving.nii.gz')
                # extract block from fixed image, save to tmp directory
                write_block(fixed_fn, start, end, '/tmp/fixed.nii.gz')
                
                # non-linear alignment     
                shell(f'/usr/bin/time -v antsRegistration -v 1 -a 1 -d 3  --initial-moving-transform {init_tfm}  -t SyN[.1] -c [500] -m GC[/tmp/fixed.nii.gz,/tmp/moving.nii.gz,1,20,Regular,1] -s 0vox -f 1  -o [/tmp/,/tmp/tfm.nii.gz, /tmp/tfm_inv.nii.gz ] ', verbose=True)
            
                # use h5 to load deformation field and save
                block_dfield = ants.read_transform(block_tfm_fn)
                dfield['field'][x:x+step,y:y+step,z:z+step] = block_dfield
                

    final_tfm = ants.transform_from_deformation_field(dfield['field'])
    ants.write_transform(final_tfm, out_tfm_fn)

'''
        




