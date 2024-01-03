import os
import re

import ants
import nibabel
import numpy as np

import brainbuilder.utils.ants_nibabel as nib
from brainbuilder.utils.utils import (
    AntsParams,
    shell,
    simple_ants_apply_tfm,
)


def w2v(c, step, start):
    return np.round((c - start) / step).astype(int)


def v2w(i, step, start):
    return start + i * step


def find_vol_min_max(vol):
    """
    Finds the min and max spatial coordinate of the srv image
    """
    profile = np.max(vol, axis=(0, 2))
    if np.sum(profile) == 0:
        print("Error : empty srv file")
        exit(1)
    srvMin = np.argwhere(profile >= 0.01)[0][0]
    srvMax = np.argwhere(profile >= 0.01)[-1][0]
    return srvMin, srvMax


def pad_volume(vol, max_factor, affine, min_voxel_size=29, direction=[1, 1, 1]):
    xdim, ydim, zdim = vol.shape

    def padded_dim(dim, max_factor, min_voxel_size):
        # min_voxel_size < dim / 2 ** (max_factor-1)
        # min_voxel_size * 2 ** (max_factor-1) < dim
        downsampled_dim = np.ceil(dim / 2 ** (max_factor - 1))
        if downsampled_dim < min_voxel_size:
            return np.ceil((min_voxel_size - downsampled_dim) / 2).astype(int)
        else:
            return 0

    x_pad = padded_dim(xdim, max_factor, min_voxel_size)
    y_pad = padded_dim(ydim, max_factor, min_voxel_size)
    z_pad = padded_dim(zdim, max_factor, min_voxel_size)

    vol_padded = np.pad(vol, ((x_pad, x_pad), (y_pad, y_pad), (z_pad, z_pad)))
    affine[0, 3] -= x_pad * abs(affine[0, 0]) * direction[0]
    affine[1, 3] -= y_pad * abs(affine[1, 1]) * direction[1]
    affine[2, 3] -= z_pad * abs(affine[2, 2]) * direction[2]
    print(np.sum(vol), np.sum(vol_padded))
    print(vol.dtype, vol_padded.dtype); 

    return vol_padded, affine


def get_ref_info(ref_rsl_fn):
    ref_img = nib.load(ref_rsl_fn)
    ref_vol = ref_img.get_fdata()
    ref_vol.shape[1]

    ref_ystep = abs(ref_img.affine[1, 1])
    ref_ystart = ref_img.affine[1, 3]
    ref_min, ref_max = list(
        map(lambda x: v2w(x, ref_ystep, ref_ystart), find_vol_min_max(ref_vol))
    )
    ref_width = ref_max - ref_min

    return ref_width, ref_min, ref_max, ref_ystep, ref_ystart


def pad_seg_vol(seg_rsl_fn, max_downsample_level):
    seg_img = nib.load(seg_rsl_fn)
    seg_vol = seg_img.get_fdata()

    ants_img = ants.image_read(seg_rsl_fn)
    direction = ants_img.direction
    com0 = ants.get_center_of_mass(ants_img)

    pad_seg_vol, pad_affine = pad_volume(
        seg_vol,
        max_downsample_level,
        seg_img.affine,
        direction=direction[[0, 1, 2], [0, 1, 2]],
    )

    seg_rsl_pad_fn = re.sub(".nii", "_padded.nii", seg_rsl_fn)

    nib.Nifti1Image(
        pad_seg_vol, pad_affine, direction=direction, dtype=np.uint8
    ).to_filename(seg_rsl_pad_fn)

    com1 = ants.get_center_of_mass(ants.image_read(seg_rsl_pad_fn))

    com_error = np.sqrt(np.sum(np.power(np.array(com0) - np.array(com1), 2)))

    assert (
        com_error < 0.1
    ), f"Error: change in ceter of mass after padding {com0}, {com1}"
    return seg_rsl_pad_fn


def get_alignment_schedule(
    resolution_list,
    resolution,
    resolution_cutoff_for_cc=0.3,
    base_nl_itr=200,
    base_lin_itr=500,
):
    # cur_res/res = 2^(f-1) --> f = 1+ log2(cur_res/res)
    # min_nl_itr = len( [ resolution for resolution in resolution_list[0:resolution_itr] if  float(resolution) <= .1 ] ) # I gues this is to limit nl alignment above 0.1mm
    base_cc_itr = np.rint(base_nl_itr / 2)

    resolution_list = [float(r) for r in resolution_list]

    # Masks work really poorly when separating the temporal lobe
    # seems to be only true with masks --> Mattes works MUCH better than GC for separating the temporal lobe from the frontal lobe
    # because of mask --> However, Mattes[bins=32] failed at 1mm
    # CC APPEARS TO BE VERY IMPORTANT, especially for temporal lobe

    cc_resolution_list = [
        r for r in resolution_list if float(r) > resolution_cutoff_for_cc
    ]
    linParams = AntsParams(resolution_list, resolution, base_lin_itr)
    print("\t\t\tLinear:")
    print("\t\t\t\t", linParams.itr_str)
    print("\t\t\t\t", linParams.f_str)
    print("\t\t\t\t", linParams.s_str)

    # resolution_list_lo = [ f for r in resolution_listdd ]
    # max_GC_resolution = 1.
    # GC_resolution = max(resolution, max_GC_resolution)

    # nlParams = AntsParams(resolution_list, resolution, base_nl_itr, max_resolution=1.0)
    nlParams = AntsParams(resolution_list, resolution, base_nl_itr)

    print("\t\t\t","Nonlinear")
    print("\t\t\t\t",nlParams.itr_str)
    print("\t\t\t\t",nlParams.s_str)
    print("\t\t\t\t",nlParams.f_str)

    max(min(cc_resolution_list), resolution)
    ccParams = AntsParams(resolution_list, resolution, base_cc_itr)

    print("\t\t\t","Nonlinear (CC)")
    print("\t\t\t\t",ccParams.itr_str)
    print("\t\t\t\t",ccParams.f_str)
    print("\t\t\t\t",ccParams.s_str)

    max_downsample_level = linParams.max_downsample_factor

    return max_downsample_level, linParams, nlParams, ccParams


def write_ref_chunk(
    sub: str,
    hemi: str,
    chunk: int,
    ref_rsl_fn: str,
    out_dir: str,
    y0w: float,
    y0: int,
    y1: int,
    resolution: float,
    ref_ystep: float,
    ref_ystart: float,
    max_downsample_level: int,
    clobber: bool = False,
) -> None:
    """
    Write a chunk from the reference volume that corresponds to the tissue chunk from the section volume
    :param sub: subject id
    :param hemi: hemisphere
    :param chunk: chunk number
    :param ref_rsl_fn: reference volume filename
    :param out_dir: output directory
    :param y0w: y0 in world coordinates
    :param y0: y0 in voxel coordinates
    :param y1: y1 in voxel coordinates
    :param resolution: resolution of the section volume
    :param ref_ystep: ystep of the reference volume
    :param ref_ystart: ystart of the reference volume
    :param max_downsample_level: maximum downsample level
    :param clobber: overwrite existing files
    :return: None
    """
    ref_chunk_fn = f"{out_dir}/{sub}_{hemi}_{chunk}_{resolution}mm_ref_{y0}_{y1}.nii.gz"
    if not os.path.exists(ref_chunk_fn) or clobber:
        # write srv chunk if it does not exist
        print(f"\t\tWriting srv chunk for file\n\n{ref_rsl_fn}")
        ref_img = nib.load(ref_rsl_fn)
        direction = np.array(ref_img.direction)
        ref_vol = ref_img.get_fdata()
        aff = ref_img.affine

        real_aff = nibabel.load(ref_rsl_fn).affine
        real_aff[1, 1]
        real_aff[1, 3]
        aff[1, 3] = direction[1, 1] * y0w

        ref_chunk = ref_vol[:, y0:y1, :]

        pad_ref_chunk, pad_aff = pad_volume(
            ref_chunk,
            max_downsample_level,
            aff,
            direction=direction[[0, 1, 2], [0, 1, 2]],
        )
        
        nib.Nifti1Image(
            pad_ref_chunk,
            pad_aff,
            direction=direction,
            dtype=np.uint8,
        ).to_filename(ref_chunk_fn)

    return ref_chunk_fn


def gen_mask(fn, clobber=False):
    out_fn = re.sub(".nii", "_mask.nii", fn)

    if not os.path.exists(out_fn) or clobber:
        from scipy.ndimage import binary_dilation

        img = nib.load(fn)
        vol = img.get_fdata()
        vol[vol > 0.00001] = 1
        vol[vol < 1] = 0

        average_resolution = np.mean(img.affine[[0, 1, 2], [0, 1, 2]])
        iterations = np.ceil(5 / average_resolution).astype(int)

        vol = binary_dilation(vol, iterations=iterations).astype(np.uint8)

        nib.Nifti1Image(
            vol, img.affine, direction=img.direction, dtype=np.uint8
        ).to_filename(out_fn)

    return out_fn


def run_alignment(
    out_dir: str,
    out_tfm_fn: str,
    out_inv_fn: str,
    out_fn: str,
    ref_rsl_fn: str,
    ref_chunk_fn: str,
    seg_rsl_fn: str,
    linParams: AntsParams,
    nlParams: AntsParams,
    ccParams: AntsParams,
    resolution: float,
    manual_affine_fn: str,
    metric: str = "GC",
    nbins: int = 32,
    use_init_tfm: bool = False,
    use_masks: bool = True,
    sampling: float = 0.9,
    clobber: bool = False,
):
    """
    Run the alignment of the tissue chunk to the chunk from the reference volume
    :param out_dir: output directory
    :param out_tfm_fn: output transformation filename
    :param out_inv_fn: output inverse transformation filename
    :param out_fn: output filename
    :param ref_rsl_fn: reference volume filename
    :param ref_chunk_fn: reference chunk filename
    :param seg_rsl_fn: segmentation volume filename
    :param linParams: linear parameters
    :param nlParams: nonlinear parameters
    :param ccParams: cross correlation parameters
    :param resolution: resolution of the section volume
    :param manual_affine_fn: manual affine filename
    :param metric: metric to use for registration
    :param nbins: number of bins for registration
    :param use_init_tfm: use initial transformation
    :param use_masks: use masks for registration
    :param sampling: sampling for registration
    :param clobber: overwrite existing files
    :return: None
    """

    prefix = re.sub("_SyN_CC_Composite.h5", "", out_tfm_fn)  # FIXME this is bad coding
    f"{prefix}/log.txt"

    prefix + "_init_"
    prefix_rigid = prefix + "_Rigid_"
    prefix_similarity = prefix + "_Similarity_"
    prefix_affine = prefix + "_Affine_"
    prefix_manual = prefix + "_Manual_"

    affine_out_fn = f"{prefix_affine}volume.nii.gz"
    affine_inv_fn = f"{prefix_affine}volume_inverse.nii.gz"
    f"{prefix_manual}Composite.nii.gz"

    seg_mask_fn = gen_mask(seg_rsl_fn, clobber=True)

    ref_tgt_fn = ref_chunk_fn
    step = 0.5
    # Parameters that worked well for chunk 1 at 0.25mm, testing them out...
    step = 0.5  # DEBUG
    nbins = 32  # DEBUG
    sampling = 0.9  # DEBUG

    # calculate SyN

    nl_metric = metric
    # if float(resolution) <= 0.3 :
    # if float(resolution) <= 0.5 :
    #    nl_metric = 'Mattes'

    nl_metric = f"Mattes[{ref_chunk_fn},{seg_rsl_fn},1,32,Random,{sampling}]"
    cc_metric = f"CC[{ref_chunk_fn},{seg_rsl_fn},1,3,Random,{sampling}]"

    syn_rate = "0.1"

    base = f"antsRegistration -v 1 -a 1 -d 3 "
    if use_masks:
        base = f"{base} --masks [{ref_mask_fn},{seg_mask_fn}] "

    def write_log(prefix, kind, cmd):
        with open(f"{prefix}/log_{kind}.txt", "w+") as F:
            F.write(cmd)

    # set initial transform
    # calculate rigid registration
    skip_manual = True

    init_str = f" --initial-moving-transform [{ref_chunk_fn},{seg_rsl_fn},1] "
    if not skip_manual or type(manual_affine_fn) == str:
        if os.path.exists(manual_affine_fn):
            ### Create init tfm to adjust for subs of very differnt sizes
            """
            if use_init_tfm and not os.path.exists(f'{prefix_init}Composite.h5'):
                s_str_0 = linParams.s_list[0] +'x'
                f_str_0 = linParams.f_list[0]
                #why does the similarity transform only have one iteration step?
                #shell(f'{base}  --initial-moving-transform [{ref_chunk_fn},{seg_rsl_fn},1]   -t Similarity[{step}]  -m {metric}[{ref_chunk_fn},{seg_rsl_fn},1,{nbins},Random,{sampling}]  -s {s_str_0} -f {f_str_0}  -c 1000   -t Affine[{step}]  -m {metric}[{ref_chunk_fn},{seg_rsl_fn},1,{nbins},Random,{sampling}]  -s {s_str_0} -f {f_str_0}  -c 1000  -o [{prefix_init},{prefix_init}volume.nii.gz,{prefix_init}volume_inverse.nii.gz]  ', verbose=False)

                shell(f'{base}  --initial-moving-transform [{ref_chunk_fn},{seg_rsl_fn},1]   -t Similarity[{step}]  -m {metric}[{ref_chunk_fn},{seg_rsl_fn},1,{nbins},Random,{sampling}]   -s {linParams.s_str} -f {linParams.f_str}  -c {linParams.itr_str}   -t Affine[{step}]  -m {metric}[{ref_chunk_fn},{seg_rsl_fn},1,{nbins},Random,{sampling}]  -s {s_str_0} -f {f_str_0}  -c 1000  -o [{prefix_init},{prefix_init}volume.nii.gz,{prefix_init}volume_inverse.nii.gz]  ', verbose=False)
                init_str = f' --initial-moving-transform {prefix_init}Composite.h5 '
            """

    # calculate rigid registration
    if not os.path.exists(f"{prefix_rigid}Composite.h5"):
        rigid_cmd = f"{base}  {init_str}  -t Rigid[{step}]  -m {metric}[{ref_chunk_fn},{seg_rsl_fn},1,{nbins},Random,{sampling}]  -s {linParams.s_str} -f {linParams.f_str}  -c {linParams.itr_str}  -o [{prefix_rigid},{prefix_rigid}volume.nii.gz,{prefix_rigid}volume_inverse.nii.gz] "
        shell(rigid_cmd, verbose=True)
        write_log(out_dir, "rigid", rigid_cmd)
    # calculate similarity registration

    if not os.path.exists(f"{prefix_similarity}Composite.h5"):
        similarity_cmd = f"{base}  --initial-moving-transform  {prefix_rigid}Composite.h5 -t Similarity[{step}]  -m {metric}[{ref_chunk_fn},{seg_rsl_fn},1,{nbins},Random,{sampling}]   -s {linParams.s_str} -f {linParams.f_str} -c {linParams.itr_str}  -o [{prefix_similarity},{prefix_similarity}volume.nii.gz,{prefix_similarity}volume_inverse.nii.gz] "
        shell(similarity_cmd, verbose=True)
        write_log(out_dir, "similarity", similarity_cmd)

    affine_init = f"--initial-moving-transform {prefix_similarity}Composite.h5"
    # else :
    #    print('\tApply manual transformation')
    #    shell(f'antsApplyTransforms -v 1 -d 3 -i {seg_rsl_fn} -r {ref_tgt_fn} -t [{manual_affine_fn},0] -o {manual_out_fn}', verbose=False)
    #    affine_init = f'--initial-moving-transform [{manual_affine_fn},0]'

    # calculate affine registration
    if not os.path.exists(f"{prefix_affine}Composite.h5"):
        affine_cmd = f"{base}  {affine_init} -t Affine[{step}] -m {metric}[{ref_tgt_fn},{seg_rsl_fn},1,{nbins},Random,{sampling}]  -s {linParams.s_str} -f {linParams.f_str}  -c {linParams.itr_str}  -o [{prefix_affine},{affine_out_fn},{affine_inv_fn}] "
        shell(affine_cmd, verbose=True)
        write_log(out_dir, "affine", affine_cmd)

    prefix_mattes_syn = f"{prefix}_SyN_Mattes_"
    mattes_syn_out_fn = f"{prefix_mattes_syn}volume.nii.gz"
    mattes_syn_inv_fn = f"{prefix_mattes_syn}volume_inverse.nii.gz"

    prefix_cc_syn = f"{prefix}_SyN_CC_"
    cc_syn_out_fn = f"{prefix_cc_syn}volume.nii.gz"
    cc_syn_inv_fn = f"{prefix_cc_syn}volume_inverse.nii.gz"

    if not os.path.exists(f"{prefix_mattes_syn}Composite.h5"):
        f"{prefix_affine}Composite.h5"
        nl_base = f"{base}  --initial-moving-transform {prefix_affine}Composite.h5 -o [{prefix_mattes_syn},{mattes_syn_out_fn},{mattes_syn_inv_fn}] "
        nl_base += f" -t SyN[{syn_rate}] -m {nl_metric}  -s {nlParams.s_str} -f {nlParams.f_str} -c {nlParams.itr_str} "
        shell(nl_base, verbose=True)
        write_log(out_dir, "syn-mattes", nl_base)

    if not os.path.exists(f"{prefix_cc_syn}Composite.h5"):
        f"{prefix_mattes_syn}Composite.h5"
        nl_base = f"{base}  --initial-moving-transform {prefix_affine}Composite.h5 -o [{prefix_cc_syn},{cc_syn_out_fn},{cc_syn_inv_fn}] "
        nl_base += f" -t SyN[{syn_rate}] -m {cc_metric} -s {ccParams.s_str} -f {ccParams.f_str} -c {ccParams.itr_str}"
        shell(nl_base, verbose=True)
        write_log(out_dir, "syn-cc", nl_base)

    prefix_syn = prefix_cc_syn
    simple_ants_apply_tfm(seg_rsl_fn, ref_rsl_fn, prefix_syn + "Composite.h5", out_fn)
    simple_ants_apply_tfm(
        ref_rsl_fn, seg_rsl_fn, prefix_syn + "InverseComposite.h5", out_inv_fn
    )


def get_max_downsample_level(resolution_list, resolution_itr):
    cur_resolution = float(resolution_list[resolution_itr])
    max_resolution = float(resolution_list[0])
    max_downsample_factor = np.floor(max_resolution / cur_resolution).astype(int)

    # log2(max_downsample_factor) + 1 = L
    max_downsample_level = np.int(np.log2(max_downsample_factor) + 1)

    return max_downsample_level


def get_manual_tfm(resolution_itr, manual_alignment_points, seg_rsl_fn, ref_rsl_fn):
    if resolution_itr == 0:
        assert os.path.exists(
            manual_alignment_points
        ), f"Need to manually create points to initialize registration between:\n\t1) {seg_rsl_fn}\n\t\2 {ref_rsl_fn}\n\tSave as:\n{manual_alignment_points}"
        # shell('')
    else:
        manual_tfm_fn = None

    return manual_tfm_fn


def align_3d(
    sub,
    hemi,
    chunk,
    seg_rsl_fn,
    ref_rsl_fn,
    out_dir,
    out_tfm_fn,
    out_tfm_inv_fn,
    out_fn,
    out_inv_fn,
    resolution,
    resolution_list,
    world_chunk_limits,
    vox_chunk_limits,
    base_nl_itr: int = 200,
    base_lin_itr: int = 500,
    manual_points_fn: str = "",
    manual_affine_fn=None,
    use_masks=False,
    clobber=False,
    verbose=True,
):

    if not os.path.exists(out_tfm_fn) or not os.path.exists(out_tfm_inv_fn) or clobber:
        print("\t\t3D Volumetric Alignment")
        chunk = int(chunk)
        # Load GM volume extracted from donor MRI.
        ref_width, ref_min, ref_max, ref_ystep, ref_ystart = get_ref_info(ref_rsl_fn)

        # get iteration schedules for the linear and non-linear portion of the ants alignment
        # get maximum number steps by which the srv image will be downsampled by
        # with ants, each step means dividing the image size by 2^(level-1)
        (
            max_downsample_level,
            linParams,
            nlParams,
            ccParams,
        ) = get_alignment_schedule(
            resolution_list, resolution, base_nl_itr=base_nl_itr, base_lin_itr=base_lin_itr
        )

        # pad the segmented volume so that it can be downsampled by the
        # ammount of times specified by max_downsample_level
        seg_pad_fn = pad_seg_vol(seg_rsl_fn, max_downsample_level)

        img = nib.load(ref_rsl_fn)
        img.shape[1]

        # extract chunk from srv and write it
        ref_chunk_fn = write_ref_chunk(
            sub,
            hemi,
            chunk,
            ref_rsl_fn,
            out_dir,
            vox_chunk_limits[0],
            world_chunk_limits[0],
            world_chunk_limits[1],
            resolution,
            ref_ystep,
            ref_ystart,
            max_downsample_level,
        )

        # run ants alignment between segmented volume (from autoradiographs) to chunk extracte
        run_alignment(
            out_dir,
            out_tfm_fn,
            out_inv_fn,
            out_fn,
            ref_rsl_fn,
            ref_chunk_fn,
            seg_pad_fn,
            linParams,
            nlParams,
            ccParams,
            resolution,
            manual_affine_fn,
            use_masks=use_masks,
            sampling=0.95,
            metric="Mattes",
        )

    return 0


"""
#this was for aligning really big chunks, block by block
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

"""
