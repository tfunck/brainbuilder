"""Functions to perform 3D volumetric alignment of the inital GM volume to the reference volume."""
import os
import re
from typing import List, Tuple

import ants
import brainbuilder.utils.ants_nibabel as nib
import nibabel
import numpy as np
from brainbuilder.utils.utils import (
    AntsParams,
    shell,
    simple_ants_apply_tfm,
)


def v2w(i: int, step: float, start: float) -> float:
    """Converts voxel coordinate to world coordinate.

    :param i: voxel coordinate
    :param step: step size
    :param start: start coordinate
    :return: world coordinate.
    """
    return start + i * step


def find_vol_min_max(vol: np.ndarray) -> tuple:
    """Finds the min and max spatial coordinate of the srv image.

    :param vol:  image volume
    :return: srvMin, srvMax
    """
    profile = np.max(vol, axis=(0, 2))
    if np.sum(profile) == 0:
        print("Error : empty srv file")
        exit(1)
    srvMin = np.argwhere(profile >= 0.01)[0][0]
    srvMax = np.argwhere(profile >= 0.01)[-1][0]
    return srvMin, srvMax


def pad_volume(
    vol: np.ndarray,
    max_factor: int,
    affine: np.ndarray,
    min_voxel_size: int = 29,
    direction: List[int] = [1, 1, 1],
) -> Tuple[np.ndarray, np.ndarray]:
    """Pad the volume so that it can be downsampled by the maximum downsample factor.

    :param vol: volume to pad
    :param max_factor: maximum downsample factor
    :param affine: affine matrix
    :param min_voxel_size: minimum voxel size
    :param direction: direction of the affine matrix
    :return: padded volume, padded affine matrix.
    """
    xdim, ydim, zdim = vol.shape

    def padded_dim(dim: int, max_factor: int, min_voxel_size: int) -> int:
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

    return vol_padded, affine


def get_ref_info(ref_rsl_fn: str) -> tuple:
    """Get reference volume information.

    Description: Get the width, min, max, ystep, and ystart of the reference volume

    :param ref_rsl_fn: reference volume filename
    :return: ref_width, ref_min, ref_max, ref_ystep, ref_ystart.
    """
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


def pad_seg_vol(seg_rsl_fn: str, max_downsample_level: str) -> str:
    """Pad a volume to center it while keeping it centered in the world coordinates.

    :param seg_rsl_fn: segmentation volume filename
    :param max_downsample_level: maximum downsample level
    :return: padded segmentation volume filename
    """
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
    resolution_list: List[int],
    resolution: int,
    resolution_cutoff_for_cc: float = 0.3,
    base_nl_itr: int = 200,
    base_lin_itr: int = 500,
) -> tuple:
    """Get the alignment schedule for the linear and nonlinear portions of the ants alignment.

    :param resolution_list: list of resolutions
    :param resolution: resolution of the section volume
    :param resolution_cutoff_for_cc: resolution cutoff for cross correlation
    :param base_nl_itr: base number of iterations for nonlinear alignment
    :param base_lin_itr: base number of iterations for linear alignment
    :return: max_downsample_level, linParams, nlParams, ccParams.
    """
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

    nlParams = AntsParams(resolution_list, resolution, base_nl_itr)

    print("\t\t\t", "Nonlinear")
    print("\t\t\t\t", nlParams.itr_str)
    print("\t\t\t\t", nlParams.s_str)
    print("\t\t\t\t", nlParams.f_str)

    max(min(cc_resolution_list), resolution)
    ccParams = AntsParams(resolution_list, resolution, base_cc_itr)

    print("\t\t\t", "Nonlinear (CC)")
    print("\t\t\t\t", ccParams.itr_str)
    print("\t\t\t\t", ccParams.f_str)
    print("\t\t\t\t", ccParams.s_str)

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
    """Write a chunk from the reference volume that corresponds to the tissue chunk from the section volume.

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
    metric: str = "GC",
    nbins: int = 32,
    use_init_tfm: bool = False,
    use_masks: bool = True,
    sampling: float = 0.9,
    clobber: bool = False,
) -> None:
    """Run the alignment of the tissue chunk to the chunk from the reference volume.

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
    :param metric: metric to use for registration
    :param nbins: number of bins for registration
    :param use_init_tfm: use initial transformation
    :param use_masks: use masks for registration
    :param sampling: sampling for registration
    :param clobber: overwrite existing files
    :return: None.
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

    ref_tgt_fn = ref_chunk_fn
    step = 0.5
    nbins = 32
    sampling = 0.9

    nl_metric = f"Mattes[{ref_chunk_fn},{seg_rsl_fn},1,32,Random,{sampling}]"
    cc_metric = f"CC[{ref_chunk_fn},{seg_rsl_fn},1,3,Random,{sampling}]"

    syn_rate = "0.1"

    base = "antsRegistration -v 1 -a 1 -d 3 "

    def write_log(prefix: str, kind: str, cmd: str) -> None:
        with open(f"{prefix}/log_{kind}.txt", "w+") as F:
            F.write(cmd)
        return None

    # set initial transform
    # calculate rigid registration

    init_str = f" --initial-moving-transform [{ref_chunk_fn},{seg_rsl_fn},1] "

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

    return None


def align_3d(
    sub: str,
    hemi: str,
    chunk: int,
    seg_rsl_fn: str,
    ref_rsl_fn: str,
    out_dir: str,
    out_tfm_fn: str,
    out_tfm_inv_fn: str,
    out_fn: str,
    out_inv_fn: str,
    resolution: int,
    resolution_list: List[int],
    world_chunk_limits: Tuple[float, float],
    vox_chunk_limits: Tuple[int, int],
    base_nl_itr: int = 200,
    base_lin_itr: int = 500,
    use_masks: bool = False,
    clobber: bool = False,
    verbose: bool = True,
) -> int:
    """Align the tissue chunk to the reference volume.

    :param sub: subject id
    :param hemi: hemisphere
    :param chunk: chunk number
    :param seg_rsl_fn: segmentation volume filename
    :param ref_rsl_fn: reference volume filename
    :param out_dir: output directory
    :param out_tfm_fn: output transformation filename
    :param out_tfm_inv_fn: output inverse transformation filename
    :param out_fn: output filename
    :param out_inv_fn: output inverse filename
    :param resolution: resolution of the section volume
    :param resolution_list: list of resolutions
    :param world_chunk_limits: world chunk limits
    :param vox_chunk_limits: voxel chunk limits
    :param base_nl_itr: base number of iterations for nonlinear alignment
    :param base_lin_itr: base number of iterations for linear alignment
    :param use_masks: use masks for registration
    :param clobber: overwrite existing files
    :param verbose: verbose output
    :return: 0.
    """
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
            resolution_list,
            resolution,
            base_nl_itr=base_nl_itr,
            base_lin_itr=base_lin_itr,
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
            use_masks=use_masks,
            sampling=0.95,
            metric="Mattes",
        )

    return 0
