"""Functions to perform 3D volumetric alignment of the inital GM volume to the reference volume."""
import logging
import os
import re
from typing import List

import ants
import brainbuilder.utils.ants_nibabel as nib
import nibabel
import numpy as np
import pandas as pd
from brainbuilder.align.align_landmarks import create_landmark_transform
from brainbuilder.utils import utils

logger = utils.get_logger(__name__)


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
        logger.critical("Error : empty srv file")
        exit(1)
    srvMin = np.argwhere(profile >= 0.01)[0][0]
    srvMax = np.argwhere(profile >= 0.01)[-1][0]
    return srvMin, srvMax


def get_ref_info(moving_fn: str) -> tuple:
    """Get reference volume information.

    Description: Get the width, min, max, ystep, and ystart of the reference volume

    :param moving_fn: reference volume filename
    :return: ref_width, ref_min, ref_max, ref_ystep, ref_ystart.
    """
    ref_img = nib.load(moving_fn)
    ref_vol = ref_img.get_fdata()
    ref_vol.shape[1]

    ref_ystep = abs(ref_img.affine[1, 1])
    ref_ystart = ref_img.affine[1, 3]
    ref_min, ref_max = list(
        map(lambda x: v2w(x, ref_ystep, ref_ystart), find_vol_min_max(ref_vol))
    )
    ref_width = ref_max - ref_min

    return ref_width, ref_min, ref_max, ref_ystep, ref_ystart


def pad_acq_volume(
    fixed_fn: str,
    seg_rsl_pad_fn: str,
    resolution: float,
    padding_offset: float = 0.15,
) -> str:
    """Pad a volume to center it while keeping it centered in the world coordinates.

    :param fixed_fn: segmentation volume filename
    :param max_downsample_level: maximum downsample level
    :param resolution: resolution of the section volume
    :return: padded segmentation volume filename
    """
    seg_img = nib.load(fixed_fn)
    seg_vol = seg_img.get_fdata()

    ants_img = ants.image_read(fixed_fn)
    direction = ants_img.direction
    com0 = ants.get_center_of_mass(ants_img)

    pad_seg_volume, pad_affine = utils.pad_volume(
        seg_vol,
        seg_img.affine,
        padding_offset=padding_offset,
        direction=direction[[0, 1, 2], [0, 1, 2]],
    )

    nib.Nifti1Image(
        pad_seg_volume, pad_affine, direction=direction, dtype=np.uint8
    ).to_filename(seg_rsl_pad_fn)

    com1 = ants.get_center_of_mass(ants.image_read(seg_rsl_pad_fn))

    com_error = np.sqrt(np.sum(np.power(np.array(com0) - np.array(com1), 2)))

    assert (
        com_error < resolution * 3
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
    base_cc_itr = np.rint(base_nl_itr / 2)

    resolution_list = [float(r) for r in resolution_list]

    # Masks work really poorly when separating the temporal lobe
    # seems to be only true with masks --> Mattes works MUCH better than GC for separating the temporal lobe from the frontal lobe
    # because of mask --> However, Mattes[bins=32] failed at 1mm
    # CC APPEARS TO BE VERY IMPORTANT, especially for temporal lobe

    cc_resolution_list = [
        r for r in resolution_list if float(r) > resolution_cutoff_for_cc
    ]
    linParams = utils.AntsParams(resolution_list, resolution, base_lin_itr)
    logger.info("\t\t\tLinear:")
    logger.info("\t\t\t\t" + linParams.itr_str)
    logger.info("\t\t\t\t" + linParams.f_str)
    logger.info("\t\t\t\t" + linParams.s_str)

    nlParams = utils.AntsParams(resolution_list, resolution, base_nl_itr)

    logger.info("\t\t\tNonlinear")
    logger.info("\t\t\t\t" + nlParams.itr_str)
    logger.info("\t\t\t\t" + nlParams.s_str)
    logger.info("\t\t\t\t" + nlParams.f_str)

    max(min(cc_resolution_list), resolution)
    ccParams = utils.AntsParams(resolution_list, resolution, base_cc_itr)

    logger.info("\t\t\tNonlinear (CC)")
    logger.info("\t\t\t\t" + ccParams.itr_str)
    logger.info("\t\t\t\t" + ccParams.f_str)
    logger.info("\t\t\t\t" + ccParams.s_str)

    max_downsample_level = linParams.max_downsample_factor

    return max_downsample_level, linParams, nlParams, ccParams


def verify_chunk_limits(
    ref_rsl_fn: str, chunk_info: pd.DataFrame, verbose: bool = False
) -> tuple:
    """Get the start and end of the chunk in the reference space.

    :param ref_rsl_fn: reference space file name
    :param verbose: verbose
    :return: (y0w, y1w) --> world coordinates; (y0, y1) --> voxel coordinates
    """
    img = nibabel.load(ref_rsl_fn)

    ystart = img.affine[1, 3]
    ystep = img.affine[1, 1]
    if "caudal_limit" in chunk_info.columns and "rostral_limit" in chunk_info.columns:
        y0w = chunk_info["caudal_limit"].values[0]
        y1w = chunk_info["rostral_limit"].values[0]

        # check if y0w and y1w are float, if not return 0 and img.shape[1]
        y0w = y0w if isinstance(y0w, float) else ystart
        y1w = y1w if isinstance(y1w, float) else ystart + ystep * img.shape[1]

        y0 = (y0w - ystart) / ystep
        y1 = (y1w - ystart) / ystep

        y0_temp = min(y0, y1)
        y1_temp = max(y0, y1)
        y0 = np.floor(y0_temp).astype(int)
        y1 = np.ceil(y1_temp).astype(int)

        assert y0 > 0, f"Error: y0 is negative: {y0}"
        assert y1 > 0, f"Error: y1 is negatove: {y1}"
        if verbose:
            print(y0w, y1w, y0, y1)

    else:
        y0w = ystart
        y1w = ystart + ystep * img.shape[1]
        y0 = 0
        y1 = img.shape[1]

    return [y0, y1], [y0w, y1w]


def crop_volume_with_indicator(
    ref_vol_fn: str,
    ref_indicator_volume: str,
    out_dir: str,
    sub: str,
    hemi: str,
    chunk: int,
    clobber: bool,
):
    """Crop the reference volume with the indicator volume to get the reference chunk."""
    ref_chunk_fn = f"{out_dir}/sub-{sub}_hemi-{hemi}_chunk-{chunk}_ref_chunk.nii.gz"
    if not os.path.exists(ref_chunk_fn) or clobber:
        ref_indicator_img = nib.load(ref_indicator_volume)
        ref_indicator_vol = ref_indicator_img.get_fdata()

        ref_img = nib.load(ref_vol_fn)
        ref_vol = ref_img.get_fdata()

        ref_chunk_vol = np.where(ref_indicator_vol > 0, ref_vol, 0).astype(np.uint8)

        nib.Nifti1Image(
            ref_chunk_vol,
            ref_img.affine,
            direction_order="lpi",
            dtype=np.uint8,
        ).to_filename(ref_chunk_fn)
    return ref_chunk_fn


def write_ref_chunk_with_fixed_limits(
    chunk_info: pd.DataFrame,
    sub: str,
    hemi: str,
    chunk: int,
    ref_vol_fn: str,
    out_dir: str,
    clobber: bool = False,
) -> str:
    (y0, y1), _ = verify_chunk_limits(ref_vol_fn, chunk_info)

    ref_chunk_fn = f"{out_dir}/sub-{sub}_hemi-{hemi}_chunk-{chunk}_ref_{y0}_{y1}.nii.gz"

    if not os.path.exists(ref_chunk_fn) or clobber:
        # write srv chunk if it does not exist
        logger.debug(f"\t\tWriting srv chunk for file\n\n{ref_vol_fn}")
        ref_img = nib.load(ref_vol_fn)
        direction = np.array(ref_img.direction)
        ref_vol = ref_img.get_fdata()

        aff = ref_img.affine

        ref_vol[:, :y0, :] = 0
        ref_vol[:, y1 + 1 :, :] = 0

        # rescale 0-255
        ref_vol = (
            255 * (ref_vol - np.min(ref_vol)) / (np.max(ref_vol) - np.min(ref_vol))
        ).astype(np.uint8)

        nib.Nifti1Image(
            ref_vol,
            aff,
            direction=direction,
            dtype=np.uint8,
        ).to_filename(ref_chunk_fn)

        print(f"Written ref chunk: {ref_chunk_fn}")

    return ref_chunk_fn


def create_indicator_volume(
    acq_landmark_volume: str, output_dir: str, clobber: bool = False
) -> str:
    acq_indicator_volume = f"{output_dir}/{os.path.basename(acq_landmark_volume).replace('.nii.gz', '_indicator.nii.gz')}"

    if not os.path.exists(acq_indicator_volume) or clobber:
        img = nib.load(acq_landmark_volume)

        vol = np.ones(img.shape, dtype=np.uint8)

        nib.Nifti1Image(
            vol, img.affine, direction_order="lpi", dtype=np.uint8
        ).to_filename(acq_indicator_volume)

    return acq_indicator_volume


def write_ref_chunk_with_landmark_transform(
    sect_info: pd.DataFrame,
    chunk_info: pd.DataFrame,
    sub: str,
    hemi: str,
    chunk: int,
    ref_vol_fn: str,
    max_resolution_3d: int,
    landmark_dir: str,
    output_dir: str,
    ref_landmark_volume: str,
    padding_offset: float = 0.15,
    clobber: bool = False,
):
    # 1 Calculate the landmark transform from acquisition to reference space
    ymax = nib.load(chunk_info["init_volume"].values[0]).shape[1]

    landmarks_out_dir = output_dir + f"/ref_chunk_landmarks/{max_resolution_3d}mm/"

    os.makedirs(landmarks_out_dir, exist_ok=True)

    acq_landmark_volume = f"{landmarks_out_dir}/sub-{sub}_hemi-{hemi}_chunk-{chunk}_acq_landmark_volume.nii.gz"

    landmark_composite_tfm_path = create_landmark_transform(
        sub,
        hemi,
        chunk,
        max_resolution_3d,
        max_resolution_3d,
        sect_info,
        chunk_info["init_volume"].values[0],  # reference volume for image dimensions
        acq_landmark_volume,  # acq landmark volume for reference
        acq_landmark_volume,  # moving landmark volume
        ref_landmark_volume,  # fixed landmark volume
        landmark_dir,
        landmarks_out_dir,
        chunk_info["init_volume"].values[0],  # qc moving volume for visualisation
        ref_vol_fn,  # qc fixed volume for visualisation
        ymax,
        chunk_info["section_thickness"].values[0],
        padding_offset=padding_offset,
        clobber=clobber,
    )

    # 2 Create a indicator .nii.gz volume of all 1s and save in space of acquisition volume
    acq_indicator_volume = create_indicator_volume(
        acq_landmark_volume, landmarks_out_dir, clobber=clobber
    )

    # 3 Apply the landmark transform to the indicator volume to get a warped indicator volume in the space of the reference volume, with nearest neighbours
    ref_indicator_volume = acq_indicator_volume.replace(".nii.gz", "_space-stx.nii.gz")

    utils.simple_ants_apply_tfm(
        acq_indicator_volume,
        ref_vol_fn,
        landmark_composite_tfm_path,
        ref_indicator_volume,
        n="NearestNeighbor",
        clobber=clobber,
    )

    # 4 Use the reference space warped indicator volume to mask the reference volume to get the reference chunk
    ref_chunk_fn = crop_volume_with_indicator(
        ref_vol_fn, ref_indicator_volume, landmarks_out_dir, sub, hemi, chunk, clobber
    )

    return ref_chunk_fn


def write_ref_chunk(
    sect_info: pd.DataFrame,
    chunk_info: pd.DataFrame,
    sub: str,
    hemi: str,
    chunk: int,
    ref_vol_fn: str,
    landmark_dir: str,
    out_dir: str,
    max_resolution_3d: int,
    ref_landmark_volume: str,
    acq_rsl_fn: str,
    padding_offset: float = 0.15,
    use_landmark_transform: bool = False,
    clobber: bool = False,
) -> None:
    """Write a chunk from the reference volume that corresponds to the tissue chunk from the section volume.

    :param sub: subject id
    :param hemi: hemisphere
    :param chunk: chunk number
    :param ref_vol_fn: reference volume filename
    :param out_dir: output directory
    :param clobber: overwrite existing files
    :return: None
    """
    if not use_landmark_transform:
        if (
            "caudal_limit" not in chunk_info.columns
            or "rostral_limit" not in chunk_info.columns
        ):
            logger.warning(
                "caudal_limit and rostral_limit not found in chunk_info, using full volume"
            )
            return ref_vol_fn

        # if landmark transform is not provided, use the fixed limits from the chunk_info.csv to get the reference chunk.
        # This is because the fixed limits may not be accurate and may not correspond to the actual tissue chunk in the reference space,
        # but it is better than nothing and will allow us to get a reference chunk that is at least in the right area of the brain.

        ref_chunk_fn = write_ref_chunk_with_fixed_limits(
            chunk_info, sub, hemi, chunk, ref_vol_fn, out_dir, clobber
        )
    else:
        # if landmark transform is provided, use the landmark transform to get the reference chunk that corresponds to the tissue chunk, instead of using the fixed limits from the chunk_info.csv. This is because the fixed limits may not be accurate and may not correspond to the actual tissue chunk in the reference space.
        # The landmark transform will allow us to get a more accurate reference chunk that corresponds to the tissue chunk in the reference space.
        ref_chunk_fn = write_ref_chunk_with_landmark_transform(
            sect_info,
            chunk_info,
            sub,
            hemi,
            chunk,
            ref_vol_fn,
            max_resolution_3d,
            landmark_dir,
            out_dir,
            ref_landmark_volume,
            padding_offset=padding_offset,
            clobber=clobber,
        )

    return ref_chunk_fn


def set_init_tfm(init_tfm, fx_fn, mv_fn) -> str:
    init_tfm_list = []

    if init_tfm is None:
        init_str = f" --initial-moving-transform [{fx_fn},{mv_fn},1] "
    else:
        mv_rsl_fn = re.sub(".nii", "_init_moving.nii", mv_fn)
        utils.simple_ants_apply_tfm(mv_fn, fx_fn, init_tfm, mv_rsl_fn, n="BSpline[2]")
        mv_fn = mv_rsl_fn
        init_str = ""
        init_tfm_list = init_tfm

    return init_str, mv_fn, init_tfm_list


def run_alignment(
    out_dir: str,
    out_tfm_fn: str,
    out_fn: str,
    mv_fn: str,
    fx_fn: str,
    linParams: utils.AntsParams,
    nlParams: utils.AntsParams,
    ccParams: utils.AntsParams,
    metric: str = "GC",
    nbins: int = None,
    init_tfm: str = None,
    sampling: float = 0.9,
    use_3d_syn_cc: bool = True,
    linear_steps: str = ["rigid", "similarity", "affine"],
    clobber: bool = False,
) -> None:
    """Run the alignment of the tissue chunk to the chunk from the reference volume.

    :param out_dir: output directory
    :param out_tfm_fn: output transformation filename
    :param out_inv_fn: output inverse transformation filename
    :param out_fn: output filename
    :param moving_fn: reference volume filename
    :param ref_chunk_fn: reference chunk filename
    :param fixed_fn: segmentation volume filename
    :param linParams: linear parameters
    :param nlParams: nonlinear parameters
    :param ccParams: cross correlation parameters
    :param resolution: resolution of the section volume
    :param metric: metric to use for registration
    :param nbins: number of bins for registration
    :param init_tfm: use initial transformation
    :param sampling: sampling for registration
    :param clobber: overwrite existing files
    :return: None.
    """
    if use_3d_syn_cc:
        prefix = re.sub(
            "_SyN_CC_Composite.h5", "", out_tfm_fn
        )  # FIXME this is bad coding
    else:
        prefix = re.sub(
            "_SyN_Mattes_Composite.h5", "", out_tfm_fn
        )  # FIXME this is bad coding

    prefix + "_init_"
    prefix_rigid = prefix + "_Rigid_"
    prefix_similarity = prefix + "_Similarity_"
    prefix_affine = prefix + "_Affine_"
    prefix_manual = prefix + "_Manual_"

    affine_out_fn = f"{prefix_affine}volume.nii.gz"
    affine_inv_fn = f"{prefix_affine}volume_inverse.nii.gz"
    f"{prefix_manual}Composite.nii.gz"

    step = 0.5
    sampling = 0.9
    nbins = 32

    orig_mv_fn = mv_fn

    init_str, mv_fn, init_tfm_list = set_init_tfm(init_tfm, fx_fn, mv_fn)

    base = "antsRegistration -v 1 -a 1 -d 3 "

    def write_log(prefix: str, kind: str, cmd: str) -> None:
        with open(f"{prefix}/log_{kind}.txt", "w+") as F:
            F.write(cmd)
        return None

    # set initial transform
    # calculate rigid registration

    verbose = 0 if logger.getEffectiveLevel() == logging.INFO else 1

    # calculate rigid registration
    if not os.path.exists(f"{prefix_rigid}Composite.h5") and "rigid" in linear_steps:
        rigid_cmd = f"{base}  {init_str}  -t Rigid[{step}]  -m {metric}[{fx_fn},{mv_fn},1,{nbins},Random,{sampling}]  -s {linParams.s_str} -f {linParams.f_str}  -c {linParams.itr_str}  -o [{prefix_rigid},{prefix_rigid}volume.nii.gz,{prefix_rigid}volume_inverse.nii.gz] "
        utils.shell(rigid_cmd, verbose=verbose)
        write_log(out_dir, "rigid", rigid_cmd)
        init_str = f"--initial-moving-transform  {prefix_rigid}Composite.h5"

    # ,{prefix_rigid}volume.nii.gz,{prefix_rigid}volume_inverse.nii.gz]

    # calculate similarity registration
    if (
        not os.path.exists(f"{prefix_similarity}Composite.h5")
        and "similarity" in linear_steps
    ):
        similarity_cmd = f"{base}  {init_str} -t Similarity[{step}]  -m {metric}[{fx_fn},{mv_fn},1,{nbins},Random,{sampling}]   -s {linParams.s_str} -f {linParams.f_str} -c {linParams.itr_str}  -o [{prefix_similarity},{prefix_similarity}volume.nii.gz,{prefix_similarity}volume_inverse.nii.gz] "
        utils.shell(similarity_cmd, verbose=verbose)
        write_log(out_dir, "similarity", similarity_cmd)
        init_str = f"--initial-moving-transform {prefix_similarity}Composite.h5"

    # calculate affine registration
    # ,{affine_out_fn},{affine_inv_fn}]
    if not os.path.exists(f"{prefix_affine}Composite.h5") and "affine" in linear_steps:
        affine_cmd = f"{base}  {init_str} -t Affine[{step}] -m {metric}[{fx_fn},{mv_fn},1,{nbins},Random,{sampling}]  -s {linParams.s_str} -f {linParams.f_str}  -c {linParams.itr_str}  -o [{prefix_affine},{affine_out_fn},{affine_inv_fn}] "
        utils.shell(affine_cmd, verbose=verbose)
        write_log(out_dir, "affine", affine_cmd)
        init_str = f"--initial-moving-transform {prefix_affine}Composite.h5"

    init_file_str = ""

    prefix_mattes_syn = f"{prefix}{init_file_str}_SyN_Mattes_"
    mattes_syn_out_fn = f"{prefix_mattes_syn}volume.nii.gz"
    mattes_syn_inv_fn = f"{prefix_mattes_syn}volume_inverse.nii.gz"

    prefix_cc_syn = f"{prefix}{init_file_str}_SyN_CC_"
    cc_syn_out_fn = f"{prefix_cc_syn}volume.nii.gz"
    cc_syn_inv_fn = f"{prefix_cc_syn}volume_inverse.nii.gz"

    prefix_syn = prefix_mattes_syn
    syn_out_fn = mattes_syn_out_fn
    syn_inv_fn = mattes_syn_inv_fn

    # calculate nonlinear registration with Mattes metric
    if use_3d_syn_cc:
        prefix_syn = prefix_cc_syn
        syn_out_fn = cc_syn_out_fn
        syn_inv_fn = cc_syn_inv_fn

    if not os.path.exists(f"{prefix_syn}Composite.h5"):
        nl_metric = f"Mattes[{fx_fn},{mv_fn},1,32,Random,{sampling}]"

        cc_metric = f"CC[{fx_fn},{mv_fn},1,3,Random,{sampling}]"

        syn_rate = "0.1"

        nl_base = f"{base}   {init_str} -o [{prefix_syn},{syn_out_fn},{syn_inv_fn}] "
        # nl_base += f" -t SyN[{syn_rate}] -m {nl_metric}  -s {nlParams.s_str} -f {nlParams.f_str} -c {nlParams.itr_str} "

        if use_3d_syn_cc:
            nl_base += f" -t SyN[{syn_rate}] -m {cc_metric} -s {ccParams.s_str} -f {ccParams.f_str} -c {ccParams.itr_str} "
            # BSplineSyN
            # nl_base += f" -t BSplineSyN[{syn_rate},5x5x5] -m {cc_metric} -s {ccParams.s_str} -f {ccParams.f_str} -c {ccParams.itr_str} "
        else:
            nl_base += f" -t SyN[{syn_rate}] -m {nl_metric}  -s {nlParams.s_str} -f {nlParams.f_str} -c {nlParams.itr_str} "
            # nl_base += f" -t BSplineSyN[{syn_rate},5x5x5] -m {nl_metric}  -s {nlParams.s_str} -f {nlParams.f_str} -c {nlParams.itr_str} "

        utils.shell(nl_base, verbose=verbose)

        write_log(out_dir, "syn-mattes", nl_base)

        init_str = f"--initial-moving-transform {prefix_syn}Composite.h5"

        assert os.path.exists(
            f"{prefix_syn}Composite.h5"
        ), f"Error: {prefix_syn}Composite.h5 does not exist"

    if init_tfm:
        out_tfm_list = [out_tfm_fn, init_tfm]
    else:
        out_tfm_list = [out_tfm_fn]

    utils.simple_ants_apply_tfm(
        orig_mv_fn, fx_fn, out_tfm_list, out_fn, n="BSpline[2]", clobber=clobber
    )

    # if os.path.exists(f"{prefix_syn}InverseComposite.h5"):
    #    utils.simple_ants_apply_tfm(
    #        fx_fn, mv_fn, prefix_syn + "InverseComposite.h5", out_inv_fn, n="BSpline[2]",
    #    )

    return out_tfm_list


def align_3d(
    chunk: int,
    moving_fn: str,
    fixed_fn: str,
    out_dir: str,
    out_tfm_fn: str,
    out_fn: str,
    resolution: int,
    resolution_list: List[int],
    init_tfm: str = None,
    base_nl_itr: int = 200,
    base_lin_itr: int = 500,
    use_3d_syn_cc: bool = True,
    linear_steps: str = ["rigid", "similarity", "affine"],
    clobber: bool = False,
) -> int:
    """Align the tissue chunk to the reference volume.

    :param sub: subject id
    :param hemi: hemisphere
    :param chunk: chunk number
    :param fixed_fn: segmentation volume filename
    :param moving_fn: reference volume filename
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
    :param clobber: overwrite existing files
    :return: 0.
    """
    logger.info("\t\t3D Volumetric Alignment")
    chunk = int(chunk)

    # get iteration schedules for the linear and non-linear portion of the ants alignment
    # get maximum number steps by which the srv image will be downsampled by
    # with ants, each step means dividing the image size by 2^(level-1)
    (
        _,
        linParams,
        nlParams,
        ccParams,
    ) = get_alignment_schedule(
        resolution_list,
        resolution,
        base_nl_itr=base_nl_itr,
        base_lin_itr=base_lin_itr,
    )

    logger.info(f"\t\tResolution: {resolution}")
    # run ants alignment between segmented volume (from autoradiographs) to chunk extracte
    vol_tfm_list = run_alignment(
        out_dir,
        out_tfm_fn,
        out_fn,
        moving_fn,
        fixed_fn,
        linParams,
        nlParams,
        ccParams,
        sampling=0.95,
        use_3d_syn_cc=use_3d_syn_cc,
        linear_steps=linear_steps,
        metric="Mattes",
        init_tfm=init_tfm,
    )

    return vol_tfm_list
