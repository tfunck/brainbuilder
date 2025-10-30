"""Performs 2D non-linear alignment of sections to sections from reference volume using ANTs."""

import glob
import json
import logging
import os
import shutil
import tempfile

import brainbuilder.utils.ants_nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from brainbuilder.qc.validate_section_alignment_to_ref import get_section_metric
from brainbuilder.utils import utils
from brainbuilder.utils.utils import (
    AntsParams,
    check_volume,
    concatenate_sections_to_volume,
    resample_to_resolution,
    shell,
    threshold,
)
from joblib import Parallel, cpu_count, delayed
from skimage.transform import resize

logger = utils.get_logger(__name__)


def resample_reference_to_sections(
    resolution: float,
    input_fn: str,
    ref_fn: str,
    tfm_inv_fn: str,
    output_dir: str,
    ymax: int, 
    section_thickness: float,
    clobber: bool = False,
) -> tuple:
    """Apply 3d transformation and resample volume into the same coordinate space as 3d receptor volume.

        The steps of this function are:
        1. Apply 3d transformation to the reference volume
        2. Resample the transformed volume to the resolution of the reconstruction
        3. Resample the transformed volume to the resolution of the section width on the y axis and the resolution of the reconstruction on the x and z axis

    Inputs:
        :param resolution:     current resolution level
        :param input_fn:     gm super-resolution volume (srv) extracted from donor brain
        :param ref_fn:     brain mask of segmented autoradiographs
        :param tfm_inv_fn:   3d transformation from mni to receptor coordinate space
        :return iso_output_fn:   gm srv volume in receptor coordinate space with isotropic voxels at <resolution>mm
        :return output_fn:  gm srv volume in receptor coordinate space with <section_thickness> dimension size along the y axis
    """
    basename = os.path.basename(input_fn).split(".")[0]

    iso_output_fn = f"{output_dir}/{basename}_{resolution}mm_iso.nii.gz"
    output_fn = f"{output_dir}/{basename}_{resolution}mm_space-nat.nii.gz"

    if (
        not os.path.exists(iso_output_fn) or not os.path.exists(output_fn) or clobber
    ):  
        # Apply 3d transformation to the reference volume
        rand = tempfile.NamedTemporaryFile().name

        rand_fn = f"{rand}.nii.gz"

        utils.simple_ants_apply_tfm(input_fn, ref_fn, tfm_inv_fn, rand_fn, ndim=3)

        img = nib.load(rand_fn)
        vol = img.get_fdata()

        assert np.sum(vol) > 0, f"Error: empty volume {iso_output_fn}"

        aff = img.affine.copy()

        # vol = (255 * (vol - vol.min()) / (vol.max() - vol.min())).astype(np.uint8)
        vol = vol.astype(np.uint8)

        # Resample the transformed volume to the resolution of the reconstruction
        #img_iso = resample_to_resolution(
        #    vol,
        #    [float(resolution)] * 3,
        #    order=3,
        #    affine=aff,
        #    dtype=np.uint8,
        #)
        ref_img = nib.load(ref_fn)
        print('ref fn', ref_fn)
        
        img_iso_ar = resize(vol, ref_img.shape, order=1, preserve_range=True).astype(np.uint8)

        img_iso = nib.Nifti1Image(img_iso_ar, ref_img.affine, direction_order="lpi")

        img_iso.to_filename(iso_output_fn)


        # Resample the transformed volume to the resolution of the section width on the y axis and the resolution of the reconstruction on the x and z axis
        vol = img_iso.get_fdata()
        affine = img_iso.affine.copy()
        affine[1, 1] = section_thickness
        
        vol = resize(
            vol.astype(float), (vol.shape[0], ymax , vol.shape[2]), order=1
        )  # .astype(np.uint8)
        img3 = nib.Nifti1Image(vol, affine, direction_order="lpi")

        img3.to_filename(output_fn)

        assert np.sum(vol) > 0, f"Error: empty volume {output_fn}"

        os.remove(rand_fn)

    return iso_output_fn, output_fn


def check_alignment_files(
    fx_fn: str, mv_fn: str, minimum_foreground_ratio: float = 0.2
) -> bool:
    """Check that the volume is not empty and return number of foreground voxels."""
    fx_foreground = check_volume(fx_fn)
    mv_foreground = check_volume(mv_fn)

    if (mv_foreground / fx_foreground < minimum_foreground_ratio) or (
        fx_foreground / mv_foreground < minimum_foreground_ratio
    ):
        logger.warning(
            f"Warning: skipping registration for due to insufficient foreground ratio between fixed and moving images:\n\tfx: {fx_fn}\n\tmv: {mv_fn}"
        )
        logger.warning(
            f" fixed image foreground voxels: {fx_foreground}, moving image foreground voxels: {mv_foreground}, minimum ratio: {minimum_foreground_ratio}"
        )
        return False
    return True


def create_identity_transform_2d(
    section_fn: str, tfm_prefix: str, output_log_fname: str
) -> str:
    """Create identity transform for 2D images.

    Description: Create identity transform for 2D images using ANTs.

    :param tfm_fn: transform filename
    :return: tfm_fn
    """
    tfm_fn = tfm_prefix + "Composite.h5"

    command_str = f"antsRegistration -v 1 -d 2 --write-composite-transform 1 -m GC[{section_fn},{section_fn},1,0,Regular,1] -t Rigid[1] -c 1 -f 1 -s 0  -o {tfm_prefix} &> {output_log_fname}"

    shell(command_str)

    assert os.path.exists(tfm_fn), f"Error: output does not exist {tfm_fn}"

    return tfm_fn


def ants_registration_2d_section(
    fx_fn: str,
    mv_fn: str,
    itr_list: str,
    s_list: str,
    f_list: str,
    prefix: str,
    transforms: list,
    metrics: list,
    bins: int = 32,
    sampling: float = 0.9,
    step: float = 0.5,
    masks: bool = False,
    init_tfm: str = None,
    write_composite_transform: int = 1,
    clobber: bool = False,
    exit_on_failure: bool = False,
    minimum_foreground_ratio: float = 0.2,
    verbose: bool = False,
) -> tuple:
    """Use ANTs to register 2d sections.

    Description: Calculate a series of transformations using ANTs to register 2d sections based on user provided parameters.

    :param fx_fn: fixed image filename
    :param mv_fn: moving image filename
    :param itr_list: list of iterations
    :param s_list: list of smoothing factors
    :param f_list: list of downsample factor sizes
    :param prefix: prefix for output files
    :param transforms: list of transforms
    :param metrics: list of metrics
    :param bins: bins
    :param sampling: sampling
    :param step: step
    :param init_tfm: initial transform
    :return: final_tfm, mv_rsl_fn
    """
    last_transform = None
    last_metric = None

    if write_composite_transform:
        final_tfm = f"{prefix}_{transforms[-1]}_{metrics[-1]}_Composite.h5"

    else:
        n_tfm = len(metrics) - 1

        if init_tfm != "identity":
            n_tfm += 1

        final_tfm = f"{prefix}_{transforms[-1]}_{metrics[-1]}_{n_tfm}Warp.nii.gz"

    mv_rsl_fn = f"{prefix}_{transforms[-1]}_{metrics[-1]}_cls_rsl.nii.gz"

    verbose = 0 if logger.getEffectiveLevel() == logging.INFO else 1

    if not os.path.exists(final_tfm) or not os.path.exists(mv_rsl_fn) or clobber:
        for transform, metric, f_str, s_str, itr_str in zip(
            transforms, metrics, f_list, s_list, itr_list
        ):
            command_log_fname = prefix + f"{transform}_{metric}_command.txt"
            output_log_fname = prefix + f"{transform}_{metric}_output.txt"

            mv_rsl_fn = f"{prefix}_{transform}_{metric}_cls_rsl.nii.gz"

            # check that the input files exist, are not empty, and have a sufficient number of foreground pixels
            if not check_alignment_files(
                fx_fn, mv_fn, minimum_foreground_ratio=minimum_foreground_ratio
            ):
                print("\tSkipping registration step.")
                if not (isinstance(init_tfm, str) and os.path.exists(init_tfm)):
                    init_tfm = create_identity_transform_2d(
                        mv_fn, prefix + "_identity_tfm_", command_log_fname
                    )
                return init_tfm, mv_fn

            if not isinstance(last_transform, type(None)):
                init_str = f"--initial-moving-transform {prefix}_{last_transform}_{last_metric}_Composite.h5"
            elif isinstance(init_tfm, str) and os.path.exists(init_tfm):
                init_str = f"--initial-moving-transform {init_tfm}"
            elif init_tfm == "identity":
                init_str = ""
            else:
                init_str = f"--initial-moving-transform [{fx_fn},{mv_fn},1]"

            command_str = f"antsRegistration -v 1 -d 2 --write-composite-transform {write_composite_transform} {init_str} -o [{prefix}_{transform}_{metric}_,{mv_rsl_fn},/tmp/out_inv.nii.gz] -t {transform}[{step}]  -m {metric}[{fx_fn},{mv_fn},1,{bins},Random,{sampling}] -s {s_str} -f {f_str} -c {itr_str} "

            if int(verbose) <= 0:
                command_str += f" &> {output_log_fname}"

            if masks:
                mask_fx = threshold(fx_fn)
                mask_mv = threshold(mv_fn)
                command_str += f" -x [{mask_fx},{mask_mv}] "

            last_transform = transform
            last_metric = metric

            logger.debug(command_str)

            with open(command_log_fname, "w") as f:
                f.write(command_str)

            shell(command_str)
            assert (
                not exit_on_failure or np.sum(np.abs(nib.load(mv_rsl_fn).dataobj)) > 0
            ), f"Error: empty volume {mv_rsl_fn}"

    assert os.path.exists(final_tfm), f"Error: output does not exist {final_tfm}"
    assert os.path.exists(mv_rsl_fn), f"Error: output does not exist {mv_rsl_fn}"

    return final_tfm, mv_rsl_fn


def affine_trials(
    fx_fn: str,
    mv_fn: str,
    linParams: AntsParams,
    prefix: str,
    n_trials: int = 5,
    verbose: bool = False,
) -> str:
    """About: Calculate affine transformation between volumes.

    Description: Try multiple affine transformations and return the best one based on dice score.
    :param fx_fn: fixed image filename
    :param mv_fn: moving image filename
    :param linParams: linear parameters
    :param prefix: prefix
    :param n_trials: number of trials
    :param verbose: verbose
    :return: affine_tfm
    """
    lin_transforms = ["Rigid", "Similarity", "Affine"]
    max_dice = 0
    best_trial = 0
    affine_tfm_trials = {}

    affine_dir = prefix + "_affines/"

    os.makedirs(affine_dir, exist_ok=True)

    for trial in range(n_trials):
        n = len(lin_transforms)
        itr_list = [linParams.itr_str] * n
        s_list = [linParams.s_str] * n
        f_list = [linParams.f_str] * n

        trial_prefix = affine_dir + f"_trial-{trial}"

        affine_tfm, mv_rsl_fn = ants_registration_2d_section(
            fx_fn=fx_fn,
            mv_fn=mv_fn,
            itr_list=itr_list,
            s_list=s_list,
            f_list=f_list,
            prefix=trial_prefix,
            transforms=lin_transforms,
            sampling=0.8,
            metrics=["Mattes"] * len(lin_transforms),
            verbose=verbose,
        )

        trial_dice, _ = get_section_metric(
            fx_fn, mv_rsl_fn, trial_prefix + "_dice.png", 0, verbose=False
        )

        best_trial = trial if trial_dice > max_dice else best_trial
        max_dice = trial_dice if trial_dice > max_dice else max_dice
        affine_tfm_trials[trial] = affine_tfm

    json.dump(affine_tfm_trials, open(f"{prefix}_affine_tfm_trials.json", "w"))

    for fn in glob.glob(f"{prefix}/*trial-*"):
        if "_trial-{best_trial}" not in fn:
            os.remove(fn)

    affine_tfm = affine_tfm_trials[best_trial]

    return affine_tfm


def align_2d_parallel(
    tfm_dir: str,
    resolution: float,
    resolution_list: list,
    row: pd.Series,
    file_to_align: str = "seg_rsl",
    use_syn: bool = True,
    base_lin_itr: int = 100,
    base_nl_itr: int = 30,
    n_affine_trials: int = 5,
    verbose: bool = False,
) -> int:
    """Align 2d sections to sections.

    Description: Calculate affine and non-linear transformations using ANTs to register 2d sections.

    :param output_dir: directory to store intermediate files
    :param mv_dir: directory to store intermediate files
    :param resolution_itr: current iteration
    :param resolution: resolution of the current iteration
    :param resolution_list: list of resolutions
    :param row: row
    :param file_to_align: file to align
    :param use_syn: use syn registration
    :param step: step
    :param bins: bins
    :param base_lin_itr: number of iterations for linear alignment
    :param base_nl_itr: number of iterations for nonlinear alignment
    :param base_cc_itr: number of iterations for cross correlation
    :param n_affine_trials: number of affine trials
    :param verbose: verbose
    :return: 0
    """
    # Set strings for alignment parameters
    base_nl_itr = 30

    linParams = AntsParams(resolution_list, resolution, base_lin_itr)

    nlParams = AntsParams(resolution_list, resolution, base_nl_itr)

    y = int(row["sample"])
    base = row["base"]

    prefix = f"{tfm_dir}/{base}_y-{y}"

    fx_fn = row["fx"]

    mv_fn = row[file_to_align]

    verbose = False
    affine_tfm = affine_trials(
        fx_fn, mv_fn, linParams, prefix, n_trials=n_affine_trials, verbose=verbose
    )

    if use_syn:
        nl_metrics = ["Mattes"]
        if resolution == resolution_list[-1]:
            nl_metrics = ["Mattes", "CC"]

        transforms = ["SyN"] * len(nl_metrics)

        syn_tfm, _ = ants_registration_2d_section(
            fx_fn=fx_fn,
            mv_fn=mv_fn,
            itr_list=[nlParams.itr_str, 20],
            s_list=[nlParams.s_str, 0],
            f_list=[nlParams.f_str, 1],
            prefix=prefix,
            transforms=transforms,
            metrics=nl_metrics,
            init_tfm=affine_tfm,
            step=0.1,
            verbose=verbose,
        )
    else:
        syn_tfm = affine_tfm

    cmd = f"antsApplyTransforms -v 0 -d 2 -n NearestNeighbor -i {mv_fn} -r {fx_fn} -t {syn_tfm} -o {row['2d_align_cls']} "

    shell(cmd, True)

    shutil.copy(syn_tfm, row["2d_tfm"])

    plt.imshow(nib.load(fx_fn).get_fdata())
    plt.imshow(
        nib.load(row["2d_align_cls"]).get_fdata(), cmap="nipy_spectral", alpha=0.4
    )
    plt.savefig(f"{prefix}_qc.png")
    plt.close()

    # assert np.sum(nib.load(prefix+'_cls_rsl.nii.gz').dataobj) > 0, 'Error: 2d affine transfromation failed'
    assert os.path.exists(
        f"{prefix}_cls_rsl.nii.gz"
    ), f"Error: output does not exist {prefix}_cls_rsl.nii.gz"

    return 0


def apply_transforms_parallel(
    tfm_dir: str,
    resolution: float,
    row: pd.Series,
    tissue_str: str = "",
    file_str: str = "img",
    interpolation: str = "Linear",
    verbose: bool = False,
) -> int:
    """Apply transforms to 2d sections.

    Description: Apply transforms to downsampled files to get the final 2d sections. A gaussian filter is used to pre-filter the images to avoid aliasing.

    :param tfm_dir: directory to store intermediate files
    :param mv_dir: directory to store intermediate files
    :param resolution: resolution of the current iteration
    :param row: row
    :return: 0
    """
    out_fn = row["2d_align" + tissue_str]

    if os.path.exists(out_fn):
        return out_fn

    y = int(row["sample"])

    base = row["base"]

    prefix = f"{tfm_dir}/{base}_y-{y}"

    img_rsl_fn = f"{prefix}_{resolution}mm{tissue_str}.nii.gz"

    img_fn = row[file_str]

    try:
        img = nib.load(img_fn)
    except Exception as e:
        logger.critical(f"Error loading image {img_fn}: {e}")
        exit(1)

    img_res = np.array([img.affine[0, 0], img.affine[1, 1]])

    # if we're not at the final resolution, we need to downsample the image
    if resolution != img_res[0] or resolution != img_res[1]:
        # sigma = utils.calculate_sigma_for_downsampling(resolution / img_res)
        # vol = img.get_fdata()
        # vol = gaussian_filter(vol, sigma)

        resample_to_resolution(
            img_fn,
            [float(resolution), float(resolution)],
            order=2,
            output_filename=img_rsl_fn,
        )

        # nib.Nifti1Image(vol, img.affine, direction_order="lpi").to_filename(img_rsl_fn)
    else:
        # if we're at the final resolution, we need to symlink the image

        # Check if the symlink exists and points to the correct file
        if os.path.islink(img_rsl_fn):
            if os.readlink(img_rsl_fn) != img_fn:
                os.remove(img_rsl_fn)
                os.symlink(img_fn, img_rsl_fn)

        elif not os.path.exists(img_rsl_fn):
            os.symlink(img_fn, img_rsl_fn)

        # the symlink is just created to help qc if the user needs to check the moving image before alignment
        # however ITK does not support symlinks so we rename img_rsl_fn to the actual file name
        img_rsl_fn = img_fn

    fx_fn = img_rsl_fn

    tfm_fn = row["2d_tfm"]

    cmd = f"antsApplyTransforms -v {int(verbose)} -d 2 -n {interpolation} -i {img_rsl_fn} -r {fx_fn} -t {tfm_fn} -o {out_fn} "

    shell(cmd, True)

    assert os.path.exists(f"{out_fn}"), "Error apply nl 2d tfm to img autoradiograph"

    return out_fn


def get_align_2d_to_do(sect_info: pd.DataFrame, clobber: bool = False) -> tuple:
    """Get files that need to be aligned.

    Description: Get list of files that need to be aligned and resampled based on whether tfm_fn and cls_fn exist.

    :param sect_info: dataframe containing section information
    :param clobber: clobber
    :return: to_do_sect_info, to_do_resample_sect_info
    """
    to_do_sect_info = []
    to_do_resample_sect_info = []

    for idx, (_, row) in enumerate(sect_info.iterrows()):
        cls_fn = row["2d_align_cls"]
        tfm_fn = row["2d_tfm"]
        out_fn = row["2d_align"]

        if not os.path.exists(tfm_fn) or not os.path.exists(cls_fn) or clobber:
            to_do_sect_info.append(row)

        if not os.path.exists(out_fn):
            to_do_resample_sect_info.append(row)

    return to_do_sect_info, to_do_resample_sect_info


def get_align_filenames(
    tfm_dir: str,
    sect_info: pd.DataFrame,
) -> pd.DataFrame:
    """Get filenames for alignment.

    :param tfm_dir: directory to store intermediate files
    :param sect_info: dataframe containing section information
    :return: sect_info
    """
    os.makedirs(tfm_dir, exist_ok=True)

    sect_info["2d_tfm_affine"] = [""] * sect_info.shape[0]
    sect_info["2d_tfm"] = [""] * sect_info.shape[0]
    sect_info["2d_align"] = [""] * sect_info.shape[0]
    sect_info["2d_align_cls"] = [""] * sect_info.shape[0]

    for idx, (_, row) in enumerate(sect_info.iterrows()):
        y = int(row["sample"])
        base = row["base"]
        prefix = f"{tfm_dir}/{base}_y-{y}"
        cls_fn = prefix + "_cls_rsl.nii.gz"
        out_fn = prefix + "_rsl.nii.gz"
        tfm_fn = prefix + "_Composite.h5"
        tfm_affine_fn = prefix + "_Affine_Composite.h5"

        sect_info.loc[idx, "2d_tfm"] = tfm_fn
        sect_info.loc[idx, "2d_tfm_affine"] = tfm_affine_fn

        sect_info.loc[idx, "2d_align_cls"] = cls_fn
        sect_info.loc[idx, "2d_align"] = out_fn

    return sect_info


def align_sections(
    sect_info: pd.DataFrame,
    output_dir: str,
    resolution: float,
    resolution_list: list,
    base_lin_itr: int = 100,
    base_nl_itr: int = 30,
    file_to_align: str = "seg_rsl",
    use_syn: bool = True,
    num_cores: int = 0,
    interpolation: str = "Linear",
    verbose: bool = False,
) -> None:
    """Align sections to sections from reference volume using ANTs.

    :param sect_info: dataframe containing information about sections
    :param rec_fn: filename of volume with 2d sections
    :param ref_fn: filename of reference volume transformed into acquisition space
    :param mv_dir: directory to store intermediate files
    :param output_dir: directory to store output files
    :param resolution: resolution to use for alignment
    :param resolution_itr: iteration of the resolution in the resolution list
    :param resolution_list: list of resolutions to use for alignment
    :param base_lin_itr: number of iterations for linear alignment
    :param base_nl_itr: number of iterations for nonlinear alignment
    :param base_cc_itr: number of iterations for cross correlation
    :param file_to_align: filename of file to align
    :param use_syn: use syn registration
    :param batch_processing: batch processing
    :param verbose: verbose
    :param clobber: clobber
    :return: None
    """
    num_cores = cpu_count() if num_cores == 0 else num_cores

    # num_cores = 1

    sect_info.reset_index(drop=True, inplace=True)

    tfm_dir = output_dir + os.sep + "tfm"

    sect_info = get_align_filenames(tfm_dir, sect_info)

    # get lists of files that need to be aligned and resampled
    to_do_sect_info, to_do_resample_sect_info = get_align_2d_to_do(sect_info)

    if len(to_do_sect_info) > 0:
        Parallel(n_jobs=num_cores, backend="multiprocessing")(
            delayed(align_2d_parallel)(
                tfm_dir,
                resolution,
                resolution_list,
                row,
                base_lin_itr=base_lin_itr,
                base_nl_itr=base_nl_itr,
                file_to_align=file_to_align,
                use_syn=use_syn,
                verbose=verbose,
            )
            for row in to_do_sect_info
        )

    if len(to_do_resample_sect_info) > 0:
        Parallel(n_jobs=num_cores, backend="multiprocessing")(
            delayed(apply_transforms_parallel)(
                tfm_dir, resolution, row, interpolation=interpolation, verbose=verbose
            )
            for row in to_do_resample_sect_info
        )

    return sect_info


def concatenate_tfm_sections_to_volume(
    sect_info: pd.DataFrame,
    rec_fn: str,
    output_dir: str,
    out_fn: str,
    target_str: str = "",
) -> pd.DataFrame:
    """Concatenate 2D sections into output volume.

    Description: Concatenate 2D sections into output volume. The steps of this function are:

    :param sect_info: dataframe containing section information
    :param rec_fn: filename of volume with 2d sections
    :param output_dir: directory to store output files
    :param out_fn: output filename
    :param target_str: target string
    :return: sect_info
    """
    hires_img = nib.load(rec_fn)
    target_name = "2d_align" + target_str

    # sect_info[target_name] = [""] * sect_info.shape[0]

    # def set_target_name(base, y):
    #    return f"{tfm_dir}/{base}_y-{y}_{target_str}.nii.gz"

    # sect_info[target_name] = sect_info.apply(
    #    lambda row: set_target_name(row["base"], int(row["sample"])), axis=1
    # )

    concatenate_sections_to_volume(
        sect_info, target_name, out_fn, hires_img.shape, hires_img.affine
    )

    return sect_info


def align_2d(
    sect_info: pd.DataFrame,
    nl_2d_dir: str,
    ref_rsl_fn: str,
    resolution: float,
    resolution_list: list,
    seg_rsl_fn: str,
    nl_3d_tfm_inv_fn: str,
    nl_2d_vol_fn: str,
    nl_2d_cls_fn: str,
    section_thickness: float,
    base_lin_itr: int = 100,
    base_nl_itr: int = 20,
    use_syn: bool = True,
    file_to_align: str = "seg_rsl",
    num_cores: int = 1,
    interpolation: str = "Linear",
    clobber: bool = False,
) -> pd.DataFrame:
    """Align 2D sections to sections from reference volume using ANTs.

    Description: Align 2D sections to sections from reference volume using ANTs. The steps of this function are:
    1. Resample reference volume to the resolution of the reconstruction and to the section width along the y-axis
    2. Create 2D sections from the resampled reference volume
    3. Align 2D sections to sections from reference volume using ANTs
    4. Concatenate 2D sections into output volumes

    :param sect_info: dataframe containing section information
    :param nl_2d_dir: directory to store intermediate files
    :param seg_dir: directory to store intermediate files
    :param ref_rsl_fn: filename of volume of template downsampled to 3D resolution
    :param resolution: resolution of the current iteration
    :param resolution_itr: current iteration
    :param resolution_list: list of resolutions
    :param chunk_info: dataframe containing chunk information
    :param resolution: resolution of the current iteration
    :param resolution_itr: current iteration
    :param resolution_list: list of resolutions
    :param file_to_align: file to align
    :param target_str: target string
    :return: sect_info
    """
    ymax = sect_info["sample"].max() + 1

    # Apply 3D transformation to reference volume and resample to the resolution
    # of the reconstruction and to the section thickness along the y-axis
    _, ref_space_nat_fn = resample_reference_to_sections(
        float(resolution),
        ref_rsl_fn,
        seg_rsl_fn,
        nl_3d_tfm_inv_fn,
        nl_2d_dir,
        ymax,
        section_thickness
    )

    # Define fixed 'fx' filenames for each section from the resampled reference volume
    sect_info["fx"] = utils.get_fx_list(sect_info, nl_2d_dir, "_fx")

    # Create 2D sections from the resampled reference volume
    utils.create_2d_sections(
        sect_info["fx"].values,
        sect_info["sample"].values,
        ref_space_nat_fn,
        nl_2d_dir,
        dtype=np.uint8,
    )

    logger.info("\t\tStep 4: 2d nl alignment")
    sect_info["base"] = sect_info["raw"].apply(
        lambda x: os.path.basename(x).split(".")[0]
    )

    # Align 2D sections to sections from reference volume using ANTs
    sect_info = align_sections(
        sect_info,
        nl_2d_dir,
        resolution,
        resolution_list,
        use_syn=use_syn,
        base_lin_itr=base_lin_itr,
        base_nl_itr=base_nl_itr,
        num_cores=num_cores,
        interpolation=interpolation,
        file_to_align=file_to_align,
    )

    # Concatenate 2D nonlinear aligned sections into output volume
    sect_info = concatenate_tfm_sections_to_volume(
        sect_info, ref_space_nat_fn, nl_2d_dir, nl_2d_vol_fn
    )

    # Concatenate 2D nonlinear aligned cls sections into an output volume
    sect_info = concatenate_tfm_sections_to_volume(
        sect_info, ref_space_nat_fn, nl_2d_dir, nl_2d_cls_fn, target_str="_cls"
    )

    return sect_info, ref_space_nat_fn
