"""Functions to run ANTs commands."""
import os
from os.path import basename
from re import sub
from shutil import copy
from typing import List, Optional, Tuple, Union

import numpy as np
import SimpleITK as sitk
from scipy.ndimage import gaussian_filter
from skimage.filters import threshold_otsu

import brainbuilder.utils.ants_nibabel as nib
from brainbuilder.utils.utils import shell, splitext


def generate_mask(fn:str, out_fn:str, sigma:float=8)->None:
    """Generate mask.
    
    :param fn: input file name
    :param out_fn: output file name
    :param sigma: sigma
    :return: None
    """
    if os.path.exists(out_fn):
        return None
    img = nib.load(fn)
    vol = img.get_fdata()
    vol = gaussian_filter(vol, sigma)

    if np.sum(vol) == 0:
        return 1
    idx = vol > threshold_otsu(vol)
    vol[idx] = 1
    vol[~idx] = 0
    nib.Nifti1Image(vol, img.affine).to_filename(out_fn)
    return 0


def ANTs(
    tfm_prefix: str,
    fixed_fn: str,
    moving_fn: str,
    moving_rsl_prefix: str,
    iterations: List[str],
    tolerance: float = 1e-08,
    metrics: Optional[List[str]] = None,
    nbins: int = 32,
    tfm_type: List[str] = ["Rigid", "Affine", "SyN"],
    rate: Optional[List[float]] = None,
    shrink_factors: Optional[List[str]] = None,
    smoothing_sigmas: Optional[List[str]] = None,
    radius: int = 3,
    init_tfm: Optional[Union[str, List[str]]] = None,
    init_inverse: bool = False,
    sampling_method: str = "Regular",
    sampling: int = 1,
    dim: int = 3,
    verbose: int = 0,
    clobber: int = 0,
    exit_on_failure: int = 0,
    fix_header: bool = False,
    generate_masks: bool = True,
    no_init_tfm: bool = False,
    mask_dir: Optional[str] = None,
    write_composite_transform: int = 1,
    collapse_output_transforms: int = 0,
    n_tries: int = 5,
    init_tfm_direction: str = "moving",
    interpolation: str = "Linear",
)->Tuple[str, str, str]:
    """Run ANTs registration using user parameters.
    
    :param tfm_prefix: transform prefix
    :param fixed_fn: fixed file name
    :param moving_fn: moving file name
    :param moving_rsl_prefix: moving reslice prefix
    :param iterations: iterations
    :param tolerance: tolerance
    :param metrics: metrics
    :param nbins: number of bins
    :param tfm_type: transform type
    :param rate: rate
    :param shrink_factors: shrink factors
    :param smoothing_sigmas: smoothing sigmas
    :param radius: radius
    :param init_tfm: initial transform
    :param init_inverse: initial inverse
    :param sampling_method: sampling method
    :param sampling: sampling
    :param dim: dimension
    :param verbose: verbose
    :param clobber: clobber
    :param exit_on_failure: exit on failure
    :param fix_header: fix header
    :param generate_masks: generate masks
    :param no_init_tfm: no initial transform
    :param mask_dir: mask directory
    :param write_composite_transform: write composite transform
    :param collapse_output_transforms: collapse output transforms
    :param n_tries: number of tries
    :param init_tfm_direction: initial transform direction
    :param interpolation: interpolation
    :return: final transform file name, final inverse transform file name, final moving reslice file name
    """
    nLevels = len(iterations)
    tfm_ext = "GenericAffine.mat"
    if "SyN" in tfm_type and write_composite_transform != 1:
        tfm_ext = f"{nLevels-1}Warp.nii.gz"
        tfm_inv_ext = f"{nLevels-1}InverseWarp.nii.gz"
    elif write_composite_transform == 1:
        tfm_ext = "Composite.h5"
        tfm_inv_ext = "InverseComposite.h5"
    final_moving_rsl_fn = (
        moving_rsl_prefix
        + "_level-"
        + str(nLevels - 1)
        + "_"
        + metrics[-1]
        + "_"
        + tfm_type[-1]
        + ".nii.gz"
    )
    final_tfm_fn = (
        f"{tfm_prefix}_level-{str(nLevels-1)}_{metrics[-1]}_{tfm_type[-1]}_{tfm_ext}"
    )
    final_tfm_inv_fn = f"{tfm_prefix}_level-{str(nLevels-1)}_{metrics[-1]}_{tfm_type[-1]}_{tfm_inv_ext}"

    # files exist, return early
    output_files = [final_moving_rsl_fn, final_tfm_fn, final_tfm_inv_fn]
    output_files_not_exists = [fn for fn in output_files if not os.path.exists(fn)]

    if len(output_files_not_exists) == 0 and clobber == 0:
        return final_tfm_fn, final_tfm_inv_fn, final_moving_rsl_fn
    else:
        print(
            "Following output files do not exist. Will run registration to create them"
        )
        print(output_files_not_exists)

    if verbose:
        print("Moving:", moving_fn)
        print("Fixed:", fixed_fn)

    if rate is None:
        rate = [0.1] * nLevels
    if shrink_factors is None:
        shrink_factors = ["4x2x1vox"] * nLevels
    if smoothing_sigmas is None:
        smoothing_sigmas = ["4.0x2.0x1.0vox"] * nLevels
    if metrics is None:
        metrics = ["Mattes"] * nLevels

    moving_mask_fn = None
    fixed_mask_fn = None
    if generate_masks:
        if mask_dir is not None:
            moving_mask_fn = (
                mask_dir + os.sep + basename(splitext(moving_fn)[0]) + "_mask.nii.gz"
            )
            fixed_mask_fn = (
                mask_dir + os.sep + basename(splitext(fixed_fn)[0]) + "_mask.nii.gz"
            )
        else:
            moving_mask_fn = moving_rsl_prefix + "_moving_mask.nii.gz"
            fixed_mask_fn = moving_rsl_prefix + "_fixed_mask.nii.gz"

        s = max(
            [
                int(float(j))
                for i in smoothing_sigmas
                for j in sub("vox|mm", "", i).split("x")
            ]
        )

        r = 0
        if not os.path.exists(moving_mask_fn):
            r += generate_mask(moving_fn, moving_mask_fn, s / np.pi)

        if not os.path.exists(fixed_mask_fn):
            r += generate_mask(fixed_fn, fixed_mask_fn, s / np.pi)

        if r != 0:
            moving_mask_fn = moving_mask_fn = None


    img_fx = nib.load(fixed_fn)
    img_mv = nib.load(moving_fn)
    # If image volume is empty, write identity matrix
    if np.sum(img_fx.get_data()) == 0 or np.sum(img_mv.get_data()) == 0:
        print("Warning: at least one of the image volume is empty")
        identity = sitk.Transform(3, sitk.sitkIdentity)
        sitk.WriteTransform(identity, final_tfm_fn)
        copy(moving_fn, final_moving_rsl_fn)

    for level in range(nLevels):
        moving_rsl_level_prefix = (
            moving_rsl_prefix
            + "_level-"
            + str(level)
            + "_"
            + metrics[level]
            + "_"
            + tfm_type[level]
        )
        tfm_level_prefix = (
            tfm_prefix
            + "_level-"
            + str(level)
            + "_"
            + metrics[level]
            + "_"
            + tfm_type[level]
            + "_"
        )

        moving_rsl_fn = moving_rsl_level_prefix + ".nii.gz"

        config_file = moving_rsl_level_prefix + "_parameters.txt"

        if (
            not os.path.exists(moving_rsl_fn)
            or not os.path.exists(final_tfm_fn)
            or clobber > 0
        ):
            # Set tfm file name
            # Warning only works if composite output == 1
            tfm_fn = tfm_level_prefix + tfm_ext

            moving_rsl_fn_inverse = moving_rsl_level_prefix + "_inverse.nii.gz"

            ### Inputs
            cmdline = "antsRegistration --verbose " + str(verbose)
            cmdline += f" --write-composite-transform {write_composite_transform} --float --collapse-output-transforms {collapse_output_transforms} --dimensionality {dim} "
            if not no_init_tfm:
                if init_tfm is None:
                    cmdline += (
                        f" --initial-{init_tfm_direction}-transform [ "
                        + fixed_fn
                        + ", "
                        + moving_fn
                        + ", 1 ] "
                    )
                elif init_tfm == "":
                    pass
                else:
                    if not isinstance(init_tfm, list):
                        init_tfm = [init_tfm]

                    if init_inverse:
                        cmdline += (
                            f" --initial-{init_tfm_direction}-transform ["
                            + ",".join(init_tfm)
                            + ",1] "
                        )
                    else:
                        cmdline += (
                            f" --initial-{init_tfm_direction}-transform "
                            + ",".join(init_tfm)
                            + " "
                        )

            if write_composite_transform == 1:
                initialize_transforms_per_stage = 1
            else:
                initialize_transforms_per_stage = 0

            cmdline += f" --initialize-transforms-per-stage {initialize_transforms_per_stage} --interpolation {interpolation} "

            # Set up smooth sigmas and shrink factors
            smooth_sigma = smoothing_sigmas[level]
            shrink_factor = shrink_factors[level]

            # Add masks
            if moving_mask_fn is not None and moving_mask_fn is not None:
                cmdline += (
                    " --masks [ " + fixed_mask_fn + " , " + moving_mask_fn + " ] "
                )

            ### Set tranform parameters for level
            cmdline += (
                " --transform " + tfm_type[level] + "[ " + str(rate[level]) + " ] "
            )
            cmdline += (
                " --metric "
                + metrics[level]
                + "["
                + fixed_fn
                + ", "
                + moving_fn
                + ", 1,"
            )
            if metrics[level] == "Mattes":
                cmdline += " " + str(nbins) + ", "
            else:
                cmdline += " " + str(radius) + ", "
            cmdline += sampling_method + " , " + str(sampling) + " ] "

            cmdline += (
                " --convergence [ "
                + iterations[level]
                + " , "
                + str(tolerance)
                + " , 20 ] "
            )
            if not isinstance(smooth_sigma, list):
                cmdline += f" --smoothing-sigmas {smooth_sigma}vox "
            else:
                cmdline += f" --smoothing-sigmas {smooth_sigma[0]}vox "
            if isinstance(shrink_factor, str):
                cmdline += " --shrink-factors " + shrink_factor
            else:
                cmdline += " --shrink-factors " + shrink_factor[0]

            ### Outputs
            cmdline += (
                " --output [ "
                + tfm_level_prefix
                + " ,"
                + moving_rsl_fn
                + ","
                + moving_rsl_fn_inverse
                + "] "
            )

            if verbose == 1:
                print(cmdline)

            errorcode = 0
            for attempt in range(n_tries):
                try:
                    # Run command line
                    stdout, stderr, errorcode = shell(cmdline, exit_on_failure=False)
                    print("Attempt:", attempt, errorcode)
                    if verbose == 1:
                        print(stdout)
                        print(stderr)
                except RuntimeError:
                    errorcode = 1
                    if exit_on_failure == 1:
                        return 1
                if errorcode == 0:
                    break
            if errorcode != 0:
                return 1

            with open(config_file, "w+") as f:
                f.write(cmdline)
            # update init_tfm
            init_tfm = [tfm_fn]
            no_init_tfm = False
            init_inverse = False

            # if fix_header and os.path.exists(moving_rsl_fn) :
            #    #nib.Nifti1Image(  nib.load(moving_rsl_fn).get_data(), nib.load(fixed_fn).affine ).to_filename(moving_rsl_fn)
            #   write_nifti( nib.load(moving_rsl_fn).get_data(), nib.load(fixed_fn).affine, moving_rsl_fn)

            # if fix_header and os.path.exists(moving_rsl_fn_inverse) :
            #    #nib.Nifti1Image(  nib.load(moving_rsl_fn_inverse).get_data(), nib.load(moving_fn).affine ).to_filename(moving_rsl_fn_inverse)
            #    write_nifti(  nib.load(moving_rsl_fn_inverse).get_data(), nib.load(moving_fn).affine, moving_rsl_fn_inverse)

    return final_tfm_fn, final_tfm_inv_fn, moving_rsl_fn


def antsApplyTransforms(
    input_image: str,
    reference_image: str,
    transform_list: List[str],
    output_image: str,
    interpolation: str = "Linear",
    input_image_type: int = 0,
    dimensionality: int = 3,
    verbose: int = 0,
)->None:
    """Apply ANTs transforms.
    
    :param input_image: input image
    :param reference_image: reference image
    :param transform_list: transform list
    :param output_image: output image
    :param interpolation: interpolation
    :param input_image_type: input image type
    :param dimensionality: dimensionality
    :param verbose: verbose
    :return: None
    """
    transforms = " -t " + " -t ".join(transform_list)
    shell(
        f"antsApplyTransforms -v {verbose} -d {dimensionality} -e {input_image_type}  -n {interpolation} -i {input_image} -r {reference_image} {transforms} -o {output_image}"
    )

    return None
