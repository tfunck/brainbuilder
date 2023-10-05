import multiprocessing
import os
import shutil
import tempfile

import nibabel
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.ndimage.filters import gaussian_filter

import brainbuilder.utils.ants_nibabel as nib
from brainbuilder.utils import utils
from brainbuilder.utils.utils import (
    AntsParams,
    gen_2d_fn,
    get_seg_fn,
    resample_to_resolution,
    shell,
)


def resample_reference_to_sections(
    resolution: float,
    input_fn: str,
    ref_fn: str,
    tfm_inv_fn: str,
    section_thickeness: float,
    output_dir: str,
    clobber: bool = False,
):
    """
    About:
        Apply 3d transformation and resample volume into the same coordinate space as 3d receptor volume.

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

    if not os.path.exists(iso_output_fn) or not os.path.exists(output_fn) or clobber:
        rand = tempfile.NamedTemporaryFile().name
        rand_fn = f"{rand}.nii.gz"
        utils.simple_ants_apply_tfm(input_fn, ref_fn, tfm_inv_fn, rand_fn, ndim=3)

        img = nib.load(rand_fn)
        vol = img.get_fdata()

        assert np.sum(vol) > 0, f"Error: empty volume {iso_output_fn}"

        aff = img.affine.copy()

        vol = (255 * (vol - vol.min()) / (vol.max() - vol.min())).astype(np.uint8)

        img_iso = resample_to_resolution(
            vol,
            [float(resolution)] * 3,
            order=0,
            aff=aff,
            dtype=np.uint16,
        )
        print("\tIso:", iso_output_fn)
        img_iso.to_filename(iso_output_fn)

        aff = img.affine.copy()
        img3 = resample_to_resolution(
            vol,
            [float(resolution), section_thickeness, float(resolution)],
            aff=aff,
            order=1,
            dtype=np.uint16,
        )
        img3.to_filename(output_fn)

        os.remove(rand_fn)

    return iso_output_fn, output_fn


def ants_registeration_2d_section(
    fx_fn: str,
    mv_fn: str,
    itr_str: str,
    s_str: str,
    f_str: str,
    prefix: str,
    transforms: list,
    metrics: list,
    bins: int = 32,
    sampling: float = 0.9,
    step: float = 0.5,
    init_str=None,
    verbose=False,
):
    """
    Use ANTs to register 2d sections
    """
    last_transform = None

    for transform, metric in zip(transforms, metrics):
        if type(last_transform) != type(None):
            init_str = (
                f"--initial-moving-transform {prefix}_{last_transform}_Composite.h5"
            )
        elif type(init_str) == str and os.path.exists(init_str) :
            init_str=f"--initial-moving-transform {init_str}"
        else:
            init_str = f"--initial-moving-transform [{fx_fn},{mv_fn},1]"
        command_str = f"antsRegistration -v {int(verbose)} -d 2    --write-composite-transform 1 {init_str} -o [{prefix}_{transform}_,{prefix}_{transform}_cls_rsl.nii.gz,/tmp/out_inv.nii.gz] -t {transform}[{step}]  -m {metric}[{fx_fn},{mv_fn},1,{bins},Random,{sampling}] -s {s_str} -f {f_str} -c {itr_str} "

        last_transform = transform

        if verbose:
            print(command_str)

        with open(prefix + f"{transform}_command.txt", "w") as f:
            f.write(command_str)

        shell(command_str)

    final_tfm = f"{prefix}_{transforms[-1]}_Composite.h5"

    return final_tfm


def align_2d_parallel(
    tfm_dir,
    mv_dir,
    resolution_itr,
    resolution,
    resolution_list,
    row,
    file_to_align="seg",
    use_syn=True,
    step=0.5,
    bins=32,
    base_lin_itr: int = 100,
    base_nl_itr: int = 20,
    base_cc_itr: int = 5,
    verbose=False,
):
    """ """
    # Set strings for alignment parameters

    cc_resolution_list = [
        r for r in resolution_list if float(r) <= 1
    ]  # CC does not work well at lower resolution
    if len(cc_resolution_list) > 0:
        cc_resolution = max(min(cc_resolution_list), resolution)
        if cc_resolution in cc_resolution_list:
            AntsParams(cc_resolution_list, cc_resolution, base_cc_itr)

    linParams = AntsParams(resolution_list, resolution, base_lin_itr)
    nlParams = AntsParams(resolution_list, resolution, base_nl_itr)

    y = int(row["sample"])

    prefix = f"{tfm_dir}/y-{y}"
    fx_fn = gen_2d_fn(prefix, "_fx")
    mv_fn = get_seg_fn(mv_dir, int(y), resolution, row[file_to_align], suffix="_rsl")

    verbose = False

    lin_transforms = ["Rigid", "Similarity", "Affine"]

    affine_tfm = ants_registeration_2d_section(
        fx_fn=fx_fn,
        mv_fn=mv_fn,
        itr_str=linParams.itr_str,
        s_str=linParams.s_str,
        f_str=linParams.f_str,
        prefix=prefix,
        transforms=lin_transforms,
        metrics=["Mattes"] * len(lin_transforms),
        verbose=verbose,
    )

    syn_tfm = ants_registeration_2d_section(
        fx_fn = fx_fn,
        mv_fn = mv_fn,
        itr_str = nlParams.itr_str,
        s_str = nlParams.s_str,
        f_str = nlParams.f_str,
        prefix = prefix,
        transforms = ["SyN", "SyN"],
        metrics = ["Mattes", "CC"],
        init_str=affine_tfm,
        step=0.1,
        verbose=verbose,
    )
    print(f"{prefix}_SyN_cls_rsl.nii.gz")
    print(row['2d_align_cls'])

    shutil.copy(f"{prefix}_SyN_cls_rsl.nii.gz", row['2d_align_cls'])
    shutil.copy(syn_tfm, row['2d_tfm'])

    # assert np.sum(nib.load(prefix+'_cls_rsl.nii.gz').dataobj) > 0, 'Error: 2d affine transfromation failed'
    print(f"\t\t\t{prefix}_Composite.h5")
    assert os.path.exists(
        f"{prefix}_cls_rsl.nii.gz"
    ), f"Error: output does not exist {prefix}_cls_rsl.nii.gz"

    return 0


def apply_transforms_parallel(tfm_dir, mv_dir, resolution_itr, resolution, row):
    y = int(row["sample"])
    prefix = f"{tfm_dir}/y-{y}"
    raw_rsl_fn = f"{prefix}_{resolution}mm.nii.gz"
    out_fn = prefix + "_rsl.nii.gz"
    fx_fn = gen_2d_fn(prefix, "_fx")


    raw_fn = row["raw"]  

    img = nib.load(raw_fn)
    img_res = np.array([img.affine[0, 0], img.affine[1, 1]])
    vol = img.get_fdata()

    sd = np.array((float(resolution) / img_res) / np.pi)
    vol = gaussian_filter(vol, sd)

    # vol = resize(vol, nib.load(fx_fn).shape, order=3)
    aff = img.affine
    nib.Nifti1Image(vol, aff, direction_order="lpi").to_filename(raw_rsl_fn)

    # fix_affine(fx_fn)
    shell(
        f"antsApplyTransforms -v 0 -d 2 -n NearestNeighbor -i {raw_rsl_fn} -r {fx_fn} -t {prefix}_Composite.h5 -o {out_fn} "
    )

    assert os.path.exists(f"{out_fn}"), "Error apply nl 2d tfm to raw autoradiograph"
    return 0


def align_sections(
    sect_info: pd.DataFrame,
    rec_fn: str,
    ref_fn: str,
    mv_dir: str,
    output_dir: str,
    resolution: float,
    resolution_itr: float,
    resolution_list: list,
    base_lin_itr: int = 100,
    base_nl_itr: int = 20,
    base_cc_itr: int = 5,
    file_to_align: str = "seg",
    use_syn: bool = True,
    batch_processing: bool = False,
    num_cores:int=1,
    verbose: bool = False,
    clobber: bool = False,
) -> None:
    """

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

    sect_info.reset_index(drop=True, inplace=True)
    sect_info.reset_index(drop=True, inplace=True)

    tfm_dir = output_dir + os.sep + "tfm"
    os.makedirs(tfm_dir, exist_ok=True)


    to_do_sect_info = []
    to_do_resample_sect_info = []

    sect_info["2d_tfm_affine"] = [""] * sect_info.shape[0]
    sect_info['2d_tfm'] = [""] * sect_info.shape[0]
    sect_info['2d_align'] = [""] * sect_info.shape[0]
    sect_info['2d_align_cls'] = [""] * sect_info.shape[0]

    for idx, (i, row) in enumerate(sect_info.iterrows()):

        y = int(row["sample"])
        prefix = f"{tfm_dir}/y-{y}"
        cls_fn = prefix + "_cls_rsl.nii.gz"
        out_fn = prefix + "_rsl.nii.gz"
        tfm_fn = prefix + "_Composite.h5"
        tfm_affine_fn = prefix + "_Affine_Composite.h5"

        sect_info["2d_tfm"].iloc[ idx ] = tfm_fn
        sect_info["2d_tfm_affine"].iloc[ idx ] = tfm_affine_fn

        sect_info["2d_align_cls"].iloc[ idx ] = cls_fn
        sect_info["2d_align"].iloc[ idx ]  = out_fn

    for idx, (i, row) in enumerate(sect_info.iterrows()):

        if not os.path.exists(tfm_fn) or not os.path.exists(cls_fn) or clobber:
            to_do_sect_info.append(row)

        if not os.path.exists(out_fn):
            to_do_resample_sect_info.append(row)

    for row in to_do_sect_info:
        align_2d_parallel(
            tfm_dir,
            mv_dir,
            resolution_itr,
            resolution,
            resolution_list,
            row,
            base_lin_itr=base_lin_itr,
            base_nl_itr=base_nl_itr,
            base_cc_itr=base_cc_itr,
            file_to_align=file_to_align,
            use_syn=use_syn,
            verbose=verbose,
        )

    if len(to_do_sect_info) > 0:
        Parallel(n_jobs=num_cores)(
            delayed(align_2d_parallel)(
                tfm_dir,
                mv_dir,
                resolution_itr,
                resolution,
                resolution_list,
                row,
                base_lin_itr=base_lin_itr,
                base_nl_itr=base_nl_itr,
                base_cc_itr=base_cc_itr,
                file_to_align=file_to_align,
                use_syn=use_syn,
                verbose=verbose,
            )
            for row in to_do_sect_info
        )

    if len(to_do_resample_sect_info) > 0:
        Parallel(n_jobs=num_cores)(
            delayed(apply_transforms_parallel)(
                tfm_dir, mv_dir, resolution_itr, resolution, row
            )
            for row in to_do_resample_sect_info
        )

    return sect_info


"""
    for i, row in sect_info.iterrows() :
        y = int(row["sample"])
        prefix = f'{tfm_dir}/y-{y}' 
        out_fn = prefix + '_rsl.nii.gz'
        tfm_fn = prefix + '_Composite.h5'
        tfm_affine_fn = prefix + '_Affine_Composite.h5'
        sect_info['tfm'].loc[ sect_info["sample"] == y ] = tfm_fn
        sect_info['tfm_affine'].loc[ sect_info["sample"] == y ] = tfm_affine_fn

    return sect_info #DEBUG untabbed this. seems like this would cause major problems
"""


def concatenate_sections_to_volume(
    sect_info, rec_fn, output_dir, out_fn, target_str="rsl"
):
    exit_flag = False
    tfm_dir = output_dir + os.sep + "tfm"

    hires_img = nib.load(rec_fn)
    out_vol = np.zeros(hires_img.shape)
    target_name = "nl_2d_" + target_str

    sect_info[target_name] = [""] * sect_info.shape[0]

    for idx, (i, row) in enumerate(sect_info.iterrows()):
        y = int(row["sample"])
        f"{tfm_dir}/y-{y}"
        fn = f"{tfm_dir}/y-{y}_{target_str}.nii.gz"

        sect_info[target_name].loc[i] = fn

    if not os.path.exists(out_fn):
        for idx, (i, row) in enumerate(sect_info.iterrows()):
            fn = sect_info[target_name].loc[i]
            y = int(row["sample"])

            try:
                # sec = nib.load(fn).get_fdata()
                sec = nibabel.load(fn).get_fdata()

                # DEBUG add this back in once the macaque raw data is fixed
                # assert np.max(sec) < 256 , 'Problem with file '+ fn + f'\n Max Value = {np.max(sec)}'
                out_vol[:, int(y), :] = sec
            except EOFError:
                print("Error:", fn)
                os.remove(fn)
                exit_flag = True

            if exit_flag:
                exit(1)

        print("\t\tWriting 3D non-linear:", out_fn)

        dtype = np.float32
        if target_str == "cls_rsl":
            dtype = np.uint8
        print(f"Writing: {out_fn} dtype: {dtype}")

        nib.Nifti1Image(
            out_vol, hires_img.affine, dtype=np.float32, direction_order="lpi"
        ).to_filename(out_fn)

    return sect_info


def align_2d(
    sect_info: pd.DataFrame,
    nl_2d_dir: str,
    seg_dir: str,
    ref_rsl_fn: str,
    resolution: float,
    resolution_itr: int,
    resolution_list: list,
    seg_rsl_fn: str,
    nl_3d_tfm_inv_fn: str,
    nl_2d_vol_fn: str,
    nl_2d_cls_fn: str,
    section_thickness: float,
    base_lin_itr=100,
    base_nl_itr=20,
    base_cc_itr=5,
    file_to_align="seg",
    num_cores: int = 1,
    clobber: bool = False,
    )->pd.DataFrame:
    """
    Align 2D sections to sections from reference volume using ANTs

    :param sect_info: dataframe containing section information
    :param chunk_info: dataframe containing chunk information
    :param resolution: resolution of the current iteration
    :param resolution_itr: current iteration
    :param resolution_list: list of resolutions
    :param file_to_align: file to align
    :param target_str: target string
    :return: sect_info
    """
    ref_iso_space_nat_fn, ref_space_nat_fn = resample_reference_to_sections(
        float(resolution),
        ref_rsl_fn,
        seg_rsl_fn,
        nl_3d_tfm_inv_fn,
        section_thickness,
        nl_2d_dir,
    )

    utils.create_2d_sections(
        sect_info, ref_space_nat_fn, float(resolution), nl_2d_dir, dtype=np.uint8
    )

    base_nl_itr = 1 #FIXME DELETEME
    base_cc_itr = 1 #FIXME DELETEME

    print("\t\tStep 4: 2d nl alignment")

    sect_info = align_sections(
        sect_info,
        ref_space_nat_fn,  # chunk_info["init_volume"],
        ref_space_nat_fn,
        seg_dir + "/2d/",
        nl_2d_dir,
        resolution,
        resolution_itr,
        resolution_list,
        base_lin_itr=base_lin_itr,
        base_nl_itr=base_nl_itr,
        base_cc_itr=base_cc_itr,
        num_cores=num_cores,
        clobber=clobber,
    )

    # Concatenate 2D nonlinear aligned sections into output volume
    sect_info = concatenate_sections_to_volume(
        sect_info, ref_space_nat_fn, nl_2d_dir, nl_2d_vol_fn
    )

    # Concatenate 2D nonlinear aligned cls sections into an output volume
    sect_info = concatenate_sections_to_volume(
        sect_info, ref_space_nat_fn, nl_2d_dir, nl_2d_cls_fn, target_str="cls_rsl"
    )

    return sect_info
