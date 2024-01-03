import os
import re
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.transform import resize

import brainbuilder.utils.ants_nibabel as nib
from brainbuilder.utils import utils
from brainbuilder.utils import validate_inputs as valinpts
from brainbuilder.utils.ANTs import ANTs

from brainbuilder.utils.utils import AntsParams

def load2d(fn):
    ar = nib.load(fn).get_fdata()
    ar = ar.reshape(ar.shape[0], ar.shape[1])
    return ar


def align_neighbours_to_fixed(
    i,
    j_list,
    df,
    transforms,
    iteration,
    shrink_factor,
    smooth_sigma,
    output_dir,
    tfm_type,
    desc,
    target_acquisition=None,
    clobber=False,
):
    # For neighbours

    # a single section is selected as fixed (ith section), then the subsequent sections are considred moving sections (jth section)
    # and are registered to the fixed section.

    i_idx = df["sample"] == i
    fixed_fn = df["img"].loc[i_idx].values[0]
    for j in j_list:
        j_idx = df["sample"] == j
        outprefix = "{}/init_transforms/{}_{}-0/".format(output_dir, j, tfm_type)

        moving_rsl_fn = outprefix + "_level-0_Mattes_{}.nii.gz".format(tfm_type)
        tfm_fn = outprefix + "_level-0_Mattes_{}_Composite.h5".format(tfm_type)
        concat_tfm_fn = (
            outprefix + "level-0_Mattes_{}_Composite_Concatenated.h5".format(tfm_type)
        )
        qc_fn = "{}/qc/{}_{}_{}_{}_{}_{}-0.png".format(
            output_dir, *desc, j, i, tfm_type
        )
        moving_fn = df["img"].loc[j_idx].values[0]

        # calculate rigid transform from moving to fixed images
        if not os.path.exists(tfm_fn) or not os.path.exists(qc_fn):
            print()
            print("\tFixed:", i, fixed_fn)
            print("\tMoving:", j, moving_fn)
            print("\tTfm:", tfm_fn, os.path.exists(tfm_fn))
            print("\tQC:", qc_fn, os.path.exists(qc_fn))
            print("\tMoving RSL:", moving_rsl_fn)

            if target_acquisition != None:
                if df["acquisition"].loc[j_idx].values[0] != target_acquisition:
                    print("\tSkipping")
                    continue

            os.makedirs(outprefix, exist_ok=True)
            inv_tfm_fn = outprefix + "level-0_Mattes_{}_InverseComposite.h5".format(
                tfm_type
            )

            ANTs(
                tfm_prefix=outprefix,
                fixed_fn=fixed_fn,
                moving_fn=moving_fn,
                moving_rsl_prefix=outprefix + "tmp",
                iterations=[iteration],
                metrics=["Mattes"],
                tfm_type=[tfm_type],
                shrink_factors=[shrink_factor],
                smoothing_sigmas=[smooth_sigma],
                init_tfm=None,
                no_init_tfm=False,
                dim=2,
                sampling_method="Random",
                sampling=0.5,
                verbose=0,
                generate_masks=False,
                clobber=True,
            )

        # concatenate the transformation files that have been applied to the fixed image and the new transform
        # that is being applied to the moving image
        if not os.path.exists(concat_tfm_fn):
            transforms_str = "-t {} ".format(" -t ".join(transforms[i] + [tfm_fn]))
            utils.shell(
                f"antsApplyTransforms -v 0 -d 2 -i {moving_fn} -r {moving_fn} {transforms_str} -o Linear[{concat_tfm_fn}] ",
            )

        # apply the concatenated transform to the moving image. this brings it into correct alignment with all of the
        # aligned images.
        if not os.path.exists(moving_rsl_fn):
            utils.shell(
                f"antsApplyTransforms -v 0 -d 2 -i {moving_fn} -r {fixed_fn} -t {concat_tfm_fn} -o {moving_rsl_fn}",
            )

        if not os.path.exists(qc_fn):
            create_qc_image(
                load2d(fixed_fn),
                load2d(moving_fn),
                load2d(moving_rsl_fn),
                i,
                j,
                df["tier"].loc[i_idx],
                df["tier"].loc[j_idx],
                df["acquisition"].loc[i_idx],
                df["acquisition"].loc[j_idx],
                qc_fn,
            )

        df["img_new"].loc[df["sample"] == j] = moving_rsl_fn
        df["init_tfm"].loc[df["sample"] == j] = concat_tfm_fn
        df["init_fixed"].loc[df["sample"] == j] = fixed_fn
        transforms[j] = [concat_tfm_fn]
    return df, transforms


def create_qc_image(
    fixed,
    moving,
    rsl,
    fixed_order,
    moving_order,
    tier_fixed,
    tier_moving,
    acquisition_fixed,
    acquisition_moving,
    qc_fn,
):
    plt.subplot(1, 2, 1)
    plt.title(
        "fixed (gray): {} {} {}".format(fixed_order, tier_fixed, acquisition_fixed)
    )
    plt.imshow(fixed, cmap=plt.cm.gray)
    plt.imshow(moving, alpha=0.35, cmap=plt.cm.hot)
    plt.subplot(1, 2, 2)
    plt.title(
        "moving (hot): {} {} {}".format(moving_order, tier_moving, acquisition_moving)
    )
    plt.imshow(fixed, cmap=plt.cm.gray)
    plt.imshow(rsl, alpha=0.35, cmap=plt.cm.hot)
    plt.tight_layout()
    plt.savefig(qc_fn)
    plt.clf()


def adjust_alignment(
    df,
    y_idx,
    mid,
    transforms,
    step,
    output_dir,
    desc,
    shrink_factor,
    smooth_sigma,
    iteration,
    tfm_type,
    target_acquisition=None,
    target_tier=1,
    clobber=False,
):
    """ """
    os.makedirs(output_dir + "/qc/", exist_ok=True)
    i = mid
    j = i
    len(y_idx)
    y_idx_tier1 = df["sample"].loc[df["tier"] == 1].values.astype(int)
    
    y_idx_tier1.sort()
    i_max = step if step < 0 else df["sample"].values.max() + 1

    # Iterate over the sections along y-axis
    for y in y_idx_tier1[mid::step]:
        j_list = []
        for j in range(int(y + step), int(i_max), int(step)):
            if j in df["sample"].loc[df["tier"] == target_tier].values.astype(int):
                j_list.append(int(j))

            if j in y_idx_tier1:
                break

        df, transforms = align_neighbours_to_fixed(
            y,
            j_list,
            df,
            transforms,
            iteration,
            shrink_factor,
            smooth_sigma,
            output_dir,
            tfm_type,
            desc,
            target_acquisition=target_acquisition,
            clobber=clobber,
        )


    # df.to_csv('{}/df_{}-0.csv'.format(output_dir,tfm_type))
    return transforms, df


def apply_transforms_to_sections(
    df, transforms, output_dir, tfm_type, target_acquisition=None, clobber=False
):
    print("Applying Transforms")
    if not target_acquisition == None:
        df = df.loc[df["acquisition"] == target_acquisition]

    for i, (rowi, row) in enumerate(df.iterrows()):
        outprefix = "{}/init_transforms/{}_{}-0/".format(
            output_dir, int(row["sample"]), tfm_type
        )
        final_rsl_fn = outprefix + "final_level-0_Mattes_{}-0.nii.gz".format(tfm_type)

        if not os.path.exists(final_rsl_fn) or clobber:
            fixed_tfm_list = transforms[row["sample"]]
            if len(fixed_tfm_list) == 0:
                continue
            transforms_str = " -t {}".format(" -t ".join(fixed_tfm_list))

            fixed_fn = row["original_img"]

            utils.shell(
                f"antsApplyTransforms -v 1 -d 2 -i {fixed_fn} -r {fixed_fn} {transforms_str} -o {final_rsl_fn}",
            )
        df["img_new"].iloc[i] = final_rsl_fn
    return df


def combine_sections_to_vol(df, y_mm, z_mm, direction, out_fn, target_tier=1):
    example_fn = df["img"].iloc[0]
    shape = nib.load(example_fn).shape
    affine = nib.load(example_fn).affine
    xstart = affine[0, 3]
    zstart = affine[1, 3]
    xstep = affine[0, 0]
    zstep = affine[1, 1]
    ystep = y_mm

    xmax = int(shape[0] )
    zmax = int(shape[1] )
    order_max = df["sample"].astype(int).max()
    df["sample"].astype(int).min()
    chunk_ymax = int(order_max + 1)  # -order_min + 1

    vol = np.zeros([xmax, chunk_ymax, zmax])
    df = df.sort_values("sample")
    for i, row in df.iterrows():
        if row["tier"] == target_tier:
            y = row["sample"]
            ar = nib.load(row["img"]).get_fdata()
            ar = ar.reshape(ar.shape[0], ar.shape[1])
            ar = resize(ar, [xmax, zmax])
            vol[:, int(y), :] = ar
            # vol[:,int(y),:] += int(y)
            del ar

    print("\n\tWriting Volume", out_fn, "\n")
    chunk_ymin = -126 + df["sample"].min() * y_mm

    affine = np.array(
        [
            [xstep, 0, 0, xstart],
            [0, ystep, 0, chunk_ymin],
            [0, 0, zstep, zstart],
            [0, 0, 0, 1],
        ]
    )
    affine = np.round(affine, 3)
    # flip the volume along the y-axis so that the image is in RAS coordinates because ANTs requires RAS
    # vol = np.flip(vol, axis=1)
    nib.Nifti1Image(vol, affine, direction_order="lpi").to_filename(out_fn)


def alignment_stage(
    sub,
    hemisphere,
    chunk,
    df,
    vol_fn_str,
    output_dir,
    transforms,
    linParams,
    desc=(0, 0, 0),
    target_acquisition=None,
    target_tier=1,
    acquisition_n=0,
    clobber=False,
):
    """
    Perform alignment of autoradiographs within a chunk. Alignment is calculated once from the middle section in the
    posterior direction and a second time from the middle section in the anterior direction.
    """
    # Set parameters for rigid transform
    tfm_type = "Rigid"
    shrink_factor = linParams.f_str #'12x10x8' #x4x2x1'
    smooth_sigma = re.sub('vox', '', linParams.s_str) # '6x5x4' #x2x1x0'
    iterations = linParams.itr_str.split(',')[0][1:]  # '100x50x25' #x100x50x20'

    shrink_factor = '4x3x2'
    smooth_sigma = '2x1.5x1'
    iterations = '100x50x25' 

    csv_fn = vol_fn_str.format(
        output_dir,
        *desc,
        target_acquisition,
        acquisition_n,
        tfm_type + "-" + str(0),
        ".csv",
    )

    if not os.path.exists(csv_fn) or not os.path.exists(out_fn) :
        df.sort_values(["sample"], inplace=True, ascending=False)

        y_idx = df["sample"].values

        y_idx_tier1 = (
            df["sample"].loc[df["tier"].astype(int) == np.min(df["tier"])].values
        )
        mid = int(len(y_idx_tier1) / 2)

        df["img_new"] = df["img"]
        df["init_tfm"] = [None] * df.shape[0]
        df["init_img"] = df['img_new'] 
        df["init_fixed"] = [None] * df.shape[0]

        # perform alignment in forward direction from middle section
        transforms, df = adjust_alignment(
            df,
            y_idx,
            mid,
            transforms,
            -1,
            output_dir,
            desc,
            shrink_factor,
            smooth_sigma,
            iterations,
            tfm_type,
            target_acquisition=target_acquisition,
            target_tier=target_tier,
            clobber=clobber,
        )

        # perform alignment in reverse direction from middle section
        transforms, df = adjust_alignment(
            df,
            y_idx,
            mid,
            transforms,
            1,
            output_dir,
            desc,
            shrink_factor,
            smooth_sigma,
            iterations,
            tfm_type,
            target_acquisition=target_acquisition,
            target_tier=target_tier,
            clobber=clobber,
        )

        # update the img so it has the new, resampled file names
        df["img"] = df["img_new"]

        out_fn = vol_fn_str.format(
            output_dir,
            *desc,
            target_acquisition,
            acquisition_n,
            tfm_type + "-" + str(0),
            "nii.gz",
        )
    # else :
    #    df = pd.read_csv(csv_fn)

    return df, transforms


def create_final_outputs(
        final_tfm_dir:str,
        df:pd.DataFrame, 
        step:int,
        )->pd.DataFrame:
    '''
    Create final outputs for the alignment stage
    :param final_tfm_dir: path to directory containing final transforms
    :param df: dataframe containing section information
    :param step: step size for iterating over sections
    :return: dataframe containing section information
    '''
    y_idx_tier = df["sample"].values.astype(int)
    y_idx_tier.sort()

    mid = int(len(y_idx_tier) / 2)
    if step < 0:
        mid -= 1
    step if step < 0 else df["sample"].values.max() + 1

    #                     c
    #            <--------------
    #                    b
    #            <---------
    #    l  l    |  a l b1 l   l
    #    l  l -> | <- l <- l   l
    #    l  l    |    l    l   l
    #   s0  s1   s2   s3   s4  s5
    #   a ( b1 (s2) )  --> s2
    #
    for i, y in enumerate(y_idx_tier[mid::step]):
        row = df.loc[y == df["sample"]]

        final_tfm_fn = "{}/{}_final_Rigid.h5".format(
            final_tfm_dir, int(row["sample"].values[0])
        )
        final_section_fn = f'{final_tfm_dir}/{int(row["sample"].values[0])}{os.path.basename(row["img"].values[0])}'
        print(row['init_img'].values)
        print(row['sample'].values)
        idx = df["sample"].values == row["sample"].values
        if not os.path.exists(final_tfm_fn) or not os.path.exists(final_section_fn):

            row["sample"].values[0].astype(int)
            if type(row["init_tfm"].values[0]) == str:
                # standard rigid transformation for moving image
                shutil.copy(row["init_tfm"].values[0], final_tfm_fn)

                if not os.path.exists(final_section_fn) and not os.path.islink(
                    final_section_fn
                ):
                    os.symlink(row["img"].values[0], final_section_fn)
            else:
                if not os.path.exists(final_section_fn) and not os.path.islink(
                    final_section_fn
                ):
                    os.symlink(row["img"].values[0], final_section_fn)
                final_tfm_fn = np.nan

        df["init_tfm"].loc[idx] = final_tfm_fn
        df['init_img'].loc[idx] = final_section_fn
    return df


def initalign(
    sect_info_csv: str,
    chunk_info_csv: str,
    output_dir: str,
    resolution_list: list,
    clobber: bool = True
) -> str:
    """
    Calulate initial rigid aligment between sections

    param sect_info_csv: path to csv file containing section information
    param chunk_info_csv: path to csv file containing chunk information
    param output_dir: path to output directory
    param clobber: overwrite existing files
    return: path to csv file containing section information
    """
    valid_inputs_npz = os.path.join(output_dir, "valid_inputs")

    # Validate Inputs
    #FIXME UNCOMMENT
    #assert valinpts.validate_csv(
    #    sect_info_csv, valinpts.sect_info_required_columns 
    #), f"Invalid section info csv file: {sect_info_csv}"
    #assert valinpts.validate_csv(
    #    chunk_info_csv, valinpts.chunk_info_required_columns
    #), f"Invalid chunk info csv file: {chunk_info_csv}"

    initalign_sect_info_csv = os.path.join(output_dir, f"initalign_sect_info.csv")
    initalign_chunk_info_csv = os.path.join(output_dir, f"initalign_chunk_info.csv")
    initalign_sect_info = pd.DataFrame({})
    initalign_chunk_info = pd.DataFrame({})

    run_stage = utils.check_run_stage(initalign_sect_info_csv, 'init_tfm', 'seg')

    linParams = AntsParams(resolution_list, resolution_list[-1], 100)  

    if (
        not os.path.exists(initalign_sect_info_csv)
        or not os.path.exists(initalign_chunk_info_csv)
        or clobber
        or run_stage #FIXME need to change initalign so that it overwrites outdated files
    ):
        sect_info = pd.read_csv(sect_info_csv)
        chunk_info = pd.read_csv(chunk_info_csv)

        initalign_sect_info = pd.DataFrame({})

        for (sub, hemisphere, chunk), curr_sect_info in sect_info.groupby(
            ["sub", "hemisphere", "chunk"]
        ):
            idx = (
                (chunk_info["sub"] == sub)
                & (chunk_info["hemisphere"] == hemisphere)
                & (chunk_info["chunk"] == chunk)
            )
            curr_chunk_info = chunk_info.loc[idx]

            curr_output_dir = f"{output_dir}/{sub}_{hemisphere}_{chunk}/"

            curr_sect_info, curr_chunk_info = align_chunk(
                sub,
                hemisphere,
                chunk,
                curr_output_dir,
                curr_sect_info,
                linParams,
                curr_chunk_info,
                clobber=clobber,
            )
            initalign_sect_info = pd.concat([initalign_sect_info, curr_sect_info])
            initalign_chunk_info = pd.concat([initalign_chunk_info, curr_chunk_info])

        initalign_sect_info.to_csv(initalign_sect_info_csv, index=False)
        initalign_chunk_info.to_csv(initalign_chunk_info_csv, index=False)

    return initalign_sect_info_csv, initalign_chunk_info_csv


def get_acquisition_contrast_order(df):
    df_list = []
    for i, acquisition_df in df.groupby(["acquisition"]):
        for j, row in acquisition_df.iterrows():
            ar = nib.load(row["img"]).dataobj
            i_max = np.max(ar)
            i_min = np.min(ar)
            contrast = (i_max - i_min) / (i_max + i_min)

            df_list.append(
                pd.DataFrame(
                    {"acquisition": [row["acquisition"]], "contrast": [contrast]}
                )
            )

    df = pd.concat(df_list)


    df_mean = df.groupby(["acquisition"]).mean()
    df_mean.reset_index(inplace=True)

    df_mean = df_mean.sort_values(by=["contrast"], ascending=False)

    acquisition_contrast_order = df_mean["acquisition"].values

    return acquisition_contrast_order


def align_chunk(
    sub,
    hemisphere,
    chunk,
    output_dir,
    sect_info,
    linParams,
    chunk_info,
    clobber=False,
):
    """
    Calulate initial rigid aligment between sections for a given chunk
    param sub: subject id
    param hemi: hemisphere
    param chunk: chunk
    param output_dir: path to output directory
    param sect_info: dataframe containing section information
    param chunk_info: dataframe containing chunk information
    param clobber: overwrite existing files
    return: dataframe containing section information

    """
    os.makedirs(output_dir, exist_ok=True)

    init_align_fn = f"{output_dir}/{sub}_{hemisphere}_{chunk}_init_align.nii.gz"
    init_tfm_csv = f"{output_dir}/{sub}_{hemisphere}_{chunk}_init_tfm.csv"

    df = sect_info.copy()

    df["original_img"] = df["img"]
    
    acquisition_contrast_order = get_acquisition_contrast_order(sect_info)
    print("\tAcquistion contrast order: ", acquisition_contrast_order)

    df["tier"] = 1

    chunk_img_fn_str = "{}/sub-{}_hemi-{}_chunk-{}_acquisition_{}_{}_init_align_{}.{}"

    n_acquisitions = len(df["acquisition"].unique())
    _, z_mm, y_mm = utils.get_chunk_pixel_size(sub, hemisphere, chunk, chunk_info)

    direction = utils.get_chunk_direction(sub, hemisphere, chunk, chunk_info)

    ###########
    # Stage 1 #
    ###########
    print("\t\tAlignment: Stage 1")
    # Perform within acquisition alignment
    output_dir_1 = output_dir + os.sep + "stage_1"

    # Iterate data frames over each acquisition
    df_acquisition = df.loc[df["acquisition"] == acquisition_contrast_order[0]]

    # Init dict with initial transforms
    transforms_1 = {}
    for i in df_acquisition["sample"]:
        transforms_1[i] = []

    df_acquisition, transforms_1 = alignment_stage(
        sub,
        hemisphere,
        chunk,
        df_acquisition,
        chunk_img_fn_str,
        output_dir_1,
        transforms_1,
        linParams,
        target_acquisition=acquisition_contrast_order[0],
        acquisition_n=0,
        target_tier=1,
        desc=(sub, hemisphere, chunk),
        clobber=clobber,
    )

    # update the master dataframe, df, with new dataframe for the acquisition
    df.loc[df["acquisition"] == acquisition_contrast_order[0]] = df_acquisition

    ###########
    # Stage 2 #
    ###########
    # Align acquisitions to one another based on mean pixel intensity. Start with highest first because these have better contrast
    output_dir_2 = output_dir + os.sep + "stage_2"
    concat_list = [
        df_acquisition.loc[
            df_acquisition["acquisition"] == acquisition_contrast_order[0]
        ]
    ]
    print("Stage 2")
    for i in range(1, n_acquisitions):
        current_acquisitions = [
            acquisition_contrast_order[0],
            acquisition_contrast_order[i],
        ]

        target_acquisition = current_acquisitions[-1]
        idx = df["acquisition"].apply(lambda x: x in current_acquisitions)
        df_acquisition = df.loc[idx]
        df_acquisition["tier"].loc[
            df_acquisition["acquisition"] == target_acquisition
        ] = 2
        df_acquisition["tier"].loc[
            df_acquisition["acquisition"] == acquisition_contrast_order[0]
        ] = 1
        # Init dict with initial transforms
        transforms = {}
        for i in df_acquisition["sample"]:
            transforms[i] = []

        df_acquisition, transforms = alignment_stage(
            sub,
            hemisphere,
            chunk,
            df_acquisition,
            chunk_img_fn_str,
            output_dir_2,
            transforms,
            linParams,
            target_acquisition=target_acquisition,
            acquisition_n=i,
            target_tier=2,
            desc=(sub, hemisphere, chunk),
            clobber=clobber,
        )

        concat_list.append(df_acquisition.loc[df_acquisition["tier"] == 2])

        # print(target_acquisition)
        # print(df_acquisition['init_tfm'].loc[ df['acquisition'] == target_acquisition ])

        # update the master dataframe, df, with new dataframe for the acquisition
        df.loc[df["acquisition"] == target_acquisition] = df_acquisition.loc[
            df["acquisition"] == target_acquisition
        ]

    stage_2_df = pd.concat(concat_list)

    ###########
    # Stage 3 #
    ###########
    final_tfm_dir = output_dir + os.sep + "final_tfm"
    os.makedirs(final_tfm_dir, exist_ok=True)
    df = stage_2_df
    df = create_final_outputs(final_tfm_dir, df, 1)
    df = create_final_outputs(final_tfm_dir, df, -1)

    print("Writing:", init_tfm_csv)

    df.to_csv(init_tfm_csv)

    df["tier"] = 1
    if not os.path.exists(init_align_fn):
        print("\tInit Align Volume:", init_align_fn)
        combine_sections_to_vol(df, y_mm, z_mm, direction, init_align_fn)

    sect_info["init_tfm"] = df["init_tfm"]
    sect_info['init_img'] = df['init_img']

    chunk_info["init_volume"] = init_align_fn

    sect_info["2d_tfm"] = sect_info["init_tfm"]

    assert not False in [
        os.path.exists(x) for x in sect_info["init_tfm"].values if not pd.isnull(x)
    ]

    return sect_info, chunk_info


if __name__ == "__main__":
    if not os.path.exists(init_align_fn):
        initalign(
            sub,
            hemisphere,
            chunk,
            init_align_fn,
            source_dir,
            output_dir,
            receptor_df_fn,
            clobber=False,
        )
