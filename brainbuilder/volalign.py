"""This module contains functions for multiresolution alignment of autoradiographs to MRI."""
import os

import matplotlib.pyplot as plt
import nibabel
import numpy as np
import pandas as pd
import seaborn as sns
from skimage.filters import threshold_otsu

import brainbuilder.utils.ants_nibabel as nib
from brainbuilder.align.align_2d import align_2d
from brainbuilder.align.align_3d import align_3d
from brainbuilder.align.intervolume import create_intermediate_volume
from brainbuilder.utils import utils
from brainbuilder.utils import validate_inputs as valinpts

global output_tuples

output_tuples = [
    ["seg_rsl_fn", "seg_dir", "_seg.nii.gz"],
    ["rec_3d_rsl_fn", "align_3d_dir", "_rec_space-mri.nii.gz"],
    ["ref_3d_rsl_fn", "align_3d_dir", "_mri_gm_space-rec.nii.gz"],
    ["nl_3d_tfm_fn", "align_3d_dir", "_rec_to_mri_SyN_CC_Composite.h5"],
    ["nl_3d_tfm_inv_fn", "align_3d_dir", "_rec_to_mri_SyN_CC_InverseComposite.h5"],
    ["nl_2d_vol_fn", "nl_2d_dir", "_nl_2d.nii.gz"],
    ["nl_2d_vol_cls_fn", "nl_2d_dir", "_nl_2d_cls.nii.gz"],  # ,
]


def get_multiresolution_filenames(
    row: pd.DataFrame,
    sub: str,
    hemisphere: str,
    chunk: int,
    resolution: float,
    pass_step: int,
    out_dir: str,
    use_3d_syn_cc: bool = True,
) -> pd.DataFrame:
    """Set filenames for each stage of multiresolution stages.

    :param row: row of dataframe
    :param sub: subject name
    :param hemisphere: hemisphere name
    :param chunk: chunk name
    :param resolution: resolution of chunk
    :param pass_step: pass step within current iteration
    :param out_dir: output directory
    :return row: row of dataframe with filenames
    """
    # Set directory names for multi-resolution alignment
    cur_out_dir = f"{out_dir}/sub-{sub}/hemi-{hemisphere}/chunk-{chunk}/{resolution}mm/pass_{pass_step}/"
    prefix = f"sub-{sub}_hemi-{hemisphere}_chunk-{chunk}_{resolution}mm"

    row["cur_out_dir"] = cur_out_dir
    row["seg_dir"] = "{}/3.1_intermediate_volume/".format(row["cur_out_dir"])
    row["align_3d_dir"] = "{}/3.2_align_3d/".format(row["cur_out_dir"])
    row["nl_2d_dir"] = "{}/3.3_align_2d".format(row["cur_out_dir"])

    output_list = []

    for output in output_tuples:
        if "CC" in output[2]:
            if not use_3d_syn_cc:
                output[2] = output[2].replace("CC", "Mattes")

        out_fn = f"{row[output[1]]}/{prefix}{output[2]}"
        row[output[0]] = out_fn
        output_list.append(out_fn)

    return row


def check_chunk_outputs(chunk_csv: str) -> None:
    """Check if chunk outputs exist, if not remove the chunk output csv and the chunk output directory.

    :param chunk_output_csv: path to chunk output csv
    :return: None
    """
    chunk_info = pd.read_csv(chunk_csv, index_col=None)

    for output, _, _ in output_tuples:
        for fn in chunk_info[output]:
            if not os.path.exists(fn):
                print("\t\tMissing", fn)
                os.remove(chunk_csv)
                return None
    return None


def multiresolution_alignment(
    hemi_info_csv: pd.DataFrame,
    chunk_info_csv: pd.DataFrame,
    sect_info_csv: pd.DataFrame,
    resolution_list: list,
    output_dir: str,
    sect_output_csv: str = "",
    chunk_output_csv: str = "",
    max_resolution_3d: float = 0.3,
    use_3d_syn_cc: bool = True,
    use_syn: bool = True,
    linear_steps: list = ["rigid", "similarity", "affine"],
    num_cores: int = 0,
    clobber: bool = False,
) -> str:
    """Multiresolution alignment of chunks.

    params: df: dataframe containing chunk information
    params: sub: subject name
    params: hemisphere: hemisphere name
    params: resolution_list: list of resolutions to align
    params: max_resolution_3d: maximum resolution to align in 3d
    returns: csv file containing chunk information
    """
    num_cores = utils.set_cores(num_cores)

    # Validate Inputs for <align_chunk>
    multi_resolution_required_columns = valinpts.chunk_info_required_columns + [
        valinpts.Column("init_volume", "volume")
    ]

    assert valinpts.validate_csv(hemi_info_csv, valinpts.hemi_info_required_columns)
    assert valinpts.validate_csv(sect_info_csv, valinpts.sect_info_required_columns)
    assert valinpts.validate_csv(chunk_info_csv, multi_resolution_required_columns)

    if sect_output_csv == "":
        sect_output_csv = f"{output_dir}/sect_info_multiresolution_alignment.csv"

    if chunk_output_csv == "":
        chunk_output_csv = f"{output_dir}/chunk_info_multiresolution_alignment.csv"

    if os.path.exists(chunk_output_csv):
        # Check if the outputs in chunk_info exist, if not delete current <chunk_output_csv>
        check_chunk_outputs(chunk_output_csv)

    qc_dir = f"{output_dir}/qc/"
    os.makedirs(qc_dir, exist_ok=True)

    if (
        not os.path.exists(sect_output_csv)
        or not os.path.exists(chunk_output_csv)
        or clobber
    ):
        hemi_info = pd.read_csv(hemi_info_csv, index_col=None)

        sect_info = pd.read_csv(sect_info_csv, index_col=None)

        chunk_info = pd.read_csv(chunk_info_csv, index_col=None)

        sect_info["nl_2d_rsl"] = ["empty"] * sect_info.shape[0]
        sect_info["nl_2d_cls_rsl"] = ["empty"] * sect_info.shape[0]

        # We create a second list of 3d resolutions that replaces values below the maximum 3D resolution with the maximum 3D resolution, because it may not be possible to perform the 3D alignment at the highest resolution due to the large memory requirements.
        resolution_list_3d = [
            float(resolution)
            if float(resolution) >= max_resolution_3d
            else max_resolution_3d
            for resolution in resolution_list
        ]

        sect_info_out = pd.DataFrame({})
        chunk_info_out = pd.DataFrame({})

        ### Reconstruct chunk for each sub, hemisphere, chunk
        groups = ["sub", "hemisphere", "chunk"]
        for (sub, hemisphere, chunk), curr_sect_info in sect_info.groupby(groups):
            idx = (
                (chunk_info["sub"] == sub)
                & (chunk_info["hemisphere"] == hemisphere)
                & (chunk_info["chunk"] == chunk)
            )

            # get chunk_info for current chunk
            curr_chunk_info = chunk_info.loc[idx]

            # get structural reference volume
            print(hemi_info.columns)
            print(sub)
            print(hemisphere)
            print(hemi_info["struct_ref_vol"])
            ref_vol_fn = (
                hemi_info["struct_ref_vol"]
                .loc[
                    (hemi_info["sub"] == sub) & (hemi_info["hemisphere"] == hemisphere)
                ]
                .values[0]
            )
            curr_chunk_info, curr_sect_info = align_chunk(
                curr_chunk_info,
                curr_sect_info,
                resolution_list,
                resolution_list_3d,
                ref_vol_fn,
                output_dir,
                num_cores=num_cores,
                use_3d_syn_cc=use_3d_syn_cc,
                use_syn=use_syn,
                linear_steps=linear_steps,
                clobber=clobber,
            )

            chunk_info_out = pd.concat([chunk_info_out, curr_chunk_info])
            sect_info_out = pd.concat([sect_info_out, curr_sect_info])

        chunk_info_out.to_csv(chunk_output_csv, index=False)

        sect_info_out.to_csv(sect_output_csv, index=False)

    # Calculate Dice between sections
    # DEBUG
    # inter_section_dice(sect_output_csv, qc_dir, clobber=clobber)
    # Calculate Dice between sections and reference volume
    # validate_section_alignment_to_ref(sect_output_csv, qc_dir, clobber=clobber)

    return chunk_output_csv, sect_output_csv


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


def alignment_qc(
    sect_output_csv: str, output_dir: str, cutoff: float = 0.7, clobber: bool = False
) -> None:
    """Create a scatter plot of dice values for each section and save the plot to <output_dir>/alignment_dice.png.

    :param sect_output_csv: path to section output csv
    :param output_dir: output directory
    :param cutoff: cutoff for bad sections
    :param clobber: clobber
    :return: None
    """
    png_fn = f"{output_dir}/alignment_dice.png"
    global_dice_csv = f"{output_dir}/global_dice.csv"

    if (
        not os.path.exists(sect_output_csv)
        or not os.path.exists(png_fn)
        or not os.path.exists(global_dice_csv)
        or not os.path.exists(png_fn)
        or utils.newer_than(sect_output_csv, png_fn)
        or utils.newer_than(sect_output_csv, global_dice_csv)
        or clobber
    ):
        df = pd.read_csv(sect_output_csv, index_col=None)

        def normalize(x: float) -> float:
            """Normalize dice values to 0-100.

            :param x: dice values
            :return: normalized dice values
            """
            return (x - x.min()) / (x.max() - x.min()) * 100

        normalized_sample = df.groupby("chunk")["sample"].transform(normalize)
        df["Coronal Section %"] = normalized_sample
        print("Coronal Section")
        df["Dice"] = df["dice"]
        df["Slab"] = df["chunk"]

        plt.clf()
        plt.close()
        plt.figure(figsize=(10, 10))
        sns.scatterplot(
            x="Coronal Section %",
            y="Dice",
            data=df,
            hue="Slab",
            palette="Set1",
            alpha=0.4,
        )
        sns.despine()
        plt.savefig(png_fn, dpi=300, bbox_inches="tight")

        dice_df = df.groupby(["sub", "hemisphere", "chunk"])
        print(dice_df["dice"].mean())
        print(dice_df["dice"].std())

        m = df["dice"].mean()
        s = df["dice"].std()

        print("\t\tDice of Aligned Sections:", m, s)
        dice_df["dice"].mean().to_csv(global_dice_csv)
        with open(f"{output_dir}/global_dice.csv", "w") as f:
            f.write(f"{m},{s}\n")

        bad_sections_df = df.loc[df["dice"] < cutoff]
        os.makedirs(f"{output_dir}/bad_sections/", exist_ok=True)

        for i, row in bad_sections_df.iterrows():
            mv = row["nl_2d_cls_rsl"]
            out_fn = f"{output_dir}/bad_sections/{os.path.basename(mv)}.jpg"
            ar = nibabel.load(mv).get_fdata()
            plt.clf()
            plt.close()
            plt.title("Dice: {:.2f}".format(row["dice"]))
            plt.imshow(ar, cmap="gray")
            plt.savefig(out_fn, dpi=300, bbox_inches="tight")
            print("\t\tBad section:", out_fn)
    return None


def calculate_distance_vol(seg_fn, out_fn):
    if not os.path.exists(out_fn):
        seg_img = nib.load(seg_fn)
        seg_data = seg_img.get_fdata()

        t = threshold_otsu(seg_data)  # Use Otsu's method for thresholding

        seg_data[seg_data <= t] = 0
        seg_data[seg_data > t] = 1  # Ensure binary segmentation

        # Find min and max GM indices for each dimension
        min_x, max_x = (
            np.min(np.where(seg_data > 0)[0]),
            np.max(np.where(seg_data > 0)[0]),
        )
        min_y, max_y = (
            np.min(np.where(seg_data > 0)[1]),
            np.max(np.where(seg_data > 0)[1]),
        )
        min_z, max_z = (
            np.min(np.where(seg_data > 0)[2]),
            np.max(np.where(seg_data > 0)[2]),
        )

        # Calculate relative distances
        x_range = np.linspace(1, 2, max_x - min_x + 1)
        y_range = np.linspace(1, 2, max_y - min_y + 1)
        z_range = np.linspace(1, 2, max_z - min_z + 1)

        seg_data[min_x : max_x + 1, :, :] *= x_range[:, None, None]  # Scale x dimension
        seg_data[:, min_y : max_y + 1, :] *= y_range[None, :, None]  # Scale y dimension
        seg_data[:, :, min_z : max_z + 1] *= z_range[None, None, :]  # Scale z dimension

        # Save the distance map
        nib.Nifti1Image(seg_data, seg_img.affine, direction_order="lpi").to_filename(
            out_fn
        )


def alignment_iteration(
    sub: str,
    hemisphere: str,
    chunk: int,
    output_dir: str,
    sect_info: pd.DataFrame,
    chunk_info: pd.DataFrame,
    ref_vol_fn: str,
    resolution_itr: int,
    resolution_list: list,
    resolution: float,
    resolution_3d: float,
    resolution_list_3d: list,
    use_syn: bool = False,
    pass_step: int = 0,
    num_cores: int = 1,
    use_3d_syn_cc: bool = True,
    linear_steps: list = ["rigid", "similarity", "affine"],
    clobber: bool = False,
) -> tuple:
    """Perform 3D-2D alignment for each resolution.

    1) Create intermediate GM volume using best availble 2D transforms
    2) Align GM volume to MRI in 3D
    3) Align autoradiographs to GM volume in 2D.
    """
    print("\t\tCreate intermediate 3d volume")
    row = chunk_info.iloc[0, :].squeeze()
    row["resolution"] = resolution

    chunk_info_out = pd.DataFrame()
    sect_info_out = pd.DataFrame()

    row = get_multiresolution_filenames(
        row,
        sub,
        hemisphere,
        chunk,
        resolution,
        pass_step,
        output_dir,
        use_3d_syn_cc=use_3d_syn_cc,
    )

    dirs_to_create = [
        row["cur_out_dir"],
        row["align_3d_dir"],
        row["seg_dir"],
        row["nl_2d_dir"],
    ]
    for dir_name in dirs_to_create:
        os.makedirs(dir_name, exist_ok=True)

    # downsample the original ref gm mask to current 3d resolution
    ref_rsl_fn = (
        row["seg_dir"]
        + f"/sub-{sub}_hemi-{hemisphere}_chunk-{chunk}_{resolution}mm_mri_gm.nii.gz"
    )

    ref_dist_fn = (
        row["align_3d_dir"]
        + f"/sub-{sub}_hemi-{hemisphere}_chunk-{chunk}_{resolution}mm_mri_gm_dist.nii.gz"
    )

    if not os.path.exists(ref_rsl_fn):
        utils.resample_to_resolution(ref_vol_fn, [resolution_3d] * 3, ref_rsl_fn)

    # insert 2d intersection alignment
    use_2d_intersection = False
    if use_2d_intersection:  # Experimental feature, not yet tested
        from brainbuilder.align.align_2d_intersection import align_2d_intersection

        sect_info = align_2d_intersection(
            sect_info,
            ref_vol_fn,
            resolution_list,
            output_dir + "/intersection/",
            n_iters=20,
            clobber=clobber,
        )

    section_thickness = chunk_info["section_thickness"]

    world_chunk_limits, vox_chunk_limits = verify_chunk_limits(ref_rsl_fn, chunk_info)

    create_intermediate_volume(
        chunk_info,
        sect_info,
        resolution_itr,
        resolution,
        resolution_list,
        resolution_3d,
        row["seg_dir"],
        row["seg_rsl_fn"],
        row["init_volume"],
        num_cores=num_cores,
        interpolation="linear",
        clobber=clobber,
    )

    ###
    ### Stage 3.2 : Align chunks to MRI
    ###
    print("\t\tAlign chunks to MRI")
    align_3d(
        sub,
        hemisphere,
        chunk,
        row["seg_rsl_fn"],
        ref_rsl_fn,
        row["align_3d_dir"],
        row["nl_3d_tfm_fn"],
        row["nl_3d_tfm_inv_fn"],
        row["rec_3d_rsl_fn"],
        row["ref_3d_rsl_fn"],
        resolution_3d,
        resolution_list_3d,
        world_chunk_limits,
        vox_chunk_limits,
        use_3d_syn_cc=use_3d_syn_cc,
        linear_steps=linear_steps,
        clobber=clobber,
    )

    ###
    ### Stage 3.3 : 2D alignment of receptor to resample MRI GM vol
    ###
    print("\t\t2D alignment of receptor to resample MRI GM vol")
    sect_info, ref_space_nat_fn = align_2d(
        sect_info,
        row["nl_2d_dir"],
        row["seg_dir"],
        ref_rsl_fn,
        resolution,
        resolution_itr,
        resolution_list,
        row["seg_rsl_fn"],
        row["nl_3d_tfm_inv_fn"],
        row["nl_2d_vol_fn"],
        row["nl_2d_vol_cls_fn"],
        row["section_thickness"],
        file_to_align="seg",
        use_syn=use_syn,
        num_cores=num_cores,
        clobber=clobber,
    )
    row["ref_space_nat"] = ref_space_nat_fn
    chunk_info_out = pd.concat([chunk_info_out, row.to_frame().T])
    sect_info_out = pd.concat([sect_info_out, sect_info])

    return chunk_info_out, sect_info_out


def align_chunk(
    chunk_info: pd.DataFrame,
    sect_info: pd.DataFrame,
    resolution_list: list,
    resolution_list_3d: list,
    ref_vol_fn: str,
    output_dir: str,
    n_passes: int = 1,
    num_cores: int = 1,
    use_3d_syn_cc: bool = True,
    linear_steps: list = ["rigid", "similarity", "affine"],
    use_syn: bool = True,
    clobber: bool = False,
) -> tuple:
    """3D-2D mutliresolution scheme.

      Steps:
      a) segments autoradiographs,
      b) aligns these to the donor mri in 3D, and then 2d.
      c) This schema is repeated for each resolution in the resolution heiarchy.

    :param chunk_info:  data frame with info for current chunk
    :param sect_info:    data frame with info for autoradiographs in current chunk
    :param files:       dictionary with all the filenames used in reconstruction
    :param resolution_list:    heirarchy of resolutions for 2d alignment
    :param resolution_list_3d: heirarchy of resolutions for 3d alignment
    :param output_dir:         output directory
    :param n_passes:           number of passes for 3d-2d alignment
    :param num_cores:          number of cores to use
    :return sect_info: updated sect_info data frame with filenames for nonlinearly 2d aligned autoradiographs
    """
    sub, hemisphere, chunk = utils.get_values_from_df(chunk_info)

    print("\tMultiresolution:", sub, hemisphere, chunk)

    qc_dir = f"{output_dir}/qc/"
    os.makedirs(qc_dir, exist_ok=True)

    chunk_pass_df = pd.DataFrame()
    sect_pass_df = pd.DataFrame()
    chunk_pass_last_df = pd.DataFrame()
    sect_pass_last_df = pd.DataFrame()

    sect_info_out = sect_info
    chunk_info_out = chunk_info

    dice_df = pd.DataFrame()

    ### Iterate over progressively finer resolution
    for resolution_itr, resolution in enumerate(resolution_list):
        resolution_3d = resolution_list_3d[resolution_itr]
        print(
            f"\tMulti-Resolution Alignement: {resolution}mm, 3D max = {resolution_3d}"
        )

        for pass_step in range(n_passes):
            # Perform alignment for each resolution, including:
            # 1) Create intermediate GM volume using best availble 2D transforms
            # 2) Align GM volume to MRI in 3D
            # 3) Align autoradiographs to GM volume in 2D
            chunk_info_out, sect_info_out = alignment_iteration(
                sub,
                hemisphere,
                chunk,
                output_dir,
                sect_info_out,
                chunk_info_out,
                ref_vol_fn,
                resolution_itr,
                resolution_list,
                resolution,
                resolution_3d,
                resolution_list_3d,
                pass_step=pass_step,
                use_3d_syn_cc=use_3d_syn_cc,
                use_syn=use_syn,
                linear_steps=linear_steps,
                num_cores=num_cores,
                clobber=clobber,
            )
            chunk_info_out["pass"] = pass_step
            sect_info_out["pass"] = pass_step

            chunk_pass_df = pd.concat([chunk_pass_df, chunk_info_out])
            sect_pass_df = pd.concat([sect_pass_df, sect_info_out])

            if pass_step == n_passes - 1 and resolution == resolution_list[-1]:
                chunk_pass_last_df = pd.concat([chunk_pass_last_df, chunk_info_out])
                sect_pass_last_df = pd.concat([sect_pass_last_df, sect_info_out])

            output_prefix = f"sub-{sub}_hemi-{hemisphere}_chunk-{chunk}_{resolution}mm_pass-{pass_step}_"

    return chunk_pass_last_df, sect_pass_last_df
