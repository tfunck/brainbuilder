"""This module contains functions for multiresolution alignment of autoradiographs to MRI."""
import os
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel
import numpy as np
import pandas as pd
import seaborn as sns
from skimage.filters import threshold_otsu

import brainbuilder.utils.ants_nibabel as nib
from brainbuilder.align.align_2d import align_2d
from brainbuilder.align.align_3d import align_3d, pad_acq_volume, write_ref_chunk
from brainbuilder.align.align_landmarks import create_landmark_transform
from brainbuilder.align.intervolume import create_acquisition_volume
from brainbuilder.utils import utils
from brainbuilder.utils import validate_inputs as valinpts
from brainbuilder.utils.paths import MultiResPaths, _multires_root_dir

logger = utils.get_logger(__name__)

def check_chunk_outputs(chunk_csv: str) -> None:
    """Check if chunk outputs exist, if not remove the chunk output csv and the chunk output directory.

    :param chunk_output_csv: path to chunk output csv
    :return: None
    """
    chunk_info = pd.read_csv(chunk_csv, index_col=None)

    for col in [
        "acq_rsl_fn",
        "ref_3d_rsl_fn",
        "nl_2d_vol_fn",
        "nl_2d_vol_cls_fn",
        "nl_3d_tfm_fn",
        "nl_3d_tfm_inv_fn",
    ]:
        for fn in chunk_info[col].values:
            if not os.path.exists(fn):
                logger.warning(
                    "Warning: %s does not exist. Deleting %s.", fn, chunk_csv
                )
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
    landmark_dir: Path = None,
    interpolation: str = "Linear",
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

    # Validate Inputs
    multi_resolution_required_columns = valinpts.chunk_info_required_columns + [
        valinpts.Column("init_volume", "volume")
    ]

    assert valinpts.validate_csv(hemi_info_csv, valinpts.hemi_info_required_columns)
    assert valinpts.validate_csv(sect_info_csv, valinpts.sect_info_required_columns)
    assert valinpts.validate_csv(chunk_info_csv, multi_resolution_required_columns)

    align_output_dir = _multires_root_dir(output_dir)

    if sect_output_csv == "":
        sect_output_csv = f"{align_output_dir}/sect_info_multiresolution_alignment.csv"

    if chunk_output_csv == "":
        chunk_output_csv = (
            f"{align_output_dir}/chunk_info_multiresolution_alignment.csv"
        )

    qc_dir = f"{align_output_dir}/qc/"
    os.makedirs(qc_dir, exist_ok=True)

    if (
        not os.path.exists(sect_output_csv)
        or not os.path.exists(chunk_output_csv)
        or clobber
    ):
        hemi_info = pd.read_csv(hemi_info_csv, index_col=None)

        sect_info = pd.read_csv(sect_info_csv, index_col=None)

        chunk_info = pd.read_csv(chunk_info_csv, index_col=None)

        # We create a second list of 3d resolutions that replaces values below the maximum 3D resolution with the maximum 3D resolution,
        # because it may not be possible to perform the 3D alignment at the highest resolution due to the large memory requirements.
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
            chunk_info_row = chunk_info.loc[idx]

            # Check that curr_chunk_info has only one row
            assert (
                len(chunk_info_row) == 1
            ), f"Error: chunk_info has multiple rows for sub-{sub}_hemi-{hemisphere}_chunk-{chunk}"

            # get structural reference volume
            ref_vol_fn = (
                hemi_info["struct_ref_vol"]
                .loc[
                    (hemi_info["sub"] == sub) & (hemi_info["hemisphere"] == hemisphere)
                ]
                .values[0]
            )

            ref_vol_fn = utils.check_volume_orientation(
                ref_vol_fn, expected_direction_str="LPI", output_dir=align_output_dir
            )

            chunk_info_row, curr_sect_info = align_chunk(
                chunk_info_row,
                curr_sect_info,
                resolution_list,
                resolution_list_3d,
                max_resolution_3d,
                ref_vol_fn,
                output_dir,
                num_cores=num_cores,
                use_3d_syn_cc=use_3d_syn_cc,
                use_syn=use_syn,
                linear_steps=linear_steps,
                interpolation=interpolation,
                landmark_dir=landmark_dir,
                clobber=clobber,
            )

            chunk_info_out = pd.concat([chunk_info_out, chunk_info_row])
            sect_info_out = pd.concat([sect_info_out, curr_sect_info])

        chunk_info_out.to_csv(chunk_output_csv, index=False)

        sect_info_out.to_csv(sect_output_csv, index=False)

    # Calculate Dice between sections
    # DEBUG
    # inter_section_dice(sect_output_csv, qc_dir, clobber=clobber)
    # Calculate Dice between sections and reference volume
    # validate_section_alignment_to_ref(sect_output_csv, qc_dir, clobber=clobber)

    return chunk_output_csv, sect_output_csv


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
    paths: MultiResPaths,
    sect_info: pd.DataFrame,
    chunk_info: pd.DataFrame,
    ref_vol_fn: str,
    resolution_itr: int,
    resolution_list: list,
    resolution: float,
    resolution_3d: float,
    max_resolution_3d: float,
    resolution_list_3d: list,
    use_syn: bool = False,
    num_cores: int = 1,
    use_3d_syn_cc: bool = True,
    linear_steps: list = ["rigid", "similarity", "affine"],
    interpolation: str = "Linear",
    padding_offset: float = 0.15,  # offset for % by which we pad the segmentation volume at the start and end of each direction
    landmark_dir: str = None,
    section_thickness: float = 0.02,
    skip_2d_alignment: bool = False,
    clobber: bool = False,
) -> tuple:
    """Perform 3D-2D alignment for each resolution.

    1) Create intermediate GM volume using best availble 2D transforms
    2) Align GM volume to MRI in 3D
    3) Align sections to GM volume in 2D.
    """
    logger.info("\t\tCreate intermediate 3d volume")

    chunk_info_out = pd.DataFrame()

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
            interpolation=interpolation,
            clobber=clobber,
        )

    sect_info = create_acquisition_volume(
        chunk_info,
        sect_info,
        resolution_itr,
        resolution,
        resolution_list,
        resolution_3d,
        paths.intermediate_volume_dir,
        paths.acq_rsl_fn,
        paths.init_volume,
        num_cores = num_cores,
        clobber=clobber,
    )

    # pad the segmented volume so that it can be downsampled by the
    # ammount of times specified by max_downsample_level
    pad_acq_volume(
        paths.acq_rsl_fn, paths.acq_pad_fn, resolution, padding_offset=padding_offset
    )

    _, landmark_composite_tfm_path = write_ref_chunk(
        sect_info,
        chunk_info,
        sub,
        hemisphere,
        chunk,
        ref_vol_fn,
        paths.ref_rsl_fn,
        landmark_dir,
        paths.align_3d_dir,
        paths.landmark_dir,
        resolution_3d,
        max_resolution_3d,
        paths.moving_landmark_volume,
        paths.fixed_landmark_volume,
        paths.moving_volume,
        paths.fixed_volume,
        paths.acq_landmark_volume,
        paths.ref_landmark_volume,
        paths.acq_rsl_fn,
        padding_offset=padding_offset,
        clobber=clobber,
    )

    ###
    ### Stage 3.2 : Align chunks to MRI
    ###
    logger.info("\t\tAlign chunks to MRI")
    vol_tfm_list = align_3d(
        chunk,
        paths.moving_volume,
        paths.fixed_volume,
        paths.align_3d_dir,
        paths.nl_3d_tfm_fn,
        paths.ref_3d_rsl_fn,
        resolution_3d,
        resolution_list_3d,
        use_3d_syn_cc=use_3d_syn_cc,
        linear_steps=linear_steps,
        init_tfm=landmark_composite_tfm_path,
        clobber=clobber,
    )

    ###
    ### Stage 3.3 : 2D alignment of receptor to resample MRI GM vol
    ###
    if not skip_2d_alignment:
        logger.info("\t\t2D alignment of receptor to resample MRI GM vol")
        sect_info, _ = align_2d(
            sect_info,
            paths.align_2d_dir,
            paths.ref_rsl_fn,
            resolution,
            resolution_list,
            paths.acq_rsl_fn,
            vol_tfm_list,
            paths.nl_2d_vol_fn,
            paths.nl_2d_vol_cls_fn,
            section_thickness,
            file_to_align="seg_rsl",
            use_syn=use_syn,
            num_cores=num_cores,
            clobber=clobber,
        )

    chunk_info_out = paths.to_dataframe()

    # Keep `nl_3d_tfm_fn` as the composite transform filename (string) for
    # downstream pipeline steps, but store the full transform chain (e.g.
    # [composite, init]) separately.
    chunk_info_out["nl_3d_tfm_list"] = [vol_tfm_list]

    align_dir = _multires_root_dir(output_dir)

    sect_info_curr_resolution_csv = f"{align_dir}/sub-{sub}_hemi-{hemisphere}_chunk-{chunk}_{resolution}mm_sect_info.csv"

    sect_info.to_csv(sect_info_curr_resolution_csv, index=False)

    chunk_info_curr_resolution_csv = f"{align_dir}/sub-{sub}_hemi-{hemisphere}_chunk-{chunk}_{resolution}mm_chunk_info.csv"

    chunk_info_out.to_csv(chunk_info_curr_resolution_csv, index=False)

    return chunk_info_out, sect_info


def align_chunk(
    chunk_info_row: pd.DataFrame,
    sect_info: pd.DataFrame,
    resolution_list: list,
    resolution_list_3d: list,
    max_resolution_3d: float,
    ref_vol_fn: str,
    output_dir: str,
    n_passes: int = 1,
    num_cores: int = 1,
    use_3d_syn_cc: bool = True,
    linear_steps: list = ["rigid", "similarity", "affine"],
    use_syn: bool = True,
    interpolation: str = "Linear",
    landmark_dir: str = None,
    clobber: bool = False,
) -> tuple:
    """3D-2D mutliresolution scheme.

      Steps:
      a) segment sections,
      b) aligns these to the donor mri in 3D, and then 2d.
      c) This schema is repeated for each resolution in the resolution heiarchy.

    :param chunk_info:  data frame with info for current chunk
    :param sect_info:    data frame with info for sections in current chunk
    :param files:       dictionary with all the filenames used in reconstruction
    :param resolution_list:    heirarchy of resolutions for 2d alignment
    :param resolution_list_3d: heirarchy of resolutions for 3d alignment
    :param output_dir:         output directory
    :param n_passes:           number of passes for 3d-2d alignment
    :param num_cores:          number of cores to use
    :return sect_info: updated sect_info data frame with filenames for nonlinearly 2d aligned sections
    """
    sub, hemisphere, chunk = utils.get_values_from_df(chunk_info_row)

    logger.info("\tAlignment to reference: %s %s %s", sub, hemisphere, chunk)

    qc_dir = f"{output_dir}/qc/"
    os.makedirs(qc_dir, exist_ok=True)

    sect_info_out = sect_info
    chunk_info_out = chunk_info_row

    ref_landmark_volume = (
        chunk_info_row["ref_landmark"].values[0]
        if "ref_landmark" in chunk_info_row.columns
        else ""
    )

    ### Iterate over progressively finer resolution
    for resolution_itr, resolution in enumerate(resolution_list):
        resolution_3d = resolution_list_3d[resolution_itr]
        logger.info(
            "\tMulti-Resolution Alignement: %smm, 3D max = %s",
            resolution,
            resolution_3d,
        )

        for pass_step in range(n_passes):
            # Perform alignment for each resolution, including:
            # 1) Create intermediate GM volume using best availble 2D transforms
            # 2) Align GM volume to MRI in 3D
            # 3) Align sections to GM volume in 2D

            section_thickness = chunk_info_row["section_thickness"].values[0]

            paths = MultiResPaths(
                sub=sub,
                hemisphere=hemisphere,
                chunk=chunk,
                resolution=resolution,
                resolution_3d=resolution_3d,
                pass_step=pass_step,
                output_dir=Path(output_dir),
                moving="ref_rsl_fn",
                fixed="acq_pad_fn",
                ref_landmark_volume=ref_landmark_volume,
                section_thickness=section_thickness,
                moving_landmark_volume=ref_landmark_volume,
                fixed_landmark="acq_landmark_volume",
                use_3d_syn_cc=use_3d_syn_cc,
            )

            chunk_info_out, sect_info_out = alignment_iteration(
                sub,
                hemisphere,
                chunk,
                output_dir,
                paths,
                sect_info_out,
                chunk_info_out,
                ref_vol_fn,
                resolution_itr,
                resolution_list,
                resolution,
                resolution_3d,
                max_resolution_3d,
                resolution_list_3d,
                use_3d_syn_cc=use_3d_syn_cc,
                use_syn=use_syn,
                linear_steps=linear_steps,
                section_thickness=section_thickness,
                num_cores=num_cores,
                interpolation=interpolation,
                landmark_dir=landmark_dir,
                clobber=clobber,
            )

    # un a final alignment
    chunk_info_out_prev = chunk_info_out.copy()

    paths = MultiResPaths(
        sub=sub,
        hemisphere=hemisphere,
        chunk=chunk,
        resolution=resolution,
        resolution_3d=resolution_3d,
        pass_step=pass_step,
        output_dir=Path(output_dir),
        stage_tag="_final",
        moving="acq_pad_fn",
        fixed="ref_rsl_fn",
        ref_landmark_volume=ref_landmark_volume,
        fixed_landmark_volume=ref_landmark_volume,
        moving_landmark="acq_landmark_volume",
        use_3d_syn_cc=use_3d_syn_cc,
    )

    logger.info(
        f"Running final alignment iteration at highest (3D) resolution: {resolution_3d}mm"
    )
    chunk_info_out, sect_info_out = alignment_iteration(
        sub,
        hemisphere,
        chunk,
        output_dir,
        paths,
        sect_info_out,
        chunk_info_out,
        ref_vol_fn,
        resolution_itr,
        resolution_list,
        resolution,
        resolution_3d,
        max_resolution_3d,
        resolution_list_3d,
        use_3d_syn_cc=use_3d_syn_cc,
        use_syn=use_syn,
        skip_2d_alignment=True,
        linear_steps=linear_steps,
        num_cores=num_cores,
        interpolation=interpolation,
        landmark_dir=landmark_dir,
        clobber=clobber,
    )

    COLS_2D = ["nl_2d_dir", "nl_2d_vol_fn", "nl_2d_vol_cls_fn"]
    for col in COLS_2D:
        if col in chunk_info_out_prev.columns:
            chunk_info_out[col] = chunk_info_out_prev[col].values
    chunk_info_out["pass"] = pass_step
    logger.info("Finished final alignment iteration at highest resolution")

    return chunk_info_out, sect_info_out
