import os

import nibabel
import numpy as np
import pandas as pd

from brainbuilder.align.align_2d import align_2d
from brainbuilder.align.align_3d import align_3d
from brainbuilder.align.intervolume import create_intermediate_volume
from brainbuilder.align.validate_alignment import validate_alignment
from brainbuilder.utils import utils
from brainbuilder.utils import validate_inputs as valinpts

global output_tuples

output_tuples = (   
        ("seg_rsl_fn", "seg_dir", "_seg.nii.gz"),
        ("rec_3d_rsl_fn","align_3d_dir","_rec_space-mri.nii.gz"),
        ("ref_3d_rsl_fn","align_3d_dir","_mri_gm_space-rec.nii.gz"),
        ("nl_3d_tfm_fn","align_3d_dir","_rec_to_mri_SyN_CC_Composite.h5"),
        ("nl_3d_tfm_inv_fn","align_3d_dir","_rec_to_mri_SyN_CC_InverseComposite.h5"),
        ("nl_2d_vol_fn","nl_2d_dir","_nl_2d.nii.gz"),
        ("nl_2d_vol_cls_fn","nl_2d_dir","_nl_2d_cls.nii.gz"),
        ("ref_space_rec_fn","nl_2d_dir","_ref_space-rec.nii.gz"),
        ("ref_iso_space_rec_fn","nl_2d_dir","_ref_space-rec_iso.nii.gz")
)

def get_multiresolution_filenames(
    row: pd.DataFrame,
    sub: str,
    hemisphere: str,
    chunk: int,
    resolution: float,
    out_dir: str,
) -> pd.DataFrame:
    """
    Set filenames for each stage of multiresolution stages

    :param row: row of dataframe
    :param sub: subject name
    :param hemisphere: hemisphere name
    :param chunk: chunk name
    :param resolution: resolution of chunk
    :param out_dir: output directory
    :return row: row of dataframe with filenames
    """

    # Set directory names for multi-resolution alignment
    cur_out_dir = f"{out_dir}/sub-{sub}/hemi-{hemisphere}/chunk-{chunk}/{resolution}mm/"
    prefix = f'sub-{sub}_hemi-{hemisphere}_chunk-{chunk}_{resolution}mm_'

    row["cur_out_dir"] = cur_out_dir
    row["seg_dir"] = "{}/3.1_intermediate_volume/".format(row["cur_out_dir"])
    row["align_3d_dir"] = "{}/3.2_align_3d/".format(row["cur_out_dir"])
    row["nl_2d_dir"] = "{}/3.3_align_2d".format(row["cur_out_dir"])

    output_list = []

    for output in output_tuples :
        out_fn = f'{row[output[1]]}/{prefix}{output[2]}'
        row[output[0]] =  out_fn
        output_list.append(out_fn)
    
    '''
    # Filenames
    row["seg_rsl_fn"] = "{}/{}_seg.nii.gz".format(
        row["seg_dir"], prefix 
    )

    row["rec_3d_rsl_fn"] = "{}/{}_rec_space-mri.nii.gz".format(
        row["align_3d_dir"], prefix 
    )
    row["ref_3d_rsl_fn"] = "{}/{}_mri_gm_space-rec.nii.gz".format(
        row["align_3d_dir"], prefix 
    )
    row[
        "nl_3d_tfm_fn"
    ] = f'{row["align_3d_dir"]}/{prefix}_rec_to_mri_SyN_CC_Composite.h5'
    row[
        "nl_3d_tfm_inv_fn"
    ] = f'{row["align_3d_dir"]}/{prefix}_rec_to_mri_SyN_CC_InverseComposite.h5'

    row["nl_2d_vol_fn"] = "{}/{}_nl_2d.nii.gz".format(
        row["nl_2d_dir"], prefix 
    )
    row["nl_2d_vol_cls_fn"] = "{}/{}_nl_2d_cls.nii.gz".format(
        row["nl_2d_dir"], prefix 
    )
    row["chunk_info_fn"] = "{}/{}_chunk_info.csv".format(
        row["cur_out_dir"], prefix
    )
    row["ref_space_rec_fn"] = "{}/{}_ref_space-rec.nii.gz".format(
        row["nl_2d_dir"], prefix 
    )
    row["ref_iso_space_rec_fn"] = "{}/{}_ref_space-rec_iso.nii.gz".format(
        row["nl_2d_dir"], prefix 
    )
    '''

    return row


def check_chunk_outputs(chunk_csv: str):
    """
    Check if chunk outputs exist, if not remove the chunk output csv and the chunk output directory
    :param chunk_output_csv: path to chunk output csv
    """

    chunk_info = pd.read_csv(chunk_csv, index_col=None)

    for output, _, _ in output_tuples:
        for fn in chunk_info[output]:
            if not os.path.exists(fn):
                print('\t\tMissing', fn)
                os.remove(chunk_csv)
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
    dice_threshold: float = 0.5,
    num_cores: int = 0,
    clobber: bool = False,
    ) -> str:
    """
    Multiresolution alignment of chunks

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

    if os.path.exists(chunk_output_csv) :
        #Check if the outputs in chunk_info exist, if not delete current <chunk_output_csv>
        check_chunk_outputs(chunk_output_csv)

    if not os.path.exists(sect_output_csv) or not os.path.exists(chunk_output_csv) or clobber or True:
        hemi_info = pd.read_csv(hemi_info_csv, index_col=None)

        sect_info = pd.read_csv(sect_info_csv, index_col=None)

        chunk_info = pd.read_csv(chunk_info_csv, index_col=None)

        sect_info["nl_2d_rsl"] = ["empty"] * sect_info.shape[0]
        sect_info["nl_2d_cls_rsl"] = ["empty"] * sect_info.shape[0]

        # create_directories(args, files, sub, hemisphere, resolution_list)

        # We create a second list of 3d resolutions that replaces values below the maximum 3D resolution with the maximum 3D resolution, because it may not be possible to perform the 3D alignment at the highest resolution due to the large memory requirements.
        resolution_list_3d = [
            float(resolution)
            for resolution in resolution_list + [max_resolution_3d]
            if float(resolution) >= max_resolution_3d
        ]

        sect_info_out = pd.DataFrame({})
        chunk_info_out = pd.DataFrame({})

        ### Reconstruct chunk for each sub, hemisphere, chunk
        groups = ["sub", "hemisphere", "chunk"]
        for (sub, hemisphere, chunk), curr_sect_info in sect_info.groupby( groups ):
            idx = (
                (chunk_info["sub"] == sub)
                & (chunk_info["hemisphere"] == hemisphere)
                & (chunk_info["chunk"] == chunk)
            )
            # get chunk_info for current chunk
            curr_chunk_info = chunk_info.loc[idx]

            # get structural reference volume
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
                    num_cores = num_cores,
                    clobber = clobber,
                    )


            chunk_info_out = pd.concat([chunk_info_out, curr_chunk_info])
            sect_info_out = pd.concat([sect_info_out, curr_sect_info])

        qc_dir = f"{output_dir}/validate_alignment/"
        maximum_resolution = resolution_list[-1]

        sect_info_out = validate_alignment(
            sect_info_out, qc_dir, maximum_resolution, clobber=clobber
        )
        #sect_info_out = sect_info_out.loc[sect_info_out["dice"] > dice_threshold]
        
        chunk_info_out.to_csv(chunk_output_csv, index=False)
        sect_info_out.to_csv(sect_output_csv, index=False)

    chunk_info = pd.read_csv(chunk_output_csv, index_col=None)
    sect_info = pd.read_csv(sect_output_csv, index_col=None)
    return chunk_output_csv, sect_output_csv


def verify_chunk_limits(ref_rsl_fn: str, chunk_info: pd.DataFrame):
    """
    Get the start and end of the chunk in the reference space
    :param ref_rsl_fn: reference space file name
    :param verbose: verbose
    :return: (y0w, y1w) --> world coordinates; (y0, y1) --> voxel coordinates
    """
    img = nibabel.load(ref_rsl_fn)

    ystart = img.affine[1, 3]
    ystep = img.affine[1, 1]

    if "caudal_limit" in chunk_info.columns and "rostral_limit" in chunk_info.columns:
        rec_points, mni_points, fn1, fn2 = read_points(manual_points_fn)
        y0w = np.min(mni_points[:, 1])
        y1w = np.max(mni_points[:, 1])

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


def align_chunk(
    chunk_info: pd.DataFrame,
    sect_info: pd.DataFrame,
    resolution_list: list,
    resolution_list_3d: list,
    ref_vol_fn: str,
    output_dir: str,
    num_cores: int = 1,
    clobber: bool = False,
)->tuple:
    """
    About:
        Mutliresolution scheme that a. segments autoradiographs, b. aligns these to the donor mri in 3D,
        and c. aligns autoradiograph sections to donor MRI in 2d. This schema is repeated for each
        resolution in the resolution heiarchy.

    Inputs:
        sect_info:    data frame with info for autoradiographs in current chunk
        files:       dictionary with all the filenames used in reconstruction
        resolution_list:    heirarchy of resolutions for 2d alignment
        resolution_list_3d: heirarchy of resolutions for 3d alignment
        output_dir:         output directory

    Outputs:
        sect_info: updated sect_info data frame with filenames for nonlinearly 2d aligned autoradiographs

    """
    sub, hemisphere, chunk = utils.get_values_from_df(chunk_info)

    chunk_info_out = pd.DataFrame()
    print("\tMultiresolution:", sub, hemisphere, chunk)

    ### Iterate over progressively finer resolution
    for resolution_itr, resolution in enumerate(resolution_list):
        resolution_3d = resolution_list_3d[resolution_itr]
        print(f"\tMulti-Resolution Alignement: {resolution}")

        row = chunk_info.iloc[0, :].squeeze()
        row["resolution"] = resolution

        row = get_multiresolution_filenames(
            row, sub, hemisphere, chunk, resolution, output_dir
        )

        dirs_to_create = [ row["cur_out_dir"], row["align_3d_dir"], row["seg_dir"], row["nl_2d_dir"]  ]
        for dir_name in dirs_to_create:
            os.makedirs(dir_name, exist_ok=True)

        # downsample the original ref gm mask to current 3d resolution
        ref_rsl_fn = (
            row["seg_dir"] + f"/sub-{sub}_hemi-{hemisphere}_chunk-{chunk}_{resolution}mm_mri_gm_.nii.gz"
        )

        if not os.path.exists(ref_rsl_fn):
            utils.resample_to_resolution( ref_vol_fn, [ resolution_3d ] * 3, ref_rsl_fn )

        world_chunk_limits, vox_chunk_limits = verify_chunk_limits(
            ref_rsl_fn, chunk_info
        )

        ###
        ### Stage 3.1 : Create intermediate 3d volume
        ###
        print('\t\tCreate intermediate 3d volume')
        [row["init_volume"]]
        create_intermediate_volume(
            chunk_info,
            sect_info,
            resolution_itr,
            resolution,
            resolution_3d,
            row["seg_dir"],
            row["seg_rsl_fn"],
            row["init_volume"],
            num_cores = num_cores,
            clobber = clobber,
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
            base_nl_itr = 2, #FIXME
            base_lin_itr = 5, #FIXME
            use_masks=False,
            clobber=clobber
        )

        ###
        ### Stage 3.3 : 2D alignment of receptor to resample MRI GM vol
        ###
        print("\t\t2D alignment of receptor to resample MRI GM vol")
        sect_info = align_2d(
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
            file_to_align = "seg",
            base_nl_itr = 2, #FIXME
            base_lin_itr = 5, #FIXME
            num_cores = num_cores,
            clobber = clobber
        )
        chunk_info_out = pd.concat([chunk_info_out, row.to_frame().T ])
    return chunk_info_out, sect_info
