import argparse
import os

import pandas as pd

from brainbuilder.initalign import initalign
from brainbuilder.segment import segment
from brainbuilder.utils import utils, validate_inputs
from brainbuilder.volalign import multiresolution_alignment
from brainbuilder.interpsections import interpolate_missing_sections

base_file_dir, fn = os.path.split(os.path.abspath(__file__))
repo_dir = f"{base_file_dir}/../"

global file_dir
base_file_dir, fn = os.path.split(os.path.abspath(__file__))
file_dir = base_file_dir + os.sep + "section_numbers" + os.sep
manual_dir = base_file_dir + os.sep + "manual_points" + os.sep


def setup_args(args):
    """
    About:
        Setup the parameters and filenames that will be used in the reconstruction
    Inputs:
        args:   user input arguments

    Outputs:
        args:   user input arguments with some additional parameters tacked on in this function
    """
    ###
    ### Parameters
    ###

    if args.chunk_info_csv == None:
        args.chunk_info_csv = base_file_dir + "/scale_factors.json"

    args.manual_tfm_dir = base_file_dir + "transforms/"

    if args.sect_info_fn == None:
        args.sect_info_fn = args.out_dir + "/autoradiograph_info_volume_order.csv"

    args.qc_dir = f"{args.out_dir}/6_quality_control/"
    os.makedirs(args.crop_dir, exist_ok=True)

    args.resolution_list = [float(r) for r in args.resolution_list]

    return args


def reconstruct(
    hemi_info_csv: str,
    chunk_info_csv: str,
    sect_info_csv: str,
    resolution_list: list,
    output_dir: str,
    pytorch_model_dir: str = f"{repo_dir}/caps/nnUNet_results/nnUNet/2d/Task502_cortex",
    interp_type: str = "surf",
    output_csv: str = "",
    n_depths: int = 0,
    dice_threshold:float = 0.5,
    num_cores: int = 0,
    batch_correction: bool = False,
    clobber: bool = False,
):
    """
    Reconstruct 2D histological sections to 3D volume using a structural reference volume (e.g., T1w MRI from brain donor, stereotaxic template)

     Processing Steps
        1. Segment
        2. Init Alignment (Rigid 2D, per chunk)
        3.1 GM MRI to autoradiograph volume (Nonlinear 3D, per chunk)
        3.2 Autoradiograph to GM MRI (2D nonlinear, per chunk)
        4. Interpolate missing vertices on sphere, interpolate back to 3D volume
        5. Quality control

    :param hemi_info_csv: str, path to csv file that contains information about hemispheres to reconstruct
    :param sect_info_csv : pandas dataframe, contains information about sections
    :param chunk_info_csv : str, path to json file that contains information about chunks
    :param resolution_list : list, resolutions to use for reconstruction
    :param output_dir : str, output directory where results will be put
    :param gm_surf_fn : str, optional, path to gm surface to use for surface interpolation
    :param wm_surf_fn : str, optional, path to wm surface to use for surface interpolation
    :param pytorch_model_dir : str, optional, path of directory of pytorch model to use for reconstruction
    :param num_cores : int, optional, number of cores to use for reconstruction, default=0 (use all cores)

    :return sect_info : pandas dataframe, contains updated information about sections with new fields for files produced
    """

    seg_dir = f"{output_dir}/1_seg/"
    initalign_dir = f"{output_dir}/2_init_align/"
    multires_align_dir = f"{output_dir}/3_multires_align/"
    interp_dir = f"{output_dir}/4_interp/"
    num_cores = utils.set_cores(num_cores)

    maximum_resolution = resolution_list[-1]

    valid_inputs = validate_inputs.validate_inputs(
        hemi_info_csv, chunk_info_csv, sect_info_csv
    )

    assert valid_inputs, "Error: invalid inputs"

    # Stage: Segment
    seg_df_csv = segment(sect_info_csv, seg_dir, maximum_resolution, clobber=clobber)

    # Stage: Initial rigid aligment of sections
    init_sect_csv, init_chunk_csv = initalign(
        seg_df_csv, chunk_info_csv, initalign_dir, clobber=clobber
    )

    # Stage: Multiresolution alignment of sections to structural reference volume
    align_chunk_info_csv, align_sect_info_csv = multiresolution_alignment(
        hemi_info_csv,
        init_chunk_csv,
        init_sect_csv,
        resolution_list,
        multires_align_dir,
        num_cores=num_cores,
        dice_threshold=dice_threshold,
        clobber=clobber,
    )

    # Stage: Interpolate missing sections
    interpolate_missing_sections(
            hemi_info_csv,
            align_chunk_info_csv,
            align_sect_info_csv,
            maximum_resolution,
            interp_dir,
            n_depths = n_depths,
            batch_correction = batch_correction,
            clobber = clobber,
    )

    return output_csv


def setup_argparse():
    """
    Get user input arguments
    """
    parser = argparse.ArgumentParser(description="Process some integers.")

    parser.add_argument(
        dest="sect_info_csv",
        type=str,
        help="Path to csv file containing section information",
    )
    parser.add_argument(
        dest="chunk_info_csv",
        type=str,
        help="Path to csv file containing chunk information",
    )
    parser.add_argument(
        dest="ref_vol_fn",
        type=str,
        help="Structural reference volume filename",
    )
    parser.add_argument(dest="output_dir", type=str, help="Output directory")

    parser.add_argument(
        "--resolutions",
        "-r",
        dest="resolution_list",
        nargs="+",
        default=[4.0, 3.0, 2.0, 1.0, 0.5, 0.25],
        type=float,
        help="Resolution list.",
    )
    parser.add_argument(
        "--gm-surf",
        dest="gm_surf_fn",
        type=str,
        default="",
        help="Gray matter surface filename",
    )
    parser.add_argument(
        "--wm-surf",
        dest="wm_surf_fn",
        type=str,
        default="",
        help="White matter surface filename",
    )
    parser.add_argument(
        "--num-cores",
        "-r",
        dest="num_cores",
        type=int,
        default=0,
        help="number of cores to use for multiprocessing",
    )

    parser.add_argument(
        "--dice-threshold",
        dest="dice_threshold",
        type=float,
        default=0.5,
        help="Cutoff dice coefficient for alignment of 2d sections (default=0.5)"
    )
    parser.add_argument(
        "--ndepths",
        dest="n_depths",
        type=int,
        default=0,
        help="number of mid-surface depths between gm and wm surface",
    )
    parser.add_argument(
        "--pytorch-model",
        "-m",
        dest="pytorch_model_dir",
        default=f"{repo_dir}/caps/nnUNet_results/nnUNet/2d/Task502_cortex",
        help="Numer of cores to use for segmentation (Default=0; This is will set the number of cores to use to the maximum number of cores availale)",
    )

    parser.add_argument(
        "--batch-correction",
        dest="batch_correction",
        default=False,
        action="store_true",
        help="Correct batch effects",
    )

    parser.add_argument(
        "--clobber",
        dest="clobber",
        default=False,
        action="store_true",
        help="Overwrite existing results",
    )
    parser.add_argument(
        "--debug", dest="debug", default=False, action="store_true", help="DEBUG mode"
    )

    return parser


if __name__ == "__main__":
    args = setup_argparse().parse_args()

    sect_info = pd.read_csv(args.sect_info_fn)

    reconstruct_hemisphere(
        args.hemi_info_csv,
        arg.chunk_info_csv,
        args.sect_info_csv,
        args.ref_vol_fn,
        resolution_list=args.resolution_list,
        output_dir=args.output_dir,
        pytorch_model_dir=args.pytorch_model_dir,
        n_depths=args.n_depths,
        gm_surf_fn=args.gm_surf_fn,
        wm_surf_fn=args.wm_surf_fn,
        dice_threshold=args.dice_threshold,
        batch_correction=args.batch_correction,
        num_cores=args.num_cores,
    )

    ### Step 0 : Crop downsampled autoradiographs
