"""Main functions for launching 3D reconstruction with BrainBuilder."""

import argparse
import os

import pandas as pd

import brainbuilder.qc.quality_control as qc
from brainbuilder.downsample import downsample_sections
from brainbuilder.initalign import initalign
from brainbuilder.interpsections import interpolate_missing_sections
from brainbuilder.qc.validate_interp_error import validate_interp_error
from brainbuilder.segment import segment
from brainbuilder.utils import utils
from brainbuilder.utils.validate_inputs import validate_inputs
from brainbuilder.volalign import multiresolution_alignment

base_file_dir, fn = os.path.split(os.path.abspath(__file__))
repo_dir = f"{base_file_dir}/../"

global file_dir
base_file_dir, fn = os.path.split(os.path.abspath(__file__))
file_dir = base_file_dir + os.sep + "section_numbers" + os.sep
manual_dir = base_file_dir + os.sep + "manual_points" + os.sep


def setup_args(args: argparse.Namespace) -> argparse.Namespace:
    """Setup the parameters and filenames that will be used in the reconstruction.

    :param args:   user input arguments
    :return args:   user input arguments with some additional parameters tacked on in this function
    """
    ###
    ### Parameters
    ###

    if args.chunk_info_csv is None:
        args.chunk_info_csv = base_file_dir + "/scale_factors.json"

    args.manual_tfm_dir = base_file_dir + "transforms/"

    if args.sect_info_fn is None:
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
    pytorch_model_dir: str = f"{repo_dir}/nnUNet/Dataset501_Brain/nnUNetTrainer__nnUNetPlans__2d/",
    interp_type: str = "surf",
    output_csv: str = "",
    n_depths: int = 0,
    use_nnunet: bool = True,
    num_cores: int = 0,
    surface_smoothing: int = 0,
    batch_correction_resolution: float = 0,
    skip_interp: bool = False,
    clobber: bool = False,
) -> None:
    """Reconstruct 2D histological sections to 3D volume using a structural reference volume (e.g., T1w MRI from brain donor, stereotaxic template).

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
    :param pytorch_model_dir : str, optional, path of directory of pytorch model to use for reconstruction
    :param num_cores : int, optional, number of cores to use for reconstruction, default=0 (use all cores)
    :return sect_info : pandas dataframe, contains updated information about sections with new fields for files produced
    """
    downsample_dir = f"{output_dir}/0_downsample/"
    seg_dir = f"{output_dir}/1_seg/"
    initalign_dir = f"{output_dir}/2_init_align/"
    multires_align_dir = f"{output_dir}/3_multires_align/"
    interp_dir = f"{output_dir}/4_interp/"
    qc_dir = f"{output_dir}/qc/"

    valid_inputs_npz = f"{output_dir}/valid_inputs"

    num_cores = int(utils.set_cores(num_cores) / 2)

    maximum_resolution = resolution_list[-1]

    print('Reconstructing 2D sections to 3D volume')
    print('\tInput files:')
    print(f'\t\tHemisphere info: {hemi_info_csv}')
    print(f'\t\tChunk info: {chunk_info_csv}')
    print(f'\t\tSection info: {sect_info_csv}')
    print(f'\t\tResolution list: {resolution_list}')
    print('\tOutput directories:')
    print(f'\t\tDownsample: {downsample_dir}')
    print(f'\t\tSegment: {seg_dir}')
    print(f'\t\tInitial alignment: {initalign_dir}')
    print(f'\t\tMultiresolution alignment: {multires_align_dir}')
    print(f'\t\tInterpolation: {interp_dir}')
    print(f'\t\tQuality control: {qc_dir}')



    valid_inputs = validate_inputs(
        hemi_info_csv, chunk_info_csv, sect_info_csv, valid_inputs_npz
    )
    assert valid_inputs, "Error: invalid inputs"

    sect_info_csv = downsample_sections(
        chunk_info_csv,
        sect_info_csv,
        min(resolution_list),
        downsample_dir,
        clobber=clobber,
    )
    qc.data_set_quality_control(sect_info_csv, qc_dir, column="img")

    # Stage: Segment
    seg_df_csv = segment(
        chunk_info_csv,
        sect_info_csv,
        seg_dir,
        maximum_resolution,
        use_nnunet=use_nnunet,
        clobber=clobber,
    )

    qc.data_set_quality_control(seg_df_csv, qc_dir, column="seg")

    # Stage: Initial rigid aligment of sections
    init_sect_csv, init_chunk_csv = initalign(
        seg_df_csv, chunk_info_csv, initalign_dir, resolution_list, clobber=clobber
    )

    # Stage: Multiresolution alignment of sections to structural reference volume
    align_chunk_info_csv, align_sect_info_csv = multiresolution_alignment(
        hemi_info_csv,
        init_chunk_csv,
        init_sect_csv,
        resolution_list,
        multires_align_dir,
        num_cores=num_cores,
        clobber=clobber,
    )

    qc.data_set_quality_control(align_sect_info_csv, qc_dir, column="init_img")

    # Stage: Interpolate missing sections
    if not skip_interp:
        reconstructed_chunk_info_csv = interpolate_missing_sections(
            hemi_info_csv,
            align_chunk_info_csv,
            align_sect_info_csv,
            maximum_resolution,
            interp_dir,
            n_depths=n_depths,
            surface_smoothing=surface_smoothing,
            batch_correction_resolution=batch_correction_resolution,
            clobber=clobber,
        )

        validate_interp_error(
            align_sect_info_csv, reconstructed_chunk_info_csv, interp_dir+'/qc', clobber=clobber
        )
    return output_csv


global ju_atlas_fn
global mni_template_fn


def setup_argparse() -> argparse.ArgumentParser:
    """Get user input arguments.

    :return parser: argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(description="Process some integers.")

    parser.add_argument(
        dest="hemi_info_csv",
        type=str,
        help="Path to csv file containing hemisphere information. Mandatory columns: [sub, hemisphere, struct_ref_vol, gm_surf, wm_surf]",
    )
    parser.add_argument(
        dest="chunk_info_csv",
        type=str,
        help="Path to csv file containing chunk informatio. Mandatory columns: [ sub, hemisphere, chunk, direction, pixel_size_0, pixel_size_1, section_thickness]",
    )
    parser.add_argument(
        dest="sect_info_csv",
        type=str,
        help="Path to csv file containing section information. Mandatory columns: [sub, hemisphere, chunk, acquisition, raw, sample] ",
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
        "--num-cores",
        "-r",
        dest="num_cores",
        type=int,
        default=0,
        help="number of cores to use for multiprocessing",
    )
    parser.add_argument(
        "--no-nnunet",
        dest="use_nnunet",
        default=True,
        action="store_false",
        help="Do not use nnUNet for segmentation",
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
        default=f"{repo_dir}/nnUNet/Dataset501_Brain/nnUNetTrainer__nnUNetPlans__2d/",
        help="Numer of cores to use for segmentation (Default=0; This is will set the number of cores to use to the maximum number of cores availale)",
    )

    parser.add_argument(
        "--batch-correction",
        dest="batch_correction_resolution",
        default=0,
        help=" Resolution at which to correct batch effects (Default=0, no correction)",
    )

    parser.add_argument(
        "surface_smoothing",
        dest="surface_smoothing",
        default=0,
        help="Use surface smoothing beore creating final volume",
    )
    parser.add_argument(
        "--clobber",
        dest="clobber",
        default=False,
        action="store_true",
        help="Overwrite existing results",
    )
    parser.add_argument(
        "--skip-interp",
        dest="skip_interp",
        default=False,
        action="store_true",
        help="Skip interpolation step",
    )
    parser.add_argument(
        "--debug", dest="debug", default=False, action="store_true", help="DEBUG mode"
    )

    return parser


if __name__ == "__main__":
    args = setup_argparse().parse_args()

    sect_info = pd.read_csv(args.sect_info_fn)

    reconstruct(
        args.hemi_info_csv,
        args.chunk_info_csv,
        args.sect_info_csv,
        resolution_list=args.resolution_list,
        output_dir=args.output_dir,
        pytorch_model_dir=args.pytorch_model_dir,
        n_depths=args.n_depths,
        use_nnunet=args.use_nnunet,
        batch_correction=args.batch_correction_resolution,
        num_cores=args.num_cores,
        skip_interp=args.skip_interp,
    )
