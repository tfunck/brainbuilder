"""Main functions for launching 3D reconstruction with BrainBuilder."""

import argparse
import logging
import os
from pathlib import Path

import pandas as pd

import brainbuilder.utils.utils as utils
from brainbuilder.downsample import downsample_sections
from brainbuilder.initalign import initalign
from brainbuilder.intensity_correction import intensity_correction
from brainbuilder.interpsections import interpolate_missing_sections
from brainbuilder.segment import segment
from brainbuilder.utils.utils import get_logger
from brainbuilder.utils.validate_inputs import validate_inputs
from brainbuilder.volalign import multiresolution_alignment

base_file_dir, fn = os.path.split(os.path.abspath(__file__))
repo_dir = f"{base_file_dir}/../"

base_file_dir, fn = os.path.split(os.path.abspath(__file__))
file_dir = base_file_dir + os.sep + "section_numbers" + os.sep
manual_dir = base_file_dir + os.sep + "manual_points" + os.sep


logger = get_logger(__name__)


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
    output_csv: str = "",
    n_depths: int = 0,
    use_3d_syn_cc: bool = True,
    use_syn: bool = True,
    linear_steps: list = ["rigid", "similarity", "affine"],
    seg_method: str = "nnunetv1",
    nnunet_model_dir: str = f"{repo_dir}/nnUNet/Dataset501_Brain/nnUNetTrainer__nnUNetPlans__2d/",
    nnunet_config_json: str = f"{repo_dir}/nnUNet/nnunet_config/primate_v1.json",
    num_cores: int = None,
    max_resolution_3d: float = 0.3,
    final_resolution: float = None,
    interp_method: str = "volumetric",
    interpolation_2d: str = "Linear",
    landmark_dir: Path = None,
    use_intensity_correction: bool = False,
    use_3d_align_stage: bool = True,
    use_interp_stage: bool = True,
    verbose: bool = False,
    clobber: bool = False,
) -> None:
    """Reconstruct 2D histological sections to 3D volume using a structural reference volume (e.g., T1w MRI from brain donor, stereotaxic template).

     Processing Steps
        1. Downsample
        2. Segment
        3. Initial Alignment (Rigid 2D, per chunk)
        4. Multiresolution Alignment (3D nonlinear, per chunk)
        5. Intensity Correction (optional)
        6. Interpolate missing sections
        7. Quality control

    :param hemi_info_csv: str, path to csv file containing hemisphere information
    :param chunk_info_csv: str, path to csv file containing chunk information
    :param sect_info_csv: str, path to csv file containing section information
    :param resolution_list: list, resolutions to use for multiresolution alignment
    :param output_dir: str, output directory where results will be saved
    :param output_csv: str, output csv file with updated section information (default="")
    :param n_depths: int, number of mid-surface depths between GM and WM surface (default=0)
    :param use_3d_syn_cc: bool, use 3D nonlinear SyN with cross-correlation (default=True)
    :param use_syn: bool, use 2D nonlinear SyN (default=True)
    :param linear_steps: list, linear transformation steps for alignment (default=['rigid', 'similarity', 'affine'])
    :param seg_method: str, segmentation method ('nnunetv1', 'nnunetv2', 'otsu', 'triangle') (default='nnunetv1')
    :param nnunet_model_dir: str, path to nnUNet model directory for segmentation
    :param nnunet_config_json: str, path to nnUNet configuration JSON file
    :param num_cores: int, number of cores for multiprocessing (default=None, uses all available)
    :param max_resolution_3d: float, maximum resolution for 3D alignment (default=0.3)
    :param final_resolution: float, final resolution for output volume (default=None)
    :param interp_method: str, interpolation method ('volumetric', 'surface') (default='volumetric')
    :param interpolation_2d: str, 2D interpolation method (default='Linear')
    :param landmark_dir: Path, directory containing landmark files (default=None)
    :param use_intensity_correction: bool, perform intensity correction (default=False)
    :param use_3d_align_stage: bool, run multiresolution alignment stage (default=True)
    :param use_interp_stage: bool, run interpolation stage (default=True)
    :param verbose: bool, verbose output for debugging (default=False)
    :param clobber: bool, overwrite existing results (default=False)
    :return: str, output_csv filename
    """
    # Set the logger level that will be used for the whole reconstruction

    if verbose:
        utils.LOG_VERBOSITY_LEVEL = logging.DEBUG
        logger.setLevel(logging.DEBUG)

    downsample_dir = f"{output_dir}/0_downsample/"
    seg_dir = f"{output_dir}/1_seg/"
    intens_corr_dir = f"{output_dir}/1.5_intensity_corr"
    initalign_dir = f"{output_dir}/2_init_align/"
    multires_align_dir = f"{output_dir}/3_multires_align/"
    interp_dir = f"{output_dir}/4_interp/"
    qc_dir = f"{output_dir}/qc/"

    valid_inputs_npz = f"{output_dir}/valid_inputs"

    if num_cores is None or num_cores == 0:
        num_cores = -1

    max_resolution_2d = resolution_list[-1]

    logger.info("Reconstructing 2D sections to 3D volume")
    logger.info("\tInputs:")
    logger.info(f"\t\tHemisphere info: {hemi_info_csv}")
    logger.info(f"\t\tChunk info: {chunk_info_csv}")
    logger.info(f"\t\tSection info: {sect_info_csv}")
    logger.info(f"\t\tMax 3D resolution: {max_resolution_3d}")
    logger.info(f"\t\tFinal resolution: {final_resolution}")
    logger.info(f"\t\tResolution list: {resolution_list}")
    logger.info(f"\t\tLinear steps: {linear_steps}")
    logger.info(f"\t\tNumber of cores: {num_cores}")
    logger.info(f"\t\tSegmentation method: {seg_method}")
    logger.info(f"\t\tUse 3D nonlinear CC: {use_3d_syn_cc}")
    logger.info(f"\t\tUse 2D nonlinear: {use_syn}")
    logger.info(f"\t\tMissing Section Interpolation method: {interp_method}")
    logger.info(f"\t\t2D interpolation method: {interpolation_2d}")
    logger.info(f"\t\tUse intensity correction: {use_intensity_correction}")
    logger.info(f"\t\tClobber: {clobber}")
    logger.info("\tStages to run:")
    logger.info(f"\t\tDownsample: {downsample_dir}")
    logger.info(f"\t\tSegment: {seg_dir}")
    logger.info(f"\t\tInitial alignment: {initalign_dir}")
    if use_3d_align_stage:
        logger.info(f"\t\tMultiresolution alignment: {multires_align_dir}")
    if use_interp_stage:
        logger.info(f"\t\tInterpolation: {interp_dir}")
    logger.info(f"\t\tQuality control: {qc_dir}")
    logger.info(f"\t\tIntensity correction: {intens_corr_dir}")

    valid_inputs = validate_inputs(
        hemi_info_csv,
        chunk_info_csv,
        sect_info_csv,
        valid_inputs_npz,
        n_jobs=num_cores,
        clobber=clobber,
    )
    assert valid_inputs, "Error: invalid inputs"

    sect_info_csv = downsample_sections(
        chunk_info_csv,
        sect_info_csv,
        min(resolution_list),
        downsample_dir,
        num_cores=num_cores,
        clobber=clobber,
    )
    # qc.data_set_quality_control(sect_info_csv, qc_dir, column="img")

    # Stage: Segment
    sect_info_csv = segment(
        sect_info_csv,
        seg_dir,
        max_resolution_2d,
        seg_method=seg_method,
        model_dir=nnunet_model_dir,
        nnunet_config_json=nnunet_config_json,
        clobber=clobber,
    )
    # qc.data_set_quality_control(seg_df_csv, qc_dir, column="seg")

    logger.info("Stage: Initial rigid alignment of sections")
    # Stage: Initial rigid aligment of sections
    sect_info_csv, chunk_info_csv = initalign(
        sect_info_csv, chunk_info_csv, output_dir, resolution_list, clobber=clobber
    )

    if use_intensity_correction:
        # Stage: Intensity correction
        logger.info("Performing intensity correction")
        sect_info_csv = intensity_correction(
            sect_info_csv, chunk_info_csv, intens_corr_dir, clobber=clobber
        )

    if use_3d_align_stage:
        # Stage: Multiresolution alignment of sections to structural reference volume
        chunk_info_csv, sect_info_csv = multiresolution_alignment(
            hemi_info_csv,
            chunk_info_csv,
            sect_info_csv,
            resolution_list,
            output_dir,
            max_resolution_3d=max_resolution_3d,
            use_3d_syn_cc=use_3d_syn_cc,
            use_syn=use_syn,
            num_cores=num_cores,
            linear_steps=linear_steps,
            interpolation=interpolation_2d,
            landmark_dir=landmark_dir,
            clobber=clobber,
        )
    # qc.data_set_quality_control(align_sect_info_csv, qc_dir, column="init_img")

    # Stage: Interpolate missing sections
    if use_interp_stage:
        chunk_info_csv = interpolate_missing_sections(
            hemi_info_csv,
            chunk_info_csv,
            sect_info_csv,
            max_resolution_2d,
            resolution_list,
            interp_dir,
            n_depths=n_depths,
            interp_method=interp_method,
            final_resolution=final_resolution,
            interpolation=interpolation_2d,
            num_cores=num_cores,
            use_final_transform=use_3d_align_stage,  # only apply the final transform if we did the 3D alignment stage
            clobber=clobber,
        )

        # validate_interp_error(
        #    align_sect_info_csv,
        #    reconstructed_chunk_info_csv,
        #    interp_dir + "/qc",
        #    clobber=clobber,
        # )

    logger.info("##### Reconstruction Complete #####")
    return output_csv


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
        "--final-resolution",
        dest="final_resolution",
        type=float,
        default=None,
        help="Final resolution for the output volume.",
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
        "--seg-method",
        dest="seg_method",
        default="nnunetv1",
        help="Use: \n\t'nnunetv1':version 1 of nnUNet segmentation,\n\t'nnunetv2': version 2 of nnUNet segmentation, \n\t'otsu': Otsu histogram thresholding, \n\t'triangle': Triangle histogram thresholding",
    )

    parser.add_argument(
        "--interp-method",
        dest="interp_method",
        default="surface",
        help="Use interpolation method: \n\t'surface': surface interpolation, \n\t'volumetric': volumetric interpolation",
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
        "--nnunet-config-json",
        dest="nnunet_config_json",
        default=f"{repo_dir}/nnUNet/nnunet_config/primate_v1.json",
        help="Path to nnUNet configuration JSON file",
    )
    # Add verbosity flag

    parser.add_argument(
        "--verbose",
        "-v",
        dest="verbose",
        default=False,
        action="store_true",
        help="Verbose output for debugging",
    )
    parser.add_argument(
        "--clobber",
        dest="clobber",
        default=False,
        action="store_true",
        help="Overwrite existing results",
    )

    parser.add_argument(
        "--no-3d-nl-cc",
        dest="use_3d_syn_cc",
        default=True,
        action="store_false",
        help="Overwrite existing results",
    )
    parser.add_argument(
        "--no-2d-nl",
        dest="use_syn",
        default=True,
        action="store_false",
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
        seg_method=args.seg_method,
        use_3d_syn_cc=args.use_3d_syn_cc,
        use_syn=args.use_syn,
        num_cores=args.num_cores,
        skip_interp=args.skip_interp,
    )
