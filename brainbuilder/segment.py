"""Create GM volume from segmented images."""
import json
import os
import re
import shutil
from glob import glob

import ants
import imageio.v3 as iio
import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
from joblib import Parallel, delayed
from skimage.filters import (
    threshold_li,
    threshold_otsu,
    threshold_triangle,
    threshold_yen,
)
from skimage.transform import resize

import brainbuilder.utils.ants_nibabel as nib
from brainbuilder.utils import utils
from brainbuilder.utils.paths import define_new_path_column

logger = utils.get_logger(__name__)

base_file_dir, fn = os.path.split(os.path.abspath(__file__))
repo_dir = f"{base_file_dir}/../"

nnUNet_dir = f"{repo_dir}/nnUNet/"

global HISTOGRAM_METHODS
HISTOGRAM_METHODS = ["triangle", "otsu", "yen", "li", "multi"]


def _gen_nnunet_out(output_dir, sub, hemi, chunk):
    out_dir = f"{output_dir}/sub-{sub}/hemi-{hemi}/chunk-{chunk}/nnunet_out/"
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def _gen_nnunet_in(output_dir, sub, hemi, chunk):
    out_dir = f"{output_dir}/sub-{sub}/hemi-{hemi}/chunk-{chunk}/nnunet/"
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def apply_threshold(img: np.ndarray, method: callable) -> np.ndarray:
    """Apply thresholding method to the image.

    :param img: np.ndarray, input image
    :param method: callable, thresholding method
    :return: np.ndarray, segmented image
    """
    thr = method(img)
    im = np.zeros_like(img)
    im[img > thr] = 1
    return im


from skimage.morphology import (
    binary_closing,
    disk,
    remove_small_holes,
    remove_small_objects,
)


def postprocess_binary(mask, min_size=128, hole_size=128, closing_radius=1):
    if closing_radius > 0:
        mask = binary_closing(mask, disk(closing_radius))
    if min_size > 0:
        mask = remove_small_objects(mask, min_size=min_size)
    if hole_size > 0:
        mask = remove_small_holes(mask, area_threshold=hole_size)
    return mask.astype(np.uint8)


def kmeans_seg(img, k=2, blur_sigma=1.0, seed=0):
    from skimage.filters import gaussian
    from sklearn.cluster import KMeans

    sm = gaussian(img, sigma=blur_sigma)
    feats = np.stack([img.ravel(), sm.ravel()], axis=1)
    km = KMeans(n_clusters=k, n_init=10, random_state=seed)
    lab = km.fit_predict(feats).reshape(img.shape)
    # choose the cluster with higher mean intensity as GM
    means = [img[lab == c].mean() if np.any(lab == c) else -np.inf for c in range(k)]
    gm_label = int(np.argmax(means))
    return lab == gm_label


def multi_threshold(img: np.ndarray) -> np.ndarray:
    """Apply multiple thresholding methods to the image.

    :param img: np.ndarray, input image
    :return: np.ndarray, segmented image
    """
    seg = np.zeros_like(img)

    methods = [threshold_li, threshold_otsu, kmeans_seg]

    pp_params = dict(min_size=128, hole_size=128, closing_radius=1)

    for method in methods:
        im_thr = apply_threshold(img, method)
        im_thr = postprocess_binary(im_thr, **pp_params)
        seg += im_thr

    # n = len(methods)
    # seg /= n
    # seg[seg < 0.5] = 0
    # seg[seg >= 0.5] = 1

    # set max to 255

    return seg


def myelin_threshold(image: np.ndarray) -> np.ndarray:
    """Apply myelin thresholding to the image."""
    t = threshold_yen(image)
    knee_point = 0.90 * image.max()

    seg = np.zeros_like(image)
    seg[(image > t) & (image < knee_point)] = 1
    return seg


def histogram_threshold(
    raw_fn: str, seg_fn: str, sd: float = 1, ref: str = None, method: str = "otsu"
) -> None:
    """Apply histogram thresholding to the cropped images.

    param: raw_fn: raw image filename
    param: seg_fn: segmentation filename
    param: sd: standard deviation
    param: ref: reference image filename
    param: method: thresholding method
    return: None
    """
    img = nib.load(raw_fn)

    affine = img.affine
    dimensions = img.shape

    ar = img.get_fdata()
    assert np.sum(np.abs(ar)) > 0, (
        "Error: empty input image with histogram thresholding " + raw_fn
    )

    if "MS" in raw_fn:  # use myelin thresholding
        out = myelin_threshold(ar)
    elif method == "triangle":
        out = apply_threshold(ar, threshold_triangle)
    elif method == "li":
        out = apply_threshold(ar, threshold_li)
    elif method == "yen":
        out = apply_threshold(ar, threshold_yen)
    else:
        out = apply_threshold(ar, threshold_otsu)
    if not isinstance(ref, type(None)):
        ref_hd = nib.load(ref)
        dimensions = ref_hd.shape  # resize to reference image
        affine = ref_hd.affine

    out = resize(out, dimensions, order=3)
    if len(out.shape) == 3:
        out = out.reshape([out.shape[0], out.shape[1]])

    assert np.sum(np.abs(out)) > 0, (
        "Error: empty segmented image with histogram thresholding " + raw_fn
    )

    # scale to 255
    out = (out / out.max() * 255).astype(np.uint8)

    nib.Nifti1Image(out, affine, direction_order="lpi").to_filename(seg_fn)

    return out


typeDataFrame = type(pd.DataFrame({}))

base_file_dir, fn = os.path.split(os.path.abspath(__file__))


def convert_2d_array_to_nnunet(
    input_filename: str,
    output_filename: str,
    res: list,
    x_scale: int = 414,
    spacing: tuple = (1, 1, 1),
    clobber: bool = False,
) -> None:
    """Converts numpy into a series of niftis.

    The image can have an arbitrary number of input channels which will be exported separately (_0000.nii.gz,
    _0001.nii.gz, etc for images and only .nii.gz for seg).
    Spacing can be ignored most of the time.
    !!!2D images are often natural images which do not have a voxel spacing that could be used for resampling. These images
    must be resampled by you prior to converting them to nifti!!!
    Datasets converted with this utility can only be used with the 2d U-Net configuration of nnU-Net
    If Transform is not None it will be applied to the image after loading.
    Segmentations will be converted to np.uint32!

    :param input_filename: str, path to numpy array
    :param output_filename: str, path to output nifti file
    :param res: list, resolution of the image
    :param spacing: tuple, spacing of the image
    :param clobber: bool, overwrite existing files
    :return: None
    """
    if (
        not os.path.exists(output_filename)
        or not utils.newer_than(output_filename, input_filename)
        or clobber
    ):
        aff = np.eye(4)
        aff[0, 0] = res[0]
        aff[1, 1] = res[1]

        # Commented out because this possible produces the wrong sized dimensinos for the the unet
        # nii_img = utils.resample_to_resolution(
        # input_filename, [0.2, 0.2], affine=aff, order=1
        # )
        # img = nii_img.get_fdata()

        # testing this downsampling for nnUNet
        img = nib.load(input_filename).get_fdata()

        scale = x_scale / img.shape[0]
        xdim = int(img.shape[0] * scale)
        ydim = int(img.shape[1] * scale)
        img = resize(img, (xdim, ydim), anti_aliasing=True)

        img = np.rot90(np.fliplr(img), -1)

        assert len(img.shape) == 2

        assert np.sum(np.abs(img)) > 0

        if ".nii.gz" in output_filename:
            img = img[None, None]  # add dimensions
            # image is now (c, y, x, z) where x=1 since it's 2d
            img = img.astype(np.float32)

            for _, i in enumerate(img):
                itk_img = sitk.GetImageFromArray(i)
                itk_img.SetSpacing(list(spacing)[::-1])
                sitk.WriteImage(itk_img, output_filename)
        else:  # use imageio to write as tiff / png / jpg / etc
            iio.imwrite(output_filename, img)

    logger.info("Wrote: " + output_filename)


def apply_histogram_threshold(
    sect_info: pd.DataFrame, num_cores: int = 1, method: str = "otsu"
) -> None:
    """Apply histogram threshold to the raw images.

    param: sect_info: dataframe with columns: raw, seg_fn
    param: num_cores: number of cores to use
    param: method: thresholding method
    return: dataframe with columns: raw, seg_fn
    """
    Parallel(n_jobs=num_cores)(
        delayed(histogram_threshold)(
            row["img"], row["seg"], ref=row["img"], method=method
        )
        for _, row in sect_info.iterrows()
    )

    return None


def get_nnunet_filename(input_fn: str, nnunet_out_dir: str) -> str:
    """Get the nnunet filename from the input filename.

    param: input_fn: input filename
    param: nnunet_out_dir: directory to save nnunet images
    return: nnunet filename
    """
    base = os.path.basename(input_fn).replace(".nii.gz", "").replace(".nii", "")

    nnunet_list = glob(f"{nnunet_out_dir}/{base}*")

    if len(nnunet_list) > 0:
        nnunet_fn = nnunet_list[0]
    else:
        nnunet_fn = ""

    return nnunet_fn


def convert_from_nnunet_list(
    sect_info: pd.DataFrame,
    output_dir: str,
    nnunet_input_str: str = "img",
    warning_flag: bool = False,
    clobber: bool = False,
) -> list:
    """Convert the nnunet images to regular nifti images.

    param: sect_info: dataframe with columns: img, seg_fn
    param: nnunet_out_dir: base output directory (subject folders live here)
    param: nnunet_input_str: column name for nnunet input
    param: clobber: overwrite existing files
    return: list of files to convert
    """
    to_do = []

    for _, row in sect_info.iterrows():
        img_fn = row[nnunet_input_str]
        seg_fn = row["seg"]

        # *** CHANGED: nnunet_out now lives inside the subject folder ***
        curr_dir = _gen_nnunet_out(
            output_dir, row["sub"], row["hemisphere"], row["chunk"]
        )

        os.makedirs(curr_dir, exist_ok=True)

        nnunet_fn = get_nnunet_filename(img_fn, curr_dir)

        if nnunet_fn == "":
            continue

        if utils.check_run_stage([seg_fn], [nnunet_fn], clobber=clobber):
            if warning_flag:
                logger.warning(f"\Could not find file: {seg_fn}")

        to_do.append((nnunet_fn, img_fn, seg_fn))

    return to_do


def convert_to_nnunet_list(
    sect_info: pd.DataFrame,
    output_dir: str,
    nnunet_input_str: str = "img",
    nnunet_ext: str = ".nii.gz",
    clobber: bool = False,
) -> list:
    """Convert the raw images to nnunet format.

    param: chunk_info: dataframe with columns: sub, hemisphere, chunk
    param: sect_info: dataframe with columns: raw, seg_fn
    param: output_dir: base directory (subject folders live here)
    param: nnunet_input_str: column name for nnunet input
    param: clobber: overwrite existing files
    return: list of files to convert
    """
    to_do = []

    for _, row in sect_info.iterrows():
        f = row[nnunet_input_str]

        # DEPRECEATED: no longer use pixel size from chunk_info, read from files
        # pixel_size_0, pixel_size_1, _ = utils.get_chunk_pixel_size(
        # row["sub"], row["hemisphere"], row["chunk"], chunk_info
        # )
        pixel_size_0, pixel_size_1 = ants.image_read(f).spacing[:2]

        fname = re.sub(".nii.gz", "", os.path.basename(f))

        # *** CHANGED: nnunet now lives inside the subject folder ***
        curr_dir = _gen_nnunet_in(
            output_dir, row["sub"], row["hemisphere"], row["chunk"]
        )

        os.makedirs(curr_dir, exist_ok=True)

        output_filename_truncated = os.path.join(curr_dir, fname)
        output_filename = output_filename_truncated + "_0000" + nnunet_ext
        if (
            not os.path.exists(output_filename)
            or not utils.newer_than(output_filename, f)
            or clobber
        ):
            to_do.append([f, pixel_size_0, pixel_size_1, output_filename])

    return to_do


def check_seg_files(
    sect_info: pd.DataFrame,
    output_dir: str,
    warning_flag: bool = False,
    nnunet_input_str: str = "img",
) -> bool:
    """Check if all the segmentation files exist.
    Also verify that they all have the same dimensions as one another.

    :param sect_info: dataframe with columns: raw, seg_fn
    :param output_dir: base output directory (subject folders live here)
    :param warning_flag: bool, optional, if True, logger.info warning message if file is missing, default=False
    :return: True if all files exist, False otherwise
    """
    all_files_valid = True
    dim_list = []

    for _, row in sect_info.iterrows():
        # check if seg file is newer than raw file, if not the seg file must be removed
        if os.path.exists(row["seg"]) and not utils.newer_than(
            row["seg"], row[nnunet_input_str]
        ):
            # *** CHANGED: compute per-row nnunet_out path to correctly locate stale files ***
            curr_nnunet_out_dir = _gen_nnunet_out(
                output_dir, row["sub"], row["hemisphere"], row["chunk"]
            )

            nnunet_filename = get_nnunet_filename(
                row[nnunet_input_str], curr_nnunet_out_dir
            )
            os.remove(row["seg"])
            if os.path.exists(nnunet_filename):
                os.remove(nnunet_filename)

        if not os.path.exists(row["seg"]):
            all_files_valid = False
            if warning_flag:
                logger.warning(f"\Could not find file: {row['seg']}")

        if os.path.exists(row["seg"]):
            dims = nib.load(row["seg"]).shape
            dim_list.append(dims)

    # Check if all dimensions are the same
    if len(dim_list) > 0:
        first_dim = dim_list[0]
        for d in dim_list:
            if d != first_dim:
                all_files_valid = False
                if warning_flag:
                    logger.warning("Dimension mismatch found in segmentation files.")
                break

    return all_files_valid


def calculate_relative_distance(sect_info, num_cores=4):
    """Calculate the relative position of each GM voxel in each dimension and then create a distance map
    that is the product of the relative distances in each dimension.
    param: sect_info: dataframe with columns: raw, seg_fn
    param: num_cores: number of cores to use
    return: sect_info with additional column 'seg_dist'
    """

    def _calculate_distance(seg_fn, y, out_fn):
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

            # min_y, max_y = np.min(np.where(seg_data > 0)[1]), np.max(np.where(seg_data > 0)[1])
            min_z, max_z = (
                np.min(np.where(seg_data > 0)[1]),
                np.max(np.where(seg_data > 0)[1]),
            )

            # Calculate relative distances
            x_range = np.linspace(1, 2, max_x - min_x + 1)
            z_range = np.linspace(1, 2, max_z - min_z + 1)

            seg_data[min_x : max_x + 1, :, :] *= x_range[:, None]  # Scale x dimension
            seg_data[:, :, min_z : max_z + 1] *= z_range[None, :]  # Scale z dimension

            seg_data *= y  # multiply by relative y position

            # Save the distance map
            nib.Nifti1Image(
                seg_data, seg_img.affine, direction_order="lpi"
            ).to_filename(out_fn)

    sect_info["seg_dist"] = sect_info["seg"].apply(  # Test
        lambda x: x.replace(".nii.gz", "_dist.nii.gz")  # Test
    )

    ymax = sect_info["sample"].max()  # Get the maximum value of the 'sample' column

    Parallel(n_jobs=num_cores)(
        _calculate_distance(row["seg"], row["sample"] / ymax, row["seg_dist"])
        for _, row in sect_info.iterrows()
    )

    sect_info["seg"] = sect_info["seg_dist"]

    return sect_info


def run_nnunet_segmentation(
    seg_method: str,
    nnunet_in_dir: str,
    nnunet_out_dir: str,
    model_dir: str,
    nnUNet_dir: str,
    datasetname: str,
    fold: int = 0,
    checkpoint: str = "checkpoint_best.pth",
    device: str = "cpu",
    nnunet_failed: bool = False,
):
    """Runs nnUNet segmentation using either nnUNet v1 or v2 based on the specified segmentation method.

    Args:
        seg_method (str): The segmentation method to use. Should contain either 'nnunetv1' or 'nnunetv2'.
        nnunet_in_dir (str): Path to the input directory containing images for segmentation.
        nnunet_out_dir (str): Path to the output directory where segmentation results will be saved.
        model_dir (str): Path to the nnUNet model directory.
        nnUNet_dir (str): Path to the nnUNet installation directory.
        nnunet_failed (bool, optional): Flag indicating if nnUNet segmentation previously failed. Defaults to False.

    Returns:
        bool: True if nnUNet segmentation failed, False otherwise.

    Raises:
        Exception: If an error occurs during segmentation, it is logged and nnunet_failed is set to True.
    File:
        segment.py
    """
    if isinstance(seg_method, str) and "nnunetv2" in seg_method:
        logger.info("\tSegmenting with nnUNet")
        try:
            utils.shell(
                f"nnUNetv2_predict_from_modelfolder --c --verbose -i {nnunet_in_dir} -o {nnunet_out_dir} -m {model_dir} -f {fold} -d {datasetname} -device {device} -chk {checkpoint}",
                exit_on_failure=False,
            )
        except Exception as e:
            logger.warning("nnUNet failed to segment")
            logger.warning(e)
            nnunet_failed = True
    elif isinstance(seg_method, str) and "nnunetv1" in seg_method:
        logger.info("\tSegmenting with nnUNet")
        try:
            os.environ["RESULTS_FOLDER"] = f"{nnUNet_dir}/../"
            utils.shell("echo results_folder $RESULTS_FOLDER")
            utils.shell(
                f"nnUNet_predict -i {nnunet_in_dir} -o {nnunet_out_dir} -t {datasetname} -m '2d'",
                exit_on_failure=False,
            )
        except Exception as e:
            logger.warning("Warning: nnUNet failed to segment")
            logger.warning(e)
            nnunet_failed = True

    return nnunet_failed


def process_nnunet_to_nifti(
    sect_info: pd.DataFrame,
    output_dir: str,
    nnunet_input_str: str,
    num_cores: int,
    seg_method: str,
    foreground_labels: list = [1],
    clobber: bool = False,
):
    """Converts nnUNet output files to standard NIfTI files using parallel processing.

    Args:
        sect_info (pd.DataFrame): DataFrame containing segmentation information.
        output_dir (str): Base directory containing subject folders with nnunet_out inside.
        num_cores (int): Number of CPU cores to use for parallel processing.
        seg_method (str): Segmentation method used.
        clobber (bool): Overwrite existing files if True.

    Returns:
        None
    """
    nnunet2nifti_to_do = convert_from_nnunet_list(
        sect_info,
        output_dir,
        nnunet_input_str=nnunet_input_str,
        warning_flag=False,
        clobber=clobber,
    )

    if len(nnunet2nifti_to_do) > 0:
        logger.info("\tConvert Files from nnUNet to standard nifti files")

        Parallel(n_jobs=num_cores)(
            delayed(convert_from_nnunet)(
                nnunet_fn,
                raw_fn,
                seg_fn,
                seg_method=seg_method,
                foreground_labels=foreground_labels,
            )
            for nnunet_fn, raw_fn, seg_fn in nnunet2nifti_to_do
        )


def copy_unsegmented_images(
    sect_info: pd.DataFrame, output_dir: str, clobber: bool = False
) -> pd.DataFrame:
    """Copies unsegmented images to the segmentation output directory.

    Args:
        sect_info (pd.DataFrame): DataFrame containing image information with 'img' column.
        output_dir (str): Directory to save copied images.
        clobber (bool): Overwrite existing files if True.

    Returns:
        pd.DataFrame: Updated DataFrame with 'seg' column containing paths to copied images.
    """
    logger.info("\tNo segmentation method specified, using unsegmented images")
    sect_info["seg"] = sect_info["img"].apply(
        lambda x: output_dir
        + "/"
        + os.path.basename(x).replace(".nii.gz", "_seg.nii.gz")
    )

    for seg_fn, img_fn in zip(sect_info["seg"], sect_info["img"]):
        if not os.path.exists(seg_fn) or clobber:
            shutil.copy(img_fn, seg_fn)
            logger.info(f"\tCopied {img_fn} to {seg_fn}")
    return sect_info


def convert_nifti_to_nnunet(
    sect_info: pd.DataFrame,
    output_dir: str,
    nnunet_input_str: str,
    x_scale: int = 414,
    nnunet_ext: str = ".nii.gz",
    num_cores: int = -1,
    clobber: bool = False,
) -> None:
    """Converts NIfTI files to nnUNet format using parallel processing.

    Args:
        sect_info (pd.DataFrame): DataFrame containing section information.
        nnunet_input_str (str): Column name for nnUNet input.
        clobber (bool): Overwrite existing files if True.
        num_cores (int): Number of CPU cores to use for parallel processing.

    Returns:
        None
    """
    nifti2nnunet_to_do = convert_to_nnunet_list(
        sect_info,
        output_dir,
        nnunet_input_str=nnunet_input_str,
        nnunet_ext=nnunet_ext,
        clobber=clobber,
    )

    Parallel(n_jobs=num_cores)(
        delayed(convert_2d_array_to_nnunet)(
            ii_fn, oo_fn, [pixel_size_0, pixel_size_1], x_scale=x_scale, clobber=clobber
        )
        for ii_fn, pixel_size_0, pixel_size_1, oo_fn in nifti2nnunet_to_do
    )


def get_nnunet_parameters(nnunet_config_json: str, model_dir: str):
    with open(nnunet_config_json, "r") as f:
        nnunet_config = json.load(f)

    x_scale = nnunet_config["x_scale"]
    foreground_labels = nnunet_config["foreground_labels"]
    datasetname = nnunet_config["datasetname"]
    fold = nnunet_config["fold"]
    checkpoint = nnunet_config["checkpoint"]

    nnunet_dataset_json = f"{model_dir}/dataset.json"
    with open(nnunet_dataset_json, "r") as f:
        nnunet_dataset = json.load(f)

    nnunet_ext = nnunet_dataset["file_ending"]

    return x_scale, foreground_labels, datasetname, fold, checkpoint, nnunet_ext

def _check_missing_segmentations(sect_info: pd.DataFrame, output_dir: str, nnunet_input_str: str) -> bool:
    missing_segmentations = False
    for (sub, hemi, chunk), sub_df in sect_info.groupby(
        ["sub", "hemisphere", "chunk"]
    ):
        
        if not check_seg_files(
            sub_df, output_dir, False, nnunet_input_str=nnunet_input_str
        ):
            missing_segmentations = True
    return missing_segmentations

def segment(
    sect_info_csv: str,
    output_dir: str,
    resolution: float,
    model_dir: str = f"{repo_dir}/nnUNet/Dataset501_Brain/nnUNetTrainer__nnUNetPlans__2d/",
    nnunet_config_json: str = f"{repo_dir}/nnUNet/nnunet_config/primate_v1.json",
    output_csv: str = "",
    num_cores: int = 0,
    seg_method: str = "nnunetv1",
    clobber: bool = False,
) -> str:
    """Segment the raw images.

    param: sect_info_csv: csv file with columns: raw, seg_fn
    param: output_dir: directory to save output
    param: resolution: resolution of the raw images
    param: model_dir: directory of the nnunet model
    param: output_csv: csv file to save output
    param: num_cores: number of cores to use
    param: seg_method: use nnunet to segment
    param: clobber: overwrite existing files
    return: csv file with columns: raw, seg_fn
    """
    nnunet_input_str = "img"
    if output_csv == "":
        output_csv = (
            output_dir
            + os.sep
            + os.path.splitext(os.path.basename(sect_info_csv))[0]
            + "_segment.csv"
        )  # check it 'seg' files are all newer than 'img' files

    sect_info = pd.read_csv(sect_info_csv, index_col=False)

   

    sect_info = define_new_path_column(
        sect_info, output_dir, tag=f"{resolution}mm_seg", col="seg", ext='.nii.gz'
    )

    run_nnunet = isinstance(seg_method, str) and "nnunet" in seg_method

    for _, df in sect_info.groupby(["hemisphere", "chunk"]):

        run_stage = utils.check_run_stage(
            df["seg"], df["img"], output_csv, clobber=clobber
        )

        if run_stage:
            # *** CHANGED: nnunet and nnunet_out now live inside each subject folder.
            #     Use output_dir as the base; per-subject subdirs are created inside
            #     convert_nifti_to_nnunet / convert_from_nnunet_list automatically. ***

            num_cores = utils.set_cores(num_cores)

            if run_nnunet:
                (
                        x_scale,
                        foreground_labels,
                        datasetname,
                        fold,
                        checkpoint,
                        nnunet_ext,
                ) = get_nnunet_parameters(nnunet_config_json, model_dir)
                
                convert_nifti_to_nnunet(
                    sect_info,
                    output_dir,
                    nnunet_input_str=nnunet_input_str,
                    x_scale=x_scale,
                    nnunet_ext=nnunet_ext,
                    clobber=clobber,
                    num_cores=num_cores,
                )

            # *** CHANGED: check segmentation files per subject using per-subject nnunet_out dirs ***
            missing_segmentations = _check_missing_segmentations(sect_info, output_dir, nnunet_input_str)   
            
            nnunet_failed = False

            device = "cuda" if torch.cuda.is_available() else "cpu"

            print("\tSegmenting with method:", seg_method)

            # *** CHANGED: run nnunet per subject so each gets its own nnunet/nnunet_out dirs ***
            if missing_segmentations or clobber:
                for (sub, hemi, chunk), sub_df in sect_info.groupby(
                    ["sub", "hemisphere", "chunk"]
                ):
                    nnunet_in_dir = _gen_nnunet_in(output_dir, sub, hemi, chunk)
                    nnunet_out_dir = _gen_nnunet_out(output_dir, sub, hemi, chunk)

                    os.makedirs(nnunet_in_dir, exist_ok=True)
                    os.makedirs(nnunet_out_dir, exist_ok=True)

                    if run_nnunet : 
                        nnunet_failed = run_nnunet_segmentation(
                            seg_method,
                            nnunet_in_dir,
                            nnunet_out_dir,
                            model_dir,
                            nnUNet_dir,
                            datasetname,
                            fold=fold,
                            checkpoint=checkpoint,
                            device=device,
                        )

                if nnunet_failed or seg_method in HISTOGRAM_METHODS:
                    # histogram thresholding fallback if nnUNet fails or if user specified histogram thresholding
                    apply_histogram_threshold(
                        sect_info, num_cores=num_cores, method=seg_method
                    )
                elif (
                    seg_method not in HISTOGRAM_METHODS and "nnunet" not in seg_method
                ):  # No segmentation method specified, use unsegmented images instead
                    sect_info = copy_unsegmented_images(sect_info, output_dir, clobber)

                if run_nnunet and not nnunet_failed:
                    process_nnunet_to_nifti(
                        sect_info,
                        output_dir,
                        nnunet_input_str,
                        num_cores,
                        seg_method,
                        foreground_labels=foreground_labels,
                        clobber=clobber,
                    )

            for _, temp_sect_info in sect_info.groupby(["sub", "hemisphere", "chunk"]):
                assert check_seg_files(
                    temp_sect_info,
                    output_dir,
                    warning_flag=True,
                    nnunet_input_str=nnunet_input_str,
                ), "Missing segmentations"

    sect_info.to_csv(output_csv, index=False)

    return output_csv


def convert_from_nnunet(
    input_fn: str,
    reference_fn: str,
    output_fn: str,
    foreground_labels=[1],
    seg_method: str = "nnunetv1",
) -> None:
    """Convert segmented files from the nnunet output to an easier to use.

    param: input_fn: input filename
    param: reference_fn: reference filename
    param: output_fn: output filename
    param: seg_dir: directory to save output
    return: None
    """
    ref_img = nib.load(reference_fn)

    if ".nii" in input_fn:
        ar = nib.load(input_fn).get_fdata()
    else:
        ar = iio.imread(input_fn)

    def _nnunet(ar, foreground_labels=foreground_labels):
        out = np.zeros_like(ar)

        for label in foreground_labels:
            out[ar == label] = 1

        return out

    if (np.sum(ar == 1) / np.product(ar.shape)) < 0.02:
        logger.info("\nWarning: Found a section that nnUNet failed to segment!\n")
        histogram_threshold(reference_fn, output_fn)
    else:
        if np.sum(ar) == 0:
            logger.info("Error: empty segmented image with nnunet")
            exit(0)

        ar = _nnunet(ar)

        ar = ar.reshape([ar.shape[0], ar.shape[1]])
        # ar = ar.T
        ar = resize(ar, ref_img.shape, order=0)
        ar = np.fliplr(
            np.flipud(ar)
        )  # WARNING: not sure if this will generalize to everyone's data, check to make sure orientation is correct

        if "hybrid" in seg_method:
            logger.info("\tUsing nnUNet hybrid segmentation")
            ar_thr = multi_threshold(ref_img.get_fdata())
            ar = ar + ar_thr
            ar /= ar.max()

        logger.info("\tWriting" + output_fn)

        # scale to 255
        ar = (ar / ar.max() * 255).astype(np.uint8)

        nib.Nifti1Image(ar, ref_img.affine, direction_order="lpi").to_filename(
            output_fn
        )

    return None
