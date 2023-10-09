import argparse
import os
from glob import glob

import numpy as np
import pandas as pd
import re 

import SimpleITK as sitk
from joblib import Parallel, delayed
from nibabel.processing import *
from scipy.ndimage import gaussian_filter
from skimage.filters import threshold_li
from skimage.transform import resize

import brainbuilder.utils.ants_nibabel as nib
from brainbuilder.utils import utils

base_file_dir, fn = os.path.split(os.path.abspath(__file__))
repo_dir = f"{base_file_dir}/../"


def histogram_threshold(raw_fn: str, seg_fn: str, sd: float = 1):
    """
    Apply histogram thresholding to the cropped images
    param: raw_fn: raw image filename
    param: seg_fn: segmentation filename
    param: sd: standard deviation
    return: None
    """

    img = nib.load(raw_fn)

    affine = img.affine
    dimensions = img.shape

    ar = img.get_fdata()
    assert np.sum(np.abs(ar)) > 0, "Error: empty input image with histogram thresholding " + raw_fn
    ar = gaussian_filter(ar, sd).astype(np.float64)
    out = np.zeros(ar.shape)
    values = ar[ar > np.min(ar)]
    idx = ar > threshold_li(values)

    out[idx] = 1

    out = resize(ar, dimensions, order=0)
    if len(out.shape) == 3:
       out = out.reshape([out.shape[0], out.shape[1]]) 

    assert np.sum(np.abs(out)) > 0, "Error: empty segmented image with histogram thresholding " + raw_fn
    nib.Nifti1Image(out, affine, direction_order="lpi").to_filename(seg_fn)

    return out


typeDataFrame = type(pd.DataFrame({}))

base_file_dir, fn = os.path.split(os.path.abspath(__file__))


def convert_2d_array_to_nifti(
    input_filename: str,
    output_filename: str,
    res: list ,
    spacing: tuple = (1, 1, 1),
    clobber = False
) -> None:
    """
    Converts numpy into a series of niftis.
    The image can have an arbitrary number of input channels which will be exported separately (_0000.nii.gz,
    _0001.nii.gz, etc for images and only .nii.gz for seg).
    Spacing can be ignored most of the time.
    !!!2D images are often natural images which do not have a voxel spacing that could be used for resampling. These images
    must be resampled by you prior to converting them to nifti!!!
    Datasets converted with this utility can only be used with the 2d U-Net configuration of nnU-Net
    If Transform is not None it will be applied to the image after loading.
    Segmentations will be converted to np.uint32!
    :param is_seg:
    :param input_array:
    :param output_filename_truncated: do not use a file ending for this one! Example: output_name='./converted/image1'. This function will add the suffix (_0000) and file ending (.nii.gz) for you.
    :param spacing:
    :return:
    """
    if not os.path.exists(output_filename) or not utils.newer_than(output_filename, input_filename)  or clobber:

        aff=np.eye(4)
        aff[0,0] = res[0]
        aff[1,1] = res[1]
        nii_img = utils.resample_to_resolution(input_filename, [.2,.2], affine=aff, order=1)
        
        img = nii_img.get_fdata()

        img = np.rot90(np.fliplr(img),-1)
        
        assert len(img.shape) == 2

        img = img[None, None]  # add dimensions
        # image is now (c, y, x, z) where x=1 since it's 2d
        img = img.astype(np.float32)

        assert np.sum(np.abs(img)) > 0

        for j, i in enumerate(img):

            itk_img = sitk.GetImageFromArray(i)
            itk_img.SetSpacing(list(spacing)[::-1])
            # sitk.WriteImage(itk_img, output_filename_truncated + "_%04.0d.nii.gz" % j)
            sitk.WriteImage(itk_img, output_filename)
            #nib.Nifti1Image(i, nii_img.affine).to_filename(output_filename)

        print("Wrote:", output_filename)


def assign_seg_filenames(df: typeDataFrame, output_dir: str):
    

    df["seg"] = df["raw"].apply(
        lambda fn: utils.gen_new_filename(fn, output_dir, "_seg.nii.gz")
    )
    return df


def apply_histogram_threshold(sect_info: typeDataFrame, num_cores: int = 1) -> None:
    """
    Apply histogram threshold to the raw images
    param: sect_info: dataframe with columns: raw, seg_fn
    return: dataframe with columns: raw, seg_fn
    """
    Parallel(n_jobs=num_cores)(
        delayed(histogram_threshold)(row["raw"], row["seg"])
        for i, row in sect_info.iterrows()
    )
    return None


def convert_from_nnunet_list(
    sect_info: typeDataFrame, 
    nnunet_out_dir: str, 
    warning_flag: bool = False,
    clobber: bool = False
) -> list:
    """
    Convert the nnunet images to regular nifti images
    param: sect_info: dataframe with columns: img, seg_fn
    param: nnunet_out_dir: directory to save nnunet images
    param: clobber: overwrite existing files
    return: list of files to convert
    """
    to_do = []
    for i, row in sect_info.iterrows():
        img_fn = row["img"]
        raw_fn = row["raw"]
        seg_fn = row["seg"]
       
        if '.nii.gz' in img_fn :
            base = re.sub('.nii.gz', '', raw_fn) 
        else:
            base = os.path.splitext(raw_fn)[0]
        base = os.path.basename( base )

        nnunet_list = glob(f"{nnunet_out_dir}/{base}*")

        print('nnunet_fn', f"{nnunet_out_dir}/{base}")
        print('seg_fn', seg_fn)
        nnunet_fn = nnunet_list[0]

        print()
        if not os.path.exists(seg_fn) or not utils.newer_than(seg_fn, img_fn) or clobber:
            if warning_flag : 
                print('\tWarning: Could not find file:', seg_fn)

            to_do.append((nnunet_fn, img_fn, seg_fn))
    return to_do


def convert_to_nnunet_list(
    chunk_info: typeDataFrame,
    sect_info: typeDataFrame,
    nnunet_in_dir: str, 
    clobber: bool = False
    ) -> list:
    """
    Convert the raw images to nnunet format
    param: sect_info: dataframe with columns: raw, seg_fn
    param: nnunet_in_dir: directory to save nnunet images
    param: clobber: overwrite existing files
    return: list of files to convert
    """
    to_do = []
    for i, row in sect_info.iterrows():
        f = row["raw"]

        idx =   (chunk_info['sub'] == row['sub']) &\
                (chunk_info['hemisphere'] == row['hemisphere']) & \
                (chunk_info['chunk'] == row['chunk'])

        pixel_size_0, pixel_size_1, _ = utils.get_chunk_pixel_size(
            row["sub"], row["hemisphere"], row["chunk"], chunk_info
        )

        fname = os.path.split(f)[1].split(".")[0]
        output_filename_truncated = os.path.join(nnunet_in_dir, fname)
        output_filename = output_filename_truncated + "_0000.nii.gz"
        if not os.path.exists(output_filename) or not utils.newer_than(output_filename, f) or clobber:
            to_do.append([f, pixel_size_0, pixel_size_1, output_filename])

    return to_do


def check_seg_files(sect_info, warning_flag=False):
    for i, row in sect_info.iterrows():
        if not os.path.exists(row["seg"]):
            if warning_flag:
                print('\tWarning: Could not find file:', row["seg"])
            return False
    return True 

def segment(
    chunk_info_csv:str,
    sect_info_csv: str,
    output_dir: str,
    resolution: float,
    model_dir: str = f"{repo_dir}/nnUNet/Dataset501_Brain/nnUNetTrainer__nnUNetPlans__2d/",
    output_csv: str = "",
    num_cores: int = 0,
    use_nnunet: bool = True,
    clobber: bool = False,
) -> str:
    """
    Segment the raw images
    param: sect_info_csv: csv file with columns: raw, seg_fn
    param: output_dir: directory to save output
    param: resolution: resolution of the raw images
    param: model_dir: directory of the nnunet model
    param: output_csv: csv file to save output
    param: num_cores: number of cores to use
    param: use_nnunet: use nnunet to segment
    param: clobber: overwrite existing files
    return: csv file with columns: raw, seg_fn
    """

    if output_csv == "":
        output_csv = (
            output_dir
            + os.sep
            + os.path.splitext(os.path.basename(sect_info_csv))[0]
            + "_segment.csv"
        )

    # check it 'seg' files are all newer than 'img' files 

    run_stage = utils.check_run_stage('seg', 'img',output_csv, clobber=clobber)

    if run_stage :
        nnunet_in_dir = f"{output_dir}/nnunet/"
        nnunet_out_dir = f"{output_dir}/nnunet_out/"
        os.makedirs(nnunet_in_dir, exist_ok=True)
        os.makedirs(nnunet_out_dir, exist_ok=True)

        num_cores = utils.set_cores(num_cores)

        sect_info = pd.read_csv(sect_info_csv, index_col=False)
        chunk_info = pd.read_csv(chunk_info_csv, index_col=False)

        sect_info = assign_seg_filenames(sect_info, output_dir)

        nifti2nnunet_to_do = convert_to_nnunet_list(
            chunk_info, sect_info, nnunet_in_dir, clobber=clobber
        )

        Parallel(n_jobs=num_cores)(
            delayed(convert_2d_array_to_nifti)(ii_fn, oo_fn, [pixel_size_0, pixel_size_1], clobber = clobber)
            for ii_fn, pixel_size_0, pixel_size_1, oo_fn in nifti2nnunet_to_do
        )

        missing_segmentations = not check_seg_files(sect_info, True) 

        use_nnunet = False
        if missing_segmentations or clobber :
            print("\tSegmenting with nnUNet")
            try:
                utils.shell(
                    f"nnUNetv2_predict_from_modelfolder --c --verbose -i {nnunet_in_dir} -o {nnunet_out_dir} -m {model_dir} -f 0  -d Dataset501_Brain -device cpu"
                )
                use_nnunet = True
            except:
                apply_histogram_threshold(sect_info, num_cores=num_cores)


        nnunet2nifti_to_do = convert_from_nnunet_list(
            sect_info, 
            nnunet_out_dir,
            warning_flag=False,
            clobber=clobber
        )

        if use_nnunet and len(nnunet2nifti_to_do) > 0:
            print("\tConvert Files from nnUNet to standard nifti files")

            Parallel(n_jobs=num_cores)(
                delayed(convert_from_nnunet)(nnunet_fn, raw_fn, seg_fn, output_dir)
                for nnunet_fn, raw_fn, seg_fn in nnunet2nifti_to_do
            )

        assert check_seg_files(sect_info, warning_flag=True), "Missing segmentations"

        sect_info.to_csv(output_csv, index=False)


    return output_csv


def convert_from_nnunet(input_fn: str, reference_fn: str, output_fn: str, seg_dir: str):
    """
    convert segmented files from the nnunet output to an easier to use

    """
    ref_img = nib.load(reference_fn)
    ar = nib.load(input_fn).get_fdata()

    if (np.sum(ar == 1) / np.product(ar.shape)) < 0.02:
        print("\nWarning: Found a section that nnUNet failed to segment!\n")
        histogram_threshold(reference_fn, output_fn)
    else:
        if np.sum(ar) == 0:
            print("Error: empty segmented image with nnunet")
            exit(0)
        ar[(ar == 3) | (ar == 4)] = 1

        gm = ar == 1
        wm = ar == 2

        ar *= 0
        ar[gm] = 1

        ar = ar.reshape([ar.shape[0], ar.shape[1]])
        #ar = ar.T
        ar = resize(ar, ref_img.shape, order=0)

        print("\tWriting", output_fn)
        nib.Nifti1Image(ar, ref_img.affine).to_filename(output_fn)

    return None


def get_args():
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument(
        "sect_info_csv", type=str, help=".csv with information about the sections"
    )
    # parser.add_argument('chunk_info_csv', type=str, help='.csv with information about the chunks')
    parser.add_argument(
        "output_dir",
        type=str,
        help="output directory where segmented images will be kept",
    )
    parser.add_argument(
        "resolution", type=float, help="max spatial resolution of reconstruction"
    )

    parser.add_argument(
        "--output-csv",
        dest="output_csv",
        default="",
        help='Path to output dataframe in .csv format (Default: store input ".csv" as "_segment.csv" in output directory)',
    )
    parser.add_argument(
        "--pytorch-model",
        "-m",
        dest="pytorch_model_dir",
        default=f"{repo_dir}/caps/nnUNet_results/nnUNet/2d/Task502_cortex",
        help="Numer of cores to use for segmentation (Default=0; This is will set the number of cores to use to the maximum number of cores availale)",
    )
    parser.add_argument(
        "--cores",
        "-n",
        dest="num_cores",
        default=0,
        help="Numer of cores to use for segmentation (Default=0; This is will set the number of cores to use to the maximum number of cores availale)",
    )
    parser.add_argument(
        "--clobber",
        "-c",
        dest="clobber",
        default=False,
        action="store_true",
        help="Overwrite existing results",
    )
    parser.parse_args()


if __name__ == "__main__":
    # create binary cortical segmentations
    df = segment(
        args.sect_info_csv,
        args.output_dir,
        args.resolution,
        args.model_dir,
        output_csv=args.output_csv,
        num_cores=args.num_cores,
    )
