"""Functions to create intermediate GM volumes for use in registration to the structural reference volume."""
import os
import shutil
from glob import glob

import brainbuilder.utils.ants_nibabel as nib
import numpy as np
import pandas as pd
from brainbuilder.align.align_2d import concatenate_sections_to_volume
from brainbuilder.interp.volinterp import (
    create_acq_atlas,
    volumetric_interpolation,
)
from brainbuilder.utils.utils import (
    get_section_intervals,
    get_seg_fn,
    resample_to_resolution,
    simple_ants_apply_tfm,
)
from joblib import Parallel, delayed
from scipy.ndimage import center_of_mass, shift
from scipy.ndimage.morphology import binary_dilation, binary_erosion
from skimage.transform import resize


def get_input_file(
    seg_fn: str,
    seg_rsl_fn: str,
    row: pd.Series,
    output_dir: str,
    resolution_2d: float,
    resolution_3d: float,
    clobber: bool = False,
) -> str:
    """Get the input file for resampling and transformation.

    :param seg_fn: the segmented file name
    :param seg_rsl_fn: the resampled segmented file name
    :param row: the row of the dataframe
    :param output_dir: the output directory
    :param resolution_2d: the 2D resolution
    :param resolution_3d: the 3D resolution
    :param clobber: whether to overwrite existing files
    :return: the input file for resampling and transformation
    """
    tfm_input_fn = seg_rsl_fn
    if not os.path.exists(seg_rsl_fn):
        resample_to_resolution(
            seg_fn,
            [resolution_2d] * 2,
            seg_rsl_fn,
            dtype=np.uint8,
        )

    if resolution_2d != resolution_3d:
        tfm_input_fn = get_seg_fn(
            output_dir, int(row["sample"]), resolution_3d, seg_fn, "_rsl"
        )
        if not os.path.exists(tfm_input_fn) or clobber:
            resample_to_resolution(
                seg_fn,
                [resolution_3d] * 2,
                tfm_input_fn,
                dtype=np.uint8,
            )
    return tfm_input_fn


def resample_and_transform(
    output_dir: str,
    resolution_itr: int,
    resolution_2d: float,
    resolution_3d: float,
    row: pd.Series,
    tfm_ref_fn: str,
    clobber: bool = False,
) -> pd.Series:
    """Resamples and transforms the segmented images to the current resolution.

    :param output_dir: output directory
    :param resolution_itr: current resolution iteration
    :param resolution_2d: current 2D resolution
    :param resolution_3d: current 3D resolution
    :param row: row of the dataframe
    :param tfm_ref_fn: reference transform file
    :param recenter_image: whether to recenter the image
    :param clobber: whether to overwrite existing files
    :return: None
    """
    seg_fn = row["seg"]

    seg_rsl_fn = get_seg_fn(
        output_dir, int(row["sample"]), resolution_2d, seg_fn, "_rsl"
    )
    seg_rsl_tfm_fn = get_seg_fn(
        output_dir, int(row["sample"]), resolution_3d, seg_fn, "_rsl_tfm"
    )
    if not os.path.exists(seg_rsl_tfm_fn) or clobber:
        tfm_input_fn = get_input_file(
            seg_fn, seg_rsl_fn, row, output_dir, resolution_2d, resolution_3d
        )

        if resolution_itr == 0:
            tfm_ref_fn = tfm_input_fn

        # get initial rigid transform
        tfm_fn = row["2d_tfm"]
        print("\tTransforming", seg_rsl_fn, "to", seg_rsl_tfm_fn)
        print("\t\twith:", tfm_fn, "\n")
        if isinstance(tfm_fn, str):
            simple_ants_apply_tfm(
                tfm_input_fn,
                tfm_ref_fn,
                tfm_fn,
                seg_rsl_tfm_fn,
                ndim=2,
                n="NearestNeighbor",
                empty_ok=True,
            )
        else:
            print("\tNo transform for", seg_rsl_fn)
            shutil.copy(tfm_input_fn, seg_rsl_tfm_fn)

    row["nl_2d_cls_rsl"] = seg_rsl_tfm_fn
    row["nl_2d_rsl"] = seg_rsl_tfm_fn

    return row


def resample_transform_segmented_images(
    sect_info: pd.DataFrame,
    resolution_itr: int,
    resolution_2d: float,
    resolution_3d: float,
    output_dir: str,
    num_cores: int = 0,
    clobber: bool = False,
) -> None:
    """Resample and transform segmented images to 3D create a 3D GM classification volume.

    :param sect_info: dataframe with information about the section
    :param chunk_info: dataframe with information about the chunk
    :param resolution_itr: resolution iteration
    :param resolution_2d: 2D resolution
    :param resolution_3d: 3D resolution
    :param output_dir: output directory
    return: None
    """
    os.makedirs(output_dir, exist_ok=True)
    os.uname()

    tfm_ref_fn = output_dir + "/2d_reference_image.nii.gz"

    if not os.path.exists(tfm_ref_fn) and resolution_itr != 0:
        ref_img = nib.load(sect_info["nl_2d_rsl"].values[0])
        xstart, zstart = ref_img.affine[[0, 1], 3]

        resample_to_resolution(
            sect_info["nl_2d_rsl"].values[0],
            [resolution_3d] * 2,
            tfm_ref_fn,
            order=0,
        )

    results = Parallel(n_jobs=num_cores, backend="multiprocessing")(
        delayed(resample_and_transform)(
            output_dir,
            resolution_itr,
            resolution_2d,
            resolution_3d,
            row,
            tfm_ref_fn,
            clobber=clobber,
        )
        for i, row in sect_info.iterrows()
    )

    sect_info = pd.DataFrame(results)

    return sect_info


def interpolate_missing_sections(
    vol: np.array, method="linear", dilate_volume: bool = False
) -> np.array:
    """Interpolates missing sections in a volume.

    :param vol (ndarray): The input volume.
    :dilate_volume (bool, optional): Whether to dilate the volume before interpolation. Defaults to False.
    :return ndarray: The volume with missing sections interpolated.
    """
    if dilate_volume:
        vol_dil = binary_erosion(
            binary_dilation(vol, iterations=4), iterations=4
        ).astype(int)
    else:
        vol_dil = vol

    intervals = get_section_intervals(vol_dil)

    out_vol = vol.copy()
    for i in range(len(intervals) - 1):
        j = i + 1
        x0, x1 = intervals[i]  # intervals of consecutive sections
        y0, y1 = intervals[j]  # intervals of consecutive sections
        x = np.mean(
            vol[:, x0:x1, :], axis=1
        )  # x is the average of the last consecutive acquired sections
        y = np.mean(
            vol[:, y0:y1, :], axis=1
        )  # y is the average of the next consecutive acquired sections
        vol[:, x0:x1, :] = np.repeat(
            x.reshape(x.shape[0], 1, x.shape[1]), x1 - x0, axis=1
        )
        for ii in range(x1, y0):
            den = y0 - x1
            assert den != 0, "Error: 0 denominator when interpolating missing sections"
            d = float(ii - x1) / den

            if method == "nearest":
                d = np.rint(d)
            print(d)
            z = x * (1 - d) + d * y
            # print(x1,d,ii,y0, '-->', np.mean(x), np.mean(z), np.mean(y))

            out_vol[:, ii, :] = z
        print()

    return out_vol


def recenter(
    vol: np.array, affine: np.array, direction: np.array = np.array([1, 1, -1])
) -> tuple:
    """Recenter the volume.

    :param vol: the volume
    :param affine: the affine
    :param direction: the direction
    :return: the recentered volume and affine
    """
    affine = np.array(affine)

    vol_sum_1 = np.sum(np.abs(vol))
    assert vol_sum_1 > 0, "Error: input volume sum is 0 in recenter"

    ndim = len(vol.shape)
    vol[pd.isnull(vol)] = 0
    coi = np.array(vol.shape) / 2
    com = center_of_mass(vol)
    d_vox = np.rint(coi - com)
    d_vox[1] = 0
    d_world = d_vox * affine[range(ndim), range(ndim)]
    d_world *= direction
    affine[range(ndim), 3] -= d_world

    print("\tShift in Segmented Volume by:", d_vox)
    vol = shift(vol, d_vox, order=0)

    affine[range(ndim), range(ndim)]
    return vol, affine


def load_2d_sections_to_volume(sect_info, in_dir, resolution_3d, dims):
    data = np.zeros(
        dims,
        dtype=np.float32,
    )

    for i, row in sect_info.iterrows():
        s0 = int(row["sample"])
        fn = get_seg_fn(
            in_dir,
            int(row["sample"]),
            resolution_3d,
            row["seg"],
            "_rsl_tfm",
        )
        img_2d = nib.load(fn).get_fdata()

        # FIXME This is not a good way to solve issue with rsl_tfm files being the wrong size. Problem is probably in the use of nibabel's resampling function in resample
        if img_2d.shape[0] != data.shape[0] or img_2d.shape[1] != data.shape[2]:
            img_2d = resize(img_2d, [data.shape[0], data.shape[2]], order=0)

        data[:, s0, :] = img_2d

    return data


def create_intermediate_volume(
    chunk_info: pd.DataFrame,
    sect_info: pd.DataFrame,
    resolution_itr: int,
    resolution: float,
    resolution_list: list,
    resolution_3d: float,
    out_dir: str,
    seg_rsl_fn: str,
    init_align_fn: str,
    interpolation: str = "nearest",
    num_cores: int = 0,
    clobber: bool = False,
) -> None:
    """Create intermediate volume for use in registration to the structural reference volume.

    param: sect_info: dataframe containing information about each section
    param: chunk_info: dataframe containing information about each chunk
    param: resolution_itr: current resolution iteration
    param: resolution: current resolution
    param: resolution_3d: current 3d resolution
    param: out_dir: output directory
    param: seg_rsl_fn: filename of the resampled segmented volume
    param: init_align_fn: filename of the initial alignment volume
    return: None
    """
    out_2d_dir = out_dir + "/2d/"

    os.makedirs(out_2d_dir, exist_ok=True)

    print("\t\tStep 2: Autoradiograph segmentation")
    if not os.path.exists(seg_rsl_fn) or clobber:
        print("\t\t\tResampling segemented sections")

        sect_info = resample_transform_segmented_images(
            sect_info,
            resolution_itr,
            resolution,
            resolution_3d,
            out_2d_dir,
            num_cores=num_cores,
            clobber=clobber,
        )

        # write 2d segmented sections at current resolution. apply initial transform
        print("\t\t\tInterpolating between segemented sections")
        # volumetric_interpolation(
        #    sect_info,
        #    init_align_fn,
        #    out_dir + "/2d/",
        #    out_dir,
        #    seg_rsl_fn,
        #    resolution_3d,
        #    chunk_info["section_thickness"].values[0],
        #    interpolation=interpolation,
        #    clobber=clobber,
        # )

        curr_align_fn = (
            out_dir
            + "/"
            + os.path.basename(seg_rsl_fn).replace(".nii.gz", "_rsl_2d.nii.gz")
        )

        img = nib.load(init_align_fn)

        example_2d_list = glob(f"{out_2d_dir}/*{resolution_3d}*rsl_tfm.nii.gz")

        assert len(example_2d_list) > 0, "Error: no files found in {}".format(
            out_2d_dir
        )

        # Example image should be at maximum 2D resolution
        example_2d_img = nib.load(example_2d_list[0])

        dims = [example_2d_img.shape[0], img.shape[1], example_2d_img.shape[1]]

        affine = img.affine.copy()
        affine[0, 0] = resolution_3d
        affine[1, 1] = img.affine[1, 1]
        affine[2, 2] = resolution_3d

        concatenate_sections_to_volume(
            sect_info, "nl_2d_rsl", curr_align_fn, dims, affine
        )

        chunk_info["nl_2d_vol_fn"] = curr_align_fn

        chunk_info = volumetric_interpolation(
            sect_info,
            chunk_info,
            curr_align_fn,
            out_dir,
            resolution,
            resolution_list,
            clobber=clobber,
        )

        print("Create Acquisition Atlas")
        create_acq_atlas(chunk_info, out_dir, seg_rsl_fn, clobber=clobber)

    return None
