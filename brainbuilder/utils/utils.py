"""Utility functions for brainbuilder."""

import contextlib
import logging
import multiprocessing
import os
import re
from collections.abc import Iterable
from subprocess import PIPE, STDOUT, Popen
from typing import List, Optional, Tuple, Union

import ants
import imageio
import nibabel
import numpy as np
import pandas as pd
import psutil
from scipy.ndimage import label
from skimage.filters import threshold_otsu
from skimage.transform import resize

import brainbuilder.utils.ants_nibabel as nib

os_info = os.uname()


LOG_VERBOSITY_LEVEL = logging.INFO  # Default logging level


def get_logger(name="brainbuilder"):
    logger = logging.getLogger(name)
    if not logger.hasHandlers():
        # Prevent duplicated messages if called multiple times
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(LOG_VERBOSITY_LEVEL)
    return logger


def convert_gm_com_dist_map(vol: np.ndarray, voxel_sizes: np.ndarray) -> np.ndarray:
    """Convert the volume to a distance map from the center of mass.

    :param vol: np.ndarray, volume
    :param voxel_sizes: np.ndarray, voxel size
    :return: np.ndarray, distance map
    """
    assert len(vol.shape) == 3, "Error: volume must be 3D"
    assert np.sum(vol) > 0, "Error: volume must not be empty"

    idx = vol > vol.min()

    # Get the center of mass
    com = np.array(np.where(idx)).mean(axis=1)

    # Create a grid of coordinates
    coords = np.array(
        np.meshgrid(
            np.arange(vol.shape[0]),
            np.arange(vol.shape[1]),
            np.arange(vol.shape[2]),
            indexing="ij",
        )
    )

    coords = coords.reshape(3, -1).T

    idx_rsl = idx.reshape(-1)

    coords = coords[idx_rsl, :]

    # Calculate the distance from the center of mass
    distances = np.linalg.norm(coords - com, axis=1)

    # Scale distances in each dimension by the array of voxel size
    # distances = distances * voxel_sizes

    # Rescale distances between [0,1] in each dimension
    distances = distances / np.max(distances, axis=0)

    # Create a new volume with the same shape as the input volume
    new_vol = np.zeros_like(vol, dtype=np.float32)

    vol = (vol - vol.min()) / (vol.max() - vol.min())

    new_vol[idx] = 1 + vol[idx] * distances

    return new_vol


def get_available_memory() -> int:
    """Get the available memory in bytes.

    :return: int
    """
    return psutil.virtual_memory()[1]


def estimate_memory_usage(n_elements: int, n_bytes_per_element: int) -> int:
    """Estimate the memory usage for a given number of elements and bytes per element.

    :param n_elements: int, number of elements
    :param n_bytes_per_element: int, number of bytes per element
    :return: int
    """
    return n_elements * n_bytes_per_element


def get_maximum_cores(
    n_elemnts_list: list, n_bytes_per_element_list: list, max_memory: float = 0.5
) -> int:
    """Get the maximum number of cores to use for a given memory limit.

    :param n_elemnts: int, number of elements
    :param n_bytes_per_element: int, number of bytes per element
    :param max_memory: float, maximum memory to us
    :return: int
    """
    available_memory = get_available_memory()
    estimated_memory = 0
    for n_elemnts, n_bytes_per_element in zip(n_elemnts_list, n_bytes_per_element_list):
        estimated_memory += estimate_memory_usage(n_elemnts, n_bytes_per_element)
    max_cores = int(available_memory / estimated_memory)
    available_memory_gb = np.round(available_memory / 1024 / 1024 / 1024, 3)
    estimated_memory_gb = np.round(estimated_memory / 1024 / 1024 / 1024, 3)

    total_cores = multiprocessing.cpu_count()
    max_cores = max([0, min([max_cores, total_cores])])

    print(
        "\tAvailable memory (Gb): ",
        available_memory_gb,
        "Estimated memory: ",
        estimated_memory_gb,
        "Max cores: ",
        max_cores,
    )
    return max_cores


def load_image(fn: str) -> np.ndarray:
    """Load an image from a file.

    :param fn: str, filename
    :return: np.ndarray
    """
    if isinstance(fn, str) and os.path.exists(fn) or os.path.islink(fn):
        if os.path.islink(fn):
            fn = os.readlink(fn)

        if ".nii" in fn:
            return ants.image_read(fn).numpy()
        else:
            return imageio.imread(fn)
    else:
        return None


def check_dimensions(fns: list, dims: tuple) -> None:
    """Check the dimensions of a list of files.

    :param fns: list, list of filenames
    :param dims: tuple, expected dimensions
    :return: None
    """
    sections_okay_flag = True
    for fn in fns:
        sec = nibabel.load(fn)

        if sec.shape[0] != dims[0] or sec.shape[1] != dims[2]:
            print("Warning: section dimensions do not match target dimensions")
            print("\tSection:", fn, sec.shape)
            print("\tTarget:", dims)
            print()

            sections_okay_flag = False

    return sections_okay_flag


def concatenate_sections_to_volume(sect_info, target_name, out_fn, dims, affine):
    """Process sections and write the 3D volume to a file."""
    if not os.path.exists(out_fn):
        out_vol = np.zeros(dims, dtype=np.float32)

        sections_okay_flag = check_dimensions(sect_info[target_name].values, dims)

        if not sections_okay_flag:
            print("Error: section dimensions do not match target dimensions")
            exit(1)

        exit_flag = False
        for i, row in sect_info.iterrows():
            fn = sect_info[target_name].loc[i]
            y = int(row["sample"])

            try:
                sec = nibabel.load(fn).get_fdata()

                sec_x = sec.shape[0]
                sec_z = sec.shape[1]

                dim_x = dims[0]
                dim_z = dims[2]
                # if the difference of section dimensions is more than 1 pixel but less than 10 pixels, resize the section to match the target dimensions
                # FIXME: this is a temporary fix for some datasets with slight mismatched section dimensions
                d0 = np.abs(sec_x - dim_x)
                d1 = np.abs(sec_z - dim_z)
                if (d0 > 0 and d0 < 10) or (d1 > 0 and d1 < 10):
                    sec = resize(
                        sec,
                        (dim_x, dim_z),
                        order=1,
                        preserve_range=True,
                        anti_aliasing=False,
                    )

                out_vol[:, int(y), :] = sec
            except EOFError:
                print("Error:", fn)
                os.remove(fn)
                exit_flag = True

            if exit_flag:
                exit(1)

        print("\t\tWriting 3D non-linear:", out_fn)

        nib.Nifti1Image(
            out_vol, affine, dtype=np.float32, direction_order="lpi"
        ).to_filename(out_fn)


def get_chunk_pixel_size(sub: str, hemi: str, chunk: str, chunk_info: str) -> tuple:
    """Get the pixel size of a chunk.

    :param sub: str, name of the subject
    :param hemi: str, hemisphere
    :param chunk: int, chunk number
    :param chunk_info: pd.DataFrame, chunk info dataframe
    :return: tuple
    """
    idx = (
        (chunk_info["sub"] == sub)
        & (chunk_info["hemisphere"] == hemi)
        & (chunk_info["chunk"] == chunk)
    )

    pixel_size_0 = None
    pixel_size_1 = None
    section_thickeness = None

    try:
        pixel_size_0 = chunk_info["pixel_size_0"][idx].values[0]
    except IndexError:
        pass

    try:
        pixel_size_1 = chunk_info["pixel_size_1"][idx].values[0]
    except IndexError:
        pass

    try:
        section_thickeness = chunk_info["section_thickness"][idx].values[0]
    except IndexError:
        pass

    return pixel_size_0, pixel_size_1, section_thickeness


def create_2d_affine(
    pixel_size_0: float, pixel_size_1: float, section_thickness: float
) -> np.ndarray:
    """Create a 2D affine.

    :param pixel_size_0: float
    :param pixel_size_1: float
    :param section_thickness: float
    :return: np.ndarray
    """
    affine = np.eye(4)
    affine[0, 0] = pixel_size_0
    affine[1, 1] = pixel_size_1
    affine[2, 2] = section_thickness
    return affine


def get_chunk_direction(
    sub: str, hemi: str, chunk: int, chunk_info: pd.DataFrame
) -> list:
    """Get the direction of a chunk.

    :param sub: str, name of the subject
    :param hemi: str, hemisphere
    :param chunk: int, chunk number
    :param chunk_info: pd.DataFrame, chunk info dataframe
    :return: list
    """
    idx = (
        (chunk_info["sub"] == sub)
        & (chunk_info["hemisphere"] == hemi)
        & (chunk_info["chunk"] == chunk)
    )

    direction = chunk_info["direction"][idx].values[0]

    return direction


def set_cores(num_cores: int) -> int:
    """Sets the number of cores to use for parallel processing.

    :param num_cores: int
    :return: int
    """
    if num_cores == 0:
        num_cores = multiprocessing.cpu_count()

    return num_cores


def get_thicken_width(
    resolution: float, section_thickness: float = 0.02, scale: float = 1
) -> int:
    """Get the thicken width.

    :param resolution: float, resolution
    :param section_thickness: float, section thickness
    :param scale : float
    :return: float, scale the width
    """
    return np.round(scale * (1 + float(resolution) / (section_thickness * 2))).astype(
        int
    )


def get_section_intervals(vol: np.ndarray) -> list:
    """Get the intervals of sections within a volume across y-axis of volume.

    :param vol: np.array, volume
    :return: list
    """
    section_sums = np.sum(vol, axis=(0, 2))
    valid_sections = section_sums > np.min(section_sums)
    labeled_sections, nlabels = label(valid_sections)
    if nlabels < 2:
        print(
            "Error: there must be a gap between thickened sections. Use higher resolution volumes."
        )

    intervals = [
        (
            np.where(labeled_sections == i)[0][0],
            np.where(labeled_sections == i)[0][-1] + 1,
        )
        for i in range(1, nlabels + 1)
    ]

    assert len(intervals) > 0, "Error: no valid intervals found for volume."
    return intervals


class AntsParams:
    """Class representing Ants parameters. Parameters are calculated based on a multi-resolution alignment schedule."""

    def __init__(
        self: object,
        resolution_list: List[float],
        resolution: float,
        base_itr: int,
        max_resolution: Optional[float] = None,
        verbose: bool = False,
    ) -> None:
        """Initialize the AntsParams object.

        :param resolution_list: List[float], list of resolutions
        :param resolution: float, resolution
        :param base_itr: int, base iteration
        :param max_resolution: Optional[float], maximum resolution (default: None)
        :param verbose: bool, verbose flag (default: False)
        :return: None
        """
        if type(max_resolution) == type(None):
            self.max_resolution = resolution
        else:
            self.max_resolution = max_resolution

        if len(resolution_list) == 0:
            return None

        self.resolution_list = resolution_list
        self.max_n = len(resolution_list)
        self.cur_n = resolution_list.index(resolution)
        self.max_itr = int((self.max_n + 1) * base_itr)

        self.f_list = self.gen_downsample_factor_list(resolution_list)
        self.max_downsample_factor = int(self.f_list[0])

        self.f_str = self.get_downsample_params(resolution_list)
        self.s_list = self.gen_smoothing_factor_list(resolution_list)
        self.s_str = self.get_smoothing_params(resolution_list)
        self.itr_str = self.gen_itr_str(self.max_itr, base_itr)

        if verbose:
            self._print()

        assert (
            len(self.f_list) == len(self.s_list) == len(self.itr_str.split("x"))
        ), f"Error: incorrect number of elements in: \n{self.f_list}\n{self.s_list}\n{self.itr_str}\n{resolution_list}"

    def print(self) -> None:
        """Print the AntsParams object.

        :return: None
        """
        print("Iterations:\t", self.itr_str)
        print("Factors:\t", self.f_str)
        print("Smoothing:\t", self.s_str)
        return None

    def gen_itr_str(self, max_itr: int, step: float) -> str:
        """Generate the iteration string.

        :param max_itr: int, maximum iteration
        :param step: float, step size
        :return: str
        """
        itr_str = "x".join(
            [
                str(int(max_itr - i * step))
                for i in range(self.cur_n + 1)
                if self.resolution_list[i] >= self.max_resolution
            ]
        )
        itr_str = "[" + itr_str + ",1e-7,20 ]"
        return itr_str

    def gen_smoothing_factor_string(self, lst: List) -> str:
        """Generate the smoothing factor string.

        :param lst: List[float], list of smoothing factors
        :return: str, the generated smoothing factor string
        """
        return "x".join([str(i) for i in lst]) + "vox"

    def gen_smoothing_factor_list(self, resolution_list: List[float]) -> List[float]:
        """Generate the list of smoothing factors.

        :param resolution_list: List[float], list of resolutions
        :return: List[float]
        """

        def smooth_f(x: float, y: float) -> float:
            """Calculate the smoothing factor.

            :param x: float, x
            :param y: float, y
            :return: float
            """
            return np.round(float(float(x) / float(y) - 1) * 0.2, 3)

        return [
            smooth_f(resolution_list[i], resolution_list[self.cur_n])
            if i != self.cur_n
            else 0
            for i in range(self.cur_n + 1)
            if self.resolution_list[i] >= self.max_resolution
        ]

    def get_smoothing_params(self, resolution_list: List[float]) -> str:
        """Generate the smoothing factor string.

        :param resolution_list: List[float], list of resolutions
        :return: str
        """
        s_list = self.gen_smoothing_factor_list(resolution_list)
        return self.gen_smoothing_factor_string(s_list)

    def calc_downsample_factor(self, cur_res: float, image_res: float) -> str:
        """Calculate the downsample factor.

        :param cur_res: float, current resolution
        :param image_res: float, image resolution
        :return: str
        """
        return (
            np.rint(float(cur_res) / float(image_res)).astype(int).astype(str)
        )  # DEBUG
        # return np.floor(1+np.log2(float(cur_res)/float(image_res))).astype(int).astype(str)

    def gen_downsample_factor_list(self, resolution_list: List[float]) -> List[str]:
        """Generate the list of downsample factors.

        :param resolution_list: List[float], list of resolutions
        :return: List[str]
        """
        factors = [
            self.calc_downsample_factor(resolution_list[i], resolution_list[self.cur_n])
            for i in range(self.cur_n + 1)
            if self.resolution_list[i] >= self.max_resolution
        ]
        return factors

    def get_downsample_params(self, resolution_list: List[float]) -> str:
        """Generate the downsample factor string.

        :param resolution_list: List[float], list of resolutions
        :return: str
        """
        return "x".join(self.gen_downsample_factor_list(resolution_list))


def check_transformation_not_empty(
    in_fn: str, ref_fn: str, tfm_fn: str, out_fn: str, empty_ok: bool = False
) -> None:
    """Check if the transformed file is empty or does not exist.

    :param in_fn: str, input filename
    :param ref_fn: str, reference filename
    :param tfm_fn: str, transformation filename
    :param out_fn: str, output filename
    :param empty_ok: bool, whether empty files are allowed
    :return: None
    """
    assert os.path.exists(out_fn), f"Error: transformed file does not exist {out_fn}"

    assert (
        np.sum(np.abs(nib.load(out_fn).dataobj)) > 0 or empty_ok
    ), f"Error in applying transformation: \n\t-i {in_fn}\n\t-r {ref_fn}\n\t-t {tfm_fn}\n\t-o {out_fn}\n"


def resample_struct_reference_volume(
    orig_struct_vol_fn: str, resolution: float, output_dir: str, clobber: bool = False
) -> str:
    """Resample the structural reference volume to the desired resolution.

    :param orig_struct_vol_fn: path to the original structural reference volume
    :param hemisphere: hemisphere
    :param resolution: resolution of the volume
    :param output_dir: path to output directory
    :param clobber: boolean indicating whether to overwrite existing files
    :return: path to the resampled structural reference volume
    """
    base_name = re.sub(
        ".nii", f"_{resolution}mm.nii", os.path.basename(orig_struct_vol_fn)
    )
    struct_vol_rsl_fn = f"{output_dir}/{base_name}"

    if not os.path.exists(struct_vol_rsl_fn) or clobber:
        resample_to_resolution(orig_struct_vol_fn, [resolution] * 3, struct_vol_rsl_fn)

    return struct_vol_rsl_fn


def simple_ants_apply_tfm(
    in_fn: str,
    ref_fn: str,
    tfm_fn: str,
    out_fn: str,
    ndim: int = 3,
    n: str = "Linear",
    empty_ok: bool = False,
    clobber: bool = False,
) -> None:
    """Apply transformation using ANTs.

    :param in_fn: str, input filename
    :param ref_fn: str, reference filename
    :param tfm_fn: str, transformation filename
    :param out_fn: str, output filename
    :param ndim: int, number of dimensions
    :param n: str, transformation -type
    :param empty_ok: bool, whether empty files are allowed
    :return: None
    """
    if not os.path.exists(out_fn) or clobber:
        str0 = f"antsApplyTransforms -n {n} -v 0 -d {ndim} -i {in_fn} -r {ref_fn} -t {tfm_fn}  -o {out_fn}"
        shell(str0, verbose=True)
        check_transformation_not_empty(in_fn, ref_fn, tfm_fn, out_fn, empty_ok=empty_ok)


def get_seg_fn(
    dirname: str, y: float, resolution: float, filename: str, suffix: str = ""
) -> str:
    """Get the segmented filename.

    :param dirname: str, directory name
    :param y: float, y value
    :param resolution: float, resolution value
    :param filename: str, original filename
    :param suffix: str, suffix to add to the filename
    :return: str, segmented filename
    """
    filename = re.sub(
        ".nii.gz",
        f"_{resolution}mm{suffix}.nii.gz",
        os.path.basename(filename),
    )
    return "{}/{}".format(dirname, filename)


def gen_2d_fn(prefix: str, suffix: str, ext: str = ".nii.gz") -> str:
    """Generate the 2D filename.

    :param prefix: str, filename prefix
    :param suffix: str, filename suffix
    :param ext: str, filename extension
    :return: str, generated filename
    """
    return f"{prefix}{suffix}{ext}"


def save_sections(
    file_list: List[Tuple[str, float]],
    vol: np.ndarray,
    aff: np.ndarray,
    dtype: int = None,
) -> None:
    """Save sections of a volume.

    :param file_list: List[Tuple[str, float]], list of filenames and y values
    :param vol: Any, volume data
    :param aff: Any, affine transformation
    :param dtype: Optional[Any], data type
    :return: None
    """
    xstep = aff[0, 0]
    ystep = aff[1, 1]
    zstep = aff[2, 2]

    xstart = aff[0, 3]
    zstart = aff[2, 3]

    affine = np.array(
        [
            [xstep, 0, 0, xstart],
            [0, zstep, 0, zstart],
            [0, 0, ystep, 0],
            [0, 0, 0, 1],
        ]
    )

    for fn, y in file_list:
        i = 0

        if np.max(vol[:, int(y), :]) < 1:
            # Create 2D srv section
            # this little while loop thing is so that if we go beyond  brain tissue in vol,
            # we find the closest y segement in vol with brain tissue
            while np.max(vol[:, int(y - i), :]) < 1:
                i += ystep / np.abs(ystep)

        sec = vol[:, int(y - i), :]

        assert np.max(sec) != np.min(sec), f"Error: empty section {fn}"

        if "damp" in fn:
            print(fn, np.max(sec))

        nib.Nifti1Image(sec, affine, dtype=dtype, direction_order="lpi").to_filename(fn)


def threshold(fn: str) -> str:
    """Threshold image.

    Description: Threshold image to create mask.

    :param fn: filename
    :return: mask_fn
    """
    img = nib.load(fn)
    vol = img.get_fdata()

    mask = np.zeros(vol.shape)
    t = threshold_otsu(vol) * 0.8
    mask[vol > t] = 1

    mask_fn = fn.replace(".nii.gz", "_mask.nii.gz")

    nib.Nifti1Image(mask, img.affine).to_filename(mask_fn)

    return mask_fn


def check_volume(fn: str) -> None:
    """Check that the file exists, can be opened with nibabel, and is not empty."""
    assert os.path.exists(fn), f"Error: file does not exist {fn}"
    try:
        img = nibabel.load(fn)
    except Exception as e:
        raise AssertionError(f"Error: file cannot be opened with nibabel {fn}\n{e}")

    data = img.get_fdata()

    assert np.min(data) != np.max(data), f"Error: file is empty {fn}"

    # Calculate Otsu threshold and check that there are voxels above the threshold
    n_foreground = np.sum(data > threshold_otsu(data))

    return n_foreground


def get_fx_list(
    df: pd.DataFrame, out_dir: str, str_var: str, ext: str = ".nii.gz"
) -> List[Tuple[str, int]]:
    """Get the to-do list for processing.

    :param df: DataFrame, input dataframe
    :param out_dir: str, output directory
    :param str_var: str, string variable
    :param ext: str, filename extension
    :return: List[Tuple[str, int]], to-do list
    """
    tfm_dir = f"{out_dir}/tfm/"

    os.makedirs(tfm_dir, exist_ok=True)

    fx_list = []
    for _, row in df.iterrows():
        y = int(row["sample"])
        base = os.path.basename(row["raw"]).split(".")[0]
        assert int(y) >= 0, f"Error: negative y value found {y}"
        prefix = f"{tfm_dir}/{base}_y-{y}"
        fn = gen_2d_fn(prefix, str_var, ext=ext)

        fx_list.append(fn)

    return fx_list


def create_2d_sections(
    fx_list: list,
    y_list: list,
    srv_fn: str,
    output_dir: str,
    dtype: int = None,
    clobber: bool = False,
) -> None:
    """Create 2D sections from a dataframe.

    :param df: pd.DataFrame, dataframe
    :param srv_fn: str, filename of the srv
    :param resolution: float, resolution
    :param output_dir: str, output directory
    :param dtype: Optional[int], data type
    :param clobber: bool, whether to overwrite existing files
    :return: None
    """
    fx_to_do = []

    tfm_dir = output_dir + os.sep + "tfm"
    os.makedirs(tfm_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    fx_to_do = [
        (fn, y) for fn, y in zip(fx_list, y_list) if (not os.path.exists(fn) or clobber)
    ]

    if len(fx_to_do) > 0:
        srv_img = nib.load(srv_fn)
        affine = srv_img.affine
        srv = srv_img.get_fdata()
        save_sections(fx_to_do, srv, affine, dtype=dtype)

    return None


def splitext(s: str) -> List:
    """Split a filename into basepath and extension.

    :param s: str, filename
    :return: List
    """
    try:
        ssplit = os.path.basename(s).split(".")
        ext = "." + ".".join(ssplit[1:])
        basepath = re.sub(ext, "", s)
        return [basepath, ext]
    except TypeError:
        return s


newlines = ["\n", "\r\n", "\r"]


def unbuffered(proc: Popen, stream: str = "stdout") -> str:
    """Read unbuffered from a process.

    :param proc: process
    :param stream: str, stream
    :yield: str
    """
    stream = getattr(proc, stream)
    with contextlib.closing(stream):
        while True:
            out = []
            last = stream.read(1)
            # Don't loop forever
            if last == "" and proc.poll() is not None:
                break
            while last not in newlines:
                # Don't loop forever
                if last == "" and proc.poll() is not None:
                    break
                out.append(last)
                last = stream.read(1)
            out = "".join(out)
            print(out)
            yield out


def shell(
    cmd: str, verbose: bool = False, exit_on_failure: bool = True, bash: bool = True
) -> tuple:
    """Run command in shell and read STDOUT, STDERR and the error code.

    :param cmd: str, command
    :param verbose: bool, optional, if True, print command, default=False
    :param exit_on_failure: bool, optional, if True, exit on failure, default=True
    """
    stdout = ""
    if bash:
        cmd = f'exec bash -c "{cmd}"'

    if verbose:
        print(cmd)

    process = Popen(
        cmd,
        shell=True,
        stdout=PIPE,
        stderr=STDOUT,
        universal_newlines=True,
    )

    for line in unbuffered(process):
        stdout = stdout + line + "\n"
        if verbose:
            print(line)

    errorcode = process.returncode
    stderr = stdout
    if errorcode != 0:
        print("Error:")
        print("Command:", cmd)
        print("Stderr:", stdout)
        if exit_on_failure:
            exit(errorcode)
    return stdout, stderr, errorcode


def gen_new_filename(fn: str, output_dir: str, new_suffix: str) -> str:
    """Generate a new filename.

    :param fn: str, filename
    :param output_dir: str, output directory
    :param new_suffix: str, new suffix
    """
    if ".nii.gz" in fn:
        old_suffix = ".nii.gz"
    else:
        old_suffix = os.path.splitext(fn)[1]

    out_fn = output_dir + "/" + re.sub(old_suffix, new_suffix, os.path.basename(fn))
    return out_fn


def get_values_from_df(
    df: pd.DataFrame, fields: List = ["sub", "hemisphere", "chunk"]
) -> list:
    """Get the unique values from a dataframe for a list of fields.

    :param df: dataframe
    :param fields: list of fields
    :return: list of unique values
    """
    out_list = []
    for field in fields:
        temp_list = np.unique(df[field])
        assert len(temp_list) == 1, f"More than one {field} in chunk " + len(temp_list)
        out_list.append(temp_list[0])

    return out_list


def get_params_from_affine(aff: np.array, ndim: int) -> tuple:
    """Get the parameters from an affine.

    :param aff: np.ndarray, affine
    :return: origin, spacing, direction
    """
    spacing = aff[range(ndim), range(ndim)]
    origin = aff[range(ndim), 3]
    direction = [
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]  # FIXME: not sure if it's good idea to hardcord LPI direction
    return origin, spacing, direction


def parse_resample_arguments(
    input_arg: Union[str, np.ndarray], output_filename: str, aff: np.ndarray, dtype: int
) -> tuple:
    """Parse the arguments for resample_to_resolution.

    :param input_arg: input file or numpy array
    :param output_filename: output filename
    :param aff: np.ndarray, affine
    :param dtype: data type
    :return: vol, dtype, output_filename, aff
    """
    if isinstance(input_arg, str):
        assert os.path.exists(
            input_arg
        ), f"Error: input file does not exist {input_arg}"

        if ".nii" in input_arg:
            img = ants.image_read(input_arg)

            vol = img.numpy()
            origin = img.origin
            spacing = img.spacing
            direction = img.direction

            aff = nib.load(input_arg).affine

            if isinstance(dtype, type(None)):
                dtype = img.dtype
        else:
            vol = load_image(input_arg)
            origin, spacing, direction = get_params_from_affine(aff, len(vol.shape))

    elif isinstance(input_arg, np.ndarray):
        vol = input_arg

        if isinstance(dtype, type(None)):
            dtype = vol.dtype

        assert (
            isinstance(aff, np.ndarray) or isinstance(aff, list)
        ), f"Error: affine must be provided as a list or numpy array but got {type(aff)}"

        assert aff is not None, "Error: affine must be provided for numpy array input"

        origin, spacing, direction = get_params_from_affine(aff, len(vol.shape))
    else:
        print("Error: input_arg must be a string or numpy array")
        exit(1)

    if not isinstance(output_filename, type(None)):
        assert isinstance(
            output_filename, str
        ), f"Error: output filename must be as string, got {type(output_filename)}"

    vol_sum = np.sum(np.abs(vol))
    assert vol_sum > 0, (
        f"Error: empty ({vol_sum}) input file for resample_to_resolution\n" + input_arg
    )
    ndim = len(vol.shape)

    if ndim == 3:
        # 2D images sometimes have dimensions (m,n,1).
        # We resample them to (m,n)
        if vol.shape[2] == 1:
            vol = vol.reshape([vol.shape[0], vol.shape[1]])

        origin = origin[:ndim]
        spacing = spacing[:ndim]
        direction = direction[:ndim]

    spacing = spacing[:ndim]

    return vol, origin, spacing, direction, dtype, output_filename, aff, ndim


def newer_than(fn1: str, fn2: str) -> bool:
    """Check if fn1 is newer than fn2.

    :param fn1: str, filename
    :param fn2: str, filename
    :return: bool
    """
    fn1 = os.readlink(fn1) if os.path.islink(fn1) else fn1
    fn2 = os.readlink(fn2) if os.path.islink(fn2) else fn2
    if pd.isnull(fn1):
        return False
    elif not os.path.exists(fn1):
        return True
    else:
        return os.path.getctime(fn1) > os.path.getctime(fn2)


def compare_timestamp_of_files(x: Union[str, List], y: Union[str, List]) -> bool:
    """Compare the timestamps of two files. If x is newer than y, return True, otherwise return False.

    :param x: str or list
    :param y: str or list
    :return: bool
    """
    if isinstance(x, str) and isinstance(y, str):
        # Compare two files
        return newer_than(x, y)
    elif isinstance(x, Iterable) and isinstance(y, Iterable):
        # Compare two lists of files
        for fn1, fn2 in zip(x, y):
            if not newer_than(fn1, fn2):
                return False
    else:
        print(
            "Error: x and y must be both strings or lists. Got {} and {}".format(
                type(x), type(y)
            )
        )
        exit(1)

    return True


def get_new_dims(
    old_resolution: Tuple[float, float, float],
    new_resolution: Tuple[float, float, float],
    old_dimensions: np.ndarray,
) -> np.ndarray:
    """Calculate new dimensions based on old and new resolutions."""
    old_resolution = np.abs(np.array(old_resolution))
    new_resolution = np.abs(new_resolution)

    scale = old_resolution / np.array(new_resolution)

    downsample_factor = 1 / scale

    new_dims = np.ceil(old_dimensions * scale)
    return new_dims, downsample_factor


def check_run_stage(
    col1: Iterable, col2: Iterable, df_csv: str = None, clobber: bool = False
) -> bool:
    """Check if a stage should be run. If the output files exist, check if they are newer than the input files.

    :param col1: column name
    :param col2: column name
    :param df_csv: path to dataframe
    :param clobber: bool, optional, if True, overwrite existing files, default=False
    :return: bool
    """
    run_stage = False

    def get_file_list(df_csv: str, col: str) -> List:
        """Get the file list from a dataframe or a list.

        :param df_csv: str, dataframe
        :param col: str, column name
        :return: list
        """
        if isinstance(col, str) and isinstance(df_csv, str) and os.path.exists(df_csv):
            df = pd.read_csv(df_csv, index_col=False)
            if col in df.columns:
                file_list = df[col].values
        elif isinstance(col, Iterable):
            file_list = col
        else:
            print("Error: col2 must be a string or list")
            exit(0)

        return file_list

    def file_check(file_list: List, nan_okay: bool = True) -> bool:
        """Check if all files exist.

        :param file_list: list of files
        :param nan_okay: bool, optional, if True, nan values are allowed, default=True
        :return: bool
        """
        for x in file_list:
            if nan_okay & pd.isnull(x):
                continue
            elif not os.path.exists(x):
                print("File does not exist: {}".format(x))
                return False
        return True

    assert (
        (isinstance(col1, Iterable) and isinstance(col2, Iterable))
        or (isinstance(col1, str) & isinstance(col2, str) & isinstance(df_csv, str))
    ), "Error: col1 and col2 must both be either strings or Iterables, but got {} and {}".format(
        type(col1), type(col2)
    )

    if isinstance(df_csv, str) and not os.path.exists(df_csv) or clobber:
        run_stage = True
    else:
        # get the list of files in col1
        file_list_1 = get_file_list(df_csv, col1)
        # get the list of files in col2
        file_list_2 = get_file_list(df_csv, col2)

        # check if all outputs for col1 exist
        all_outputs_exist = file_check(file_list_1)

        if not all_outputs_exist:
            # if not all outputs exist, run stage
            run_stage = True
            print("Not all files exist for {}".format(col1))

        # files in col1 should be newer than the ones in col2
        # check if this is the case, if not run stage
        if not compare_timestamp_of_files(file_list_1, file_list_2):
            print("All files exist but some are older than {}".format(col2))
            run_stage = True

    return run_stage


def pad_to_max_dims(vol: np.ndarray, max_dims: np.ndarray) -> np.ndarray:
    """Pad a volume to the maximum dimensions."""
    print(vol.shape, max_dims)
    offset0 = int(max_dims[0] - vol.shape[0])
    offset1 = int(max_dims[1] - vol.shape[1])
    offset2 = int(max_dims[2] - vol.shape[2]) if len(vol.shape) == 3 else 0

    print("offsets", offset0, offset1, offset2)

    if len(vol.shape) == 2:
        vol = np.pad(vol, ((0, offset0), (0, offset1)), mode="constant")
    elif len(vol.shape) == 3:
        vol = np.pad(vol, ((0, offset0), (0, offset1), (0, offset2)), mode="constant")
    return vol


def calculate_sigma_for_downsampling(new_pixel_spacing: float) -> float:
    """Calculate the standard deviation of a Gaussian pre-filter for downsampling.

    Parameters:
    original_pixel_spacing (float): The pixel spacing in the original image (in units like mm/pixel).

    Returns:
    float: The standard deviation for the Gaussian pre-filter.
    """
    # Nyquist frequency for the downsampled image
    nyquist_frequency_downsampled = 1 / (2 * new_pixel_spacing)
    # Standard deviation of the Gaussian filter
    sigma = 1 / (2 * np.pi * nyquist_frequency_downsampled)

    sigma[sigma <= 1] = 0

    return sigma


def resample_to_resolution(
    input_arg: Union[str, np.ndarray],
    new_resolution: Tuple[float, float, float],
    output_filename: Optional[str] = None,
    dtype: Optional[np.dtype] = None,
    affine: Optional[np.ndarray] = None,
    direction_order: str = "lpi",
    order: int = 1,
    factor: float = 1,
    max_dims: Optional[np.ndarray] = None,
) -> nib.Nifti1Image:
    """Resample a volume to a new resolution.

    :param input_arg: input file or numpy array
    :param new_resolution: new resolution
    :param output_filename: output filename
    :param dtype: data type
    :param affine: np.ndarray, affine
    :param order: order of interpolation
    :return: img_out
    """
    (
        vol,
        origin,
        old_resolution,
        direction,
        dtype,
        output_filename,
        affine,
        ndim,
    ) = parse_resample_arguments(input_arg, output_filename, affine, dtype)

    new_dims, downsample_factor = get_new_dims(
        old_resolution, new_resolution, vol.shape
    )

    if order != 0:
        sigma = calculate_sigma_for_downsampling(downsample_factor)
    else:
        sigma = 0

    # sigma[sigma <= 1] = 0

    vol = resize(
        vol.astype(float),
        new_dims,
        order=order,
        anti_aliasing=True,
        anti_aliasing_sigma=sigma,
    ).astype(dtype)

    if max_dims is not None:
        vol = pad_to_max_dims(vol, max_dims)

    vol *= factor

    assert np.sum(np.abs(vol)) > 0, (
        "Error: empty output array for prefilter_and_downsample\n" + output_filename
    )

    affine = np.eye(4, 4)
    dim_range = range(ndim)
    affine[dim_range, dim_range] = new_resolution
    affine[dim_range, 3] = origin

    img_out = nib.Nifti1Image(vol, affine, dtype=dtype, direction_order=direction_order)

    if isinstance(output_filename, str):
        img_out.to_filename(output_filename)

    return img_out
