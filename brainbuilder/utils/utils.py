import bisect
import contextlib
import os
import re
import time

import ants
import imageio
import matplotlib
import nibabel as nb
import pandas as pd

import brainbuilder.utils.ants_nibabel as nib

matplotlib.use("Agg")
import multiprocessing
from glob import glob
from os.path import basename
from re import sub
from subprocess import PIPE, STDOUT, Popen
from time import time

import h5py as h5
import matplotlib.pyplot as plt
import nibabel
import numpy as np
from joblib import Parallel, delayed
from scipy.ndimage import center_of_mass, label
from skimage.transform import resize
from sklearn.cluster import KMeans
from scipy.ndimage import shift

from brainbuilder.utils.mesh_io import load_mesh, save_mesh

os_info = os.uname()


def load_image(fn):
    """
    Load an image from a file.
    :param fn: str, filename
    :return: np.ndarray
    """
    if '.nii' in fn:
        return ants.image_read(fn).numpy()
    else:
        return imageio.imread(fn)

def get_chunk_pixel_size(sub, hemi, chunk, chunk_info):
    """
    Get the pixel size of a chunk.
    :param sub: str, name of the subject
    :param hemi: str, hemisphere
    :param chunk: int, chunk number
    :param chunk_info: pd.DataFrame, chunk info dataframe
    :return: float
    """
    idx = (
        (chunk_info["sub"] == sub)
        & (chunk_info["hemisphere"] == hemi)
        & (chunk_info["chunk"] == chunk)
    )
    pixel_size_0 = chunk_info["pixel_size_0"][idx].values[0]
    pixel_size_1 = chunk_info["pixel_size_1"][idx].values[0]

    return pixel_size_0, pixel_size_1


def get_chunk_direction(sub, hemi, chunk, chunk_info):
    """
    Get the direction of a chunk.
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



def set_cores(num_cores):
    """
    Sets the number of cores to use for parallel processing.
    :param num_cores: int
    :return: int
    """
    if num_cores == 0:
        num_cores = multiprocessing.cpu_count()

    return num_cores

def imshow_images(
    out_fn,
    images,
    rows,
    columns,
    wspace=0.1,
    hspace=0,
    titles=[],
    facecolor="black",
    figsize=(10, 10),
    rotn=0,
):
    fig, axes = plt.subplots(rows, columns, figsize=figsize)

    if titles == []:
        titles = [""] * len(images)

    assert len(titles) == len(
        images
    ), "Error: images and titles dont have same number of elements, {len(images)} and {len(titles)}"

    fig.patch.set_facecolor(facecolor)

    for i, (img, title) in enumerate(zip(images, titles)):
        ax = axes.ravel()[i]

        img = np.rot90(img, rotn)
        ax.imshow(img, cmap="gray_r")
        if title != "" and type(title) != type(None):
            ax.set_title(title)

        ax.spines[["right", "top"]].set_visible(False)
        ax.axis("off")
        ax.set_facecolor("black")
        ax.set_aspect("equal")


    r = rows
    c = columns

    fig.subplots_adjust(wspace=wspace, hspace=hspace)
    fig.set_figheight(fig.get_figwidth() * ax.get_data_ratio() * r / c)
    plt.savefig(out_fn)
    plt.cla()
    plt.clf()


def get_thicken_width(resolution, section_thickness):
    return np.round(1 * (1 + float(resolution) / (section_thickness * 2))).astype(int)


def get_section_intervals(vol):
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
    def __init__(
        self,
        resolution_list,
        resolution,
        base_itr,
        max_resolution=None,
        start_resolution=None,
        verbose=False,
    ):
        if type(max_resolution) == type(None):
            self.max_resolution = resolution
        else:
            self.max_resolution = max_resolution

        if start_resolution != None:
            resolution_list = [
                r for r in resolution_list if float(r) < float(start_resolution)
            ]
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

    def _print(self):
        print("Iterations:\t", self.itr_str)
        print("Factors:\t", self.f_str)
        print("Smoothing:\t", self.s_str)

    def gen_itr_str(self, max_itr, step):
        itr_str = "x".join(
            [
                str(int(max_itr - i * step))
                for i in range(self.cur_n + 1)
                if self.resolution_list[i] >= self.max_resolution
            ]
        )
        itr_str = "[" + itr_str + ",1e-7,20 ]"
        return itr_str

    def gen_smoothing_factor_string(self, lst):
        return "x".join([str(i) for i in lst]) + "vox"

    def gen_smoothing_factor_list(self, resolution_list):
        # smooth_f = lambda x, y:  np.round(float(float(x)/float(y))/np.pi,2)
        smooth_f = lambda x, y: np.round(float(float(x) / float(y) - 1) * 0.2, 3)
        # smooth_f = lambda x, y:  (float(x)/float(y))/2
        return [
            smooth_f(resolution_list[i], resolution_list[self.cur_n])
            if i != self.cur_n
            else 0
            for i in range(self.cur_n + 1)
            if self.resolution_list[i] >= self.max_resolution
        ]

    def get_smoothing_params(self, resolution_list):
        s_list = self.gen_smoothing_factor_list(resolution_list)
        return self.gen_smoothing_factor_string(s_list)

    def calc_downsample_factor(self, cur_res, image_res):
        return (
            np.rint(float(cur_res) / float(image_res)).astype(int).astype(str)
        )  # DEBUG
        # return np.floor(1+np.log2(float(cur_res)/float(image_res))).astype(int).astype(str)

    def gen_downsample_factor_list(self, resolution_list):
        factors = [
            self.calc_downsample_factor(resolution_list[i], resolution_list[self.cur_n])
            for i in range(self.cur_n + 1)
            if self.resolution_list[i] >= self.max_resolution
        ]
        return factors

    def get_downsample_params(self, resolution_list):
        return "x".join(self.gen_downsample_factor_list(resolution_list))


def check_transformation_not_empty(in_fn, ref_fn, tfm_fn, out_fn, empty_ok=False):
    assert os.path.exists(out_fn), f"Error: transformed file does not exist {out_fn}"
    assert (
        np.sum(np.abs(nib.load(out_fn).dataobj)) > 0 or empty_ok
    ), f"Error in applying transformation: \n\t-i {in_fn}\n\t-r {ref_fn}\n\t-t {tfm_fn}\n\t-o {out_fn}\n"


def simple_ants_apply_tfm(
    in_fn, ref_fn, tfm_fn, out_fn, ndim=3, n="Linear", empty_ok=False
):
    if not os.path.exists(out_fn):
        str0 = f"antsApplyTransforms -v 0 -d {ndim} -i {in_fn} -r {ref_fn} -t {tfm_fn}  -o {out_fn}"
        shell(str0, verbose=True)
        check_transformation_not_empty(in_fn, ref_fn, tfm_fn, out_fn, empty_ok=empty_ok)

def get_seg_fn(dirname, y, resolution, filename, suffix=""):
    filename = re.sub(
        ".nii.gz",
        # f"_y-{int(y)}_{resolution}mm{suffix}.nii.gz",
        f"_{resolution}mm{suffix}.nii.gz",
        os.path.basename(filename),
    )
    return "{}/{}".format(dirname, filename)


def gen_2d_fn(prefix, suffix, ext=".nii.gz"):
    return f"{prefix}{suffix}{ext}"


def save_sections(file_list, vol, aff, dtype=None):
    xstep = aff[0, 0]
    ystep = aff[1, 1]
    zstep = aff[2, 2]

    xstart = aff[0, 3]
    zstart = aff[2, 3]

    for fn, y in file_list:
        affine = np.array(
            [
                [xstep, 0, 0, xstart],
                [0, zstep, 0, zstart],
                [0, 0, ystep, 0],
                [0, 0, 0, 1],
            ]
        )
        i = 0
        if np.sum(vol[:, int(y), :]) == 0:
            # Create 2D srv section
            # this little while loop thing is so that if we go beyond  brain tissue in vol,
            # we find the closest y segement in vol with brain tissue
            while np.sum(vol[:, int(y - i), :]) == 0:
                i += (ystep / np.abs(ystep)) * 1
        sec = vol[:, int(y - i), :]
        nib.Nifti1Image(sec, affine, dtype=dtype, direction_order="lpi").to_filename(fn)


def get_to_do_list(df, out_dir, str_var, ext=".nii.gz"):
    to_do_list = []
    for idx, (i, row) in enumerate(df.iterrows()):
        y = int(row["sample"])
        assert int(y) >= 0, f"Error: negative y value found {y}"
        prefix = f"{out_dir}/y-{y}"
        fn = gen_2d_fn(prefix, str_var, ext=ext)
        if not os.path.exists(fn):
            to_do_list.append([fn, y])
    return to_do_list


def create_2d_sections(df, srv_fn, resolution, output_dir, dtype=None, clobber=False):
    fx_to_do = []

    tfm_dir = output_dir + os.sep + "tfm"
    os.makedirs(tfm_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    fx_to_do = get_to_do_list(df, tfm_dir, "_fx")

    if len(fx_to_do) > 0:
        print("srv_fn: ", srv_fn)
        srv_img = nib.load(srv_fn)
        affine = srv_img.affine
        srv = srv_img.get_fdata()
        save_sections(fx_to_do, srv, affine, dtype=dtype)


def splitext(s):
    try:
        ssplit = os.path.basename(s).split(".")
        ext = "." + ".".join(ssplit[1:])
        basepath = re.sub(ext, "", s)
        return [basepath, ext]
    except TypeError:
        return s


newlines = ["\n", "\r\n", "\r"]
def unbuffered(proc, stream="stdout"):
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


def shell(cmd, verbose=False, exit_on_failure=True):
    """Run command in shell and read STDOUT, STDERR and the error code"""
    stdout = ""
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


def gen_new_filename(fn, output_dir, new_suffix):
    if '.nii.gz'  in fn :  
        old_suffix = '.nii.gz'
    else :
        old_suffix = os.path.splitext(fn)[1]

    out_fn = output_dir + "/" + re.sub(old_suffix, new_suffix, os.path.basename(fn))
    return out_fn


def world_center_of_mass(vol, affine):
    affine = np.array(affine)
    ndim = len(vol.shape)
    r = np.arange(ndim).astype(int)
    com = center_of_mass(vol)
    steps = affine[r, r]
    starts = affine[r, 3]
    wcom = com * steps + starts
    return wcom




def get_values_from_df(df, fields=["sub", "hemisphere", "chunk"]):
    """
    Get the unique values from a dataframe for a list of fields
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


def recenter(vol, affine, direction=np.array([1, 1, -1])):
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


def get_params_from_affine(aff, ndim):
    '''
    Get the parameters from an affine
    :param aff: np.ndarray, affine
    :return: origin, spacing, direction
    '''
    spacing = aff[range(ndim), range(ndim)]
    origin = aff[range(ndim), 3]
    direction = [
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]  # FIXME: not sure if it's good idea to hardcord LPI direction
    return origin, spacing, direction

def parse_resample_arguments(input_arg, output_filename, aff, dtype) -> tuple:
    """
    Parse the arguments for resample_to_resolution
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

        if '.nii' in input_arg:

            img = ants.image_read(input_arg)

            vol = img.numpy()
            origin = img.origin
            spacing = img.spacing
            direction = img.direction
            aff = nib.load(input_arg).affine

            if isinstance(dtype, type(None)):
                dtype = img.dtype
        else :
            vol = load_image(input_arg)
            origin, spacing, direction = get_params_from_affine(aff, len(vol.shape))

    elif type(input_arg) == np.ndarray:
        vol = input_arg
        if type(dtype) == type(None):
            dtype = vol.dtype

        assert (
            type(aff) == np.ndarray or type(aff) == list
        ), f"Error: affine must be provided as a list or numpy array but got {type(aff)}"

        origin, spacing, direction = get_params_from_affine(aff, len(vol.shape))
    else:
        print("Error: input_arg must be a string or numpy array")
        exit(1)


    if not isinstance(output_filename, type(None)):
        assert (
            isinstance(output_filename, str)
        ), f"Error: output filename must be as string, got {type(output_filename)}"

    return vol, origin, spacing, direction, dtype, output_filename, aff

def resample_to_resolution(
        input_arg,
        new_resolution,
        output_filename=None,
        dtype=None,
        aff=None,
        direction_order="lpi",
        order=1,
    ) :
    '''
    Resample a volume to a new resolution

    :param input_arg: input file or numpy array
    :param new_resolution: new resolution
    :param output_filename: output filename
    :param dtype: data type
    :param aff: np.ndarray, affine
    :param order: order of interpolation
    :return: img_out
    '''
    (
        vol,
        origin,
        old_resolution,
        direction,
        dtype,
        output_filename,
        aff,
    ) = parse_resample_arguments(input_arg, output_filename, aff, dtype)

    assert np.sum(np.abs(vol)) > 0, (
        "Error: empty input file for prefilter_and_downsample\n" + input_filename
    )
    ndim = len(vol.shape)

    if ndim == 3:
        # 2D images sometimes have dimensions (m,n,1).
        # We resample them to (m,n)
        if vol.shape[2] == 1:
            vol = vol.reshape([vol.shape[0], vol.shape[1]])

    scale = old_resolution / np.array(new_resolution)

    new_dims = np.ceil(vol.shape * scale)

    sigma = (new_resolution / np.array(old_resolution)) / 5

    vol = resize(vol, new_dims, order=order, anti_aliasing=True, anti_aliasing_sigma=sigma)

    assert np.sum(np.abs(vol)) > 0, (
        "Error: empty output array for prefilter_and_downsample\n" + output_filename
    )

    affine = np.eye(4, 4)
    dim_range = range(ndim)
    affine[dim_range, dim_range] = new_resolution
    affine[dim_range, 3] = origin

    img_out = nib.Nifti1Image(vol, affine, dtype = dtype, direction_order = direction_order)

    if type(output_filename) == str:
        img_out.to_filename(output_filename)

    return img_out





