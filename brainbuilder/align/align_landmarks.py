# === BEGIN: align/align_landmarks.py =========================================
from __future__ import annotations

import os
import shutil
import subprocess
from glob import glob

import ants
import numpy as np
import pandas as pd
from brainbuilder.utils import ants_nibabel as nib
from brainbuilder.utils import utils
from brainbuilder.utils.utils import pad_volume, simple_ants_apply_tfm
from joblib import Parallel, delayed
from scipy.ndimage import binary_dilation, center_of_mass
from skimage.transform import resize

logger = utils.get_logger(__name__)


# ---------- helpers


def get_unique_values(path):
    img = nib.load(path)
    data = np.array(img.dataobj)
    return np.unique(data)[1:]


def _strip_ext(p: p) -> str:
    """Return root (without .nii/.nii.gz)."""
    s = os.path.basename(p)
    s = s.replace(".nii.gz", "").replace(".nii", "")
    return s


def _label_ids(img_data: np.ndarray) -> np.ndarray:
    labs = np.unique(img_data.astype(np.int64))
    return labs[labs > 0]


def dilate_labels(
    label_binary: np.array,
    structure: np.array,
    scaling: float,
    idx_range: int,
    min_size: int = 10,
):
    """Dilate label_binary in x or z direction if needed to prevent loss during resizing."""
    ratio = idx_range / scaling
    assert ratio > 0, "Dilation ratio must be positive."
    if ratio < min_size:
        # (factor + range) / scaling = min_size =>  factor = min_size * scaling -range
        x_dilation_factor = (
            np.ceil(min_size * scaling - idx_range).astype(int) // 2
        )  # divide by 2 because we dilate on both sides

        print(
            "\tDilating label with factor",
            x_dilation_factor,
            "to prevent loss during resizing.",
        )
        label_binary = binary_dilation(
            label_binary, structure=structure, iterations=x_dilation_factor
        )

    return label_binary


def adjust_label_sizes(
    warped: np.array,
    unique_labels: np.array,
    x_structure: np.array,
    z_structure: np.array,
    scaling: np.array,
    x_step: float,
    z_step: float,
) -> np.array:
    """Adjust label sizes in warped slice to prevent loss during resizing."""
    x_scaling = scaling[0]
    z_scaling = scaling[2]

    for label in unique_labels:
        # get the extent of the label in dim 0 and 1 (x and z)
        label_binary = warped == label

        coords = np.argwhere(label_binary)
        if coords.size == 0:
            return warped

        x_min, z_min = coords.min(axis=0)
        x_max, z_max = coords.max(axis=0) + 1  # add 1 to include the max index

        x_range = (x_max - x_min) * x_step
        z_range = (z_max - z_min) * z_step

        # check if the label is too small, that is, if it might disappear during resizing
        label_binary = dilate_labels(label_binary, x_structure, x_scaling, x_range)
        label_binary = dilate_labels(label_binary, z_structure, z_scaling, z_range)

        warped[label_binary > 0] = label

    return warped


def set_scaling(
    dims: np.array, target_dims: np.array, max_scaling_allowed: float = 100.0
):
    """Set scaling factors and adjust target dimensions if scaling is too large."""
    scaling = np.array(dims) / np.array(target_dims)
    max_scaling = np.max(scaling)
    if max_scaling > max_scaling_allowed:
        scale_adjust_factor = max_scaling_allowed / scaling
        target_dims = np.ceil(np.array(dims) * scale_adjust_factor).astype(int)
        scaling = np.array(dims) / np.array(target_dims)

    return scaling, target_dims


def _init_parameters(
    sect_info: pd.DataFrame,
    fixed_origin: np.ndarray,
    section_thickness: float,
    resolution: float,
    ymax: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Initialize affine and dimensions for the sparse landmark volume."""
    example_raw_img = nib.load(sect_info["raw"].values[0])

    steps = [
        example_raw_img.affine[0, 0],
        section_thickness,
        example_raw_img.affine[1, 1],
    ]

    affine = np.ones((4, 4))
    affine[0, 0] = resolution
    affine[1, 1] = resolution
    affine[2, 2] = resolution
    affine[0:3, 3] = fixed_origin

    # dims for sparse landmark volume
    dims = [example_raw_img.shape[0], ymax, example_raw_img.shape[1]]

    scaling = resolution / np.array(steps)

    target_dims = np.rint(dims / scaling).astype(int)

    r = np.max([1, scaling[1] * 10]).astype(int)
    print("Building sparse landmark volume with section thickness ratio", r)

    return affine, dims, target_dims, scaling, r, steps


def get_transform_type(labels: np.ndarray) -> str:
    n_landmarks = len(np.unique(labels))

    if n_landmarks < 2:
        print(
            f"Not enough landmarks ({n_landmarks}) found for chunk {chunk}, skipping."
        )
        return None
    elif n_landmarks < 12:
        transform_type = "rigid"
    elif n_landmarks < 20:
        transform_type = "affine"
    else:
        transform_type = "bspline"

    print(f"Using transform type '{transform_type}' with {n_landmarks} landmarks.")

    return transform_type


# ---------- main API
def _process_and_save_sparse_landmark_volume(
    sect_info: pd.DataFrame,
    out_vol_path: str,
    fixed_origin: np.ndarray,
    section_thickness: float,
    ymax: int,
    resolution_3d: float,
    padding_offset: float = 0.15,
):
    """Create and save sparse 3D landmark volume by pasting warped 2D landmark slices."""
    # Initialize affine and dimensions
    affine, dims, target_dims, scaling, r, steps = _init_parameters(
        sect_info, fixed_origin, section_thickness, resolution_3d, ymax
    )

    out_data = np.zeros(dims, dtype=np.uint32)

    x_structure = np.zeros([3, 3])
    x_structure[1, :] = 1

    z_structure = np.zeros([3, 3])
    z_structure[:, 1] = 1

    unique_labels_list = []
    for _, row in sect_info.iterrows():
        y = int(row["sample"])

        xmax, zmax = np.array(nib.load(row["raw"]).shape)

        warped_slice_path = row["landmark_2d_rsl"]

        if not warped_slice_path:
            continue

        print("\tProcessing warped slice:", warped_slice_path)

        warped = nib.load(str(warped_slice_path)).get_fdata().astype(np.uint32)

        unique_labels = np.unique(warped[warped > 0])
        unique_labels = unique_labels[unique_labels != 0]

        warped = adjust_label_sizes(
            warped, unique_labels, x_structure, z_structure, scaling, steps[0], steps[2]
        )

        y0 = int(max(0, y - r))
        y1 = int(min(dims[1], y0 + r))

        # repeat warped to match y1-y0
        warped_rep = np.repeat(warped[:, np.newaxis, :], y1 - y0, axis=1)

        idx = warped_rep > 0
        # this is not ideal because of potential label conflicts but is necessary to prevent loss of labels
        # during transformation
        out_data[:, y0:y1, :][idx] = warped_rep[idx]

        unique_labels_list += unique_labels.tolist()

    out_data = resize(out_data, target_dims, anti_aliasing=False, order=0)

    out_data, _ = pad_volume(out_data, np.ones([4, 4]), padding_offset=padding_offset)

    unique_labels_rsl = np.unique(out_data)[1:]

    assert (
        set(unique_labels_list) == set(unique_labels_rsl)
    ), f"Some labels are missing after resizing sparse landmark volume. Before: {unique_labels_list}, after: {unique_labels_rsl}"

    print("Writing sparse landmark volume to", out_vol_path)
    nib.Nifti1Image(out_data, affine, direction_order="lpi").to_filename(out_vol_path)


def apply_tfm_and_check(
    input_file: str, ref_file: str, tfm_file: str, output_file: str
) -> None:
    """Apply 2D transformation to input landmarks and check that all the landmarks are present in the output.
    If not, open the landmark file and dilate all the labels by n before applying the transform again. Repeat
    until all labels are present or max_dilation is reached.
    """
    max_dim = max(nib.load(input_file).shape)
    max_dilation = max_dim // 2

    out_dir = os.path.dirname(output_file) + "/"

    dilated_input_file = out_dir + os.path.basename(input_file).replace(
        ".nii.gz", "_dilate_tmp.nii.gz"
    )

    current_input_file = input_file

    dilate_count = 0
    while dilate_count <= max_dilation:
        print()
        print("Current input file:", current_input_file)
        simple_ants_apply_tfm(
            current_input_file,
            ref_file,
            tfm_file,
            output_file,
            ndim=2,
            n="NearestNeighbor",
            clobber=True,
            empty_ok=True,
        )

        # check that every label in input_file is present in output_file
        input_img = nib.load(current_input_file)
        out_img = nib.load(output_file)
        input_labels = _label_ids(input_img.get_fdata())
        out_labels = _label_ids(out_img.get_fdata())

        missing_labels = set(input_labels) - set(out_labels)

        if len(missing_labels) == 0:
            return

        print(
            f"Missing labels {missing_labels} in warped landmark {output_file}. Dilating and retrying..."
        )

        # dilate all labels in input_file by 1
        input_data = np.squeeze(np.array(input_img.dataobj))
        dilated_data = np.zeros_like(input_data)

        for label in input_labels:
            label_binary = input_data == label
            print(f"\tDilating label {label}...{np.sum(label_binary)} voxels")
            label_binary_dilated = binary_dilation(label_binary, iterations=2)
            dilated_data[label_binary_dilated] = label

        dilated_img = nib.Nifti1Image(
            dilated_data, input_img.affine, direction_order="lpi"
        )
        dilated_img.to_filename(dilated_input_file)

        dilate_count += 2

        current_input_file = dilated_input_file

    print("out_labels", out_labels)
    print("max_dilations = ", max_dilation)

    raise RuntimeError(
        f"Failed to warp landmark {input_file} after {max_dilation} dilations."
    )


def process_row(row: pd.Series, clobber: bool = False) -> None:
    """Process a single row: warp 2D landmark and paste into out_data."""
    landmark_2d_rsl = str(row["landmark_2d_rsl"])

    lm_path = row["landmark"]
    if not lm_path or not os.path.exists(lm_path):
        return None

    tfm_path = row["2d_tfm"]
    raw_path = row["raw"]

    # landmark_tfm = row["landmark_2d_tfm"]

    if not isinstance(tfm_path, str) or (
        isinstance(tfm_path, str) and not os.path.exists(tfm_path)
    ):
        # Copy landmark as is (no transform)
        print(f"No transform found for section {lm_path}, copying landmark as is.")
        # shutil.copy(str(lm_path), str(landmark_tfm))
        shutil.copy(str(lm_path), str(landmark_2d_rsl))

    if landmark_2d_rsl and not os.path.exists(landmark_2d_rsl) or clobber:
        print(
            f"\tWarping landmark {lm_path} to {landmark_2d_rsl} using transform {tfm_path}."
        )

        apply_tfm_and_check(lm_path, raw_path, tfm_path, landmark_2d_rsl)


def build_sparse_landmark_volume(
    out_vol_path: str,
    fixed_qc_vol_path: str,
    resolution: float,
    resolution_3d: float,
    sect_info: pd.DataFrame,
    output_dir: str,
    section_thickness: float,
    ymax: int,
    fixed_origin: np.ndarray = None,
    padding_offset: float = 0.15,
    clobber: bool = False,
) -> pd.DataFrame:
    """Create a sparse 3D landmark volume in the 'init_volume' space by warping
    each available 2D landmark label image using the best available 2D transform.

    - sect_info: rows for a single (subject, hemi, chunk)
      must contain columns: ['raw', tfm_field_name, 'init_volume', 'landmark'] (landmark optional)
    - template_vol_path: path to the volume that carries the desired shape/header
      (e.g., the same path you pass to create_intermediate_volume(); typically row['init_volume'])
    """
    output_2d_dir = output_dir + "/landmark_2d_warped/"

    fixed_origin = nib.load(fixed_qc_vol_path).affine[0:3, 3]

    os.makedirs(output_2d_dir, exist_ok=True)

    # Create a new column with paths to warped 2D landmarks in output_dir. add {resolution}mm_rsl suffix
    sect_info["landmark_2d_rsl"] = sect_info["landmark"].apply(
        lambda p: f"{output_2d_dir}/{_strip_ext(p)}_itr-{resolution}mm_rsl.nii.gz"
        if isinstance(p, str)
        else None
    )

    sect_info["landmark_2d_tfm"] = sect_info["landmark"].apply(
        lambda p: f"{output_2d_dir}/{_strip_ext(p)}_itr-{resolution}mm_tfm.nii.gz"
        if isinstance(p, str)
        else None
    )

    if not os.path.exists(out_vol_path) or clobber:
        Parallel(n_jobs=-1)(
            delayed(process_row)(row, clobber=clobber)
            for _, row in sect_info.iterrows()
        )

        _process_and_save_sparse_landmark_volume(
            sect_info,
            out_vol_path,
            fixed_origin,
            section_thickness,
            ymax,
            resolution_3d,
            padding_offset=padding_offset,
        )

    return sect_info


def convert_transform(
    nl_tfm: str,
    affine_tfm: str,
    out_tfm_h5: str,
    clobber: bool = False,
) -> None:
    """Convert ANTs transforms to h5 format."""
    print("Converting landmark transforms to h5 format...")
    if os.path.exists(out_tfm_h5) and not clobber:
        return

    nl_tfm_h5 = os.path.splitext(nl_tfm)[0] + ".h5"

    print(nl_tfm)

    def_field = ants.image_read(nl_tfm)
    print("Def field shape:", def_field.shape)

    ants_nl_tfm = ants.transform_from_displacement_field(def_field)
    print("Writing non-linear landmark transform to", nl_tfm_h5)

    ants.write_transform(ants_nl_tfm, nl_tfm_h5)

    assert os.path.exists(nl_tfm_h5) and ants.read_transform(
        nl_tfm_h5
    ), f"Converted landmark transform not created: {nl_tfm_h5}"

    cmd = f"antsApplyTransforms -d 3 -t {nl_tfm_h5} -t {affine_tfm} -o [{out_tfm_h5},1] -r {nl_tfm} --float 1"
    logger.info(f"[ANTs] {cmd}")
    subprocess.run(cmd, shell=True, executable="/bin/bash")

    assert os.path.exists(out_tfm_h5) and ants.read_transform(
        out_tfm_h5
    ), f"Converted landmark transform not created: {out_tfm_h5}"


def init_landmark_transform(
    out_tfm: str,
    fixed_landmarks: str,
    moving_landmarks: str,
    output_dir: str,
    transform_type: str = "bspline",  # 'rigid'|'similarity'|'affine'|'bspline'
    mesh_size: str = "5x5x5",
    min_labels_required: int = 12,  # rigid/similarity: 3, affine: 12, bspline: 12 (for now)
    fixed_qc_vol_path: str | None = None,
    moving_qc_vol_path: str | None = None,
    qc_dir: str | None = None,
    clobber: bool = False,
) -> str:
    """Run antsLandmarkBasedTransformInitializer in two stages: affine and the bspline."""
    # if os.path.exists(out_tfm_h5) and not clobber:
    #    return out_tfm_h5

    affine_vol_fn = output_dir + os.path.basename(out_tfm).replace(
        ".h5", "_affine_landmark_init.nii.gz"
    )

    affine_tfm = output_dir + os.path.basename(out_tfm).replace(".h5", "_affine.h5")

    nl_tfm = output_dir + os.path.basename(out_tfm).replace(
        ".h5", f"_{mesh_size}_nl.nii.gz"
    )

    if not os.path.exists(fixed_landmarks):
        raise RuntimeError(f"Reference landmarks not found: {fixed_landmarks}")

    if not os.path.exists(moving_landmarks):
        raise RuntimeError(f"Chunk sparse landmarks not found: {moving_landmarks}")

    ### 1) Run affine alignment
    if not os.path.exists(affine_tfm) or clobber:
        print("Running affine landmark-based alignment...")
        cmd = f"antsLandmarkBasedTransformInitializer 3 {fixed_landmarks} {moving_landmarks}  'affine' {affine_tfm} "

        logger.info(f"[ANTs] {cmd}")

        subprocess.run(cmd, shell=True, executable="/bin/bash")

        assert os.path.exists(
            affine_tfm
        ), f"Affine landmark transform not created: {affine_tfm}"

    ### 2) Apply affine to moving landmarks to get intermediate volume
    print("Applying affine transform to moving landmarks...")

    simple_ants_apply_tfm(
        moving_landmarks,
        fixed_landmarks,
        affine_tfm,
        affine_vol_fn,
        ndim=3,
        n="NearestNeighbor",
        clobber=clobber,
    )

    if fixed_qc_vol_path and moving_qc_vol_path:
        moving_qc_affine_path = moving_qc_vol_path.replace(
            ".nii.gz", "_affine_landmark_qc.nii.gz"
        )

    assert transform_type in [
        "rigid",
        "affine",
        "bspline",
    ], f"Invalid transform type: {transform_type}"

    print("\nApplying transforms to moving QC volume for visual inspection...")
    simple_ants_apply_tfm(
        moving_qc_vol_path,
        fixed_qc_vol_path,
        affine_tfm,
        moving_qc_affine_path,
        ndim=3,
        n="Linear",
        clobber=clobber,
    )
    print(f"wrote: {moving_qc_affine_path}\n")

    # assert check that the affin_vol_fn has same values as moving_landmarks after transform
    affine_labels = set(np.unique(nib.load(affine_vol_fn).get_fdata())[1:])
    moving_labels = set(np.unique(nib.load(moving_landmarks).get_fdata())[1:])
    assert (
        affine_labels == moving_labels
    ), f"Affine transformed landmarks do not match moving landmarks.\n\tAffine: {affine_labels}\n\tMoving: {moving_labels}"

    ### 3) Run non-linear alignment
    if not os.path.exists(nl_tfm) or clobber:
        print("Running non-linear landmark-based alignment...")
        cmd = f"antsLandmarkBasedTransformInitializer 3 {fixed_landmarks} {affine_vol_fn}   'bspline' {nl_tfm} {mesh_size}"

        stdio = subprocess.run(cmd, shell=True, executable="/bin/bash")
        logger.info(f"[ANTs] {cmd}")
        logger.info(f"[ANTs] stdout: {stdio.stdout}")

        subprocess.run(cmd, shell=True, executable="/bin/bash")

        assert os.path.exists(
            nl_tfm
        ), f"Non-linear landmark transform not created: {nl_tfm}"

    ### 4)  Concatenate transforms and convert to h5
    if not os.path.exists(out_tfm) or clobber:
        utils.concat_transforms_to_h5([nl_tfm, affine_tfm], out_tfm)

    moving_qc_nl_final_path = output_dir + os.path.basename(moving_qc_vol_path).replace(
        ".nii.gz", f"_{mesh_size}_nl_landmark_qc.nii.gz"
    )

    if fixed_qc_vol_path and moving_qc_vol_path:
        print(
            "\nApplying non-linear transforms to moving QC volume for visual inspection..."
        )

        print("-i", moving_qc_vol_path)
        print("-t", out_tfm)
        print("-r", fixed_qc_vol_path)
        print("-o", moving_qc_nl_final_path)

        # if  '0.125' in moving_qc_nl_final_path and 'final' in moving_qc_nl_final_path: #FIXME

        simple_ants_apply_tfm(
            moving_qc_vol_path,
            fixed_qc_vol_path,
            out_tfm,
            moving_qc_nl_final_path,
            ndim=3,
            n="Linear",
            clobber=clobber,
        )

        print(f"wrote: {moving_qc_nl_final_path}\n")

    return affine_tfm, nl_tfm, out_tfm


def find_landmark_files(sect_info: pd.DataFrame, landmark_dir: str) -> pd.Series:
    """Find landmark files for each section in the landmark directory.

    :param sect_info: section info dataframe
    :param landmark_dir: directory containing landmark files
    :return: series of landmark file paths
    """
    output_landmark_files = []

    for i, row in sect_info.iterrows():
        # strip path and extension from raw filename
        raw_basename = os.path.basename(row["raw"])
        raw_root = _strip_ext(raw_basename)

        landmark_str = f"{landmark_dir}/{raw_root}*.nii.gz"

        landmark_list = glob(landmark_str)

        if len(landmark_list) == 0:
            output_landmark_files.append(None)
        elif len(landmark_list) == 1:
            landmark_fn = landmark_list[0]
            output_landmark_files.append(str(landmark_fn))
        else:
            raise ValueError(
                f'Multiple landmark files found for section {row["sample"]} with pattern {landmark_str}.'
            )

    assert len(output_landmark_files) > 0, f"No landmark files found in {landmark_dir}."
    return pd.Series(output_landmark_files)


def load(path: str, dtype: int = np.uint32) -> Tuple:
    img = nib.load(path)
    orig = img.affine[0:3, 3]
    # step = np.abs(np.array([img.affine[0,0], img.affine[1,1], img.affine[2,2]]))
    step = np.array([img.affine[0, 0], img.affine[1, 1], img.affine[2, 2]])
    vol = np.array(img.dataobj, dtype=dtype)
    return img, vol, orig, step


def get_com(vol: np.ndarray, label: int) -> np.ndarray:
    """Get center of mass for a given label in a volume."""
    assert np.any(vol == label), f"Label {label} not found in the volume."
    com = np.array(center_of_mass((vol == label).astype(np.uint32)))
    # print("Center of mass for label", label, ":", com)
    return com


def w2v(idx: np.ndarray, orig: np.ndarray, step: np.ndarray) -> np.ndarray:
    """World to voxel coordinates."""
    ndim = len(idx)
    w = np.rint((idx - orig[0:ndim]) / step[0:ndim]).astype(int)
    return w


def v2w(idx: np.ndarray, orig: np.ndarray, step: np.ndarray) -> np.ndarray:
    """Voxel to world coordinates."""
    ndim = len(idx)
    v = idx * step[0:ndim] + orig[0:ndim]
    return v


def write_vtk_points(points_lps: np.ndarray, vtk_path: str):
    P = np.asarray(points_lps, float)
    with open(vtk_path, "w") as f:
        f.write("# vtk DataFile Version 3.0\npoints\nASCII\nDATASET POLYDATA\n")
        f.write(f"POINTS {len(P)} float\n")
        for x, y, z in P:
            f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")
        f.write(f"VERTICES {len(P)} {len(P)*2}\n")
        for i in range(len(P)):
            f.write(f"1 {i}\n")


def check_for_identical_landmark_values(
    landmark_series: pd.Series, ref_landmark_path: str
) -> None:
    """Check that labels in landmark_series are identical to those in ref_landmark_path and that each label in landmark_series appears in only one file.

    :param landmark_series: series of landmark file paths
    :param ref_landmark_path: reference landmark file path
    :return: None
    """
    ref_img = nib.load(ref_landmark_path)
    ref_data = np.array(ref_img.dataobj)
    ref_labels = set(np.unique(ref_data)[1:])

    error_flag = False

    all_labels = set()
    for landmark_path in landmark_series:
        if not landmark_path or not os.path.exists(landmark_path):
            continue

        lm_img = nib.load(landmark_path)
        lm_data = np.array(lm_img.dataobj)
        lm_labels = set(np.unique(lm_data)[1:])

        # check that all labels in lm_labels are in ref_labels
        assert lm_labels.issubset(
            ref_labels
        ), f"Landmark file {landmark_path} contains labels not present in reference landmark file {ref_landmark_path}.\n\tLandmark labels: {lm_labels}\n\tReference labels: {ref_labels}"

        # check that no label in lm_labels is already in all_labels
        intersection = all_labels.intersection(lm_labels)
        if len(intersection) != 0:
            print(f"\tERROR: Labels {intersection} appear in multiple landmark files.")
            error_flag = True
        all_labels.update(lm_labels)

    if error_flag:
        raise ValueError("Some labels appear in multiple landmark files.")


def validate_landmark_labels(
    fixed_landmark_path: str, moving_landmark_path: str
) -> None:
    """Validate that acquisition and reference landmark volumes have matching labels.

    :param acq_landmark_path: path to acquisition landmark volume
    :param ref_landmark_path: path to reference landmark volume
    :raises AssertionError: if label sets don't match
    """
    ar0_img = nib.load(fixed_landmark_path)
    ar1_img = nib.load(moving_landmark_path)

    ar0 = np.array(ar0_img.dataobj)
    ar1 = np.array(ar1_img.dataobj)

    ar0_labels = np.unique(ar0)[1:]
    ar1_labels = np.unique(ar1)[1:]

    assert (
        set(ar0_labels) == set(ar1_labels)
    ), f"Source and target (ref) landmark volumes have different labels.\n\t{fixed_landmark_path}: {ar0_labels}\n\t{moving_landmark_path}: {ar1_labels}"


def adjust_reference_landmark_labels(
    ref_landmark_path: str,
    output_dir: str,
    clobber: bool = False,
) -> str:
    """Adjust reference landmark labels by dilating them to ensure better coverage.

    :param ref_landmark_path: path to reference landmark file
    :param output_dir: output directory
    :param clobber: overwrite existing files
    :return: path to adjusted reference landmark file
    """
    adjusted_ref_landmark_path = (
        f"{output_dir}/adjusted_{os.path.basename(ref_landmark_path)}"
    )

    if os.path.exists(adjusted_ref_landmark_path) and not clobber:
        return adjusted_ref_landmark_path

    ref_img = nib.load(ref_landmark_path)
    ref_data = np.array(ref_img.dataobj)

    unique_labels = np.unique(ref_data)
    unique_labels = unique_labels[unique_labels != 0]  # exclude background

    adjusted_data = np.zeros_like(ref_data)

    for label in unique_labels:
        # dilate labels to ensure better coverage
        label_mask = ref_data == label

        label_mask_dilated = binary_dilation(label_mask, iterations=2)

        adjusted_data[label_mask_dilated] = label

    adjusted_img = nib.Nifti1Image(adjusted_data, ref_img.affine, direction_order="lpi")

    adjusted_img.to_filename(adjusted_ref_landmark_path)

    print(
        f"Adjusted reference landmark labels and saved to {adjusted_ref_landmark_path}"
    )

    return adjusted_ref_landmark_path


def create_landmark_transform(
    sub: str,
    hemisphere: str,
    chunk: int,
    resolution: float,
    resolution_3d: float,
    sect_info: pd.DataFrame,
    acq_rsl_fn: str,
    acq_landmark_path: str,
    moving_landmark_path: str,
    fixed_landmark_path: str,
    source_landmark_dir: str,
    output_landmark_dir: str,
    moving_qc_vol_path,
    fixed_qc_vol_path,
    ymax: int,
    section_thickness: float,
    num_cores: int = -1,
    transform_type="bspline",
    fixed_origin: np.ndarray = None,
    padding_offset: float = 0.15,
    mesh_size: str = "3x3x3",
    clobber: bool = False,
) -> str:
    """Process landmarks for alignment.

    :param sub: subject name
    :param hemisphere: hemisphere name
    :param chunk: chunk number
    :param resolution: resolution
    :param sect_info: section information dataframe
    :param init_volume: initial volume path
    :param ref_landmark_path: reference landmark path
    :param num_cores: number of cores to use
    :param clobber: overwrite existing files
    :return: path to the landmark transform file
    """
    transform_type = "bspline"

    landmark_tfm_path = f"{output_landmark_dir}/sub-{sub}_hemi-{hemisphere}_chunk-{chunk}_landmark_init_itr-{resolution}mm_{transform_type}_{mesh_size}_Composite.h5"

    os.makedirs(output_landmark_dir, exist_ok=True)

    if os.path.exists(landmark_tfm_path) and not clobber:
        print(f"Landmark transform already exists: {landmark_tfm_path}")
        return landmark_tfm_path

    # TODO move this to a separate pre-processing step
    # moving_landmark_path = adjust_reference_landmark_labels(
    #    moving_landmark_path, output_dir, clobber=clobber
    # )
    # check_for_identical_landmark_values(sect_info["landmark"], moving_landmark_path)

    sect_info["landmark"] = find_landmark_files(sect_info, source_landmark_dir)

    # check that at least some landmarks are found
    assert (
        sect_info["landmark"].notnull().sum() > 0
    ), f"No landmark files found in {source_landmark_dir}."

    sect_info = build_sparse_landmark_volume(
        acq_landmark_path,
        acq_rsl_fn,
        resolution,
        resolution_3d,
        sect_info,
        output_landmark_dir,
        section_thickness,
        ymax,
        padding_offset=padding_offset,
        fixed_origin=fixed_origin,
        clobber=clobber,
    )

    validate_landmark_labels(fixed_landmark_path, moving_landmark_path)

    print("Creating landmark transform...")
    affine_tfm, nl_tfm, composite_tfm = init_landmark_transform(
        landmark_tfm_path,
        fixed_landmark_path,
        moving_landmark_path,
        output_landmark_dir,
        transform_type=transform_type,
        qc_dir=output_landmark_dir + "/qc",
        fixed_qc_vol_path=fixed_qc_vol_path,
        moving_qc_vol_path=moving_qc_vol_path,
        mesh_size=mesh_size,
        clobber=clobber,
    )
    print("Done")

    return composite_tfm
    landmark_volume_rsl_path = f"{output_dir}/sub-{sub}_hemi-{hemisphere}_chunk-{chunk}_acq_landmarks_{resolution}mm_{transform_type}_rsl.nii.gz"

    simple_ants_apply_tfm(
        acq_landmark_path,
        ref_landmark_path,
        init_tfms,
        landmark_volume_rsl_path,
        ndim=3,
        n="NearestNeighbor",
        clobber=clobber,
    )

    ar2_labels = get_unique_values(landmark_volume_rsl_path)

    assert (
        set(ar1_labels) == set(ar2_labels)
    ), f"Acq rsl and ref landmark volumes have different labels.\n\tRef values: {ref_landmark_path}\n\t{ar1_labels}\n\tLandmark rsl values:\n\t{landmark_volume_rsl_path}\n\t{ar2_labels}"

    moving_qc_vol_rsl_path = f"{qc_dir}/" + os.path.basename(
        moving_qc_vol_path
    ).replace(".nii", "_landmark-rsl.nii")

    if (
        moving_qc_vol_path
        and fixed_qc_vol_path
        and os.path.exists(moving_qc_vol_path)
        and os.path.exists(fixed_qc_vol_path)
    ):
        if not os.path.exists(moving_qc_vol_rsl_path) or clobber:
            simple_ants_apply_tfm(
                moving_qc_vol_path,
                fixed_qc_vol_path,
                init_tfms,
                str(moving_qc_vol_rsl_path),
                ndim=3,
                n="Linear",
                clobber=clobber,
            )
            print("Saved moving landmark rsl to:\n\t", moving_qc_vol_rsl_path)
            print("Compare to fixed qc vol:\n\t", fixed_qc_vol_path)

    create_qc_images(
        sect_info,
        landmark_volume_rsl_path,
        ref_landmark_path,
        moving_qc_vol_rsl_path,
        fixed_qc_vol_path,
        qc_dir,
        clobber=clobber,
    )
    return init_tfms


# === END: align/align_landmarks.py ===========================================
