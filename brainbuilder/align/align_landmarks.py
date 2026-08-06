# === BEGIN: align/align_landmarks.py =========================================
from __future__ import annotations

import os
import shutil
import subprocess
from glob import glob
from typing import Tuple

import ants
import numpy as np
import pandas as pd
from brainbuilder.utils import ants_nibabel as nib
from brainbuilder.utils import utils
from brainbuilder.utils.axis_utils import (
    DEFAULT_SECTION_AXIS,
    inplane_axes,
    volume_shape,
)
from brainbuilder.utils.utils import pad_volume, simple_ants_apply_tfm
from joblib import Parallel, delayed
from scipy.ndimage import binary_dilation, center_of_mass

logger = utils.get_logger(__name__)


# ---------- helpers


def get_unique_values(path):
    img = nib.load(path)
    data = np.array(img.dataobj)
    return np.unique(data)[1:]


def _strip_ext(p: str) -> str:
    """Return root (without .nii/.nii.gz)."""
    s = os.path.basename(p)
    s = s.replace(".nii.gz", "").replace(".nii", "")
    return s


def _label_ids(img_data: np.ndarray) -> np.ndarray:
    labs = np.unique(img_data.astype(np.int64))
    return labs[labs > 0]


def _stamp_label_sphere(
    vol: np.ndarray, center: np.ndarray, radius: int, label: int
) -> None:
    """Stamp a solid isotropic sphere of ``label`` into ``vol`` centred at voxel
    ``center`` (clipped to the volume bounds).

    Landmark registration only uses each label's centre of mass, so a small,
    fixed-size sphere placed at the label centroid is a faithful and
    resolution-independent representation of the landmark.
    """
    center = np.asarray(center, dtype=int)
    shape = np.asarray(vol.shape)
    lo = np.maximum(center - radius, 0)
    hi = np.minimum(center + radius + 1, shape)
    if np.any(hi <= lo):
        return
    sub = vol[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]]
    grids = np.ogrid[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]]
    dist2 = sum((g - center[a]) ** 2 for a, g in enumerate(grids))
    sub[dist2 <= radius * radius] = label


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
    axis: int = DEFAULT_SECTION_AXIS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Initialize affine and dimensions for the sparse landmark volume."""
    example_raw_img = nib.load(sect_info["raw"].values[0])

    inplane = inplane_axes(axis)
    steps = [0.0, 0.0, 0.0]
    steps[axis] = section_thickness
    steps[inplane[0]] = example_raw_img.affine[0, 0]
    steps[inplane[1]] = example_raw_img.affine[1, 1]

    affine = np.ones((4, 4))
    affine[0, 0] = resolution
    affine[1, 1] = resolution
    affine[2, 2] = resolution
    affine[0:3, 3] = fixed_origin

    # dims for sparse landmark volume
    dims = list(
        volume_shape((example_raw_img.shape[0], example_raw_img.shape[1]), ymax, axis)
    )
    scaling = resolution / np.array(steps)

    target_dims = np.rint(dims / scaling).astype(int)

    return affine, dims, target_dims, scaling, steps


def get_transform_type(labels: np.ndarray) -> str:
    n_landmarks = len(np.unique(labels))

    if n_landmarks < 2:
        logger.warning(f"Not enough landmarks ({n_landmarks}) found, skipping.")
        return None
    elif n_landmarks < 12:
        transform_type = "rigid"
    elif n_landmarks < 20:
        transform_type = "affine"
    else:
        transform_type = "bspline"

    logger.info(
        f"Using transform type '{transform_type}' with {n_landmarks} landmarks."
    )

    return transform_type


def _process_and_save_sparse_landmark_volume(
    sect_info: pd.DataFrame,
    out_vol_path: str,
    reference_vol_path: str,
    section_thickness: float,
    ymax: int,
    resolution_3d: float,
    padding_offset: float = 0.15,
    axis: int = DEFAULT_SECTION_AXIS,
    sphere_radius: int = 2,
):
    """Create and save a sparse 3D landmark volume by stamping each landmark
    label as a small sphere at its centre of mass, directly in the target grid.

    Landmark-based registration (antsLandmarkBasedTransformInitializer) only uses
    each label's centre of mass, so every label is represented by a compact,
    fixed-size sphere placed at its centroid. Building directly in the output
    (target) grid avoids the previous dilate -> downsample -> overwrite pipeline,
    which could silently drop small labels and behaved differently at each
    resolution.
    """
    reference_origin = nib.load(reference_vol_path).affine[0:3, 3]
    reference_direction = ants.image_read(reference_vol_path).direction[
        [0, 1, 2], [0, 1, 2]
    ]
    print(f"Reference origin: {reference_origin}, direction: {reference_direction}")

    # Initialize affine and dimensions
    affine, dims, target_dims, scaling, steps = _init_parameters(
        sect_info, reference_origin, section_thickness, resolution_3d, ymax, axis=axis
    )

    inplane = inplane_axes(axis)
    dims = np.asarray(dims, dtype=int)
    target_dims = np.asarray(target_dims, dtype=int)

    out_data = np.zeros(target_dims, dtype=np.uint32)

    unique_labels_list = []
    label_centers = {}  # label id -> centre voxel in the target grid
    for _, row in sect_info.iterrows():
        y = int(row["sample"])

        warped_slice_path = row["landmark_2d_rsl"]

        if not warped_slice_path:
            continue

        print("\tProcessing warped slice:", warped_slice_path)

        warped = np.squeeze(nib.load(str(warped_slice_path)).get_fdata()).astype(
            np.uint32
        )

        if warped.ndim != 2:
            raise ValueError(
                f"Expected 2D warped landmark slice, got shape {warped.shape}: {warped_slice_path}"
            )

        # Target index along the sectioning axis for this section (same mapping
        # the previous per-section resize applied: full-resolution axis -> target).
        y_t = int(
            np.clip(round(y * target_dims[axis] / dims[axis]), 0, target_dims[axis] - 1)
        )

        for label in np.unique(warped[warped > 0]):
            coords = np.argwhere(warped == label)
            if coords.size == 0:
                continue

            # Centroid in the warped slice, mapped into the target in-plane grid.
            c_in = coords.mean(axis=0)
            center = np.zeros(3, dtype=int)
            center[axis] = y_t
            center[inplane[0]] = int(
                np.clip(
                    round(c_in[0] * target_dims[inplane[0]] / warped.shape[0]),
                    0,
                    target_dims[inplane[0]] - 1,
                )
            )
            center[inplane[1]] = int(
                np.clip(
                    round(c_in[1] * target_dims[inplane[1]] / warped.shape[1]),
                    0,
                    target_dims[inplane[1]] - 1,
                )
            )

            label_centers[int(label)] = center
            unique_labels_list.append(int(label))

    # Stamp all spheres, then force each centroid voxel so that overlapping
    # spheres from neighbouring landmarks can never erase a label's centre.
    for label, center in label_centers.items():
        _stamp_label_sphere(out_data, center, sphere_radius, label)
    for label, center in label_centers.items():
        out_data[tuple(center)] = label

    # we need to pad because if we transform into the acquisition space, then landmarks at the edge might be cut off
    out_data, affine = pad_volume(
        out_data, affine, reference_direction, padding_offset=padding_offset
    )

    unique_labels_rsl = np.unique(out_data)[1:]

    assert set(unique_labels_list) == set(unique_labels_rsl), (
        "Some labels are missing after building sparse landmark volume "
        "(two landmarks may map to the same voxel at this resolution). "
        f"Before: {sorted(set(unique_labels_list))}, "
        f"after: {sorted(int(v) for v in unique_labels_rsl)}"
    )

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

    original_img = nib.load(input_file)
    original_data = np.squeeze(np.array(original_img.dataobj))
    required_labels = _label_ids(original_data)

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
        out_img = nib.load(output_file)
        out_labels = _label_ids(out_img.get_fdata())

        missing_labels = set(required_labels) - set(out_labels)

        if len(missing_labels) == 0:
            return

        # print(
        #    f"Missing labels {missing_labels} in warped landmark {output_file}. Dilating and retrying..."
        # )

        # Rebuild the retry input from the original labels so the success criteria
        # stay anchored to the source landmark set instead of the prior dilation.
        dilated_data = np.array(original_data, copy=True)

        for label in sorted(missing_labels):
            label_binary = original_data == label
            # print(f"\tDilating label {label}...{np.sum(label_binary)} voxels")
            label_binary_dilated = binary_dilation(label_binary, iterations=2)
            dilated_data[label_binary_dilated] = label

        dilated_img = nib.Nifti1Image(
            dilated_data, original_img.affine, direction_order="lpi"
        )
        dilated_img.to_filename(dilated_input_file)

        dilate_count += 2

        current_input_file = dilated_input_file

    raise RuntimeError(
        f"Failed to warp landmark {input_file} after {max_dilation} dilations. Missing labels: {sorted(missing_labels)}"
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
    reference_vol_path: str,
    resolution: float,
    resolution_3d: float,
    sect_info: pd.DataFrame,
    output_dir: str,
    section_thickness: float,
    ymax: int,
    axis: int = DEFAULT_SECTION_AXIS,
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
            reference_vol_path,
            section_thickness,
            ymax,
            resolution_3d,
            padding_offset=padding_offset,
            axis=axis,
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

    def_field = ants.image_read(nl_tfm)

    ants_nl_tfm = ants.transform_from_displacement_field(def_field)

    ants.write_transform(ants_nl_tfm, nl_tfm_h5)

    assert os.path.exists(nl_tfm_h5) and ants.read_transform(
        nl_tfm_h5
    ), f"Converted landmark transform not created: {nl_tfm_h5}"

    cmd = [
        "antsApplyTransforms",
        "-d",
        "3",
        "-t",
        nl_tfm_h5,
        "-t",
        affine_tfm,
        "-o",
        f"[{out_tfm_h5},1]",
        "-r",
        nl_tfm,
        "--float",
        "1",
    ]
    logger.info(f"[ANTs] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    assert os.path.exists(out_tfm_h5) and ants.read_transform(
        out_tfm_h5
    ), f"Converted landmark transform not created: {out_tfm_h5}"


def calculate_dist_between_labels(
    filepath1: str, filepath2: str, output_csv: str | None = None
) -> float:
    """For two label volumes, calculate distances between matching label COMs (mm).

    Prints per-label distances plus mean/std; returns the mean distance (mm).
    """
    img1 = nib.load(filepath1)
    img2 = nib.load(filepath2)

    vol1 = np.asarray(img1.dataobj)
    vol2 = np.asarray(img2.dataobj)

    def _to_int_labels(vol: np.ndarray, path: str) -> np.ndarray:
        # This function is intended for discrete label images.
        if np.issubdtype(vol.dtype, np.integer):
            return vol.astype(np.int64, copy=False)

        flat = vol.reshape(-1)
        if flat.size == 0:
            return vol.astype(np.int64)

        # Avoid a full-volume pass for huge arrays.
        if flat.size > 1_000_000:
            stride = max(1, flat.size // 1_000_000)
            flat = flat[::stride]

        frac = np.abs(flat - np.rint(flat))
        max_frac = float(np.nanmax(frac))
        if not np.isfinite(max_frac) or max_frac > 1e-3:
            raise ValueError(
                f"{path} does not look like a discrete label image (non-integer values detected). "
                "Did you accidentally pass a transform/displacement field instead of a label volume?"
            )

        return np.rint(vol).astype(np.int64)

    vol1 = _to_int_labels(vol1, filepath1)
    vol2 = _to_int_labels(vol2, filepath2)

    steps = np.abs(img1.affine[[0, 1, 2], [0, 1, 2]])  # voxel size in x, y, z (mm)

    labels1 = _label_ids(vol1)
    labels2 = _label_ids(vol2)

    shared_labels = sorted(set(labels1.tolist()).intersection(set(labels2.tolist())))

    assert shared_labels, f"No shared labels between volumes. Vol1 labels: {labels1}, Vol2 labels: {labels2}"

    dists = []

    df = pd.DataFrame(
        columns=[
            "label",
            "distance_mm",
            "com1_x",
            "com1_y",
            "com1_z",
            "com2_x",
            "com2_y",
            "com2_z",
        ]
    )
    for label in shared_labels:
        com1 = get_com(vol1, label)
        com2 = get_com(vol2, label)

        dist = float(np.sqrt(np.sum(((com1 - com2) * steps) ** 2)))

        dists.append(dist)

        print(
            f"Label {label}: distance = {dist:.2f} mm\t(com1: {np.round(com1,1)}, com2: {np.round(com2,1)})"
        )

        tdf = pd.DataFrame(
            {
                "label": [label],
                "distance_mm": [dist],
                "com1_x": [com1[0]],
                "com1_y": [com1[1]],
                "com1_z": [com1[2]],
                "com2_x": [com2[0]],
                "com2_y": [com2[1]],
                "com2_z": [com2[2]],
            }
        )

        df = pd.concat([df, tdf])

    if output_csv:
        df.to_csv(output_csv, index=False)

    mean_dist = float(np.mean(dists))
    std_dist = float(np.std(dists))

    print(f"Mean distance: {mean_dist:.2f} mm")
    print(f"Standard deviation: {std_dist:.2f} mm")

    return mean_dist


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
    use_com_qc: bool = False,
    qc_dir: str | None = None,
    clobber: bool = False,
) -> str:
    """Run antsLandmarkBasedTransformInitializer in two stages: affine and the bspline."""
    # if os.path.exists(out_tfm_h5) and not clobber:
    #    return out_tfm_h5

    os.makedirs(output_dir, exist_ok=True)

    affine_vol_fn = output_dir + os.path.basename(out_tfm).replace(
        ".h5", "_affine_landmark_init.nii.gz"
    )

    affine_tfm = output_dir + os.path.basename(out_tfm).replace(".h5", "_affine.h5")

    nl_tfm = output_dir + os.path.basename(out_tfm).replace(".h5", "_nl.nii.gz")

    if not os.path.exists(fixed_landmarks):
        raise RuntimeError(f"Reference landmarks not found: {fixed_landmarks}")

    if not os.path.exists(moving_landmarks):
        raise RuntimeError(f"Chunk sparse landmarks not found: {moving_landmarks}")

    ### 1) Run affine alignment
    if not os.path.exists(affine_tfm) or clobber:
        print("Running affine landmark-based alignment...")
        cmd = [
            "antsLandmarkBasedTransformInitializer",
            "3",
            fixed_landmarks,
            moving_landmarks,
            "affine",
            affine_tfm,
        ]

        logger.info(f"[ANTs] {' '.join(cmd)}")

        subprocess.run(cmd, check=True)

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

    if use_com_qc:
        point_qc2_csv = f"{output_dir}/{os.path.basename(affine_vol_fn).replace('.nii.gz', '_qc.csv')}"
        calculate_dist_between_labels(fixed_landmarks, affine_vol_fn, point_qc2_csv)

    moving_qc_affine_path = None
    if fixed_qc_vol_path and moving_qc_vol_path:
        moving_qc_affine_path = output_dir + os.path.basename(
            moving_qc_vol_path
        ).replace(".nii.gz", "_affine_landmark_qc.nii.gz")

    assert transform_type in [
        "rigid",
        "affine",
        "bspline",
    ], f"Invalid transform type: {transform_type}"

    if fixed_qc_vol_path and moving_qc_vol_path:
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
    nl_vol_fn = output_dir + os.path.basename(out_tfm).replace(
        ".h5", f"_{mesh_size}_nl_landmark_init.nii.gz"
    )

    if not os.path.exists(nl_tfm) or clobber:
        print("Running non-linear landmark-based alignment...")

        cmd = [
            "antsLandmarkBasedTransformInitializer",
            "3",
            fixed_landmarks,
            affine_vol_fn,
            "bspline",
            nl_tfm,
            mesh_size,
        ]

        logger.info(f"[ANTs] {' '.join(cmd)}")
        stdio = subprocess.run(cmd, check=True, capture_output=True, text=True)
        if stdio.stdout:
            logger.info(f"[ANTs] stdout: {stdio.stdout}")
        if stdio.stderr:
            logger.info(f"[ANTs] stderr: {stdio.stderr}")

        assert os.path.exists(
            nl_tfm
        ), f"Non-linear landmark transform not created: {nl_tfm}"

    if not os.path.exists(nl_vol_fn) or clobber:
        print("Applying non-linear transform to affine-warped landmarks...")
        simple_ants_apply_tfm(
            affine_vol_fn,
            fixed_landmarks,
            nl_tfm,
            nl_vol_fn,
            ndim=3,
            n="NearestNeighbor",
            clobber=clobber,
        )

    if use_com_qc:
        point_qc3_csv = (
            f"{output_dir}/{os.path.basename(nl_vol_fn).replace('.nii.gz', '_qc.csv')}"
        )
        calculate_dist_between_labels(fixed_landmarks, nl_vol_fn, point_qc3_csv)

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

    for _, row in sect_info.iterrows():
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
    landmark_series = np.array(
        output_landmark_files
    )  # pd.Series(output_landmark_files)
    # print("sum", landmark_series.notnull().sum())
    return landmark_series


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
    acq_rsl_fn: str,  # reference volume for the sparse landmark volume (e.g., the output of create_intermediate_volume())
    acq_landmark_path: str,  # path to the sparse landmark volume created by build_sparse_landmark_volume()
    moving_landmark_path: str,
    fixed_landmark_path: str,
    source_landmark_dir: str,
    output_landmark_dir: str,
    moving_qc_vol_path,
    fixed_qc_vol_path,
    ymax: int,
    section_thickness: float,
    axis: int = DEFAULT_SECTION_AXIS,
    num_cores: int = -1,
    transform_type="bspline",
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

    landmark_fwd_tfm_path = f"{output_landmark_dir}/sub-{sub}_hemi-{hemisphere}_chunk-{chunk}_landmark_init_itr-{resolution}mm_{transform_type}_{mesh_size}_Composite.h5"
    landmark_inv_tfm_path = f"{output_landmark_dir}/sub-{sub}_hemi-{hemisphere}_chunk-{chunk}_landmark_init_itr-{resolution}mm_{transform_type}_{mesh_size}_InverseComposite.h5"

    os.makedirs(output_landmark_dir, exist_ok=True)

    if (
        os.path.exists(landmark_fwd_tfm_path)
        and os.path.exists(landmark_inv_tfm_path)
        and not clobber
    ):
        logger.info(f"Landmark transform already exists: {landmark_fwd_tfm_path}")
        return landmark_fwd_tfm_path, landmark_inv_tfm_path

    logger.info(
        f"Creating landmark transform for sub-{sub} hemi-{hemisphere} chunk-{chunk} with {transform_type} transform..."
    )
    logger.info(f"Acquisition landmark path: {acq_landmark_path}")
    logger.info(f"Fixed landmark path: {fixed_landmark_path}")
    logger.info(f"Moving landmark path: {moving_landmark_path}")

    # TODO move this to a separate pre-processing step
    # moving_landmark_path = adjust_reference_landmark_labels(
    #    moving_landmark_path, output_dir, clobber=clobber
    # )
    # check_for_identical_landmark_values(sect_info["landmark"], moving_landmark_path)

    sect_info["landmark"] = find_landmark_files(sect_info, source_landmark_dir)

    logger.info(
        f"Found {sect_info['landmark'].notnull().sum()} landmark files in {source_landmark_dir}."
    )
    for fn in sect_info.loc[sect_info["landmark"].notnull(), ["landmark"]].values:
        logger.info(f"\t{fn}")

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
        axis=axis,
        padding_offset=padding_offset,
        clobber=clobber,
    )

    validate_landmark_labels(fixed_landmark_path, moving_landmark_path)

    logger.info("Creating forward landmark transform...")
    _, _, fwd_composite_tfm = init_landmark_transform(
        landmark_fwd_tfm_path,
        fixed_landmark_path,  # fixed
        moving_landmark_path,  # moving
        output_landmark_dir,
        transform_type=transform_type,
        qc_dir=output_landmark_dir + "/qc",
        fixed_qc_vol_path=fixed_qc_vol_path,
        moving_qc_vol_path=moving_qc_vol_path,
        mesh_size=mesh_size,
        clobber=clobber,
    )

    logger.info("Creating inverse landmark transform...")
    _, _, inv_composite_tfm = init_landmark_transform(
        landmark_inv_tfm_path,
        moving_landmark_path,  # fixed
        fixed_landmark_path,  # moving
        output_landmark_dir,
        transform_type=transform_type,
        qc_dir=output_landmark_dir + "/qc",
        fixed_qc_vol_path=moving_qc_vol_path,
        moving_qc_vol_path=fixed_qc_vol_path,
        mesh_size=mesh_size,
        clobber=clobber,
    )

    return fwd_composite_tfm, inv_composite_tfm
