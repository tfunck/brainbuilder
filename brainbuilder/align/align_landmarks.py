# === BEGIN: align/align_landmarks.py =========================================
from __future__ import annotations

import os
import shutil
import subprocess
from glob import glob

import numpy as np
import pandas as pd
from brainbuilder.utils import ants_nibabel as nib

# If you already have a logger, import it; else fallback to print
from brainbuilder.utils import utils
from brainbuilder.utils.utils import simple_ants_apply_tfm
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from scipy.ndimage import center_of_mass

logger = utils.get_logger(__name__)


# ---------- helpers


def _strip_ext(p: p) -> str:
    """Return root (without .nii/.nii.gz)."""
    s = os.path.basename(p)
    s = s.replace(".nii.gz", "").replace(".nii", "")
    return s


def _label_ids(img_data: np.ndarray) -> np.ndarray:
    labs = np.unique(img_data.astype(np.int64))
    return labs[labs > 0]


# ---------- main API


def build_sparse_landmark_volume(
    out_vol_path: str,
    origins,
    resolution,
    sect_info,
    output_dir: str,
    section_thickness: float,
    ref_section_thickness: float,
    ymax: int,
    tfm_field_name: str = "2d_tfm",
    n_jobs: int = 1,
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

    if not os.path.exists(out_vol_path) or clobber:
        example_raw_img = nib.load(sect_info["raw"].values[0])
        steps = [
            example_raw_img.affine[0, 0],
            section_thickness,
            example_raw_img.affine[1, 1],
        ]
        dims = [example_raw_img.shape[0], ymax, example_raw_img.shape[1]]

        out_data = np.zeros(dims, dtype=np.uint32)

        unique_values_list = []
        unique_values_rsl_list = []

        # Iterate sections; paste warped 2D landmark slice
        def process_row(row: pd.Series, clobber: bool = False) -> None:
            """Process a single row: warp 2D landmark and paste into out_data."""
            output_path = str(row["landmark_2d_rsl"])

            if not output_path:
                return None

            raw_path = row["raw"]
            lm_path = row["landmark"]
            tfm_path = row.get(tfm_field_name, None) or row.get("init_tfm", None)
            if output_path and not os.path.exists(output_path) or clobber:
                if not isinstance(lm_path, str) or (
                    isinstance(lm_path, str) and not os.path.exists(lm_path)
                ):
                    return None

                if not isinstance(tfm_path, str) or (
                    isinstance(tfm_path, str) and not os.path.exists(tfm_path)
                ):
                    # Copy landmark as is (no transform)
                    print(
                        f"No transform found for section {lm_path}, copying landmark as is."
                    )
                    shutil.copy(str(lm_path), str(output_path))

                if output_path and not os.path.exists(output_path) or clobber:
                    simple_ants_apply_tfm(
                        lm_path,
                        raw_path,
                        tfm_path,
                        output_path,
                        ndim=2,
                        n="NearestNeighbor",
                    )

                    # check that every label in lm_path is present in output_path
                    lm_img = nib.load(lm_path)
                    out_img = nib.load(output_path)
                    lm_labels = _label_ids(lm_img.get_fdata())
                    out_labels = _label_ids(out_img.get_fdata())

                    missing_labels = set(lm_labels) - set(out_labels)
                    assert (
                        len(missing_labels) == 0
                    ), f"Missing labels {missing_labels} in warped landmark {output_path}."

        Parallel(n_jobs=1)(
            delayed(process_row)(row, clobber=clobber)
            for _, row in sect_info.iterrows()
        )

        r = max(1, np.rint(ref_section_thickness / section_thickness).astype(int))

        print("Building sparse landmark volume with section thickness ratio", r)

        for _, row in sect_info.iterrows():
            y = int(row["sample"])
            warped_slice_path = row["landmark_2d_rsl"]

            if not warped_slice_path:
                continue

            warped = nib.load(str(warped_slice_path)).get_fdata().astype(np.uint32)

            y0 = int(max(0, y - r))
            y1 = int(min(dims[1], y0 + r))

            # repeat warped to match y1-y0
            warped_rep = np.repeat(warped[:, np.newaxis, :], y1 - y0, axis=1)

            idx = warped_rep > 0

            # this is not ideal because of potential label conflicts but is necessary to prevent loss of labels
            # during transformation
            out_data[:, y0:y1, :][idx] = warped_rep[idx]

        affine = np.eye(4)
        affine[0, 0], affine[1, 1], affine[2, 2] = steps
        affine[0, 3], affine[1, 3], affine[2, 3] = origins

        print("Writing sparse landmark volume to", out_vol_path)

        nib.Nifti1Image(
            out_data.astype(np.uint32), affine, direction_order="lpi"
        ).to_filename(str(out_vol_path))

    return sect_info


def init_landmark_transform(
    out_tfm_h5: str,
    fixed_ref_landmarks: str,
    moving_chunk_landmarks: str,
    transform_type: str = "bspline",  # 'rigid'|'similarity'|'affine'|'bspline'
    mesh_size: str = "5x5x5",
    min_labels_required: int = 12,  # rigid/similarity: 3, affine: 12, bspline: 12 (for now)
    qc_dir: str | None = None,
    clobber: bool = False,
) -> str:
    """Run antsLandmarkBasedTransformInitializer and convert .tfm -> .h5.
    Also (optionally) generate QC: per-label CoM deltas and quick overlays.
    """
    if os.path.exists(out_tfm_h5) and not clobber:
        return out_tfm_h5

    fixed_ref_landmarks = fixed_ref_landmarks
    moving_chunk_landmarks = moving_chunk_landmarks
    out_tfm = out_tfm_h5

    if not os.path.exists(fixed_ref_landmarks):
        raise RuntimeError(f"Reference landmarks not found: {fixed_ref_landmarks}")
    if not os.path.exists(moving_chunk_landmarks):
        raise RuntimeError(
            f"Chunk sparse landmarks not found: {moving_chunk_landmarks}"
        )

    # Build temporary copies that keep only overlapping labels (others -> 0).

    cmd = f"antsLandmarkBasedTransformInitializer 3 {fixed_ref_landmarks} {moving_chunk_landmarks}  {transform_type.lower()} {out_tfm} "

    if transform_type.lower() == "bspline":
        cmd += f" {mesh_size} "

    logger.info(f"[ANTs] {cmd}")
    subprocess.run(cmd, shell=True, executable="/bin/bash")

    assert os.path.exists(out_tfm), f"Landmark transform not created: {out_tfm}"

    return out_tfm


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


def create_qc_images(
    sect_info: pd.DataFrame,
    landmark_volume_rsl_path: str,
    ref_landmark_path: str,
    moving_qc_vol_path: str,
    fixed_qc_vol_path: str,
    qc_dir: str,
    r1: int = 10,
    clobber: bool = False,
) -> None:
    """For each label, create qc image with subplots for 4 coronal sections showing:
    1) <label> from "landmark" overlayed on "raw" image in sect_info
    2) <label> from "landmark_rsl" overlayed on "2d_align" in sect_info
    3) <label> from landmark_volume_rsl_path overlayed on "fixed_qc_vol_path"
    4) <label> from ref_landmark_path overlayed on "fixed_qc_vol_path"
    """
    acq_img, acq_vol, acq_orig, acq_step = load(landmark_volume_rsl_path)
    ref_img, ref_vol, ref_orig, ref_step = load(ref_landmark_path)
    mv_img, mv_vol, mv_orig, mv_step = load(moving_qc_vol_path, dtype=float)
    fx_img, fx_vol, fx_orig, fx_step = load(fixed_qc_vol_path, dtype=float)

    for _, row in sect_info.iterrows():
        landmark_path = row["landmark"]
        landmark_rsl_path = row["landmark_2d_rsl"]

        # if landmark_path is None, continue
        if not landmark_path or not os.path.exists(landmark_path):
            print(f"Landmark path not found for section {row['sample']}, skipping QC.")
            continue

        landmark = np.array(nib.load(landmark_path).dataobj, np.uint32)

        raw = np.array(nib.load(row["raw"]).dataobj, np.uint32)

        _, align_2d, align_2d_orig, align_2d_step = load(row["2d_align"])
        _, landmark_rsl, landmark_rsl_orig, landmark_rsl_step = load(landmark_rsl_path)

        unique_labels = np.unique(landmark.astype(np.uint32))[1:]

        for label in unique_labels:
            label = int(label)

            qc_output_path = f"{qc_dir}/qc_sub-{row['sub']}_hemi-{row['hemisphere']}_chunk-{row['chunk']}_sample-{row['sample']}_label-{label}_qc.png"

            if os.path.exists(qc_output_path) and not clobber:
                print("QC image already exists, skipping:", qc_output_path)
                continue

            # Get center of mass for current label in landmark
            idx0 = np.argwhere(landmark == label)
            assert idx0.size > 0, f"Label {label} not found in landmark."

            idx1 = np.argwhere(landmark_rsl == label)
            assert idx1.size > 0, f"Label {label} not found in landmark_rsl."

            x0_com, z0_com = get_com(landmark, label).astype(int)
            x1_com, z1_com = w2v(
                v2w(
                    get_com(landmark_rsl, label).astype(int),
                    landmark_rsl_orig,
                    landmark_rsl_step,
                ),
                align_2d_orig,
                align_2d_step,
            )

            x2_com, y2_com, z2_com = w2v(
                v2w(get_com(acq_vol, label), acq_orig, acq_step), fx_orig, fx_step
            )
            x3_com, y3_com, z3_com = w2v(
                v2w(get_com(ref_vol, label), ref_orig, ref_step), fx_orig, fx_step
            )

            if (
                np.any(np.isnan([x2_com, y2_com, z2_com, x3_com, y3_com, z3_com]))
                or np.min([x2_com, y2_com, z2_com, x3_com, y3_com, z3_com]) < 0
            ):
                print(
                    f"Warning: label {label} not found in landmark volume for section {row['sample']}, skipping QC."
                )
                continue

            r0 = max(1, int(0.02 * np.min(raw.shape)))
            r1 = max(1, int(0.02 * np.min(align_2d.shape)))
            r2 = max(1, int(0.02 * np.min(mv_vol.shape)))
            r3 = max(1, int(0.02 * np.min(fx_vol.shape)))

            mask_0 = np.zeros_like(raw)
            mask_0[x0_com - r0 : x0_com + r0, z0_com - r0 : z0_com + r0] = 10.0

            mask_1 = np.zeros_like(align_2d)
            mask_1[x1_com - r1 : x1_com + r1, z1_com - r1 : z1_com + r1] = 10.0

            mask_2 = np.zeros_like(fx_vol[:, y2_com, :])
            mask_2[x2_com - r2 : x2_com + r2, z2_com - r2 : z2_com + r2] = 10.0

            mask_3 = np.zeros_like(fx_vol[:, y3_com, :])
            mask_3[x3_com - r3 : x3_com + r3, z3_com - r3 : z3_com + r3] = 10.0

            fig = plt.figure(figsize=(10, 10))
            plt.suptitle(
                f"Subject: {row['sub']}, Hemisphere: {row['hemisphere']}, Chunk: {row['chunk']}, Sample: {row['sample']}, Label: {label}"
            )

            plt.subplot(2, 2, 1)
            plt.title("Original Landmark on Raw")
            plt.imshow(raw.T, cmap="gray", origin="lower")
            plt.imshow(mask_0.T, cmap="Reds", alpha=0.6, origin="lower")

            plt.subplot(2, 2, 2)
            plt.title("2D Aligned Landmark on 2D Align")
            plt.imshow(align_2d.T, cmap="gray", origin="lower")
            plt.imshow(mask_1.T, cmap="Reds", alpha=0.8, origin="lower")

            plt.subplot(2, 2, 3)
            plt.title("Acq Landmark RSL on Fixed QC Volume")
            plt.imshow(fx_vol[:, y2_com, :].T, cmap="gray", origin="lower")
            plt.imshow(mask_2.T, cmap="Reds", alpha=0.8, origin="lower")

            plt.subplot(2, 2, 4)
            plt.title("Ref Landmark on Fixed QC Volume")
            plt.imshow(fx_vol[:, y3_com, :].T, cmap="gray", origin="lower")
            plt.imshow(mask_3.T, cmap="Reds", alpha=0.8, origin="lower")

            plt.tight_layout()
            plt.savefig(qc_output_path, dpi=150)
            plt.close(fig)
            print(f"Saved QC image to {qc_output_path}")


def create_landmark_transform(
    sub: str,
    hemisphere: str,
    chunk: int,
    resolution: float,
    output_dir: str,
    sect_info: pd.DataFrame,
    ref_landmark_path: str,
    landmark_dir: str,
    moving_qc_vol_path,
    fixed_qc_vol_path,
    ymax: int,
    section_thickness,
    num_cores: int = -1,
    transform_type="bspline",
    clobber: bool = False,
) -> str:
    """Process landmarks for alignment.

    :param sub: subject name
    :param hemisphere: hemisphere name
    :param chunk: chunk number
    :param resolution: resolution
    :param output_dir: output directory
    :param sect_info: section information dataframe
    :param init_volume: initial volume path
    :param ref_landmark_path: reference landmark path
    :param num_cores: number of cores to use
    :param clobber: overwrite existing files
    :return: path to the landmark transform file
    """
    qc_dir = f"{output_dir}/qc/"

    os.makedirs(qc_dir, exist_ok=True)

    transform_type = "rigid"  # FIXME

    acq_landmark_path = f"{output_dir}/sub-{sub}_hemi-{hemisphere}_chunk-{chunk}_acq_landmarks_itr-{resolution}mm.nii.gz"
    landmark_tfm_path = f"{output_dir}/sub-{sub}_hemi-{hemisphere}_chunk-{chunk}_landmark_init_itr-{resolution}mm_{transform_type}_Composite.h5"

    if landmark_dir:
        sect_info["landmark"] = find_landmark_files(sect_info, landmark_dir)

    clobber = True
    ref_section_thickness = (nib.load(ref_landmark_path).affine[1, 1],)

    sect_info = build_sparse_landmark_volume(
        acq_landmark_path,
        nib.load(moving_qc_vol_path).affine[0:3, 3],  # origins
        resolution,
        sect_info,
        f"{output_dir}/",
        section_thickness,
        ref_section_thickness,
        ymax,
        n_jobs=num_cores,
        clobber=clobber,
    )

    if not os.path.exists(landmark_tfm_path) or clobber:
        init_landmark_transform(
            landmark_tfm_path,
            ref_landmark_path,
            acq_landmark_path,
            transform_type=transform_type,
            qc_dir=qc_dir,
            clobber=clobber,
        )

    landmark_volume_rsl_path = f"{output_dir}/sub-{sub}_hemi-{hemisphere}_chunk-{chunk}_acq_landmarks_{resolution}mm_rsl.nii.gz"

    simple_ants_apply_tfm(
        acq_landmark_path,
        ref_landmark_path,
        landmark_tfm_path,
        landmark_volume_rsl_path,
        ndim=3,
        n="NearestNeighbor",
        clobber=clobber,
    )

    ar0_img = nib.load(acq_landmark_path)
    ar1_img = nib.load(landmark_volume_rsl_path)
    ar2_img = nib.load(ref_landmark_path)

    ar0, ar1, ar2 = (
        np.array(ar0_img.dataobj),
        np.array(ar1_img.dataobj),
        np.array(ar2_img.dataobj),
    )

    ar0_labels, ar1_labels, ar2_labels = (
        np.unique(ar0)[1:],
        np.unique(ar1)[1:],
        np.unique(ar2)[1:],
    )

    assert set(ar0_labels) == set(
        ar1_labels
    ), "Acq and acq rsl landmark volumes have different labels."
    assert set(ar1_labels) == set(
        ar2_labels
    ), "Acq rsl and ref landmark volumes have different labels."

    moving_qc_vol_rsl_path = f"{qc_dir}/" + os.path.basename(
        moving_qc_vol_path
    ).replace(".nii", "_landmark-rsl.nii")

    if (
        True
        or moving_qc_vol_path
        and fixed_qc_vol_path
        and os.path.exists(moving_qc_vol_path)
        and os.path.exists(fixed_qc_vol_path)
    ):
        if True or not moving_qc_vol_rsl_path.exists() or clobber:
            simple_ants_apply_tfm(
                moving_qc_vol_path,
                fixed_qc_vol_path,
                landmark_tfm_path,
                str(moving_qc_vol_rsl_path),
                ndim=3,
                n="Linear",
                clobber=clobber,
            )
            print("Saved moving landmark rsl to", moving_qc_vol_rsl_path)

    create_qc_images(
        sect_info,
        landmark_volume_rsl_path,
        ref_landmark_path,
        moving_qc_vol_rsl_path,
        fixed_qc_vol_path,
        qc_dir,
        clobber=clobber,
    )
    return landmark_tfm_path


# === END: align/align_landmarks.py ===========================================
