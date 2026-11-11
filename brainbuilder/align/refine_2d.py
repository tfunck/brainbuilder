"""Stage 3.4: refine the 2D section alignment with morphint's section-to-section refinement.

The volume assembled from the 2D-aligned sections in stage 3.3 is passed to
``morphint`` with ``refine_2d_alignment_flag=True`` and
``intersection_fill_flag=False``, so morphint only refines the alignment of each
section to its neighbours and does not interpolate the gaps between sections.
morphint returns one cumulative displacement field per refined section; these are
concatenated with the stage 3.3 transforms so that ``2d_tfm`` maps the
downsampled section directly onto the refined, 2D-aligned section.
"""

import os
import shutil
from glob import glob

import ants
import nibabel
import numpy as np
import pandas as pd
from brainbuilder.align.align_2d import (
    apply_transforms_parallel,
    concatenate_tfm_sections_to_volume,
)
from brainbuilder.utils import utils
from brainbuilder.utils.axis_utils import DEFAULT_SECTION_AXIS, section_profile
from brainbuilder.utils.utils import shell, single_threaded_itk
from joblib import Parallel, cpu_count, delayed, parallel_backend
from morphint.morphint import morphint
from skimage.transform import resize

logger = utils.get_logger(__name__)


def _strip_nii_ext(fn: str) -> str:
    return os.path.basename(fn).replace(".nii.gz", "").replace(".nii", "")


def get_morphint_refine_outputs(
    nl_2d_vol_fn: str, output_dir: str, iterations_per_resolution: int
) -> tuple:
    """Return the paths morphint writes when refining <nl_2d_vol_fn> in <output_dir>.

    These mirror the naming used by ``morphint.morphint`` / ``section_refine.refine_nii``.

    :param nl_2d_vol_fn: volume passed to morphint
    :param output_dir: ``curr_output_dir`` passed to morphint
    :param iterations_per_resolution: number of refinement iterations morphint runs
    :return: refined volume filename, directory with the final cumulative warps (None if absent)
    """
    refine_dir = f"{output_dir}/refine_alignment"
    refined_vol_fn = f"{refine_dir}/{_strip_nii_ext(nl_2d_vol_fn)}_refined-2d.nii.gz"

    # A single-level schedule runs <iterations_per_resolution> iterations, written to
    # iter_<i>_<resolution>mm/. The cumulative warps of the last one are the final warps.
    last_itr = max(int(iterations_per_resolution), 1) - 1
    warp_dirs = sorted(glob(f"{refine_dir}/section_refine/iter_{last_itr}_*mm"))

    assert len(warp_dirs) <= 1, (
        f"Error: expected one final morphint iteration directory, found {warp_dirs}"
    )

    warp_dir = warp_dirs[0] if len(warp_dirs) == 1 else None

    return refined_vol_fn, warp_dir


def get_refined_section_ids(nl_2d_vol_fn: str, axis: int = DEFAULT_SECTION_AXIS) -> set:
    """Return the section indices that morphint refines.

    morphint treats all-zero sections as missing and does not refine the first and
    last acquired sections, because they lack a neighbour on one side.

    :param nl_2d_vol_fn: volume passed to morphint
    :param axis: sectioning axis
    :return: set of refined section indices along <axis>
    """
    vol = nibabel.load(nl_2d_vol_fn).get_fdata()
    valid_idx = np.where(section_profile(vol, axis, reduction=np.max) > 0)[0]
    return {int(y) for y in valid_idx[1:-1]}


def warp_to_section_frame(warp_fn: str, section_fn: str, out_fn: str) -> str:
    """Re-express a morphint section warp in the physical frame of a 2D section image.

    morphint cuts sections out of the 3D volume and only gives them the volume's
    in-plane origin and spacing, whereas the 2D-aligned section images carry their
    own origin, spacing and direction. Both share the same voxel grid, because
    stage 3.3 stacks the sections into the volume index-for-index, so the
    displacements are mapped through voxel space.

    :param warp_fn: morphint displacement field (morphint section frame)
    :param section_fn: 2D-aligned section defining the target frame and grid
    :param out_fn: output displacement field in the frame of <section_fn>
    :return: out_fn
    """
    warp = ants.image_read(warp_fn)
    section = ants.image_read(section_fn)

    disp = warp.numpy().astype(np.float64)

    # physical -> voxel displacements: v = S^-1 D^T d
    disp_vox = (disp @ np.asarray(warp.direction)) / np.asarray(warp.spacing)

    section_shape = tuple(section.shape)
    if disp_vox.shape[:2] != section_shape:
        # Stage 3.3 resizes sections that are a few voxels off the volume grid
        # before stacking them, so resample the warp with the same mapping.
        scale = np.asarray(section_shape) / np.asarray(disp_vox.shape[:2])
        disp_vox = (
            resize(
                disp_vox,
                section_shape + (disp_vox.shape[-1],),
                order=1,
                preserve_range=True,
                anti_aliasing=False,
            )
            * scale
        )

    # voxel -> physical displacements in the section frame: d = D S v
    direction = np.asarray(section.direction)
    disp_section = (disp_vox * np.asarray(section.spacing)) @ direction.T

    ants.image_write(
        ants.from_numpy(
            disp_section.astype(np.float32),
            origin=list(section.origin),
            spacing=list(section.spacing),
            direction=direction,
            has_components=True,
        ),
        out_fn,
    )
    return out_fn


def concatenate_refined_transform(
    warp_fn: str, tfm_2d_fn: str, ref_fn: str, out_fn: str
) -> str:
    """Concatenate a section refinement warp with the stage 3.3 2D transform.

    The refinement warp acts on the 2D-aligned section, i.e. it is applied to the
    output grid points before the 3.3 transform maps them into the downsampled
    section. In antsApplyTransforms order this is ``-t warp -t tfm_2d`` (same
    convention as ``-t Warp -t Affine``).

    :param warp_fn: refinement displacement field in the 2D-aligned section frame
    :param tfm_2d_fn: stage 3.3 transform (downsampled section -> 2D-aligned section)
    :param ref_fn: 2D-aligned section that defines the output grid
    :param out_fn: output composite transform (.h5)
    :return: out_fn
    """
    tfm_str = f"-t {warp_fn}"

    if isinstance(tfm_2d_fn, str) and os.path.exists(tfm_2d_fn):
        tfm_str += f" -t {tfm_2d_fn}"

    shell(
        f"antsApplyTransforms -v 0 -d 2 -r {ref_fn} {tfm_str} -o CompositeTransform[{out_fn}]"
    )

    assert os.path.exists(out_fn), f"Error: refined transform not created {out_fn}"

    return out_fn


def _refine_section_transform(
    warp_fn: str,
    section_fn: str,
    tfm_2d_fn: str,
    section_warp_fn: str,
    refined_tfm_fn: str,
) -> str:
    warp_to_section_frame(warp_fn, section_fn, section_warp_fn)
    return concatenate_refined_transform(
        section_warp_fn, tfm_2d_fn, section_fn, refined_tfm_fn
    )


def get_refine_filenames(tfm_dir: str, row: pd.Series) -> dict:
    """Return the per-section stage 3.4 output filenames.

    Named from the row's ``img`` basename so that the base downsample resolution is
    part of the cache key.

    :param tfm_dir: stage 3.4 transform directory
    :param row: section row
    :return: dictionary of output filenames
    """
    prefix = f"{tfm_dir}/{_strip_nii_ext(row['img'])}_y-{int(row['sample'])}"
    return {
        "section_warp": f"{prefix}_refine_warp.nii.gz",
        "tfm": f"{prefix}_refined_Composite.h5",
        "rsl": f"{prefix}_refined_rsl.nii.gz",
        "cls_rsl": f"{prefix}_refined_cls_rsl.nii.gz",
    }


def refine_2d_alignment(
    sect_info: pd.DataFrame,
    nl_2d_vol_fn: str,
    output_dir: str,
    resolution: float,
    nl_2d_refined_vol_fn: str,
    nl_2d_refined_vol_cls_fn: str,
    axis: int = DEFAULT_SECTION_AXIS,
    relax: float = 0.9,
    iterations_per_resolution: int = 4,
    base_itr: int = 10,
    num_cores: int = 0,
    interpolation: str = "Linear",
    cleanup: bool = True,
    clobber: bool = False,
) -> pd.DataFrame:
    """Refine the stage 3.3 2D alignment by aligning each section to its neighbours.

    Steps:
    1. Run morphint's section refinement on <nl_2d_vol_fn> using only the current
       resolution (single-level schedule) and without filling the gaps between sections.
    2. Convert each section's cumulative morphint warp into the frame of its
       2D-aligned section and concatenate it with the section's 3.3 ``2d_tfm``.
    3. Resample the downsampled sections with the refined transforms and rebuild
       the 2D-aligned volumes.

    Sections that morphint does not refine (first/last acquired section) keep their
    stage 3.3 transform and aligned images.

    :param sect_info: section dataframe after stage 3.3
    :param nl_2d_vol_fn: volume of 2D-aligned sections created in stage 3.3
    :param output_dir: stage 3.4 output directory
    :param resolution: current 2D alignment resolution
    :param nl_2d_refined_vol_fn: output volume of refined sections
    :param nl_2d_refined_vol_cls_fn: output volume of refined sections (uint8)
    :param axis: sectioning axis
    :param relax: morphint per-iteration damping of the refinement increment
    :param iterations_per_resolution: number of refinement iterations at the current resolution
    :param base_itr: base number of ANTs iterations for the section-to-section registrations
    :param num_cores: number of cores, 0 uses all available cores
    :param interpolation: interpolation used to resample the refined sections
    :param cleanup: remove morphint's per-iteration scratch once the refined transforms exist
    :param clobber: overwrite existing outputs
    :return: sect_info with ``2d_tfm``, ``2d_align`` and ``2d_align_cls`` pointing to the refined outputs
    """
    num_cores = cpu_count() if num_cores in (0, None) else num_cores

    sect_info = sect_info.reset_index(drop=True)

    tfm_dir = f"{output_dir}/tfm"
    os.makedirs(tfm_dir, exist_ok=True)

    refined_ids = get_refined_section_ids(nl_2d_vol_fn, axis=axis)

    is_refined = sect_info["sample"].astype(int).isin(refined_ids)

    filenames = {
        idx: get_refine_filenames(tfm_dir, row)
        for idx, row in sect_info.loc[is_refined].iterrows()
    }

    if clobber:
        for fns in filenames.values():
            for fn in fns.values():
                if os.path.exists(fn):
                    os.remove(fn)
        for fn in [nl_2d_refined_vol_fn, nl_2d_refined_vol_cls_fn]:
            if os.path.exists(fn):
                os.remove(fn)

    to_do = [idx for idx, fns in filenames.items() if not os.path.exists(fns["tfm"])]

    if len(to_do) > 0:
        logger.info("\t\t\tRefining 2D alignment with morphint")

        morphint(
            nl_2d_vol_fn,
            output_dir,
            resolution,
            resolution_list=[resolution],
            axis=axis,
            refine_2d_alignment_flag=True,
            intersection_fill_flag=False,
            base_ants_itr=base_itr,
            n_resolutions=1,
            iterations_per_resolution=iterations_per_resolution,
            relax=relax,
            num_jobs=num_cores,
            clobber=clobber,
        )

        _, warp_dir = get_morphint_refine_outputs(
            nl_2d_vol_fn, output_dir, iterations_per_resolution
        )

        warp_fns = {
            idx: f"{warp_dir}/cumulative_{int(sect_info.loc[idx, 'sample'])}.nii.gz"
            for idx in to_do
        }

        missing = [
            fn for fn in warp_fns.values() if warp_dir is None or not os.path.exists(fn)
        ]
        if len(missing) > 0:
            raise RuntimeError(
                f"Missing {len(missing)} morphint refinement warps (e.g. {missing[0]}). "
                "If morphint's scratch directory was removed, rerun with clobber=True."
            )

        with single_threaded_itk(), parallel_backend("loky", inner_max_num_threads=1):
            Parallel(n_jobs=num_cores, backend="loky", prefer="processes")(
                delayed(_refine_section_transform)(
                    warp_fns[idx],
                    sect_info.loc[idx, "2d_align"],
                    sect_info.loc[idx, "2d_tfm"],
                    filenames[idx]["section_warp"],
                    filenames[idx]["tfm"],
                )
                for idx in to_do
            )

    for idx, fns in filenames.items():
        sect_info.loc[idx, "2d_tfm"] = fns["tfm"]
        sect_info.loc[idx, "2d_align_out"] = fns["rsl"]
        sect_info.loc[idx, "2d_align_cls_out"] = fns["cls_rsl"]

    refined_rows = [row for _, row in sect_info.loc[is_refined].iterrows()]

    for tissue_str in ["", "_cls"]:
        with single_threaded_itk(), parallel_backend("loky", inner_max_num_threads=1):
            Parallel(n_jobs=num_cores, backend="loky", prefer="processes")(
                delayed(apply_transforms_parallel)(
                    tfm_dir,
                    resolution,
                    row,
                    tissue_str=tissue_str,
                    interpolation=interpolation,
                )
                for row in refined_rows
            )

    sect_info["2d_align"] = sect_info["2d_align_out"]
    sect_info["2d_align_cls"] = sect_info["2d_align_cls_out"]

    # nl_2d_vol_fn shares the shape and affine of the 3.3 reference volume, so it
    # serves as the template for the refined volumes.
    sect_info = concatenate_tfm_sections_to_volume(
        sect_info, nl_2d_vol_fn, output_dir, nl_2d_refined_vol_fn, axis=axis
    )
    sect_info = concatenate_tfm_sections_to_volume(
        sect_info,
        nl_2d_vol_fn,
        output_dir,
        nl_2d_refined_vol_cls_fn,
        target_str="_cls",
        axis=axis,
        output_dtype=np.uint8,
    )

    if cleanup:
        # The refined composite transforms embed the warps, so the morphint
        # per-iteration sections, pair warps and cumulative warps are no longer needed.
        shutil.rmtree(
            f"{output_dir}/refine_alignment/section_refine", ignore_errors=True
        )
        for fns in filenames.values():
            if os.path.exists(fns["section_warp"]):
                os.remove(fns["section_warp"])

    return sect_info
