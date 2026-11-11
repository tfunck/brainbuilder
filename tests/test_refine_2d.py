"""Tests for stage 3.4: refinement of the 2D alignment with morphint."""

import os
import subprocess
from pathlib import Path

import ants
import nibabel
import numpy as np
import pandas as pd
import pytest
from brainbuilder.align.align_2d import (
    apply_transforms_parallel,
    concatenate_tfm_sections_to_volume,
    get_align_filenames,
    get_base_from_raw,
)
from brainbuilder.align.refine_2d import (
    get_morphint_refine_outputs,
    get_refine_filenames,
    refine_2d_alignment,
    warp_to_section_frame,
)
from brainbuilder.utils import ants_nibabel as nib
from brainbuilder.utils.axis_utils import get_section, volume_shape
from scipy.ndimage import gaussian_filter

RESOLUTION = 0.4
SECTION_SHAPE = (40, 48)
SAMPLES = [0, 2, 4, 6, 9, 11, 13, 15]
# Non-trivial in-plane frame so that the morphint frame (volume in-plane origin,
# identity direction) and the section frame differ.
SECTION_ORIGIN = [7.5, -4.0]
SECTION_DIRECTION = np.array([[-1.0, 0.0], [0.0, 1.0]])


def _anatomy(shape: tuple) -> np.ndarray:
    rng = np.random.default_rng(1)
    img = np.zeros(shape, np.float32)
    img[8:-8, 8:-8] = 1
    img += gaussian_filter(rng.random(shape), 2).astype(np.float32)
    img = gaussian_filter(img, 1.5)
    return (255 * (img - img.min()) / np.ptp(img)).astype(np.float32)


def _make_stage_3_3(tmp_path: Path, axis: int) -> tuple:
    """Write a synthetic chunk as it looks after stage 3.3."""
    raw_dir = tmp_path / "img"
    align_2d_dir = tmp_path / "3.3_align_2d"
    raw_dir.mkdir()
    align_2d_dir.mkdir()

    anatomy = _anatomy(SECTION_SHAPE)
    rng = np.random.default_rng(2)

    rows = []
    for y in SAMPLES:
        img_fn = str(raw_dir / f"sec_{y}_{RESOLUTION}mm.nii.gz")
        ants.image_write(
            ants.from_numpy(
                anatomy,
                origin=SECTION_ORIGIN,
                spacing=[RESOLUTION] * 2,
                direction=SECTION_DIRECTION,
            ),
            img_fn,
        )

        # per-section 2D transforms that leave the sections slightly misaligned
        # with respect to each other
        theta = np.deg2rad(rng.uniform(-3, 3))
        tx = ants.create_ants_transform(
            transform_type="AffineTransform",
            dimension=2,
            matrix=np.array(
                [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
            ),
            translation=tuple(rng.uniform(-0.75, 0.75, 2)),
            center=(-2.5, 6.0),
        )
        mat_fn = str(raw_dir / f"sec_{y}.mat")
        tfm_fn = str(raw_dir / f"sec_{y}_Composite.h5")
        ants.write_transform(tx, mat_fn)
        subprocess.run(
            f"antsApplyTransforms -d 2 -r {img_fn} -t {mat_fn} -o CompositeTransform[{tfm_fn}]",
            shell=True,
            check=True,
        )

        rows.append(
            {
                "sub": "01",
                "hemisphere": "L",
                "chunk": 1,
                "acquisition": "synth",
                "raw": f"/raw/sec_{y}.tif",
                "sample": y,
                "img": img_fn,
                "section_axis": axis,
                "tfm_3_3": tfm_fn,
            }
        )

    sect_info = pd.DataFrame(rows)
    sect_info["base"] = sect_info["raw"].apply(get_base_from_raw)

    # mimic align_2d.align_sections: 2d_tfm + resampled 2d_align(_cls) sections
    tfm_dir = str(align_2d_dir / "tfm")
    sect_info = get_align_filenames(tfm_dir, sect_info)
    for idx, row in sect_info.iterrows():
        os.rename(row["tfm_3_3"], row["2d_tfm"])
    for tissue_str in ["", "_cls"]:
        for _, row in sect_info.iterrows():
            apply_transforms_parallel(tfm_dir, RESOLUTION, row, tissue_str=tissue_str)
    sect_info["2d_align"] = sect_info["2d_align_out"]
    sect_info["2d_align_cls"] = sect_info["2d_align_cls_out"]

    # 3.3 volume, written 'lpi' like the reference volume in native space
    affine = np.eye(4)
    affine[:3, 3] = [-10.0, 3.0, 12.0]
    for a in range(3):
        affine[a, a] = RESOLUTION
    affine[axis, axis] = 0.2
    template_fn = str(align_2d_dir / "ref_space-nat.nii.gz")
    dims = volume_shape(SECTION_SHAPE, max(SAMPLES) + 1, axis)
    nib.Nifti1Image(
        np.zeros(dims, np.float32), affine, direction_order="lpi"
    ).to_filename(template_fn)
    nl_2d_vol_fn = str(align_2d_dir / "chunk_nl_2d.nii.gz")
    concatenate_tfm_sections_to_volume(
        sect_info, template_fn, str(align_2d_dir), nl_2d_vol_fn, axis=axis
    )

    return sect_info, nl_2d_vol_fn


def test_warp_to_section_frame_reproduces_morphint_resampling(tmp_path: Path) -> None:
    """A warp converted to the section frame resamples the same voxels as morphint."""
    rng = np.random.default_rng(0)
    img = _anatomy(SECTION_SHAPE)
    spacing = [RESOLUTION] * 2

    morphint_origin = [-33.0, 4.0]
    disp = gaussian_filter(rng.normal(size=SECTION_SHAPE + (2,)), (4, 4, 0)) * 4
    ants.image_write(
        ants.from_numpy(
            disp.astype(np.float32),
            origin=morphint_origin,
            spacing=spacing,
            has_components=True,
        ),
        str(tmp_path / "warp_morphint.nii.gz"),
    )
    section_morphint = ants.from_numpy(img, origin=morphint_origin, spacing=spacing)
    expected = ants.apply_transforms(
        section_morphint, section_morphint, [str(tmp_path / "warp_morphint.nii.gz")]
    ).numpy()

    section_fn = str(tmp_path / "section.nii.gz")
    ants.image_write(
        ants.from_numpy(
            img, origin=SECTION_ORIGIN, spacing=spacing, direction=SECTION_DIRECTION
        ),
        section_fn,
    )
    warp_fn = warp_to_section_frame(
        str(tmp_path / "warp_morphint.nii.gz"),
        section_fn,
        str(tmp_path / "warp_section.nii.gz"),
    )
    section = ants.image_read(section_fn)
    result = ants.apply_transforms(section, section, [warp_fn]).numpy()

    assert np.abs(expected - img).max() > 10
    np.testing.assert_allclose(result, expected, atol=1e-3)


@pytest.mark.parametrize("axis", [1, 2])
def test_refine_2d_alignment(tmp_path: Path, axis: int) -> None:
    """Refined transforms take the downsampled sections to morphint's refined sections."""
    sect_info, nl_2d_vol_fn = _make_stage_3_3(tmp_path, axis)
    sect_info_3_3 = sect_info.copy()

    output_dir = str(tmp_path / "3.4_refine_2d")
    refined_vol_fn = f"{output_dir}/chunk_nl_2d_refined.nii.gz"
    refined_cls_fn = f"{output_dir}/chunk_nl_2d_refined_cls.nii.gz"

    kwargs = dict(
        axis=axis,
        iterations_per_resolution=2,
        base_itr=10,
        num_cores=2,
    )

    sect_info_out = refine_2d_alignment(
        sect_info.copy(),
        nl_2d_vol_fn,
        output_dir,
        RESOLUTION,
        refined_vol_fn,
        refined_cls_fn,
        cleanup=False,
        **kwargs,
    )

    # morphint refined the 3.3 volume without filling the gaps between sections
    morphint_vol_fn, warp_dir = get_morphint_refine_outputs(nl_2d_vol_fn, output_dir, 2)
    assert warp_dir is not None and os.path.exists(morphint_vol_fn)
    # single-level schedule at the current resolution
    assert os.path.basename(warp_dir) == f"iter_1_{RESOLUTION}mm"
    morphint_vol = nibabel.load(morphint_vol_fn).get_fdata()
    for y in set(range(max(SAMPLES) + 1)) - set(SAMPLES):
        assert np.max(get_section(morphint_vol, y, axis)) == 0

    endpoints = {min(SAMPLES), max(SAMPLES)}
    nl_2d_vol = nibabel.load(nl_2d_vol_fn).get_fdata()
    refinement, error, error_wrong_order = [], [], []
    for idx, row in sect_info_out.iterrows():
        y = int(row["sample"])

        if y in endpoints:
            for col in ["2d_tfm", "2d_align", "2d_align_cls"]:
                assert row[col] == sect_info_3_3.loc[idx, col]
            continue

        assert row["2d_tfm"].startswith(output_dir)
        assert row["2d_tfm"].endswith("_refined_Composite.h5")
        assert row["2d_align"].startswith(output_dir) and os.path.exists(
            row["2d_align"]
        )
        assert os.path.exists(row["2d_align_cls"])

        morphint_section = get_section(morphint_vol, y, axis)
        section_3_3 = get_section(nl_2d_vol, y, axis)

        # The morphint warp, converted to the section frame and applied to the 3.3
        # section, reproduces morphint's refined section exactly.
        section_warp_fn = get_refine_filenames(f"{output_dir}/tfm", row)["section_warp"]
        aligned_3_3 = ants.image_read(sect_info_3_3.loc[idx, "2d_align"])
        two_step = ants.apply_transforms(
            aligned_3_3, aligned_3_3, [section_warp_fn]
        ).numpy()
        np.testing.assert_allclose(two_step, morphint_section, atol=1e-3)

        # The concatenated transform maps the downsampled image straight onto the
        # refined section, so it only differs from morphint by one interpolation.
        img = ants.image_read(row["img"])
        refined = ants.apply_transforms(img, img, [row["2d_tfm"]]).numpy()
        np.testing.assert_allclose(
            refined, ants.image_read(row["2d_align"]).numpy(), atol=1e-3
        )

        wrong_order_fn = str(tmp_path / f"wrong_order_{y}.h5")
        subprocess.run(
            f"antsApplyTransforms -d 2 -r {row['img']} -t {sect_info_3_3.loc[idx, '2d_tfm']} "
            f"-t {section_warp_fn} -o CompositeTransform[{wrong_order_fn}]",
            shell=True,
            check=True,
        )
        wrong_order = ants.apply_transforms(img, img, [wrong_order_fn]).numpy()

        refinement.append(np.abs(morphint_section - section_3_3).mean())
        error.append(np.abs(morphint_section - refined).mean())
        error_wrong_order.append(np.abs(morphint_section - wrong_order).mean())

    assert np.mean(refinement) > 1
    assert np.mean(error) < 0.3 * np.mean(refinement)
    assert np.mean(error) < 0.75 * np.mean(error_wrong_order)

    # refined volumes are assembled on the 3.3 grid
    refined_img = nibabel.load(refined_vol_fn)
    assert refined_img.shape == nibabel.load(nl_2d_vol_fn).shape
    np.testing.assert_allclose(refined_img.affine, nibabel.load(nl_2d_vol_fn).affine)
    assert nibabel.load(refined_cls_fn).get_data_dtype() == np.uint8

    # cleanup removes morphint's scratch and reruns reuse the refined transforms
    sect_info_rerun = refine_2d_alignment(
        sect_info_3_3.copy(),
        nl_2d_vol_fn,
        output_dir,
        RESOLUTION,
        refined_vol_fn,
        refined_cls_fn,
        cleanup=True,
        **kwargs,
    )
    assert get_morphint_refine_outputs(nl_2d_vol_fn, output_dir, 2)[1] is None
    pd.testing.assert_frame_equal(sect_info_rerun, sect_info_out)
