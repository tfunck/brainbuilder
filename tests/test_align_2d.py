"""Tests for initializing the 2D alignment from the previous resolution level."""

import subprocess
from pathlib import Path

import ants
import brainbuilder.align.align_2d as align_2d
import numpy as np
import pandas as pd
import pytest
from scipy.ndimage import gaussian_filter

RESOLUTION = 0.5


def _section(tmp_path: Path) -> tuple:
    """Write a fixed section, a moving section rotated by 70 degrees, and the transform aligning them."""
    rng = np.random.default_rng(3)
    shape = (64, 64)
    fx = np.zeros(shape, np.float32)
    fx[10:50, 14:40] = 1
    fx[30:56, 30:52] = 0.6
    fx = gaussian_filter(fx + 0.3 * gaussian_filter(rng.random(shape), 2), 1.0)
    fx_img = ants.from_numpy(
        fx.astype(np.float32), origin=[0, 0], spacing=[RESOLUTION] * 2
    )
    fx_fn = str(tmp_path / "fx.nii.gz")
    ants.image_write(fx_img, fx_fn)

    # A rotation this large is not recovered by a registration that only runs the
    # finest level of the schedule, so the result depends on the initial transform.
    theta = np.deg2rad(70)
    tfm = ants.create_ants_transform(
        transform_type="AffineTransform",
        dimension=2,
        matrix=np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
        ),
        translation=(2.0, -1.5),
        center=(16.0, 16.0),
    )
    mat_fn = str(tmp_path / "prev.mat")
    ants.write_transform(tfm, mat_fn)
    prev_tfm_fn = str(tmp_path / "prev_Composite.h5")
    subprocess.run(
        f"antsApplyTransforms -d 2 -r {fx_fn} -t {mat_fn} -o CompositeTransform[{prev_tfm_fn}]",
        shell=True,
        check=True,
    )

    mv_fn = str(tmp_path / "mv.nii.gz")
    ants.image_write(
        ants.apply_transforms(fx_img, fx_img, [mat_fn], whichtoinvert=[True]), mv_fn
    )
    return fx_fn, mv_fn, prev_tfm_fn


def test_get_init_tfm(tmp_path: Path) -> None:
    """The previous 2D transform takes precedence over the initalign transform."""
    prev_fn = tmp_path / "prev.h5"
    init_fn = tmp_path / "init.h5"
    prev_fn.touch()
    init_fn.touch()
    row = pd.Series({"init_tfm": str(init_fn)})

    assert align_2d.get_init_tfm(row, str(prev_fn)) == str(prev_fn)
    assert align_2d.get_init_tfm(row, np.nan) == str(init_fn)
    assert align_2d.get_init_tfm(row, str(tmp_path / "missing.h5")) is None
    assert align_2d.get_init_tfm(pd.Series({"init_tfm": np.nan}), None) is None


def test_align_sections_initializes_from_previous_transform(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """use_init_tfm initializes from the 2D transform of the previous resolution level."""
    fx_fn, mv_fn, prev_tfm_fn = _section(tmp_path)

    sect_info = pd.DataFrame(
        [
            {
                "sample": 5,
                "raw": "/raw/sec_5.tif",
                "base": "sec_5",
                "fx": fx_fn,
                "seg_rsl": mv_fn,
                "img": mv_fn,
                "2d_tfm": prev_tfm_fn,
            }
        ]
    )

    init_tfms = []
    affine_trials = align_2d.affine_trials

    def spy_affine_trials(*args, **kwargs) -> str:  # noqa: ANN002, ANN003
        init_tfms.append(kwargs.get("init_tfm"))
        return affine_trials(*args, **kwargs)

    # num_cores=1 runs the joblib workers in this process, so the spy sees the calls
    monkeypatch.setattr(align_2d, "affine_trials", spy_affine_trials)

    sect_info = align_2d.align_sections(
        sect_info,
        str(tmp_path / "align_2d"),
        RESOLUTION,
        [1.0, RESOLUTION],
        file_to_align="seg_rsl",
        use_syn=False,
        use_init_tfm=True,
        num_cores=1,
    )

    assert init_tfms == [prev_tfm_fn]

    # the new transform is written to this level's directory and includes the
    # initial transform, so the moving section ends up aligned to the fixed one
    assert sect_info.loc[0, "2d_tfm"] != prev_tfm_fn
    fx = ants.image_read(fx_fn).numpy()
    aligned = ants.image_read(sect_info.loc[0, "2d_align"]).numpy()
    assert np.corrcoef(fx.ravel(), aligned.ravel())[0, 1] > 0.95
