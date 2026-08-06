"""Helpers for handling an arbitrary sectioning axis (sagittal/coronal/axial).

BrainBuilder assembles 2D histological sections into a 3D volume by stacking them
along a *sectioning axis*.  Historically this axis was hardcoded to numpy axis ``1``
(coronal sectioning).  These helpers generalize every volume operation so that the
sections can be stacked along any axis:

===========  =============  ==========================
Label        numpy axis     Anatomical plane (LPI vol)
===========  =============  ==========================
``x`` / 0    0              sagittal
``y`` / 1    1              coronal   (default, legacy)
``z`` / 2    2              axial / horizontal
===========  =============  ==========================

The output volume orientation (``direction_order="lpi"``) is unchanged: only *which*
axis the sections stack along changes.  A section's two in-plane dimensions always map
to the two volume axes other than the sectioning axis, kept in ascending order.

The sectioning axis is stored per chunk in the optional ``section_axis`` column of
``chunk_info.csv``.  When the column is absent (legacy datasets) the axis defaults to
``1`` (coronal) so existing pipelines are byte-for-byte unaffected.
"""
from typing import Sequence, Tuple, Union

import numpy as np
import pandas as pd

#: numpy axis used when no ``section_axis`` information is available (coronal).
DEFAULT_SECTION_AXIS = 1

#: Accepted user-facing labels mapped to numpy axis indices.
_AXIS_LABEL_TO_INDEX = {
    "x": 0,
    "y": 1,
    "z": 2,
    "0": 0,
    "1": 1,
    "2": 2,
    0: 0,
    1: 1,
    2: 2,
}

#: Inverse mapping (numpy axis index -> canonical letter label).
AXIS_INDEX_TO_LABEL = {0: "x", 1: "y", 2: "z"}


def map_axis(value: Union[str, int, float, None]) -> int:
    """Map a user-facing sectioning-axis label to a numpy axis index.

    Accepts ``x``/``y``/``z`` (case-insensitive) or ``0``/``1``/``2`` (int or str).
    ``None``/NaN returns :data:`DEFAULT_SECTION_AXIS`.

    :param value: axis label from the user / chunk_info
    :return: numpy axis index (0, 1, or 2)
    :raises ValueError: if the label is not recognized
    """
    if value is None:
        return DEFAULT_SECTION_AXIS

    key: Union[str, int] = value
    if isinstance(value, float):
        if np.isnan(value):
            return DEFAULT_SECTION_AXIS
        key = int(value)
    elif isinstance(value, str):
        key = value.strip().lower()

    try:
        return _AXIS_LABEL_TO_INDEX[key]
    except (KeyError, TypeError):
        raise ValueError(f"Invalid section_axis '{value}'. Use one of x/y/z or 0/1/2.")


def get_section_axis(
    chunk_info: pd.DataFrame,
    sub: object = None,
    hemi: object = None,
    chunk: object = None,
) -> int:
    """Return the sectioning axis for a chunk from ``chunk_info``.

    Falls back to :data:`DEFAULT_SECTION_AXIS` (coronal) when the ``section_axis``
    column is missing or the value is empty, preserving legacy behaviour.

    :param chunk_info: chunk info dataframe (or a single-row selection)
    :param sub: optional subject to filter on
    :param hemi: optional hemisphere to filter on
    :param chunk: optional chunk to filter on
    :return: numpy axis index (0, 1, or 2)
    """
    if chunk_info is None:
        return DEFAULT_SECTION_AXIS

    columns = getattr(chunk_info, "columns", [])
    if "section_axis" not in columns:
        return DEFAULT_SECTION_AXIS

    df = chunk_info
    if sub is not None:
        df = df[df["sub"] == sub]
    if hemi is not None:
        df = df[df["hemisphere"] == hemi]
    if chunk is not None:
        df = df[df["chunk"] == chunk]

    if len(df) == 0:
        return DEFAULT_SECTION_AXIS

    return map_axis(df["section_axis"].values[0])


def section_axis_from_row(row: Union[pd.Series, dict]) -> int:
    """Return the sectioning axis stored in a single chunk/section row.

    :param row: a mapping (dataframe row) that may contain ``section_axis``
    :return: numpy axis index (0, 1, or 2)
    """
    try:
        value = row["section_axis"]
    except (KeyError, TypeError, IndexError):
        return DEFAULT_SECTION_AXIS
    return map_axis(value)


def section_index(
    axis: int, position: Union[int, slice], ndim: int = 3
) -> Tuple[Union[int, slice], ...]:
    """Build an indexing tuple selecting ``position`` along ``axis``.

    ``position`` may be an ``int`` (single section) or a ``slice`` (range of sections).

    :param axis: sectioning axis (numpy index)
    :param position: index or slice along the sectioning axis
    :param ndim: number of dimensions of the target volume
    :return: tuple usable to index a numpy array
    """
    index: list = [slice(None)] * ndim
    index[axis] = position
    return tuple(index)


def get_section(vol: np.ndarray, position: Union[int, slice], axis: int) -> np.ndarray:
    """Read the section (or slab) at ``position`` along ``axis``.

    :param vol: 3D volume
    :param position: index or slice along the sectioning axis
    :param axis: sectioning axis (numpy index)
    :return: the 2D section (int position) or sub-volume (slice position)
    """
    return vol[section_index(axis, position, vol.ndim)]


def set_section(
    vol: np.ndarray, section: np.ndarray, position: Union[int, slice], axis: int
) -> None:
    """Write ``section`` into ``vol`` at ``position`` along ``axis``.

    :param vol: 3D volume (modified in place)
    :param section: 2D section (int position) or sub-volume (slice position)
    :param position: index or slice along the sectioning axis
    :param axis: sectioning axis (numpy index)
    """
    # vol[section_index(axis, position, vol.ndim)] = section
    if axis == 0:
        vol[position, :, :] = section
    elif axis == 1:
        vol[:, position, :] = section
    elif axis == 2:
        vol[:, :, position] = section

    return vol


def add_section(
    vol: np.ndarray, section: np.ndarray, position: Union[int, slice], axis: int
) -> None:
    """Accumulate ``section`` into ``vol`` at ``position`` along ``axis``.

    :param vol: 3D volume (modified in place)
    :param section: 2D section or sub-volume to add
    :param position: index or slice along the sectioning axis
    :param axis: sectioning axis (numpy index)
    """
    vol[section_index(axis, position, vol.ndim)] += section


def inplane_axes(axis: int, ndim: int = 3) -> Tuple[int, ...]:
    """Return the axes orthogonal to the sectioning ``axis`` (the in-plane axes).

    :param axis: sectioning axis (numpy index)
    :param ndim: number of dimensions of the volume
    :return: tuple of in-plane axis indices, ascending
    """
    return tuple(a for a in range(ndim) if a != axis)


def section_profile(vol: np.ndarray, axis: int, reduction=np.max) -> np.ndarray:
    """Reduce over the in-plane axes, giving a 1D profile along the sectioning axis.

    Equivalent to the legacy ``np.max(vol, axis=(0, 2))`` for ``axis == 1``.

    :param vol: 3D volume
    :param axis: sectioning axis (numpy index)
    :param reduction: numpy reduction accepting an ``axis`` tuple (default ``np.max``)
    :return: 1D array with one value per section
    """
    return reduction(vol, axis=inplane_axes(axis, vol.ndim))


def volume_shape(
    section_shape: Sequence[int], n_sections: int, axis: int
) -> Tuple[int, ...]:
    """Build a 3D volume shape by inserting ``n_sections`` at ``axis``.

    ``section_shape`` holds the two in-plane dimensions in ascending volume-axis order.

    :param section_shape: 2D section shape (two in-plane dims)
    :param n_sections: number of sections along the sectioning axis
    :param axis: sectioning axis (numpy index)
    :return: 3D shape tuple
    """
    shape = list(section_shape)
    shape.insert(axis, int(n_sections))
    return tuple(shape)


def alloc_volume(
    section_shape: Sequence[int],
    n_sections: int,
    axis: int,
    dtype: object = np.float32,
) -> np.ndarray:
    """Allocate a zero volume with ``n_sections`` stacked along ``axis``.

    :param section_shape: 2D section shape (two in-plane dims)
    :param n_sections: number of sections along the sectioning axis
    :param axis: sectioning axis (numpy index)
    :param dtype: numpy dtype of the volume
    :return: zero-filled 3D volume
    """
    return np.zeros(volume_shape(section_shape, n_sections, axis), dtype=dtype)


def repeat_section(section: np.ndarray, n: int, axis: int) -> np.ndarray:
    """Insert a size-1 axis at ``axis`` and repeat the 2D ``section`` ``n`` times.

    Equivalent to the legacy ``np.repeat(section.reshape([x, 1, z]), n, axis=1)``.

    :param section: 2D in-plane section
    :param n: number of repetitions along the sectioning axis
    :param axis: sectioning axis (numpy index)
    :return: 3D array of shape with ``n`` along ``axis``
    """
    return np.repeat(np.expand_dims(section, axis), n, axis=axis)


def set_affine_spacing(
    affine: np.ndarray,
    axis: int,
    section_thickness: float,
    inplane_resolution: float,
) -> np.ndarray:
    """Set diagonal spacings: ``section_thickness`` on ``axis``, resolution in-plane.

    :param affine: 4x4 affine (modified in place)
    :param axis: sectioning axis (numpy index)
    :param section_thickness: spacing along the sectioning axis
    :param inplane_resolution: spacing along the two in-plane axes
    :return: the modified affine
    """
    for a in range(3):
        affine[a, a] = section_thickness if a == axis else inplane_resolution
    return affine


def get_affine_spacing(affine: np.ndarray, axis: int) -> float:
    """Return the spacing (voxel size) along the sectioning ``axis``.

    :param affine: 4x4 affine
    :param axis: sectioning axis (numpy index)
    :return: spacing along the sectioning axis
    """
    return affine[axis, axis]


def get_affine_origin(affine: np.ndarray, axis: int) -> float:
    """Return the world-space origin along the sectioning ``axis``.

    :param affine: 4x4 affine
    :param axis: sectioning axis (numpy index)
    :return: origin coordinate along the sectioning axis
    """
    return affine[axis, 3]
