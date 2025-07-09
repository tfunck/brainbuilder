"""Create thickened volumes for each acquisition and each chunk."""
import os
import re
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter, label

import brainbuilder.utils.ants_nibabel as nib
from brainbuilder.utils.utils import (
    get_thicken_width,
    simple_ants_apply_tfm,
)


def setup_section_normalization(
    acquisition: str, sect_info: pd.DataFrame, array_src: np.ndarray
) -> Tuple[np.ndarray, bool]:
    """This function is not for chunk normalization but for section normalization based on surrounding sections.

    :param acquisition: acquisition type
    :param sect_info: dataframe with section information
    :param array_src: source array
    :return: array_src, normalize_sections
    """
    normalize_sections = False
    mean_list = []
    std_list = []
    y_list = []

    sect_info = sect_info.sort_values(["sample"])
    if acquisition in [] or normalize_sections:
        print("Normalizing", acquisition)

        for row_i, row in sect_info.iterrows():
            y = int(row["sample"])

            # Conversion of radioactivity values to receptor density values
            section = array_src[:, y, :]

            idx = section >= 0

            mean_list.append(np.mean(section[idx]))
            std_list.append(np.std(section[idx]))
            y_list.append(y)

        mean_list = np.array(mean_list)
        pad = 2
        new_mean_list = []
        i_list = range(len(mean_list))

        new_mean_list.append(mean_list[0])
        for i, y, mean in zip(i_list[1:-1], y_list[1:-1], mean_list[1:-1]):
            i0 = max(i - pad, 0)
            i1 = min(i + pad, len(mean_list) - 1)
            x = list(y_list[i0:i]) + list(y_list[i + 1 : i1 + 1])
            z = list(mean_list[i0:i]) + list(mean_list[i + 1 : i1 + 1])
            kind_dict = {
                1: "nearest",
                2: "linear",
                3: "quadratic",
                4: "cubic",
                5: "cubic",
            }
            interp_f = interp1d(x, z, kind=kind_dict[min(5, len(x))])
            new_mean = interp_f(y)
            new_mean_list.append(new_mean)
        new_mean_list.append(mean_list[-1])

        for y, new_mean in zip(y_list, new_mean_list):
            section = array_src[:, y, :]
            section[section > 0] = (
                new_mean + section[section > 0] - np.mean(section[section > 0])
            )
            array_src[:, y, :] = section

    return array_src, normalize_sections


def thicken_sections_within_chunk(
    thickened_fn: str,
    source_image_fn: str,
    section_thickness: float,
    acquisition: str,
    chunk_sect_info: pd.DataFrame,
    resolution: float,
    tissue_type: str = None,
    gaussian_sd: float = 0,
    width: int = None,
) -> None:
    """Thicken sections within a chunk. A thickened section is simply a section that is expanded along the y axis to the resolution of the reconstruction.

    :param thickened_fn: path to thickened volume
    :param source_image_fn: path to source image
    :param section_thickness: section thickness
    :param acquisition: acquisition type
    :param chunk_sect_info: dataframe with chunk information
    :param resolution: resolution of the interpolated volumes
    :param tissue_type: tissue type of the interpolated volumes
    :param gaussian_sd: standard deviation of gaussian filter
    :return: None
    """
    array_img = nib.load(source_image_fn)
    array_src = array_img.get_fdata()

    assert np.sum(array_src) != 0, (
        "Error: source volume for thickening sections is empty\n" + source_image_fn
    )

    if width is None:
        width = get_thicken_width(resolution, section_thickness, scale=1)

    print("\t\tThickening sections to ", 0.02 * width * 2)

    dim = [array_src.shape[0], 1, array_src.shape[2]]

    rec_vol = np.zeros_like(array_src)
    n = np.zeros_like(array_src)

    for row_i, row in chunk_sect_info.iterrows():
        y = int(row["sample"])

        # Conversion of radioactivity values to receptor density values
        if tissue_type != None:  # FIXME Not great coding
            target_section = f"nl_2d_{tissue_type}_rsl"
        else:
            target_section = "nl_2d_rsl"

        nl_2d_rsl = row[target_section]
        print(target_section)
        print(nl_2d_rsl)

        assert os.path.exists(
            nl_2d_rsl
        ), f"Error: thickened section file {nl_2d_rsl} does not exist\n"

        section = nib.load(nl_2d_rsl).get_fdata().copy()

        if np.sum(section) == 0:
            print(f"Warning: empty frame {row_i} {row}\n")

        y0 = int(y) - width if int(y) - width > 0 else 0
        y1 = (
            1 + int(y) + width
            if 1 + int(y) + width < array_src.shape[1]
            else array_src.shape[1]
        )

        rep = np.repeat(section.reshape(dim), y1 - y0, axis=1)

        rec_vol[:, y0:y1, :] += rep

        n[:, y0:y1, :] += 1

    # normalize by number of sections within range

    assert np.sum(rec_vol) != 0, "Error: thickened volume is empty"

    rec_vol[n > 0] = rec_vol[n > 0] / n[n > 0]
    rec_vol[n == 0] = 0

    if np.sum(gaussian_sd) > 0:
        empty_voxels = rec_vol < np.max(rec_vol) * 0.05
        rec_vol = gaussian_filter(rec_vol, gaussian_sd)
        rec_vol[empty_voxels] = 0

    if "batch_offset" in chunk_sect_info.columns:
        batch_offset = chunk_sect_info["batch_offset"].values[0]
        rec_vol = rec_vol + batch_offset

    assert np.sum(rec_vol) != 0, "Error: thickened volume is empty"
    print(
        "\t\t\tThickened volume:\t",
        np.min(rec_vol),
        np.mean(rec_vol[rec_vol > rec_vol.min()]),
        np.max(rec_vol),
    )

    print("\tthickened_fn", thickened_fn)
    nib.Nifti1Image(rec_vol, array_img.affine, direction_order="lpi").to_filename(
        thickened_fn
    )

    return rec_vol


def check_all_thickened_files_exist(output_csv: str) -> bool:
    """Check if all thickened files exist.

    :param output_csv: path to csv file containing chunk information
    :return: True if all thickened files exist, False otherwise
    """
    if not os.path.exists(output_csv):
        return False

    chunk_info = pd.read_csv(output_csv)
    for (chunk, acquisition), chunk_info_row in chunk_info.groupby(
        [
            "chunk",
            "acquisition",
        ]
    ):
        if not os.path.exists(chunk_info_row["thickened"].values[0]):
            return False

        if not os.path.exists(chunk_info_row["thickened_stx"].values[0]):
            return False
    return True


def get_section_intervals(vol: np.ndarray) -> list:
    """Get the intervals of sections within a volume across y-axis of volume.

    :param vol: np.array, volume
    :return: list
    """
    section_max = np.max(vol, axis=(0, 2))
    section_min = np.min(vol, axis=(0, 2))

    valid_sections = section_max != section_min

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


def create_distance_volume(volume_filename: str, distance_filename: str) -> np.ndarray:
    """Create a volume that represents distances from acquired sections.

    :param volume_filename: path to volume
    :param distance_filename: path to distance volume
    :return: np.array
    """
    img = nib.load(volume_filename)
    vol = img.get_fdata()

    intervals = get_section_intervals(vol)

    out_vol = np.zeros(vol.shape)
    for i in range(len(intervals) - 1):
        j = i + 1
        x0, x1 = intervals[i]
        y0, y1 = intervals[j]
        x = np.mean(vol[:, x0:x1, :], axis=1)
        vol[:, x0:x1, :] = np.repeat(
            x.reshape(x.shape[0], 1, x.shape[1]), x1 - x0, axis=1
        )
        for ii in range(x1, y0):
            den = y0 - x1
            assert den != 0, "Error: 0 denominator when interpolating missing sections"
            d0 = ii - x1
            d1 = y1 - ii
            d = min(d0, d1)
            out_vol[:, ii, :] = d

    nib.Nifti1Image(out_vol, img.affine, direction_order="lpi").to_filename(
        distance_filename
    )

    return out_vol


def transform_chunk_volumes(
    df: pd.DataFrame,
    struct_vol_rsl_fn: str,
    output_dir: str,
    vol_str="thickened",
    vol_stx_str="thickened_stx",
    clobber: bool = False,
) -> str:
    """Transform thickened chunk volumes to structural volume."""
    output_csv = f"{output_dir}/chunk_info_thickened_stx.csv"

    if not os.path.exists(output_csv) or clobber:
        for (sub, hemisphere, chunk, acquisition), chunk_df in df.groupby(
            ["sub", "hemisphere", "chunk", "acquisition"]
        ):
            thickened_fn = chunk_df[vol_str].values[0]
            thickened_stx_fn = chunk_df[vol_stx_str].values[0]

            nl_3d_tfm_fn = chunk_df["nl_3d_tfm_fn"].values[0]

            if not os.path.exists(thickened_stx_fn) or clobber:
                print(
                    f"\tTransforming {thickened_fn} to stx space:\n\t {thickened_stx_fn}"
                )

                simple_ants_apply_tfm(
                    thickened_fn,
                    struct_vol_rsl_fn,
                    nl_3d_tfm_fn,
                    thickened_stx_fn,
                    n="NearestNeighbor",
                    clobber=clobber,
                )

            idx = (
                (df["sub"] == sub)
                & (df["hemisphere"] == hemisphere)
                & (df["chunk"] == chunk)
                & (df["acquisition"] == acquisition)
            )

            df[vol_stx_str].loc[idx] = thickened_stx_fn

        df.to_csv(output_csv, index=False)
    else:
        df = pd.read_csv(output_csv)

    return df


def create_thickened_volumes(
    output_dir: str,
    chunk_info: pd.DataFrame,
    sect_info: pd.DataFrame,
    resolution: float,
    struct_vol_rsl_fn: str,
    tissue_type: str = None,
    gaussian_sd: float = 0,
    width: int = None,
    clobber: bool = False,
) -> str:
    """Create thickened volumes for each acquisition and each chunk.

    A thickened volume is simply a volume consisting of the sections for a particular acquisition that are expanded along the
    y axis to the resolution of the reconstruction.

    :param output_dir: directory for interpolated volumes
    :param chunk_info_csv: dataframe with chunk information
    :param sect_info_csv: dataframe with section information
    :param resolution: resolution of the interpolated volumes
    :param tissue_type: tissue type of the interpolated volumes
    :param gaussian_sd: standard deviation of gaussian filter
    :param clobber: overwrite existing files
    :return: None
    """
    os.makedirs(output_dir, exist_ok=True)

    output_csv = f"{output_dir}/chunk_info_thickened_{resolution}mm.csv"

    thickened_files_exist = False

    if os.path.exists(output_csv) or clobber:
        thickened_files_exist = check_all_thickened_files_exist(output_csv)

    if not os.path.exists(output_csv) or not thickened_files_exist or clobber:
        chunk_info_out = pd.DataFrame({})

        for (
            sub,
            hemisphere,
            chunk,
            acquisition,
        ), chunk_sect_info in sect_info.groupby(
            [
                "sub",
                "hemisphere",
                "chunk",
                "acquisition",
            ]
        ):
            idx = chunk_info["chunk"] == chunk
            chunk_info_row = chunk_info[idx].iloc[0]

            if tissue_type is not None:
                thickened_fn = f"{output_dir}/sub-{sub}_hemi-{hemisphere}_{int(chunk)}_{acquisition}_{resolution}_{tissue_type}_thickened.nii.gz"
            else:
                thickened_fn = f"{output_dir}/sub-{sub}_hemi-{hemisphere}_{int(chunk)}_{acquisition}_{resolution}_thickened.nii.gz"

            thickened_stx_fn = re.sub(".nii.gz", "_space-stx.nii.gz", thickened_fn)

            chunk_info_row["acquisition"] = acquisition
            chunk_info_row["thickened"] = thickened_fn
            chunk_info_row["thickened_stx"] = thickened_stx_fn

            if not os.path.exists(thickened_fn) or clobber:
                thicken_sections_within_chunk(
                    chunk_info_row["thickened"],
                    chunk_info_row["nl_2d_vol_fn"],
                    chunk_info_row["section_thickness"],
                    acquisition,
                    chunk_sect_info,
                    resolution,
                    tissue_type=tissue_type,
                    gaussian_sd=gaussian_sd,
                    width=width,
                )

            chunk_info_out = pd.concat([chunk_info_out, chunk_info_row.to_frame().T])

        chunk_info_out.to_csv(output_csv, index=False)

    return output_csv
