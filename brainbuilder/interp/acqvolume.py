import os

import numpy as np
import pandas as pd
import stripy as stripy
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter

import brainbuilder.utils.ants_nibabel as nib
from brainbuilder.utils.utils import get_thicken_width


def setup_section_normalization(acquisition, sect_info, array_src):
    # this function is not for chunk normalization but for section normalization
    # # based on surrounding sections
    normalize_sections = False
    mean_list = []
    std_list = []
    y_list = []
    group_mean = 0
    group_std = 0

    sect_info = sect_info.sort_values(["sample"])
    if acquisition in []:  # [ 'cellbody' , 'myelin' ] :
        print("Normalizing", acquisition)

        normalize_sections = True
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
            )  # + new_mean
            array_src[:, y, :] = section
        # plt.plot(mean_list,c='r')
        # plt.plot(new_mean_list,c='b')
        # plt.savefig('/tmp/tmp.png')
        # group_mean=np.mean(mean_list)
        # group_std = np.std(std_list)

    return array_src, normalize_sections


def thicken_sections_within_chunk(
    thickened_fn,
    source_image_fn,
    section_thickness,
    acquisition,
    chunk_sect_info,
    resolution,
    tissue_type="",
    gaussian_sd=0,
):
    print(thickened_fn)
    print(source_image_fn)
    array_img = nib.load(source_image_fn)
    array_src = array_img.get_fdata()

    print(array_src.shape)
    exit(0)

    ystart = array_img.affine[1, 3]

    assert np.sum(array_src) != 0, (
        "Error: source volume for thickening sections is empty\n" + source_image_fn
    )

    array_src, normalize_sections = setup_section_normalization(
        acquisition, chunk_sect_info, array_src
    )

    # get the thicken widith fot the section. The resolution is halfed because we are thickening in both directions
    width = get_thicken_width(resolution, section_thickness)
    print("\t\tThickening sections to ", 0.02 * width * 2)

    dim = [array_src.shape[0], 1, array_src.shape[2]]
    rec_vol = np.zeros_like(array_src)
    n = np.zeros_like(array_src)

    use_conversion_factor = False
    if "conversion_factor" in chunk_sect_info.columns:
        use_conversion_factor = True

    for row_i, row in chunk_sect_info.iterrows():
        y = int(row["sample"])

        # Conversion of radioactivity values to receptor density values
        nl_2d_rsl = row["nl_2d_rsl"]
        section = nib.load(nl_2d_rsl).get_fdata().copy()

        if use_conversion_factor:
            conversion_factor = row["conversion_factor"]
            print(
                "\t\t\tRadio. to Dens.:\t",
                y,
                np.min(section),
                np.max(section),
                conversion_factor,
            )
            section *= conversion_factor

        if np.sum(section) == 0:
            print(f"Warning: empty frame {row_i} {row}\n")

        y0 = int(y) - width if int(y) - width > 0 else 0
        y1 = (
            1 + int(y) + width
            if 1 + int(y) + width < array_src.shape[1]
            else array_src.shape[1]
        )

        # put acquisition sections into rec_vol
        yrange = list(range(y0, y1))

        print("y = ", y, y0, y1)
        rep = np.repeat(section.reshape(dim), len(yrange), axis=1)

        rec_vol[:, yrange, :] += rep
        n[:, yrange, :] += 1

    # normalize by number of sections within range

    assert np.sum(rec_vol) != 0, "Error: thickened volume is empty"
    rec_vol[n > 0] = rec_vol[n > 0] / n[n > 0]
    rec_vol[n == 0] = 0

    if np.sum(gaussian_sd) > 0:
        empty_voxels = rec_vol < np.max(rec_vol) * 0.05
        rec_vol = gaussian_filter(rec_vol, gaussian_sd)
        rec_vol[empty_voxels] = 0

    if "batch_offset" in chunk_sect_info.columns:
        # conversion_factor = chunk_info["conversion_factor"].values[0]
        batch_offset = chunk_sect_info["batch_offset"].values[0]
        # rec_vol = rec_vol * conversion_factor + conversion_offset
        print("\t\t\tbatch_offset", batch_offset)
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
    """Check if all thickened files exist
    :param output_csv: path to csv file containing chunk information
    :return: True if all thickened files exist, False otherwise
    """
    chunk_info = pd.read_csv(output_csv)
    for (chunk, acquisition), chunk_info_row in chunk_info.groupby(
        [
            "chunk",
            "acquisition",
        ]
    ):
        if not os.path.exists(chunk_info_row["thickened"].values[0]):
            return False
    return True


def create_thickened_volumes(
    output_dir: str,
    chunk_info: pd.DataFrame,
    sect_info: pd.DataFrame,
    resolution: float,
    tissue_type: str = "",
    gaussian_sd=0,
    clobber: bool = False,
):
    """Create thickened volumes for each acquisition and each chunk. A thickened volume is simply
    a volume consisting of the sections for a particular acquisition that are expanded along the
    y axis to the resolution of the reconstruction

    :param output_dir: directory for interpolated volumes
    :param chunk_info_csv: dataframe with chunk information
    :param sect_info_csv: dataframe with section information
    :param resolution: resolution of the interpolated volumes
    :param tissue_type: tissue type of the interpolated volumes
    :return: None
    """
    output_csv = f"{output_dir}/chunk_info_thickened_{resolution}mm.csv"

    thickened_files_exist = False

    if os.path.exists(output_csv):
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

            thickened_fn = f"{output_dir}/sub-{sub}_hemi-{hemi}_{int(chunk)}_{acquisition}_{resolution}{tissue_type}_thickened.nii.gz"

            chunk_info_row["acquisition"] = acquisition
            chunk_info_row["thickened"] = thickened_fn

            if not os.path.exists(thickened_fn) or clobber:
                thicken_sections_within_chunk(
                    chunk_info_row["thickened"],
                    chunk_info_row["nl_2d_vol_fn"],
                    chunk_info_row["section_thickness"],
                    acquisition,
                    chunk_sect_info,
                    resolution,
                    gaussian_sd=gaussian_sd,
                )

            chunk_info_out = pd.concat([chunk_info_out, chunk_info_row.to_frame().T])

        chunk_info_out.to_csv(output_csv, index=False)
    return output_csv
