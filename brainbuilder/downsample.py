"""Script for downsampling raw files to reconstruction resolution."""

import os
import re

import numpy as np
import pandas as pd
from joblib import Parallel, cpu_count, delayed

import brainbuilder.utils.ants_nibabel as nib
from brainbuilder.utils import utils


def downsample_sections(
    chunk_info_csv: str,
    sect_info_csv: str,
    resolution: str,
    output_dir: str,
    num_cores: int = None,
    clobber: bool = False,
) -> str:
    """Downsample sections to the lowest resolution in the resolution list.

    :param chunk_info_csv: path to chunk_info.csv
    :param sect_info_csv: path to sect_info.csv
    :param resolution: resolution to downsample to
    :param output_dir: path to output directory
    :param clobber: bool, optional, if True, overwrite existing files, default=False
    :return sect_info_csv: path to updated sect_info.csv
    """
    chunk_info = pd.read_csv(chunk_info_csv)
    sect_info = pd.read_csv(sect_info_csv)

    def get_base(x: str) -> str:
        """Get base filename."""
        if ".nii.gz" not in x:
            x = os.path.splitext(x)[0] + f"_{resolution}mm.nii.gz"
        else:
            x = re.sub(".nii.gz", f"_{resolution}mm.nii.gz", x)

        x = os.path.basename(x)
        return output_dir + "/" + x

    # define downsample img filenames based on current resolution
    sect_info["img"] = sect_info["raw"].apply(get_base)

    os.makedirs(output_dir, exist_ok=True)

    sect_info_csv = output_dir + "/downsample_sect_info.csv"

    run_stage = utils.check_run_stage(
        sect_info["img"], sect_info["raw"], sect_info_csv, clobber=clobber
    )

    if run_stage:
        to_do = []
        for i, row in sect_info.iterrows():
            raw_file = row["raw"]
            downsample_file = row["img"]

            sub = row["sub"]
            hemi = row["hemisphere"]
            chunk = row["chunk"]

            try:
                conversion_factor = row["conversion_factor"]
            except KeyError:
                conversion_factor = 1

            pixel_size_0, pixel_size_1, section_thickness = utils.get_chunk_pixel_size(
                sub, hemi, chunk, chunk_info
            )

            affine = utils.create_2d_affine(
                pixel_size_0, pixel_size_1, section_thickness
            )

            if utils.check_run_stage([downsample_file], [raw_file], clobber=clobber):
                to_do.append(
                    (raw_file, downsample_file, affine, resolution, conversion_factor)
                )

        if num_cores is None or num_cores == 0:
            num_cores = cpu_count()

        Parallel(n_jobs=num_cores, backend="multiprocessing")(
            delayed(utils.resample_to_resolution)(
                raw_file,
                [resolution, resolution],
                downsample_file,
                affine=affine,
                order=1,
                factor=factor,
            )
            for raw_file, downsample_file, affine, resolution, factor in to_do
        )

        sect_info.to_csv(sect_info_csv, index=False)

    for (sub, hemisphere, chunk), chunk_sect_info in sect_info.groupby(
        [
            "sub",
            "hemisphere",
            "chunk",
        ]
    ):
        vol_fn = f"{output_dir}/sub-{sub}_hemi-{hemisphere}_chunk-{chunk}_{resolution}mm.nii.gz"

        if not os.path.exists(vol_fn) or clobber:
            ydim = chunk_sect_info["sample"].max() + 1

            example_img = chunk_sect_info["img"].iloc[0]
            xdim, zdim = nib.load(example_img).shape

            print("Allocate Volume")
            vol = np.zeros((xdim, ydim, zdim), dtype=np.float32)

            for _, tdf in chunk_sect_info.groupby(["acquisition"]):
                for _, row in tdf.iterrows():
                    y = row["sample"]
                    vol[:, y, :] = nib.load(row["img"]).get_fdata()

            pixel_size_0, pixel_size_1, section_thickness = utils.get_chunk_pixel_size(
                sub, hemisphere, chunk, chunk_info
            )
            affine = np.eye(4)
            affine[0, 0] = resolution
            affine[1, 1] = section_thickness
            affine[2, 2] = resolution

            nib.Nifti1Image(vol, affine, direction_order="lpi").to_filename(vol_fn)
    return sect_info_csv
