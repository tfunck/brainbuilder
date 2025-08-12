"""Script for downsampling raw files to reconstruction resolution."""

import os
import re

import numpy as np
import pandas as pd
from joblib import Parallel,  delayed
import nibabel
import brainbuilder.utils.ants_nibabel as nib
from brainbuilder.utils import utils

def compute_max_new_dims(files, resolution, output_dir, num_cores=None):
    """Compute the maximum new dimensions for downsampling."""
    new_resolution = np.array([resolution] * 2)
    max_dim_0 = 0
    max_dim_1 = 0

    num_cores = num_cores if num_cores is not None else -1

    results_file = f"{output_dir}/max_dims_{resolution}mm.npy"

    if os.path.exists(results_file):
        max_dims = np.load(results_file)
        max_dim_0, max_dim_1 = max_dims[0], max_dims[1]
    else:
        def get_image_dims(raw_file):
            vol_shape = nib.load(raw_file).shape
            old_resolution = nibabel.load(raw_file).header.get_zooms()[:3]
            new_dims, _ = utils.get_new_dims(old_resolution, new_resolution, vol_shape)
            print(vol_shape)
            return new_dims

        results = Parallel(n_jobs=num_cores)(
            delayed(get_image_dims)(raw_file) for raw_file in files
        )

        max_dim_0 = max([res[0] for res in results])
        max_dim_1 = max([res[1] for res in results])

        np.save(results_file, np.array([max_dim_0, max_dim_1]))
   
    
    print(f"\t\tMax dimensions for downsampling to {resolution}mm: {max_dim_0}, {max_dim_1}") 

    return max_dim_0, max_dim_1

def assemble_downsampled_volume(sect_info, output_dir, resolution, chunk_info, clobber=False):
    """Assemble downsampled sections into 3D volumes per chunk."""
    for (sub, hemisphere, chunk), chunk_sect_info in sect_info.groupby(
        ["sub", "hemisphere", "chunk"]
    ):
        print("\tAssembling downsampled volume for sub:", sub, "hemisphere:", hemisphere, "chunk:", chunk)
        vol_fn = f"{output_dir}/sub-{sub}_hemi-{hemisphere}_chunk-{chunk}_{resolution}mm.nii.gz"

        if not os.path.exists(vol_fn) or clobber:
            ydim = chunk_sect_info["sample"].max() + 1

            example_img = chunk_sect_info["img"].iloc[0]
            xdim, zdim = nib.load(example_img).shape

            print("Allocate Volume")
            vol = np.zeros((xdim, ydim, zdim), dtype=np.uint8)

            for _, tdf in chunk_sect_info.groupby(["acquisition"]):
                for _, row in tdf.iterrows():
                    y = row["sample"]
                    section = nib.load(row["img"]).get_fdata()
                    vol[:, y, :] = section.astype(np.uint8)

            _, _, section_thickness = utils.get_chunk_pixel_size(
                sub, hemisphere, chunk, chunk_info
            )
            affine = np.eye(4)
            affine[0, 0] = resolution
            affine[1, 1] = section_thickness
            affine[2, 2] = resolution

            nib.Nifti1Image(vol, affine, direction_order="lpi").to_filename(vol_fn)

def downsample_within_chunk(sect_info, chunk_info, resolution, max_dim_0, max_dim_1, num_cores=None, clobber=False):

    to_do = []

    for _, row in sect_info.iterrows():
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

        Parallel(n_jobs=num_cores, backend="multiprocessing")(
            delayed(utils.resample_to_resolution)(
                raw_file,
                [resolution, resolution],
                downsample_file,
                affine=affine,
                order=1,
                factor=factor,
                max_dims=(max_dim_0, max_dim_1),
            )
            for raw_file, downsample_file, affine, resolution, factor in to_do
        )


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
    
    if num_cores is None or num_cores == 0:
        num_cores = -1 

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
        
        for (sub, hemi, chunk), sect_info_sub_hemi_chunk in sect_info.groupby(["sub", "hemisphere", "chunk"]):
            print(f"\tDownsampling sub: {sub}, hemisphere: {hemi}, chunk: {chunk}")

            max_dim_0, max_dim_1 = compute_max_new_dims(sect_info_sub_hemi_chunk['raw'], resolution, output_dir, num_cores)
          
            print("\t\t\tdownsampling within chunk...") 
            downsample_within_chunk(
                sect_info_sub_hemi_chunk,
                chunk_info,
                resolution,
                max_dim_0,
                max_dim_1,     
                num_cores=num_cores,
                clobber=clobber,
            ) 
    
        sect_info.to_csv(sect_info_csv, index=False)

        assemble_downsampled_volume(sect_info, output_dir, resolution, chunk_info, clobber=clobber)

    return sect_info_csv
