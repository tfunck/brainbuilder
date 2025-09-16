"""Main function for performing interpolation between acquired 2D sections."""
import os

import numpy as np
import pandas as pd

from brainbuilder.interp.surfinterp import (
    combine_volumes,
    create_final_reconstructed_volume,
    volumes_to_surface_profiles,
)
from brainbuilder.interp.volinterp import volumetric_pipeline
from brainbuilder.utils.utils import get_logger

logger = get_logger(__name__)

METHOD_SURFACE = "surface"
METHOD_VOLUMETRIC = "volumetric"


def interpolate_missing_sections(
    hemi_info_csv: str,
    chunk_info_csv: str,
    sect_info_csv: str,
    resolution: float,
    resolution_list: list,
    output_dir: str,
    n_depths: int = 0,
    final_resolution: float = None,
    interp_method: str = METHOD_VOLUMETRIC,
    interpolation: str = "Linear",
    num_cores: int = -1,
    clobber: bool = False,
) -> str:
    """Interpolate missing sections in a volume.

    :param sect_info: a dataframe with the following columns:
    :param brain_mask_fn: path to the brain mask
    :param resolution: resolution of the volume
    :param clobber: if True, overwrite existing files
    :return: path to the interpolated volume
    """
    logger.info("\n\tInterpolate Missing Sections\n")

    sect_info = pd.read_csv(sect_info_csv, index_col=False)
    chunk_info = pd.read_csv(chunk_info_csv, index_col=False)
    hemi_info = pd.read_csv(hemi_info_csv, index_col=False)

    os.makedirs(output_dir, exist_ok=True)

    interp_chunk_info = volumetric_pipeline(
        sect_info,
        chunk_info,
        hemi_info,
        resolution,
        resolution_list,
        output_dir,
        final_resolution=final_resolution,
        interpolation=interpolation,
        num_cores=num_cores,
        clobber=clobber,
    )
    assert (
        "acquisition" in interp_chunk_info.columns
    ), "Error: 'acquisition' column not found in chunk info"

    n_chunks = interp_chunk_info["chunk"].nunique()

    if n_chunks > 0:
        for (sub, hemisphere, acq), curr_sect_info in sect_info.groupby(
            ["sub", "hemisphere", "acquisition"]
        ):
            curr_hemi_info = hemi_info.loc[
                (hemi_info["sub"] == sub) & (hemi_info["hemisphere"] == hemisphere)
            ]
            assert len(curr_hemi_info) > 0, "Error: no hemisphere info found"

            acq_chunk_info = interp_chunk_info.loc[
                (interp_chunk_info["sub"] == sub)
                & (interp_chunk_info["hemisphere"] == hemisphere)
                & (interp_chunk_info["acquisition"] == acq)
            ]
            assert len(acq_chunk_info) > 0, "Error: no chunk info found"

            ref_vol_fn = curr_hemi_info["struct_ref_vol"].values[0]

            curr_output_dir = f"{output_dir}/sub-{sub}/hemi-{hemisphere}/acq-{acq}/"

            os.makedirs(curr_output_dir, exist_ok=True)

            logger.info(
                f"\t\tInterpolating sections for sub-{sub}, hemi-{hemisphere}, acq-{acq} at {resolution}mm resolution with method {interp_method}"
            )

            if interp_method == METHOD_VOLUMETRIC:
                reconstructed_cortex_fn = f"{curr_output_dir}/sub-{sub}_hemi-{hemisphere}_acq-{acq}_{resolution}mm_cortex.nii.gz"

                logger.info(
                    f"\t\t\tReconstructed cortex file: {reconstructed_cortex_fn}"
                )

                ref_vol_rsl_fn = acq_chunk_info["ref_vol_rsl_fn"].values[
                    0
                ]  # if 'ref_vol_rsl_fn' in acq_chunk_info.columns else ref_vol_fn

                combine_volumes(
                    acq_chunk_info["interp_stx"].values,
                    reconstructed_cortex_fn,
                    mask_fn=ref_vol_rsl_fn,
                    clobber=clobber,
                )

            elif interp_method == METHOD_SURFACE:
                reconstructed_cortex_fn = f"{curr_output_dir}/sub-{sub}_hemi-{hemisphere}_acq-{acq}_{resolution}mm_l{n_depths}_cortex.nii.gz"

                depth_list = np.round(np.linspace(0, 1, int(n_depths)), 3)

                gm_surf_fn = curr_hemi_info["gm_surf"].values[0]
                wm_surf_fn = curr_hemi_info["wm_surf"].values[0]

                (
                    stx_vol_dict,
                    _,
                    _,
                    profiles_fn,
                    chunk_info,
                    struct_vol_rsl_fn,
                ) = volumes_to_surface_profiles(
                    acq_chunk_info,
                    curr_sect_info,
                    resolution,
                    curr_output_dir,
                    output_dir + "/surfaces/",
                    ref_vol_fn,
                    gm_surf_fn,
                    wm_surf_fn,
                    depth_list=depth_list,
                    volume_type="interp_nat",
                    interp_order=1,
                    clobber=clobber,
                )

                create_final_reconstructed_volume(
                    reconstructed_cortex_fn,
                    acq_chunk_info,
                    struct_vol_rsl_fn,
                    resolution,
                    stx_vol_dict,
                    profiles_fn,
                    volume_type="interp_stx",
                    clobber=True,  # clobber
                )

    chunk_info_csv = f"{output_dir}/reconstructed_chunk_info.csv"
    interp_chunk_info.to_csv(chunk_info_csv, index=False)

    return chunk_info_csv


if __name__ == "__main__":
    pass
