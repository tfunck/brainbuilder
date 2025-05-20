"""Validate inputs to BrainBuilder."""
import os
from typing import Union

import numpy as np
import pandas as pd
from joblib import Parallel, cpu_count, delayed

import brainbuilder.utils.utils as utils
from brainbuilder.utils.mesh_io import load_mesh_ext

global chunk_info_required_columns
global sect_info_required_columns


class Column:
    """Class to represent a column in a dataframe."""

    def __init__(self, name: str, kind: str, required: bool = True) -> None:
        """Initialize a column.

        :param name: name of the column
        :param kind: kind of column, either "volume" or "surface"
        :param required: boolean indicating whether the column is required
        """
        self.name = str(name)
        self.kind = kind
        self.required = required

    def validate_rows_in_column(self, rows: pd.DataFrame, n_jobs: int = None) -> bool:
        """Validate that the rows in a column based on whether they are volumes or surfaces.

        :param rows: list of rows in the column
        :return: boolean indicating whether the inputs have been validated
        """
        valid_inputs = True

        if self.required:

            def val_func(x: Union[int, str, float, bool]) -> bool:
                """Return inputs."""
                return x

            if self.kind == "volume":
                val_func = validate_volume
            elif self.kind == "surface":
                pass
            elif self.kind == int:

                def val_func(x: Union[int, float, str]) -> bool:
                    """Validate entry by verifying that it is an integer."""
                    return isinstance(x, np.integer)
            elif self.kind == float:

                def val_func(x: Union[int, float, str]) -> bool:
                    """Validate entry by verifying that it is a float."""
                    return isinstance(x, float)
            else:
                return True

            if n_jobs is None:
                n_jobs = int(cpu_count() )

            validated_rows = Parallel(n_jobs=n_jobs)(
                delayed(val_func)(var) for var in rows
            )
            valid_inputs = np.product(np.array(validated_rows))

            if not valid_inputs:
                invalid_rows = rows[~np.array(validated_rows)]
                invalid_row_types = [type(row) for row in invalid_rows]

        return valid_inputs


# Strings
sub = Column("sub", None)
hemisphere = Column("hemisphere", None)
chunk = Column("chunk", None)
direction = Column("direction", None)
pixel_size_0 = Column("pixel_size_0", float)
pixel_size_1 = Column("pixel_size_1", float)
section_thickness = Column("section_thickness", float)
acquisition = Column("acquisition", None)
sample = Column("sample", int)

# Volumes
raw = Column("raw", "volume")
struct_ref_vol = Column("struct_ref_vol", "volume")

# Surfaces
gm_surf = Column("gm_surf", "surface", False)
wm_surf = Column("wm_surf", "surface", False)


chunk_info_required_columns = [
    sub,
    hemisphere,
    chunk,
    direction,
    pixel_size_0,
    pixel_size_1,
    section_thickness,
]
sect_info_required_columns = [acquisition, sub, hemisphere, chunk, raw, sample]
hemi_info_required_columns = [sub, hemisphere, struct_ref_vol, gm_surf, wm_surf]


def validate_dataframe(
    df: pd.DataFrame, required_columns: list, n_jobs: int = None
) -> bool:
    """Validate that the dataframe has the required columns.

    :param df: dataframe to validate
    :param required_columns: list of required columns
    :return: boolean indicating whether the inputs have been validated
    """
    valid_columns = []
    for column in required_columns:
        if column.name not in df.columns:
            valid_inputs = False
            print(f"\tMissing input: required field <{column.name}> not found in .csv")
        else:
            validated = column.validate_rows_in_column(
                df[column.name].values, n_jobs=n_jobs
            )
            print(column.name, validated)
            valid_columns.append(validated)

    valid_inputs = np.product(np.array(valid_columns))
    return valid_inputs


def validate_csv(df_csv: str, required_columns: list, n_jobs: int = None) -> bool:
    """Validate the entries of a .csv file.

    :param df_csv:             Path to .csv file with information on sections to reconstruct
    :param required_columns:   List of column names that must be present in the .csv file
    :return valid_inputs:        boolean indicating whether the inputs have been validated
    """
    valid_inputs = True

    if not os.path.exists(df_csv):
        valid_inputs = False
        print(f"\tMissing input: .csv with section info does not exist {df_csv}")
    else:
        print(f"\tValidating {df_csv}")
        df = pd.read_csv(df_csv)
        valid_inputs = validate_dataframe(df, required_columns, n_jobs=n_jobs)

    return valid_inputs


def validate_surface(surface_filename: str) -> bool:
    """Validate that a surface exists and that it is not empty.

    :param surface_filename:       Path to .nii.gz file that serves as structural reference volume to use for reconstruction
    :param volume_filename:       Path to .nii.gz file that corresponds to surface_filename
    :param valid_inputs:        boolean indicating whether the inputs have been validated
    """
    valid_inputs = True

    if not isinstance(surface_filename, str):
        print(f"\tMissing input: surface_filename is not a string {surface_filename}")
        valid_inputs = False
        return valid_inputs

    if not os.path.exists(surface_filename):
        print(f"\tMissing input: file does not exist {surface_filename}")
        valid_inputs = False
    else:
        try:
            coords, faces = load_mesh_ext(surface_filename)
        except Exception as e:
            print(f"\tError: could not load surface file {surface_filename}")
            print(e)
            valid_inputs = False

    return valid_inputs


def validate_volume(fn: str) -> bool:
    """Validate that a volume exists and that it is not empty.

    :param fn : str, Path to .nii.gz file that serves as structural reference volume to use for reconstruction
    :return: valid_inputs bool
    :return:     boolean indicating whether the inputs have been validated
    """
    valid_inputs = True

    if not os.path.exists(fn):
        print(f"\tMissing input: file does not exist {fn}")
        valid_inputs = False
    else:
        try:
            vol = utils.load_image(fn)

            if np.sum(np.abs(vol)) == 0:
                valid_inputs = False
                print(f"\tMissing input: empty template file {fn}")
        except Exception as e:
            valid_inputs = False
            print(
                f"\tIncorrect format: incorrect format, expected path to volume, but got {fn}"
            )
            print(e)

    return valid_inputs


def validate_inputs(
    hemi_info_csv: str,
    chunk_info_csv: str,
    sect_info_csv: str,
    valid_inputs_npz: str = "",
    n_jobs: int = None,
    clobber: bool = False,
) -> bool:
    """Validate that the inputs to reconstruct are valid.

    :param hemi_info_csv: str, path to csv file that contains information about hemispheres to reconstruct :param chunk_info_csv: str,
    :param: sect_info_csv             Path to .csv file with information on sections to reconstruct
    :param: template_fn       Path to .nii.gz file that serves as structural reference volume to use for reconstruction
    :param: valid_inputs_npz         Path to .npz file that contains boolean indicating whether the inputs have been validated
    :param: clobber                   Boolean indicating whether to overwrite existing npz file
    :return: valid_inputs        boolean indicating whether the inputs have been validated
    """
    if (
        valid_inputs_npz != ""
        and os.path.exists(valid_inputs_npz + ".npz")
        and not clobber
    ):
        valid_inputs = np.load(valid_inputs_npz + ".npz")["valid_inputs"]
    else:
        valid_inputs = False

    if n_jobs == None or n_jobs == 0:
        n_jobs = int(cpu_count() / 2)

    if not valid_inputs:
        print("\nValidating Hemi Info")
        hemi_info_valid = validate_csv(
            hemi_info_csv, hemi_info_required_columns, n_jobs=n_jobs
        )
        print("\tHemi Info Valid =", bool(hemi_info_valid))

        print("\nValidating Chunk Info")
        chunk_info_valid = validate_csv(
            chunk_info_csv, chunk_info_required_columns, n_jobs=n_jobs
        )
        print("\tChunk Info Valid =", bool(chunk_info_valid))

        print("\nValidating Sect Info")
        sect_info_valid = validate_csv(
            sect_info_csv, sect_info_required_columns, n_jobs=n_jobs
        )
        print("\tSect Info Valid =", bool(sect_info_valid))

        valid_inputs = sect_info_valid * chunk_info_valid * hemi_info_valid
        print("Valid Inputs", valid_inputs)

        if valid_inputs_npz != "":
            os.makedirs(os.path.dirname(valid_inputs_npz), exist_ok=True)
            np.savez(valid_inputs_npz, valid_inputs=valid_inputs)

    return valid_inputs
