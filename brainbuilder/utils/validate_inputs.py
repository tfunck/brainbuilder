import os

import nibabel as nib
import numpy as np
import pandas as pd

global chunk_info_required_columns
global sect_info_required_columns


class Column:
    def __init__(self, name: str, kind: str, required: bool = True):
        """
        :param name: name of the column
        :param kind: kind of column, either "volume" or "surface"
        :param required: boolean indicating whether the column is required
        """

        self.name = str(name)
        self.kind = kind
        self.required = required

    def validate_rows_in_column(self, rows) -> bool:
        """
        Validate that the rows in a column based on whether they are volumes or surfaces
        :param rows: list of rows in the column
        :return: boolean indicating whether the inputs have been validated
        """

        valid_inputs = True

        if self.required:
            val_func = lambda x: x

            if self.kind == "volume":
                val_func = validate_volume
            elif self.kind == "surface":
                pass
            else:
                return True

            validated_rows = [val_func(fn) for fn in rows]
            valid_inputs = np.product(np.array(validated_rows))

        return valid_inputs


# Strings
sub = Column("sub", None)
hemisphere = Column("hemisphere", None)
chunk = Column("chunk", None)
direction = Column("direction", None)
pixel_size_0 = Column("pixel_size_0", None)
pixel_size_1 = Column("pixel_size_1", None)
section_thickness = Column("section_thickness", None)
acquisition = Column("acquisition", None)
sample = Column("sample", None)

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


def validate_raw(df: pd.DataFrame) -> bool:
    pass  # TODO


def validate_dataframe(df: pd.DataFrame, required_columns: list) -> bool:
    """
    Validate that the dataframe has the required columns
    :param df: dataframe to validate
    :param required_columns: list of required columns
    :return: boolean indicating whether the inputs have been validated
    """
    # TODO Update with BIDS fields
    valid_columns = []

    for column in required_columns:
        if not column.name in df.columns:
            valid_inputs = False
            print(f"\tMissing input: required field <{column.name}> not found in .csv")
        else:
            validated = column.validate_rows_in_column(df[column.name].values)
            valid_columns.append(validated)

    valid_inputs = np.product(np.array(valid_columns))
    return valid_inputs


def validate_csv(df_csv: str, required_columns: list) -> bool:
    """
    Inputs:
        df_csv             Path to .csv file with information on sections to reconstruct
        required_columns   List of column names that must be present in the .csv file
    Outputs:
        valid_inputs        boolean indicating whether the inputs have been validated
    """

    valid_inputs = True

    if not os.path.exists(df_csv):
        valid_inputs = False
        print(f"\tMissing input: .csv with section info does not exist {df_csv}")
    else:
        print(f"\tValidating {df_csv}")
        df = pd.read_csv(df_csv)
        valid_inputs = validate_dataframe(df, required_columns)

    return valid_inputs


def validate_surface(surface_filename: str) -> bool:
    """
    Inputs:
        surface_filename       Path to .nii.gz file that serves as structural reference volume to use for reconstruction
        volume_filename       Path to .nii.gz file that corresponds to surface_filename

    Outputs:
        valid_inputs        boolean indicating whether the inputs have been validated
    """
    valid_inputs = True

    if type(surface_filename) != str:
        print(f"\tMissing input: surface_filename is not a string {surface_filename}")
        valid_inputs = False
        return valid_inputs

    if not os.path.exists(surface_filename):
        print(f"\tMissing input: file does not exist {sect_info_csv}")
        valid_inputs = False
    else:
        try:
            coords, faces = load_mesh_ext(surface_filename)
        except:
            print(f"\tError: could not load surface file {surface_filename}")
            valid_inputs = False

    return valid_inputs


def validate_volume(template_fn: str) -> bool:
    """
    Validate that a volume exists and that it is not empty
    Inputs
    ------
        template_fn : str
        Path to .nii.gz file that serves as structural reference volume to use for reconstruction

    Outputs
    -------
        valid_inputs : bool
        boolean indicating whether the inputs have been validated
    """

    valid_inputs = True

    if not os.path.exists(template_fn):
        print(f"\tMissing input: file does not exist {template_fn}")
        valid_inputs = False
    else:
        try:
            img = nib.load(template_fn)
            vol = img.dataobj

            if np.sum(np.abs(vol)) == 0:
                valid_inputs = False
                print(f"\tMissing input: empty template file {template_fn}")
        except:
            valid_inputs = False
            print(
                f"\tIncorrect format: incorrect format, expected path to template, but got {template_fn}"
            )

    return valid_inputs


def validate_inputs(
    hemi_info_csv: str,
    chunk_info_csv: str,
    sect_info_csv: str,
) -> bool:
    """
    Validate that the inputs to reconstruct are valid
    :param hemi_info_csv: str, path to csv file that contains information about hemispheres to reconstruct :param chunk_info_csv: str,
    :param: sect_info_csv             Path to .csv file with information on sections to reconstruct
    :param: template_fn       Path to .nii.gz file that serves as structural reference volume to use for reconstruction
    :return: valid_inputs        boolean indicating whether the inputs have been validated
    """

    hemi_info_valid = validate_csv(hemi_info_csv, hemi_info_required_columns)

    chunk_info_valid = validate_csv(chunk_info_csv, chunk_info_required_columns)

    sect_info_valid = validate_csv(sect_info_csv, sect_info_required_columns)

    valid_inputs = sect_info_valid * chunk_info_valid * hemi_info_valid

    return valid_inputs