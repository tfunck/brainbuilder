# CELL 1
from brainbuilder.tests.generate_synthetic_data import generate_synthetic_data
import os

default_path = os.path.realpath(os.path.dirname(os.path.abspath(__file__))+'/../')+'/'

generate_synthetic_data( 
    input_fn = f'{default_path}data/mni_icbm152_01_tal_nlin_asym_09c.nii.gz',
    out_dir = f'{default_path}data/mni152_test',
    gm_surf_fn = f'{default_path}data/MR1_gray_surface_R_81920.surf.gii',
    wm_surf_fn = f'{default_path}data/MR1_white_surface_R_81920.surf.gii',
    clobber = True
    )


def realpath(x) : return os.path.realpath(default_path+x)

# CELL 2
import pandas as pd

sect_info_csv = f"{default_path}data/mni152_test/sect_info.csv"
chunk_info_csv = f"{default_path}data/mni152_test/chunk_info.csv"
hemi_info_csv = f"{default_path}data/mni152_test/hemi_info.csv"


# Load the sect_info csv and visualize contents. Stores section-wise information
print('Sect Info:')
sect_info_df = pd.read_csv(sect_info_csv)
print(sect_info_df)

# Load the chunk_info csv and visualize contents. Stores chunk-wise information
print('Chunk Info:')
chunk_info_df = pd.read_csv(chunk_info_csv)
print(chunk_info_df)

# Load the hemi_info csv and visualize contents. Stores hemispheric information
print('Hemi Info:')
hemi_info_df = pd.read_csv(hemi_info_csv)
print(hemi_info_df)


# CELL 3

from brainbuilder.reconstruct import reconstruct    

resolution_list = [4, 3, 2, 1]

df = reconstruct(
        hemi_info_csv,
        chunk_info_csv,
        sect_info_csv,
        [4, 3, 2, 1],
        f"{default_path}/data/mni152_test_output/",
        clobber=False,
    )