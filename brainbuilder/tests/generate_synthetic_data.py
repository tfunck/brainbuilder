"""Functions for creating synthetic data."""
import argparse
import os

import nibabel as nib
import numpy as np
import pandas as pd


def load(fn:str)->tuple:
    """Load a NIfTI file."""
    img = nib.load(fn)
    vol = img.get_fdata()
    return img, vol

"""
def rotate3d(vol, angles):
    # r = Rotation.from_euler('xyz', angles, degrees=True)
    # vol = r.apply(vol)
    # vol = Rotate(vol, angles[0], (1,0))
    # vol = Rotate(vol, angles[1], (1,2))
    # vol = Rotate(vol, angles[2], (0,2))

    return vol
"""

def save_section(vol:str, y:int, affine:np.array, out_fn:str)->None:
    """Save a 2D section of a 3D volume as a NIfTI file.

    Args:
        vol (ndarray): The 3D volume.
        y (int): The y-coordinate of the section to save.
        affine (ndarray): The affine transformation matrix.
        out_fn (str): The output file name.

    Returns:
        None
    """
    section = vol[:, y, :]
    # imageio.imwrite(out_fn, section)
    nib.Nifti1Image(section, affine).to_filename(out_fn)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", dest="input_fn", type=str)
    parser.add_argument("--output-dir", dest="out_dir", type=str)
    parser.add_argument("--gm-surf", dest="gm_surf_fn", type=str)
    parser.add_argument("--wm-surf", dest="wm_surf_fn", type=str)
    parser.add_argument("--sub", dest="sub", type=str)
    parser.add_argument("--hemisphere", dest="hemisphere", type=str)

    args = parser.parse_args()
    sub = args.sub
    hemisphere = args.hemisphere
    out_dir = args.out_dir


def save_coronal_sections(input_fn:str, out_dir:str, raw_dir:str, sub:str, hemisphere:str, chunk:int, ystep:int=4, clobber:bool=False)->str:
    """Save coronal sections of a volume as NIfTI files."""
    input_img, input_vol = load(input_fn)

    ymax = input_img.shape[1]

    sect_info_csv = f"{out_dir}/sect_info.csv"

    #angles = np.random.uniform(-30, 30, 3)
    #input_vol = rotate3d(input_vol, angles)
    #gm_vol = rotate3d(gm_vol, angles)

    if not os.path.exists(sect_info_csv) or clobber:
        affine = input_img.affine

        df = pd.DataFrame({})

        section_max = np.max([np.sum(input_vol[:, y, :]) for y in range(0, ymax, 4)])

        for y in range(0, ymax, ystep):
            raw_sec_fn = f"{raw_dir}/sub-{sub}_chunk-{chunk}_sample-{y}_synth.nii.gz"

            if np.sum(input_vol[:, y, :]) < section_max * 0.05:
                continue

            save_section(input_vol, y, affine, raw_sec_fn)
            print(raw_sec_fn)

            row_dict = {
                "raw": [raw_sec_fn],
                "sub": [sub],
                "hemisphere": [hemisphere],
                "acquisition": ["synth"],
                "sample": [y],
                "chunk": [chunk],
            }

            df = pd.concat([df, pd.DataFrame(row_dict)])

        df.to_csv(sect_info_csv, index=False)
    
    else :
        df = pd.read_csv(sect_info_csv)

    return df

def generate_synthetic_data(
    input_fn: str ='data/mni_icbm152_01_tal_nlin_asym_09c.nii.gz',
    out_dir: str ='/tmp/brainbuilder/test_output',
    gm_surf_fn: str='data/MR1_gray_surface_R_81920.surf.gii',
    wm_surf_fn: str='data/MR1_white_surface_R_81920.surf.gii',
    sub: str='01',
    hemisphere: str='both',
    chunk:int=1,
    ystep:int=4,
    clobber:bool=False,
)->tuple:
    """Generate synthetic data using an input volume file and surface files.

    :param input_fn: Input volume file.
    :param out_dir: Output directory.
    :param gm_surf_fn: Gray matter surface file.
    :param wm_surf_fn: White matter surface file.
    :param sub: Subject ID.
    :param hemisphere: Hemisphere.
    :param chunk: Chunk number.
    :param clobber: Overwrite existing files.
    :return: tuple of section info CSV file, chunk info CSV file, and hemisphere info CSV file. 
    """
    print('Generating synthetic data')
    raw_dir = f"{out_dir}/raw_dir/"
    hemi_info_csv = f"{out_dir}/hemi_info.csv"
    sect_info_csv = f"{out_dir}/sect_info.csv"
    chunk_info_csv = f"{out_dir}/chunk_info.csv"

    chunk = 1

    for dir_path in [out_dir, raw_dir]:
        os.makedirs(dir_path, exist_ok=True)


    save_coronal_sections(input_fn, out_dir, raw_dir, sub, hemisphere, chunk, ystep=ystep, clobber=clobber )


    chunk_info_df = pd.DataFrame(
        {
            "sub": [sub],
            "chunk": [chunk],
            "hemisphere": [hemisphere],
            "pixel_size_0": [1.0],
            "pixel_size_1": [1.0],
            "section_thickness": [1.0],
            "direction": ["caudal_to_rostral"],
        }
    )

    chunk_info_df.to_csv(chunk_info_csv, index=False)

    hemi_info_df = pd.DataFrame(
        {
            "sub": [sub],
            "hemisphere": [hemisphere],
            "struct_ref_vol": [input_fn],
            "gm_surf": [gm_surf_fn],
            "wm_surf": [wm_surf_fn],
        }
    )

    hemi_info_df.to_csv(hemi_info_csv, index=False)

    return sect_info_csv, chunk_info_csv, hemi_info_csv
