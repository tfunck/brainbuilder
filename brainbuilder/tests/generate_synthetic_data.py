"""Functions for creating synthetic data."""
import argparse
import os
from glob import glob

# import nibabel as nib
import brainbuilder.utils.ants_nibabel as nib
import numpy as np
import pandas as pd
from skimage.filters import threshold_otsu


def load(fn: str) -> tuple:
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


def save_section(vol: str, y: int, affine: np.array, out_fn: str) -> None:
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
    nib.Nifti1Image(section, affine, direction_order="lpi").to_filename(out_fn)


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


def save_coronal_sections(
    input_fn: str,
    out_dir: str,
    raw_dir: str,
    sub: str,
    hemisphere: str,
    chunk: int,
    ystep: int = 4,
    clobber: bool = False,
) -> str:
    """Save coronal sections of a volume as NIfTI files."""
    input_img, input_vol = load(input_fn)

    ymax = input_img.shape[1]

    sect_info_csv = f"{out_dir}/sect_info.csv"

    # angles = np.random.uniform(-30, 30, 3)
    # input_vol = rotate3d(input_vol, angles)
    # gm_vol = rotate3d(gm_vol, angles)

    if not os.path.exists(sect_info_csv) or clobber:
        affine = input_img.affine
        affine[:, 3] = [0, 0, 0, 1]  # set origin to 0,0,0

        df = pd.DataFrame({})

        y_list = np.array(range(0, ymax, ystep))

        section_max = np.max([np.sum(input_vol[:, y, :]) for y in y_list])

        for y in y_list:
            raw_sec_fn = f"{raw_dir}/sub-{sub}_chunk-{chunk}_sample-{y}_synth.nii.gz"

            if np.sum(input_vol[:, y, :]) < section_max * 0.05:
                continue

            section = input_vol[:, y, :]

            if np.sum(np.abs(section) > section.min()) < 200:  # skip empty sections
                continue

            nib.Nifti1Image(section, affine, direction_order="lpi").to_filename(
                raw_sec_fn
            )
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

    else:
        df = pd.read_csv(sect_info_csv)

    return df


def create_landmarks(
    df: pd.DataFrame,
    input_fn: str,
    reference_fn: str,
    out_landmark_fn: str,
    out_dir: str,
    n=30,
    clobber: bool = False,
) -> None:
    """Create random landmarks for the synthetic data.
    Random samples from the sections in df are used as landmarks.
    In n random sections, choose a random point with intensity > threshold as a landmark.
    Convert voxel coordinates to world coordinates using the input image affine and then convert
    to voxel values in the reference volume.
    The random sections must have unique values and must be identical in the sections and reference landmarks.
    """
    os.makedirs(out_dir, exist_ok=True)

    # check the number of existing landmarks in out_dir
    n_landmarks = len(glob(f"{out_dir}/*.nii.gz"))

    print(out_dir)
    print(n_landmarks)
    print(os.path.exists(out_landmark_fn))
    print(not clobber)

    if n_landmarks >= n and os.path.exists(out_landmark_fn) and not clobber:
        print(f"Landmarks already exist in {out_dir}, skipping creation.")
        return

    y_list = df["sample"].values
    section_list = df["raw"].values

    full_range = range(len(y_list))
    section_chosen_idx = np.random.choice(full_range, size=n, replace=False)
    section_chosen = section_list[section_chosen_idx]
    y_chosen = y_list[section_chosen_idx]

    input_img, _ = load(input_fn)
    reference_img, reference_vol = load(reference_fn)

    ref_landmark_volume = np.zeros(reference_vol.shape)

    vmax = df["sample"].max()

    values_list = []

    affine = input_img.affine
    affine[:, 3] = [0, 0, 0, 1]  # set origin to 0,0,0

    for y, section_path in zip(y_chosen, section_chosen):
        output_landmark_2d_fn = f"{out_dir}/{os.path.basename(section_path).replace('.nii.gz','_landmarks.nii.gz')}"

        section_img = nib.load(section_path)
        section = section_img.get_fdata()

        threshold = threshold_otsu(section)

        xs, zs = np.where(section > threshold)

        if len(xs) == 0:
            continue

        idx = np.random.choice(len(xs))

        x = xs[idx]
        z = zs[idx]

        r = 5

        landmark_2d = np.zeros(section.shape)
        landmark_2d[x - r : x + r, z - r : z + r] = y

        values_list.append(y)

        nib.Nifti1Image(landmark_2d, affine, direction_order="lpi").to_filename(
            output_landmark_2d_fn
        )

        xw = x * input_img.affine[0, 0]
        yw = y * input_img.affine[1, 1]
        zw = z * input_img.affine[2, 2]

        xv = np.rint(xw / reference_img.affine[0, 0]).astype(int)
        yv = np.rint(yw / reference_img.affine[1, 1]).astype(int)
        zv = np.rint(zw / reference_img.affine[2, 2]).astype(int)

        ref_x = xv
        ref_y = yv
        ref_z = zv

        # print(y, '-->', yw, '-->', yv)
        # print(input_img.affine[1,1], input_img.affine[1,3])
        # print(reference_img.affine[1,1], reference_img.affine[1,3])

        ref_landmark_volume[
            ref_x - r : ref_x + r, ref_y - r : ref_y + r, ref_z - r : ref_z + r
        ] = y

        import matplotlib.pyplot as plt

        plt.subplot(1, 2, 1)
        plt.imshow(section.T, cmap="gray", origin="lower")
        plt.imshow(
            landmark_2d.T,
            cmap="nipy_spectral",
            origin="lower",
            alpha=0.5,
            vmin=0,
            vmax=vmax,
        )

        plt.subplot(1, 2, 2)
        plt.imshow(reference_vol[:, yv, :].T, cmap="gray", origin="lower")
        plt.imshow(
            ref_landmark_volume[:, yv, :].T, cmap="Reds", origin="lower", alpha=0.7
        )

        plt.savefig(output_landmark_2d_fn.replace(".nii.gz", ".png"))
        plt.close()

    ref_values = np.unique(ref_landmark_volume)[1:]

    # check that values_list and ref_values have the same values
    assert set(ref_values) == set(values_list)

    print(ref_values)
    print(np.sort(values_list))

    nib.Nifti1Image(
        ref_landmark_volume, reference_img.affine, direction_order="lpi"
    ).to_filename(out_landmark_fn)


def generate_synthetic_data(
    input_fn: str,
    reference_fn: str,
    out_dir: str = "/tmp/brainbuilder/test_output",
    gm_surf_fn: str = "data/MR1_gray_surface_R_81920.surf.gii",
    wm_surf_fn: str = "data/MR1_white_surface_R_81920.surf.gii",
    sub: str = "01",
    hemisphere: str = "both",
    chunk: int = 1,
    ystep: int = 4,
    landmark_dir: str = None,
    clobber: bool = False,
) -> tuple:
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
    print("Generating synthetic data")
    raw_dir = f"{out_dir}/raw_dir/"
    hemi_info_csv = f"{out_dir}/hemi_info.csv"
    sect_info_csv = f"{out_dir}/sect_info.csv"
    chunk_info_csv = f"{out_dir}/chunk_info.csv"

    chunk = 1

    for dir_path in [out_dir, raw_dir]:
        os.makedirs(dir_path, exist_ok=True)

    df = save_coronal_sections(
        input_fn, out_dir, raw_dir, sub, hemisphere, chunk, ystep=ystep, clobber=clobber
    )

    ref_landmark_path = f"{out_dir}/ref_landmarks_volume.nii.gz"

    if landmark_dir:
        create_landmarks(
            df, input_fn, reference_fn, ref_landmark_path, landmark_dir, clobber=clobber
        )

    img = nib.load(input_fn)
    xstep, ystep, zstep = img.affine[0, 0], img.affine[1, 1], img.affine[2, 2]

    chunk_info_df = pd.DataFrame(
        {
            "sub": [sub],
            "chunk": [chunk],
            "hemisphere": [hemisphere],
            "pixel_size_0": [xstep],
            "pixel_size_1": [zstep],
            "section_thickness": [ystep],
            "ref_landmark": [ref_landmark_path],
            "direction": ["caudal_to_rostral"],
        }
    )

    chunk_info_df.to_csv(chunk_info_csv, index=False)

    hemi_info_df = pd.DataFrame(
        {
            "sub": [sub],
            "hemisphere": [hemisphere],
            "struct_ref_vol": [reference_fn],
            "gm_surf": [gm_surf_fn],
            "wm_surf": [wm_surf_fn],
        }
    )

    hemi_info_df.to_csv(hemi_info_csv, index=False)

    return sect_info_csv, chunk_info_csv, hemi_info_csv
