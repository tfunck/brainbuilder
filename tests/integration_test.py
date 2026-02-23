import argparse
import os
import urllib.request

import nibabel as nib
import numpy as np
from brainbuilder.reconstruct import reconstruct
from brainbuilder.tests.generate_synthetic_data import generate_synthetic_data


def preprocess_input_data(input_path, output_path):
    """Preprocess the input data by downloading and preparing necessary files."""
    img = nib.load(input_path)
    ar = img.get_fdata()

    # invert intensities
    ar = ar.max() - ar

    # Normalize to 0-255 and convert to uint8
    ar = (ar / ar.max() * 255).astype("uint8")

    # Crop to non-zero region
    # valid_x_indices = (ar.sum(axis=(1,2)) > 0).nonzero()[0]
    # valid_y_indices = (ar.sum(axis=(0, 2)) > 0).nonzero()[0]
    # valid_z_indices = (ar.sum(axis=(0,1)) > 0).nonzero()[0]

    # ar_cropped = ar[:, valid_y_indices[0] : valid_y_indices[-1] + 1, :]

    print(f"Writing preprocessed data to {output_path}...")
    nib.Nifti1Image(ar, img.affine).to_filename(output_path)


input_http_path = "https://ftp.bigbrainproject.org/bigbrain-ftp/BigBrainRelease.2015/3D_Volumes/MNI-ICBM152_Space/nii/full16_200um_2009b_sym.nii.gz"
cls_http_path = "https://ftp.bigbrainproject.org/bigbrain-ftp/BigBrainRelease.2015/3D_Classified_Volumes/MNI-ICBM152_Space/nii/full_cls_400um_2009b_sym.nii.gz"

tmp_dir = "/tmp/brainbuilder_test_data"

input_local_path = f"{tmp_dir}/full16_200um_2009b_sym.nii.gz"
cls_local_path = f"{tmp_dir}/full_cls_400um_2009b_sym.nii.gz"
gm_local_path = f"{tmp_dir}/gm_mask.nii.gz"
preproc_input_path = f"{tmp_dir}/preprocessed_input.nii.gz"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Integration test for brainbuilder.")
    parser.add_argument(
        "--clobber", action="store_true", help="Whether to overwrite existing files."
    )
    args = parser.parse_args()

    clobber = args.clobber

    if clobber:
        # ask user if they are sure they want to remove existing data. Require y/Y/n/N input
        resp = input(
            "Clobber is set. This will remove existing temporary data. Are you sure? (y/n): "
        )
        if resp.lower() != "y":
            print("Aborting.")
            exit(0)

        if os.path.exists(tmp_dir):
            # remove dir, even if it is not empty
            import shutil

            shutil.rmtree(tmp_dir)

    os.makedirs(tmp_dir, exist_ok=True)

    # Download the input file if it does not exist
    if not os.path.exists(input_local_path) or clobber:
        print(f"Downloading {input_http_path} to {input_local_path}...")
        urllib.request.urlretrieve(input_http_path, input_local_path)
        print("Download complete.")

    if not os.path.exists(cls_local_path) or clobber:
        print(f"Downloading {cls_http_path} to {cls_local_path}...")
        urllib.request.urlretrieve(cls_http_path, cls_local_path)
        print("Download complete.")

    if not os.path.exists(gm_local_path) or clobber:
        print(f"Generating gray matter mask at {gm_local_path}...")
        cls_img = nib.load(cls_local_path)
        cls_ar = cls_img.get_fdata()
        gm_labels = [2, 5, 6]
        # Gray matter labels in the classified volume
        gm_mask = np.sum([(cls_ar == lbl).astype("uint8") for lbl in gm_labels], axis=0)

        nib.Nifti1Image(gm_mask.astype(np.uint8), cls_img.affine).to_filename(
            gm_local_path
        )
        print("Gray matter mask generation complete:", gm_local_path)

    if not os.path.exists(preproc_input_path) or clobber:
        preprocess_input_data(input_local_path, preproc_input_path)

    landmark_dir = f"{tmp_dir}/landmarks/"
    os.makedirs(landmark_dir, exist_ok=True)

    generate_synthetic_data(
        preproc_input_path,
        gm_local_path,
        out_dir=tmp_dir,
        gm_surf_fn="data/MR1_gray_surface_R_81920.surf.gii",
        wm_surf_fn="data/MR1_white_surface_R_81920.surf.gii",
        landmark_dir=landmark_dir,
        clobber=clobber,
    )

    sect_info_csv = f"{tmp_dir}/sect_info.csv"
    chunk_info_csv = f"{tmp_dir}/chunk_info.csv"
    hemi_info_csv = f"{tmp_dir}/hemi_info.csv"

    df = reconstruct(
        hemi_info_csv,
        chunk_info_csv,
        sect_info_csv,
        [4, 3, 2, 1],
        f"{tmp_dir}/reconstruction/",
        landmark_dir=f"{tmp_dir}/landmarks/",
        clobber=clobber,
    )
