import glob
import os

import ants
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from brainbuilder.qc.validate_interp_error import calculate_regional_averages
from scipy.stats import pearsonr, spearmanr


def calculate_2d_average(fingerprint_3d_df, fingerprint_2d_df, conversion_df):
    fingerprint_3d_df["average_2d"] = np.nan

    # Iterate over the receptor densitites extracted using a 3D atlas
    for i, row in fingerprint_3d_df.iterrows():
        label = row["label"]  # 3D atlas region number
        receptor = row["receptor"]  # receptor (ligand) name

        # check if region is in the conversion csv
        if label in conversion_df["Label"].values:
            # identify 2d region name based on row == "receptor" and column == "region"
            region_2d_names = conversion_df.loc[conversion_df["Label"] == label][
                ["2D_0", "2D_1", "2D_2"]
            ].values[0]

            # eliminate blank '' values
            region_2d_names = [
                name for name in region_2d_names if name != "" and name is not np.nan
            ]

            if len(region_2d_names) > 0:
                avg2d = 0
                for area_2d in region_2d_names:
                    avg = fingerprint_2d_df.loc[
                        (fingerprint_2d_df["ligand"] == receptor),
                        (fingerprint_2d_df.columns == area_2d),
                    ]

                    avg2d += avg.values[0]

                avg2d = avg2d / len(region_2d_names)

                fingerprint_3d_df["average_2d"].loc[
                    (fingerprint_3d_df["label"] == label)
                    & (fingerprint_3d_df["receptor"] == receptor)
                ] = avg2d

    # remove nan
    receptor_df = fingerprint_3d_df.dropna(subset=["average_2d"])

    return receptor_df


def compare_2d_vs_3d(receptor_df, output_dir):
    # Calculate error between average_2d and average_3d
    receptor_df["mrse"] = np.sqrt(
        (receptor_df["average_2d"] - receptor_df["average_3d"]) ** 2
    )
    receptor_df.sort_values(by=["mrse"], ascending=False, inplace=True)

    print(receptor_df)
    print(spearmanr(receptor_df["average_3d"], receptor_df["average_2d"]))
    print(pearsonr(receptor_df["average_3d"], receptor_df["average_2d"]))

    # plot the 3D vs 2D averages with regression line
    output_png = f"{output_dir}/receptor_fingerprint_3D_vs_2D.png"

    plt.figure(figsize=(12, 12))
    plt.title("Receptor fingerprint 3D vs 2D")
    sns.lmplot(
        x="average_3d",
        y="average_2d",
        data=receptor_df,
        fit_reg=True,
        scatter_kws=dict(alpha=0.25),
    )
    # rename y axis as Original (2D)
    plt.ylabel("Original (2D)")
    # rename x axis as Reconstructed (3D)
    plt.xlabel("Reconstructed (3D)")
    plt.savefig(output_png, dpi=200)

    print(output_png)

    output_csv = f"{output_dir}/receptor_fingerprint_3D_vs_2D.csv"
    print(output_csv)
    receptor_df.to_csv(output_csv)

    return receptor_df


def validate_alignment(
    atlas_filename,
    template_filename,
    receptor_files,
    ref_volume_filename,
    fingerprint_2d_csv,
    conversion_csv,
    output_dir,
    clobber=False,
):
    os.makedirs(output_dir, exist_ok=True)

    # 1. Align template_filename to ref_volume_filename with ANTs
    outprefix = f"{output_dir}/template_to_ref_"
    tfm_fn = f"{outprefix}Composite.h5"
    if not os.path.exists(tfm_fn) or clobber:
        reg = ants.registration(
            fixed=ants.image_read(ref_volume_filename),
            moving=ants.image_read(template_filename),
            type_of_transform="SyN",
            outprefix=outprefix,
            write_composite_transform=True,
            verbose=True,
        )

    # 2. Apply transformation to atlas_filename
    atlas_warped_filename = f"{output_dir}/atlas_warped.nii.gz"
    if not os.path.exists(atlas_warped_filename) or clobber or True:
        atlas_warped = ants.apply_transforms(
            fixed=ants.image_read(receptor_files[0]),
            moving=ants.image_read(atlas_filename),
            transformlist=[tfm_fn],
            interpolator="nearestNeighbor",
            verbose=True,
        )
        ants.image_write(atlas_warped, atlas_warped_filename)
        # nib.Nifti1Image(atlas_warped.numpy(), nib.load(atlas_warped_filename).affine, direction_order='lpi').to_filename(atlas_warped_filename)
    print(atlas_warped_filename)
    print(receptor_files[0])
    # 3. Calculate regional averages in the transformed atlas for each receptor file
    df_list = []
    clobber = True
    for receptor_file in receptor_files:
        curr_receptor_df_filename = f'{output_dir}/{os.path.basename(receptor_file).replace(".nii.gz", "_averages.csv")}'
        if not os.path.exists(curr_receptor_df_filename) or clobber:
            curr_receptor_df = calculate_regional_averages(
                receptor_file, atlas_warped_filename
            )
            receptor = os.path.basename(receptor_file).split("_")[2].replace("acq-", "")
            curr_receptor_df["receptor"] = [receptor] * len(curr_receptor_df)
            curr_receptor_df.to_csv(curr_receptor_df_filename, index=False)
        else:
            curr_receptor_df = pd.read_csv(curr_receptor_df_filename)

        df_list.append(curr_receptor_df)

    fingerprint_3d_df = pd.concat(df_list)

    # 4. Normalize average by label
    fingerprint_3d_df["average_3d"] = fingerprint_3d_df.groupby("label")[
        "average"
    ].transform(lambda x: (x - x.mean()) / x.std())

    # 5. Save as a csv file with rows = regions and columns = receptors
    fingerprint_3d_df.to_csv(
        f"{output_dir}/receptor_fingerprint_averages_3D.csv", index=False
    )

    fingerprint_2d_df = pd.read_csv(fingerprint_2d_csv)

    # zscore fingerprint_2d_df by column
    fingerprint_2d_df.iloc[:, 3:] = fingerprint_2d_df.iloc[:, 3:].apply(
        lambda x: (x - x.mean()) / x.std(), axis=0
    )

    conversion_df = pd.read_csv(conversion_csv)

    fingerprints_df = calculate_2d_average(
        fingerprint_3d_df, fingerprint_2d_df, conversion_df
    )

    compare_2d_vs_3d(fingerprints_df, output_dir)


if __name__ == "__main__":
    atlas_filename = "/data/receptor/human/civet/JuBrain_Map_v30.nii"
    template_filename = (
        "/data/receptor/human/civet/mni_icbm152_t1_tal_nlin_asym_09c.nii.gz"
    )
    receptor_files = glob.glob(
        "/data/receptor/human/brainbuilder_human_out/4_interp/sub-MR1/hemi-R/acq-*/*_cortex.nii.gz"
    )
    ref_volume_filename = "/data/receptor/human/civet/mri1/final/mri1_t1_tal.nii"
    output_dir = "/data/receptor/human/brainbuilder_human_out/qc/receptor_fingerprint/"
    fingerprint_2d_csv = "/data/receptor/human/dens_mr1_r_all_receptors_areas.csv"
    conversion_csv = "/data/receptor/human/civet/jubrain_regions.csv"
    clobber = False

    validate_alignment(
        atlas_filename,
        template_filename,
        receptor_files,
        ref_volume_filename,
        fingerprint_2d_csv,
        conversion_csv,
        output_dir,
        clobber,
    )
