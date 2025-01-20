"""Validate interpolation error by comparing original autoradiographs to 2D and 3D interpolated autoradiographs."""
import os

import brainbuilder.utils.ants_nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from brainbuilder.utils.utils import shell
from joblib import Parallel, delayed
from scipy.stats import spearmanr
from skimage.segmentation import slic


def calculate_regional_averages(
    acquisition_fn: str,
    atlas_fn: str,
    conversion_factor: float = 1,
) -> pd.DataFrame:
    """Calculate regional averages for each label in the 2d images."""
    assert os.path.exists(acquisition_fn), f"Error: {acquisition_fn} does not exist"
    assert os.path.exists(atlas_fn), f"Error: {atlas_fn} does not exist"

    # Read Atlas file
    atlas_img = nib.load(atlas_fn)
    atlas_volume = atlas_img.get_fdata().astype(int)
    assert np.sum(atlas_volume) > 0, "Error: empty atlas volume"

    print("Atlas:", atlas_fn, acquisition_fn)


    # Read file with receptor density info
    acquisition_img = nib.load(acquisition_fn)
    reconstructed_volume = acquisition_img.get_fdata()
    assert np.sum(reconstructed_volume) > 0, "Error: empty reconstructed volume"

    atlas_volume[reconstructed_volume == 0] = 0

    reconstructed_volume = reconstructed_volume * conversion_factor

    assert not np.isnan(
        np.sum(reconstructed_volume)
    ), f"Error :  NaN in {acquisition_fn}"

    averages_out, labels_out, n_voxels = average_over_label(
        atlas_volume.reshape(
            -1,
        ),
        reconstructed_volume.reshape(
            -1,
        ),
    )

    volume = n_voxels * (atlas_img.affine[0, 0] * atlas_img.affine[1, 1] * 0.02)

    df = pd.DataFrame({"label": labels_out, "average": averages_out, "volume": volume})

    return df


def calculate_regional_averages_by_row(
    row: pd.Series,
    section_str: str,
    label_str: str,
    conversion_factor: float = 1,
) -> pd.DataFrame:
    """Calculate regional averages for each label in the 2d images."""
    acquisition_fn = row[section_str]
    atlas_fn = row[label_str]

    avg_df = calculate_regional_averages(
        acquisition_fn, atlas_fn, conversion_factor=conversion_factor
    )
    print('Average DF')
    print(avg_df)

    # Create dataframe that repeats rows for each label
    df = row.to_frame().T
    df = df.loc[df.index.repeat(avg_df.shape[0])].reset_index(drop=True)
    df["label"] = avg_df["label"]
    df["average"] = avg_df["average"]
    df["volume"] = avg_df["volume"]

    print('hello', acquisition_fn)
    return df


def average_over_label(
    labels: np.array, values: np.array, conversion_factor: float = 1
) -> tuple:
    """Average over label."""
    averages_out = []
    labels_out = []
    unique_labels = np.unique(labels)[1:]

    labels_remapped = np.zeros_like(labels)
    assert (
        labels.shape == values.shape
    ), "Error: mismatch in label and values array dimensions"

    for i, label in enumerate(unique_labels):
        labels_remapped[label == labels] = i + 1

    averages_out = np.bincount(labels_remapped, values)[1:]
    labels_out = unique_labels
    n_out = np.bincount(labels_remapped)[1:]
    averages_out = (averages_out / n_out) * conversion_factor

    return np.array(averages_out), np.array(labels_out), np.array(n_out)


def cluster_sections(
    sect_info: pd.DataFrame, output_dir: str, clobber: bool = False
) -> pd.DataFrame:
    """Cluster sections."""
    os.makedirs(output_dir, exist_ok=True)

    # Define cluster function
    def cluster(
        section_filename: str,
        mask_filename: str,
        output_filename: str,
        sample: int,
        n_segments: int = 100,
        compactness: float = 10.0,
        max_num_iter: int = 10,
        sigma: float = 0,
        mask: bool = None,
        clobber: bool = False,
    ) -> str:
        """Cluster a section using ."""
        if not os.path.exists(output_filename) or clobber:
            img_hd = nib.load(section_filename)
            img = img_hd.get_fdata()
            mask = nib.load(mask_filename).get_fdata()

            cluster_image = slic(
                img,
                n_segments=n_segments,
                compactness=compactness,
                max_num_iter=max_num_iter,
                sigma=sigma,
                mask=mask,
                channel_axis=None,
            )

            idx = cluster_image > 0

            cluster_image[idx] = cluster_image[idx] + sample * n_segments

            cluster_image = cluster_image.astype(np.uint32)

            nib.Nifti1Image(
                cluster_image, img_hd.affine, direction_order="lpi"
            ).to_filename(output_filename)

        return output_filename

    # Define cluster filenames
    sect_info["cluster"] = sect_info["img"].apply(
        lambda x: output_dir
        + "/"
        + os.path.basename(x).replace(".nii.gz", "_cluster.nii.gz")
    )

    Parallel(n_jobs=8)(
        delayed(cluster)(
            row["img"], row["seg"], row["cluster"], row["sample"], clobber=clobber
        )
        for i, row in sect_info.iterrows()
    )

    return sect_info


def regional_averages(
    sect_info: pd.DataFrame,
    section_string: str,
    labels_string: str,
    output_dir: str,
    source: str = None,
    conversion_factor: float = 1,
    clobber: bool = False,
) -> pd.DataFrame:
    """Calculate regional values."""
    os.makedirs(output_dir, exist_ok=True)

    if source is not None:
        output_filename = f"{output_dir}/regional_values_{source}.csv"
    else:
        output_filename = f"{output_dir}/regional_values.csv"

    if not os.path.exists(output_filename) or clobber:
        if "cluster_volume_3d" in sect_info.columns:
            for i, row in sect_info.iterrows():
                print(row["cluster_volume_3d"])
                print(row["reconstructed_filename"])
        
        # Define regional values function
        res = Parallel(n_jobs=-1)(
            delayed(calculate_regional_averages_by_row)(
                row,
                section_string,
                labels_string,
                conversion_factor=conversion_factor,
            )
            for i, row in sect_info.iterrows()
        )
        print(res[0])
        # Concatenate results
        print("Concatenating results")
        regional_values_df = pd.concat(res)

        print("Saving results")
        regional_values_df["source"] = [source] * regional_values_df.shape[0]

        regional_values_df.to_csv(output_filename)
    else:
        regional_values_df = pd.read_csv(output_filename)

    return regional_values_df


def apply_2d_transformations(df: pd.DataFrame, clobber: bool = False) -> pd.DataFrame:
    """Apply 2D transformations to cluster images."""
    df["cluster_2d"] = df["cluster"].apply(lambda x: x.replace(".nii", "_2d.nii"))

    def apply_transform(row: pd.Series) -> None:
        """Apply 2D transformation."""
        tfm = row["2d_tfm"]
        img = row["cluster"]
        ref = row["nl_2d_rsl"]
        out_fn = row["cluster_2d"]

        if not os.path.exists(out_fn) or clobber:
            shell(
                f"antsApplyTransforms -v 0 -n NearestNeighbor -d 2 -i {img} -o {out_fn} -t {tfm} -r {ref}"
            )
            hd = nib.load(out_fn)
            ar = hd.get_fdata()
            plt.imshow(ar)
            plt.savefig(out_fn.replace(".nii.gz", ".png"))

    Parallel(n_jobs=8)(delayed(apply_transform)(row) for i, row in df.iterrows())

    return df


def apply_3d_transformations(
    chunk_info: pd.DataFrame, clobber: bool = False
) -> pd.DataFrame:
    """Apply 2D transformations to cluster images."""
    chunk_info["cluster_volume_3d"] = chunk_info["cluster_volume_2d"].apply(
        lambda x: x.replace("_2d.nii", "_3d.nii")
    )

    for _, row in chunk_info.iterrows():
        tfm = row["nl_3d_tfm_fn"]
        img = row["cluster_volume_2d"]
        out_fn = row["cluster_volume_3d"]

        ref_volume_filename = row["reconstructed_filename"]
        if not os.path.exists(out_fn) or clobber:
            shell(
                f"antsApplyTransforms -v 1 -n NearestNeighbor -d 3 -i {img} -o {out_fn} -t {tfm} -r {ref_volume_filename}", verbose=True
            )
        
        assert os.path.exists(out_fn), f"Error: {out_fn} does not exist"
        assert np.sum(nib.load(out_fn).get_fdata()) > 0, "Error: empty output volume"

    return chunk_info


def concat_sections_to_volume(
    df: pd.DataFrame,
    volume_filename: str,
    ref_volume_filename: str,
    target_string: str = "cluster_2d",
    clobber: bool = False,
) -> str:
    """Concatenate sections to volume."""

    # Define volume filename
    def load_section(row: pd.Series) -> tuple:
        """Load section."""
        y = int(row["sample"])
        section = nib.load(row[target_string]).get_fdata().astype(np.uint32)
        return y, section

    if not os.path.exists(volume_filename) or clobber:
        img = nib.load(ref_volume_filename)
        ydim = img.shape[1]
        xdim = nib.load(df[target_string].values[0]).shape[0]
        zdim = nib.load(df[target_string].values[0]).shape[1]

        out_volume = np.zeros([xdim, ydim, zdim]).astype(np.uint32)

        # Iterate over df rows in chunks of 100
        for i, row in df.iterrows():
            y, section = load_section(row)
            out_volume[:, y, :] = section

        assert np.sum(out_volume) > 0, "Error: empty output volume"
        nib.Nifti1Image(out_volume, img.affine, direction_order="lpi").to_filename(
            volume_filename
        )

    return volume_filename


def create_3d_chunk_volumes(
    sect_info: pd.DataFrame,
    chunk_info: pd.DataFrame,
    output_dir: str,
    target_string: str = "cluster_2d",
    clobber: bool = False,
) -> pd.DataFrame:
    """Create 3D volumes of 2D warped classified autoradiographs in each individual chunks."""
    output_filename = f"{output_dir}/cluster_volume_chunk_info.csv"
    if not os.path.exists(output_filename) or clobber:
        chunk_info["cluster_volume_2d"] = [None] * chunk_info.shape[0]

        for col in ["label", "average", "volume", "Unnamed: 0"]:
            if col in sect_info.columns:
                del sect_info[col]

        # Remove duplicate rows
        sect_info = sect_info.drop_duplicates()
        print(chunk_info.columns)
        output_chunk_info_list = []
        for (sub, hemisphere, chunk, acquisition), chunk_sect_info in sect_info.groupby(
            ["sub", "hemisphere", "chunk", "acquisition"]
        ):
            idx = (
                (chunk_info["chunk"] == chunk)
                & (chunk_info["hemisphere"] == hemisphere)
                & (chunk_info["sub"] == sub)
                & (chunk_info["acquisition"] == acquisition)
            )
            ref_volume_filename = chunk_info.loc[idx, "nl_2d_vol_fn"].values[0]
            cluster_2d_volume_filename = f"{output_dir}/{sub}_{hemisphere}_{chunk}_{acquisition}_cluster_2d.nii.gz"

            concat_sections_to_volume(
                chunk_sect_info,
                cluster_2d_volume_filename,
                ref_volume_filename,
                target_string=target_string,
                clobber=clobber,
            )
            print("\t", cluster_2d_volume_filename)
            temp_chunk_info = chunk_info.loc[idx].copy()
            temp_chunk_info["acquisition"] = acquisition
            temp_chunk_info["cluster_volume_2d"] = cluster_2d_volume_filename
            output_chunk_info_list.append(temp_chunk_info)

        chunk_info = pd.concat(output_chunk_info_list)

        chunk_info.to_csv(f"{output_dir}/cluster_volume_chunk_info.csv")
    else:
        chunk_info = pd.read_csv(output_filename)
    print(output_filename)
    return chunk_info


def prepare_dataframe(
    sect_info: pd.DataFrame, min_average: float, min_volume: float, source: str
) -> pd.DataFrame:
    """Prepare dataframe for plotting."""
    sect_info = sect_info.loc[sect_info["source"] == source]

    columns = []
    for col in ["sub", "hemisphere", "chunk", "sample", "label"]:
        if col in sect_info.columns:
            columns.append(col)

    sect_info.sort_values(columns, inplace=True)

    # sect_info = sect_info.loc[ (sect_info['average'] > min_average) ]

    return sect_info


def plot_validation_interp_error(
    sect_info_orig: pd.DataFrame,
    sect_info_2d: pd.DataFrame,
    chunk_info_3d: pd.DataFrame,
    output_dir: str,
) -> None:
    """Plot validation interpolation error between same labels."""
    min_average = np.max(sect_info_orig["average"]) * 0.05
    min_volume = 0.5**2

    sect_info_orig = prepare_dataframe(
        sect_info_orig, min_average, min_volume, "original"
    )
    sect_info_2d = prepare_dataframe(sect_info_2d, min_average, min_volume, "2d")
    chunk_info_3d = prepare_dataframe(chunk_info_3d, min_average, min_volume, "3d")

    # downsample the dataframe rows
    sect_info_orig = sect_info_orig.sample(frac=0.1)

    df = pd.DataFrame([])
    for i, (_, row) in enumerate(sect_info_orig.iterrows()):
        if i % 100 == 0:
            print(f"\t{np.round(100.*i/sect_info_orig.shape[0],1)}", end="\r")
        print(i)
        sub = row["sub"]
        hemisphere = row["hemisphere"]
        chunk = row["chunk"]
        label = row["label"]
        sample = row["sample"]
        average_orig = row["average"]
        volume = row["volume"]

        print('\tA',volume, average_orig)
        if volume < min_volume or average_orig < min_average:
            continue

        idx2d = (
            (sect_info_2d["sub"] == sub)
            & (sect_info_2d["hemisphere"] == hemisphere)
            & (sect_info_2d["chunk"] == chunk)
            & (sect_info_2d["label"] == label)
        )
        print('\tB',idx2d.sum())
        if idx2d.sum() == 0:
            continue

        average_2d = sect_info_2d.loc[idx2d, "average"].values[0]
        volume_2d = sect_info_2d.loc[idx2d, "volume"].values[0]

        print('\tC',average_2d, volume_2d)
        if average_2d < min_average and volume_2d < min_volume:
            continue

        idx3d = (
            (chunk_info_3d["sub"] == sub)
            & (chunk_info_3d["hemisphere"] == hemisphere)
            & (chunk_info_3d["chunk"] == chunk)
            & (chunk_info_3d["label"] == label)
        )
        print(sub, hemisphere, chunk, label)
        print(chunk_info_3d["sub"].unique())
        print(chunk_info_3d["hemisphere"].unique())
        print(chunk_info_3d["chunk"].unique())
        print(chunk_info_3d["label"].unique())
        print('\tD',idx3d.sum())
        if idx3d.sum() == 0:
            continue

        average_3d = chunk_info_3d.loc[idx3d, "average"].values[0]
        volume_3d = chunk_info_3d.loc[idx3d, "volume"].values[0]

        if average_3d < min_average and volume_3d < min_volume:
            continue

        print('hello')
        row = pd.DataFrame(
            {
                "sub": [sub],
                "hemisphere": [hemisphere],
                "chunk": [chunk],
                "sample": [sample],
                "label": [label],
                "average_2d": [average_2d],
                "average_orig": [average_orig],
                "average_3d": [average_3d],
                "volume": [volume],
            }
        )
        df = pd.concat([df, row])
    assert df.shape[0] > 0, "Error: empty dataframe"

    df.to_csv(f"{output_dir}/validation_interp_error.csv")
    print(f"{output_dir}/validation_interp_error.csv")

    df["% Interp. Error"] = (
        np.abs(df["average_orig"] - df["average_2d"]) / df["average_orig"]
    )

    rho_0 = spearmanr(df["average_orig"], df["average_2d"])
    rho_1 = spearmanr(df["average_orig"], df["average_3d"])
    print("Spearmans rho (2d):", rho_0)
    print("Spearmans rho (3d):", rho_1)
    plt.subplot(1, 2, 1)
    sns.regplot(x="average_orig", y="average_2d", data=df, scatter_kws={"s": 1})
    plt.subplot(1, 2, 2)
    sns.regplot(x="average_orig", y="average_3d", data=df, scatter_kws={"s": 1})
    plt.savefig(f"{output_dir}/validation_interp_error.png")
    print(f"{output_dir}/validation_interp_error.png")


def validate_interp_error(
    sect_info_csv: str, chunk_info_csv: str, output_dir: str, clobber: bool = False
) -> None:
    """Validate interpolation error by comparing original autoradiographs to 2D and 3D interpolated autoradiographs."""
    output_sect_info_csv = f"{output_dir}/validate_interp_error_sect_info.csv"
    sect_info = pd.read_csv(sect_info_csv)
    chunk_info = pd.read_csv(chunk_info_csv)

    # Only keep highest resolution in chunk_info
    chunk_info = chunk_info.loc[
        chunk_info["resolution"] == chunk_info["resolution"].min()
    ]
    print("Chunk Info:", chunk_info_csv)

    # Unsupervised clustering of autoradiographs
    print("\tClustering sections")
    sect_info = cluster_sections(sect_info, output_dir, clobber=clobber)

    # Apply 2D transformations to segmented autoradiotraphs
    print("\tApply 2D transformations to clusters")
    sect_info_2d = apply_2d_transformations(sect_info, clobber=clobber)

    # Calculate regional averages for each label in the autoradiographs
    print("\tCalculate regional averages")
    sect_info_orig = regional_averages(
        sect_info, "img", "cluster", output_dir, "original", clobber=clobber
    )

    # Calculate regional averages for each label in the 2D warped classified autoradiographs
    print("\tCalculate regional averages for 2D warped classified autoradiographs")
    sect_info_2d = regional_averages(
        sect_info_2d,
        "nl_2d_rsl",
        "cluster_2d",
        output_dir,
        "2d",
        conversion_factor=1,
        clobber=clobber,
    )

    sect_info = pd.concat([sect_info_orig, sect_info_2d])

    sect_info.to_csv(output_sect_info_csv)

    # Create 3D volumes of 2D warped classified autoradiographs
    print("\tCreate 3D volumes of 2D warped classified autoradiographs")
    chunk_info = create_3d_chunk_volumes(
        sect_info_2d.copy(), chunk_info, output_dir, clobber=clobber
    )
    print(chunk_info.columns)
    print(chunk_info['cluster_volume_2d'].values)
    clobber=True
    # Apply 3D transformations to 3D volumes of 2D warped classified autoradiographs
    chunk_info = apply_3d_transformations(chunk_info, clobber=clobber)
    print(chunk_info['cluster_volume_3d'].values)

    # Calculate regional averages for each label in the 3D volumes
    chunk_info_3d = regional_averages(
        chunk_info,
        "reconstructed_filename",
        "cluster_volume_3d",
        output_dir,
        "3d",
        clobber=True,
    )

    # Plot original label values versus 2d warped label values
    plot_validation_interp_error(
        sect_info_orig, sect_info_2d, chunk_info_3d, output_dir
    )
