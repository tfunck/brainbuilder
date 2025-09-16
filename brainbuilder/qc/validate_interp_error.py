"""Validate interpolation error by comparing original autoradiographs to 2D and 3D interpolated autoradiographs."""
import os

import brainbuilder.utils.ants_nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from brainbuilder.utils.utils import shell
from joblib import Parallel, delayed
from skimage import morphology
from skimage.measure import label
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
    assert np.sum(np.abs(atlas_volume)) > 0, "Error: empty atlas volume"

    print("Atlas:", atlas_fn, acquisition_fn)

    # Read file with receptor density info
    acquisition_img = nib.load(acquisition_fn)
    reconstructed_volume = acquisition_img.get_fdata()
    assert (
        np.sum(np.abs(reconstructed_volume)) > 0
    ), f"Error: empty reconstructed volume, {acquisition_fn}"

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

    if len(atlas_volume.shape) == 2:
        spacing = atlas_img.affine[0, 0] * atlas_img.affine[1, 1]
    elif len(atlas_volume.shape) == 3:
        spacing = atlas_img.affine[0, 0] * atlas_img.affine[2, 2]

    volume = n_voxels * spacing
    # print("Volume:", volume, '=', n_voxels, 'x', atlas_img.affine[0, 0],'x', atlas_img.affine[1, 1])
    print(labels_out.shape, averages_out.shape, volume.shape)
    df = pd.DataFrame({"label": labels_out, "average": averages_out, "volume": volume})

    return df


def calculate_regional_averages_by_row(
    row: pd.Series,
    section_str: str,
    label_str: str,
    output_dir: str,
    clobber: bool = False,
    conversion_factor: float = 1,
) -> pd.DataFrame:
    """Calculate regional averages for each label in the 2d images."""
    acquisition_fn = row[section_str]
    atlas_fn = row[label_str]
    row_output_filename = f'{output_dir}/{atlas_fn.split(".")[0]}.csv'

    if not os.path.exists(row_output_filename) or clobber:
        avg_df = calculate_regional_averages(
            acquisition_fn, atlas_fn, conversion_factor=conversion_factor
        )

        # Create dataframe that repeats rows for each label
        df = row.to_frame().T
        df = df.loc[df.index.repeat(avg_df.shape[0])].reset_index(drop=True)
        df["label"] = avg_df["label"]
        df["average"] = avg_df["average"]
        df["volume"] = avg_df["volume"]
    else:
        df = pd.read_csv(row_output_filename)

    return df


def _average_over_label(
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


def average_over_label(
    labels: np.array, values: np.array, conversion_factor: float = 1
) -> tuple:
    # Average over label
    unique_labels = np.unique(labels)
    averages_out = np.zeros(len(unique_labels) - 1)
    labels_out = unique_labels[1:]
    n_out = np.zeros(averages_out.shape)

    for i, label in enumerate(labels_out):
        mask = labels == label
        averages_out[i] = np.mean(values[mask])
        n_out[i] = np.sum(mask)

    averages_out *= conversion_factor

    return averages_out, labels_out, n_out


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
        compactness: float = 1.0,
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
            mask = np.rint(mask)
            # perform 1 morphological erosion to remove edge voxels on mask
            mask = np.rint(mask)
            mask = morphology.binary_erosion(mask, morphology.disk(1))

            n_segments = np.random.uniform(30, 100, 1).astype(int)[0]

            print(section_filename)
            print(np.sum(img))
            print(mask_filename)
            print(np.sum(mask))

            cluster_image = slic(
                img,
                n_segments=n_segments,
                compactness=compactness,
                max_num_iter=max_num_iter,
                sigma=sigma,
                mask=mask,
                channel_axis=None,
            )
            # use <label> function to create numpy array with each connected region identified by a unique integer
            cluster_image = label(cluster_image, connectivity=1)

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
    # Remove rows where 'img' does not exist
    sect_info = sect_info.loc[sect_info["img"].apply(os.path.exists)]

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
    n_jobs: int = 2,
    clobber: bool = False,
) -> pd.DataFrame:
    """Calculate regional values."""
    os.makedirs(output_dir, exist_ok=True)

    row_output_dir = output_dir + "/rows/"
    os.makedirs(row_output_dir, exist_ok=True)

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
        res = Parallel(n_jobs=n_jobs)(
            delayed(calculate_regional_averages_by_row)(
                row,
                section_string,
                labels_string,
                row_output_dir,
                conversion_factor=conversion_factor,
                clobber=clobber,
            )
            for i, row in sect_info.iterrows()
            if row["chunk"] == 1
        )
        # Concatenate results
        print("Concatenating results")
        regional_values_df = pd.concat(res)

        print("Saving results")
        regional_values_df["source"] = [source] * regional_values_df.shape[0]

        regional_values_df.to_csv(output_filename)
    else:
        regional_values_df = pd.read_csv(output_filename)

    return regional_values_df, output_filename


def apply_2d_transformations(df: pd.DataFrame, clobber: bool = False) -> pd.DataFrame:
    """Apply 2D transformations to cluster images."""
    df["cluster_2d"] = df["cluster"].apply(lambda x: x.replace(".nii", "_2d.nii"))

    def apply_transform(row: pd.Series) -> None:
        """Apply 2D transformation."""
        tfm = row["2d_tfm"]
        img = row["cluster"]
        ref = row["2d_align"]
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

    def _apply_transform(row: pd.Series) -> None:
        tfm = row["nl_3d_tfm_fn"]
        img = row["cluster_volume_2d"]
        out_fn = row["cluster_volume_3d"]

        ref_volume_filename = row["reconstructed_filename"]
        if not os.path.exists(out_fn) or clobber:
            shell(
                f"antsApplyTransforms -v 1 -n NearestNeighbor -d 3 -i {img} -o {out_fn} -t {tfm} -r {ref_volume_filename}",
                verbose=True,
            )

            assert os.path.exists(out_fn), f"Error: {out_fn} does not exist"
            assert (
                np.sum(nib.load(out_fn).get_fdata()) > 0
            ), "Error: empty output volume"

    Parallel(n_jobs=8)(
        delayed(_apply_transform)(row) for i, row in chunk_info.iterrows()
    )

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
        output_chunk_info_list = []
        for (sub, hemisphere, chunk, acquisition), chunk_sect_info in sect_info.groupby(
            ["sub", "hemisphere", "chunk", "acquisition"]
        ):
            print(sub, hemisphere, chunk, acquisition)
            assert chunk_sect_info.shape[0] > 0

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


def prepare_dataframe(sect_info: pd.DataFrame, source: str) -> pd.DataFrame:
    """Prepare dataframe for plotting."""
    sect_info = sect_info.loc[sect_info["source"] == source]

    columns = []
    for col in ["sub", "hemisphere", "chunk", "sample", "label"]:
        if col in sect_info.columns:
            columns.append(col)

    sect_info.sort_values(columns, inplace=True)

    return sect_info


def get_interp_error_df(
    sect_info_orig: pd.DataFrame,
    sect_info_2d: pd.DataFrame,
    sect_info_3d: pd.DataFrame,
    output_dir: str,
    clobber: bool = False,
) -> pd.DataFrame:
    interp_error_csv = f"{output_dir}/validation_interp_error.csv"

    if not os.path.exists(interp_error_csv) or clobber:
        min_average, max_average = np.percentile(
            sect_info_orig["average"], [5, 95]
        )  # np.max(sect_info_orig["average"]) * 0.05
        # min_average = 10
        max_average = np.inf
        min_volume = 10**2 * 0.25**2
        print("Min Average:", min_average)
        print("Min Volume:", min_volume)

        sect_info_orig = prepare_dataframe(sect_info_orig, "original")
        sect_info_2d = prepare_dataframe(sect_info_2d, "2d")
        sect_info_3d = prepare_dataframe(sect_info_3d, "3d")

        df = pd.DataFrame([])

        volume_3d_skip = 0
        volume_2d_skip = 0
        volume_orig_skip = 0
        average_3d_skip = 0
        average_2d_skip = 0
        average_orig_skip = 0

        for i, (_, row) in enumerate(sect_info_3d.iterrows()):
            if i % 100 == 0:
                print(f"\t{np.round(100.*i/sect_info_orig.shape[0],1)}", end="\r")
            sub = row["sub"]
            hemisphere = row["hemisphere"]
            chunk = row["chunk"]
            label = row["label"]
            average_3d = row["average"]
            volume = row["volume"]
            cluster = row["cluster_volume_3d"]

            idx2d = (
                (sect_info_2d["sub"] == sub)
                & (sect_info_2d["hemisphere"] == hemisphere)
                & (sect_info_2d["chunk"] == chunk)
                & (sect_info_2d["label"] == label)
            )
            if idx2d.sum() == 0:
                print("No 2D found")
                print(sub, hemisphere, chunk, label)
                exit(0)
                continue

            average_2d = sect_info_2d.loc[idx2d, "average"].values[0]
            volume_2d = sect_info_2d.loc[idx2d, "volume"].values[0]
            cluster = sect_info_2d.loc[idx2d, "cluster"].values[0]
            cluster_2d = sect_info_2d.loc[idx2d, "cluster_2d"].values[0]

            idxorig = (
                (sect_info_orig["sub"] == sub)
                & (sect_info_orig["hemisphere"] == hemisphere)
                & (sect_info_orig["chunk"] == chunk)
                & (sect_info_orig["label"] == label)
            )
            if idxorig.sum() == 0:
                print(sub, hemisphere, chunk, label)
                exit(0)
                continue

            average_orig = sect_info_orig.loc[idxorig, "average"].values[0]
            volume_orig = sect_info_orig.loc[idxorig, "volume"].values[0]
            sample = sect_info_orig.loc[idxorig, "sample"].values[0]
            raw = sect_info_orig.loc[idxorig, "img"].values[0]

            if (
                average_3d < min_average or average_3d > max_average
            ):  # volume < min_volume or
                # print(f"Skipping (orig): {sub} {hemisphere} {chunk} {label} ")
                if False and volume < min_volume:
                    print(f"Volume: {volume}, min volume: {min_volume}")
                    print(volume_2d, volume_orig)
                    volume_3d_skip += 1
                    x = np.sum(nib.load(cluster).get_fdata() == label)
                    print(cluster_2d)
                    print("Cluster sum", x, x * 0.25**2)
                    exit(0)

                if average_3d < min_average:
                    print(f"Average: {average_3d}, min average: {min_average}")
                    average_3d_skip += 1
                continue
            if (
                average_2d < min_average
                or volume_2d < min_volume
                or average_2d > max_average
            ):
                print(f"Skipping (2d): {sub} {hemisphere} {chunk} {label} ")
                if volume_2d < min_volume:
                    print(f"2D Volume: {volume_2d}, min volume: {min_volume}")
                    volume_2d_skip += 1
                if average_2d < min_average:
                    print(f"2D Average: {average_2d}, min average: {min_average}")
                    average_2d_skip += 1
                continue

            if (
                average_orig < min_average
                or volume_orig < min_volume
                or average_orig > max_average
            ):
                # print(f"Skipping (orig): {sub} {hemisphere} {chunk} {label} {sample}")
                if volume_orig < min_volume:
                    print(f"Orig Volume: {volume_orig}, min volume: {min_volume}")
                    volume_orig_skip += 1
                if average_orig < min_average:
                    print(f"Orig Average: {average_orig}, min average: {min_average}")
                    average_orig_skip += 1
                continue

            row = pd.DataFrame(
                {
                    "sub": [sub],
                    "hemisphere": [hemisphere],
                    "chunk": [chunk],
                    "raw": [raw],
                    "sample": [sample],
                    "acquisition": [row["acquisition"]],
                    "label": [label],
                    "average_2d": [average_2d],
                    "average_orig": [average_orig],
                    "average_3d": [average_3d],
                    "volume": [volume_orig],
                    "volume_3d": [volume],
                    "cluster": [cluster],
                    "cluster_2d": [cluster_2d],
                }
            )
            df = pd.concat([df, row])
        assert df.shape[0] > 0, "Error: empty dataframe"
        df["error"] = (
            100.0 * (df["average_orig"] - df["average_3d"]) / df["average_orig"]
        )
        df["error_abs"] = np.abs(df["error"])
        df.sort_values(["error_abs"], inplace=True, ascending=False)
        df.to_csv(interp_error_csv)
        print(f"Volume 3D skipped: {volume_3d_skip}")
        print(f"Volume 2D skipped: {volume_2d_skip}")
        print(f"Volume Orig skipped: {volume_orig_skip}")
        print(f"Average 3D skipped: {average_3d_skip}")
        print(f"Average 2D skipped: {average_2d_skip}")
        print(f"Average Orig skipped: {average_orig_skip}")
    else:
        df = pd.read_csv(interp_error_csv)

    return df


def plot_validation_interp_error(
    sect_info_orig: pd.DataFrame,
    sect_info_2d: pd.DataFrame,
    sect_info_3d: pd.DataFrame,
    output_dir: str,
    clobber: bool = False,
) -> None:
    """Plot validation interpolation error between same labels."""
    print("Plot validation interp error")

    # get rid of duplicate rows based on sub, hemisphere, chunk, sample, label
    sect_info_orig = sect_info_orig.drop_duplicates(
        subset=["sub", "hemisphere", "chunk", "sample", "label"], keep=False
    )
    sect_info_2d = sect_info_2d.drop_duplicates(
        subset=["sub", "hemisphere", "chunk", "sample", "label"], keep=False
    )
    sect_info_3d = sect_info_3d.drop_duplicates(
        subset=["sub", "hemisphere", "chunk", "label"], keep=False
    )

    df = get_interp_error_df(
        sect_info_orig, sect_info_2d, sect_info_3d, output_dir, clobber=clobber
    )

    df["volume_change"] = 100.0 * np.abs(
        (df["volume_3d"] - df["volume"]) / df["volume"]
    )

    print(df.groupby(["acquisition", "chunk", "sample"])["error_abs"].mean())

    print(f"{output_dir}/validation_interp_error.csv")

    df.sort_values("error_abs", inplace=True, ascending=False)
    print(df)

    # calculate linear regression and get slope and intercept
    from scipy.stats import linregress

    slope_0, _, r2_0, _, _ = linregress(df["average_orig"], df["average_2d"])
    slope_1, _, r2_1, _, _ = linregress(df["average_orig"], df["average_3d"])

    slope_0 = np.round(slope_0, 3)
    slope_1 = np.round(slope_1, 3)

    r2_0 = np.round(r2_0, 3)
    r2_1 = np.round(r2_1, 3)

    print("Pearson r2 (2d):", r2_0)
    print("Pearson r2 (3d):", r2_1)

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    sns.regplot(x="average_orig", y="average_2d", data=df, scatter_kws={"s": 1})
    plt.xlabel("Raw Autoradiograph ROI (fmol/mg protein)")
    plt.ylabel("2D Interpolated ROI (fmol/mg protein)")
    plt.text(
        0.5,
        0.85,
        f"r\u00b2 = {r2_0}, slope={slope_0}\np<0.001",
        ha="center",
        va="center",
        transform=plt.gca().transAxes,
    )
    sns.despine(fig=None, ax=None, top=True, right=True)

    plt.subplot(1, 2, 2)
    sns.regplot(x="average_orig", y="average_3d", data=df, scatter_kws={"s": 1})
    plt.xlabel("Raw Autoradiograph ROI (fmol/mg protein)")
    plt.ylabel("3D Interpolated ROI (fmol/mg protein)")
    plt.text(
        0.5,
        0.85,
        f"r\u00b2 = {r2_1} slope={slope_1}\np<0.001",
        ha="center",
        va="center",
        transform=plt.gca().transAxes,
    )
    sns.despine(fig=None, ax=None, top=True, right=True)

    print("Volume size:")
    print(np.mean(df["volume"]), "+/-", np.std(df["volume"]))
    print(np.percentile(df["volume"], [25, 50, 75]))

    df["error_2d"] = (
        100.0 * (df["average_orig"] - df["average_2d"]) / df["average_orig"]
    )
    print("Error:")
    print(np.mean(df["error_2d"]), "+/-", np.std(df["error_2d"]))
    print(np.mean(df["error"]), "+/-", np.std(df["error"]))
    plt.suptitle("Validation of 2D & 3D Interpolation Error", fontsize=16)
    plt.savefig(f"{output_dir}/validation_interp_error.png", dpi=300)
    print(f"{output_dir}/validation_interp_error.png")
    plt.close()
    plt.cla()
    plt.clf()

    plt.figure(figsize=(10, 10))
    # plot interpolation error versus volume
    # sns.scatterplot(x="volume", y="error", data=df, alpha=0.1) # kwargs={'alpha': 0.2})
    # plt.subplot(1,2,1)
    sns.histplot(x="volume", y="error", data=df)  # kwargs={'alpha': 0.2})
    sns.despine(fig=None, ax=None, top=True, right=True)
    plt.ylabel("Interpolation Error %")
    plt.xlabel("Area (mm^2)")

    # plt.subplot(1,2,2)
    # sns.scatterplot(x="volume_change", y="error", data=df, alpha=0.1)
    # sns.despine(fig=None, ax=None, top=True, right=True)
    # plt.ylabel('Interpolation Error %')
    # plt.xlabel('Volume Change %')

    plt.savefig(f"{output_dir}/validation_interp_error_vs_volume.png", dpi=300)
    print(f"{output_dir}/validation_interp_error_vs_volume.png")

    # plt.subplot(2, 2, 4)
    # plot interpolation error versus volume
    # sns.scatterplot(x="volume", y="average_orig", data=df, alpha=0.1) #kwargs={'alpha': 0.2})
    # sns.despine(fig=None, ax=None, top=True, right=True)
    # plt.ylabel('Interpolation Error %')
    # plt.xlabel('Receptor Density (fmol/mg protein)')

    # add a sup title for the figure


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

    # sample 10% of the data
    sect_info = sect_info.sample(frac=0.1)

    # Unsupervised clustering of autoradiographs
    print("\tClustering sections")
    sect_info = cluster_sections(sect_info, output_dir, clobber=clobber)

    print("N samples:", sect_info.shape[0])
    # Apply 2D transformations to segmented autoradiotraphs
    print("\tApply 2D transformations to clusters")
    sect_info_2d = apply_2d_transformations(sect_info, clobber=clobber)

    # Calculate regional averages for each label in the autoradiographs
    print("\tCalculate regional averages")
    sect_info_orig, _ = regional_averages(
        sect_info, "img", "cluster", output_dir, "original", n_jobs=10, clobber=clobber
    )

    # Calculate regional averages for each label in the 2D warped classified autoradiographs
    print("\tCalculate regional averages for 2D warped classified autoradiographs")
    sect_info_2d, _ = regional_averages(
        sect_info_2d,
        "align_2d",
        "cluster_2d",
        output_dir,
        "2d",
        conversion_factor=1,
        n_jobs=10,
        clobber=clobber,
    )

    sect_info = pd.concat([sect_info_orig, sect_info_2d])

    # sect_info.to_csv(output_sect_info_csv)

    # Create 3D volumes of 2D warped classified autoradiographs
    print("\tCreate 3D volumes of 2D warped classified autoradiographs")
    chunk_info = create_3d_chunk_volumes(
        sect_info_2d.copy(), chunk_info, output_dir, clobber=clobber
    )

    # Apply 3D transformations to 3D volumes of 2D warped classified autoradiographs
    print("Apply 3D transformations")
    chunk_info = apply_3d_transformations(chunk_info, clobber=clobber)

    # Calculate regional averages for each label in the 3D volumes
    print("Calculate regional averages for label in 3D volumes")
    sect_info_3d, _ = regional_averages(
        chunk_info,
        "reconstructed_filename",
        "cluster_volume_3d",
        output_dir,
        "3d",
        n_jobs=5,
        clobber=clobber,
    )

    # Plot original label values versus 2d warped label values
    plot_validation_interp_error(sect_info_orig, sect_info_2d, sect_info_3d, output_dir)
