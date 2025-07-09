import os

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
from scipy.ndimage import gaussian_filter
from skimage.transform import resize

global GM_LABEL
GM_LABEL = 1


def get_section_percentiles(
    img_path: str,
    seg_path: str,
    y: int,
    acquisition: str,
    chunk: int,
    output_dir: str,
    clobber: bool = False,
):
    auto_basename = os.path.basename(img_path).replace(".nii.gz", "")
    csv_file = f"{output_dir}/{auto_basename}.csv"

    if not os.path.exists(csv_file) or clobber:
        seg_vol = nib.load(seg_path).get_fdata()
        auto_vol = nib.load(img_path).get_fdata()

        assert (
            np.sum(seg_vol > 0) > 0
        ), f"1 Segmentation volume is all zero for {img_path} and {seg_path}"

        # seg_vol = np.flip(resize(seg_vol, auto_vol.shape, order=0), axis=(0,1))
        seg_vol = resize(seg_vol.astype(float), auto_vol.shape, order=0)
        assert (
            np.sum(seg_vol > 0) > 0
        ), f"2 Segmentation volume is all zero for {img_path} and {seg_path}"

        seg_vol[auto_vol <= 0] = 0

        # check that the segmentation volume is not all zero
        if np.sum(seg_vol > 0) == 0:
            print(f" Segmentation volume is all zero for {img_path} and {seg_path}")
            return None

        gm_deciles = np.zeros(9)
        # if GM_LABEL in np.unique(seg_vol):
        values = auto_vol[seg_vol > 0]
        # calculate deciles of gm
        gm_deciles = np.percentile(values, [10, 20, 30, 40, 50, 60, 70, 80, 90])

        # check that the percentiles are not all zero
        assert not np.all(
            gm_deciles == 0
        ), f"4 All percentiles are zero for {img_path} and {seg_path} with {gm_deciles}"

        row = pd.DataFrame(
            {
                "img": [img_path],
                "seg": [seg_path],
                "chunk": [chunk],
                "acquisition": [acquisition],
                "gm_10": [gm_deciles[0]],
                "gm_20": [gm_deciles[1]],
                "gm_30": [gm_deciles[2]],
                "gm_40": [gm_deciles[3]],
                "gm_50": [gm_deciles[4]],
                "gm_60": [gm_deciles[5]],
                "gm_70": [gm_deciles[6]],
                "gm_80": [gm_deciles[7]],
                "gm_90": [gm_deciles[8]],
                "y": [y],
            }
        )

        row.to_csv(csv_file, index=False)
    else:
        row = pd.read_csv(csv_file)

    return row


def get_dataset_percentiles(
    sect_info_df: pd.DataFrame,
    chunk_info_df: pd.DataFrame,
    percentiles_csv: str,
    output_dir: str,
    clobber: bool = False,
):
    """Get the percentiles of the dataset and save to csv file"""
    if not os.path.exists(percentiles_csv) or clobber:
        os.makedirs(output_dir, exist_ok=True)

        results = Parallel(n_jobs=1)(
            delayed(get_section_percentiles)(
                row["img"],
                row["seg"],
                row["sample"],
                row["acquisition"],
                row["chunk"],
                output_dir,
                clobber=clobber,
            )
            for _, row in sect_info_df.iterrows()
        )

        df_list = []
        for result in results:
            if result is not None:
                df_list.append(result)

        df = pd.concat(df_list)

        df["y0"] = df["y"]

        for (chunk, direction), _ in chunk_info_df.groupby(["chunk", "direction"]):
            if direction == "caudal_to_rostral":
                df.loc[df["chunk"] == chunk, "y0"] = (
                    df.loc[df["chunk"] == chunk, "y"].max()
                    - df.loc[df["chunk"] == chunk, "y"]
                )

        # drop rows with NaN values
        df = df.dropna()

        min_chunk = df["chunk"].min()
        max_chunk = df["chunk"].max()

        # for chunks 1 - 6, add the max y0 of the previous chunk to the next chunk
        for chunk in range(min_chunk, max_chunk):
            max_y0 = df.loc[df["chunk"] == chunk, "y0"].max()
            df.loc[df["chunk"] == chunk + 1, "y0"] = (
                df.loc[df["chunk"] == chunk + 1, "y0"] + max_y0
            )

        # melt the dataframe to long format
        df = pd.melt(
            df,
            id_vars=["img", "chunk", "acquisition", "y0", "seg"],
            value_vars=[f"gm_{dec}" for dec in range(10, 100, 10)],
            var_name="type",
            value_name="density",
        )

        df.to_csv(percentiles_csv, index=False)
    else:
        df = pd.read_csv(percentiles_csv)

    return df


def plot_correction_over_chunk(
    df, plot_dir, n_points: int = 10, order: int = 1, clobber: bool = False
):
    # create custom Red Green seaborn color palette
    # palette = sns.color_palette("RdGn", n_colors=2)

    os.makedirs(plot_dir, exist_ok=True)

    palette = sns.color_palette(
        ["#1f77b4", "#ff7f0e", "#2ca02c"]
    )  # Blue for "Uncorrected", Orange for "Batch Corrected", and Green for "Real"

    df["density_corr"] = df["density"] * df["slope"] + df["intercept"]

    def get_median_density(row, i):
        """Get the median density of the image"""
        if i % 100 == 0:
            print(f"{100*i/len(df)}")

        img_path = row["img_corr"]
        seg_path = row["seg"]
        img_data = nib.load(img_path).get_fdata()
        seg_data = nib.load(seg_path).get_fdata()
        seg_data = resize(seg_data.astype(float), img_data.shape, order=0)

        row["density_real"] = np.median(img_data[seg_data > 0])
        return row

    df["density_real"] = 0.0  # Initialize the column
    # res = Parallel(n_jobs=-1)(
    #    delayed(get_median_density)(row,i) for i, (_, row) in enumerate(df.iterrows())
    # )
    # df = pd.DataFrame(res)  # Ensure the result is a DataFrame
    # print(df)

    # melt density and density_corr to long format
    df = pd.melt(
        df,
        id_vars=["chunk", "acquisition", "y0"],
        value_vars=["density", "density_corr", "density_real"],
        var_name="state",
        value_name="density_value",
    )

    # rename density to "Uncorrected"
    df.loc[df["state"] == "density", "state"] = "Uncorrected"
    df.loc[df["state"] == "density_corr", "state"] = "Batch Corrected"
    df.loc[df["state"] == "density_real", "state"] = "Real"

    for acquisition, group in df.groupby("acquisition"):
        out_png = f"{plot_dir}/{acquisition}_{n_points}_order-{order}.png"

        if not os.path.exists(out_png) or clobber:
            plt.figure(figsize=(12, 6))
            sns.lineplot(
                data=group,
                x="y0",
                y="density_value",
                style="chunk",
                hue="state",
                palette=palette,
            )
            plt.title(f"{acquisition} means over y")
            plt.xlabel("Y index")
            plt.ylabel("Receptor Density (Percentiles)")
            plt.legend(title="chunk and Type")
            # place legend outside of plot
            plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
            plt.tight_layout()

            plt.savefig(out_png)
            plt.close()

            print("\tWriting", out_png)


def inter_chunk_regr(x: np.array, y: np.array, order: int = 1):
    assert len(x) == len(y), f"length of x: {len(x)}, length of y: {len(y)}"

    # calculate linear regression of dfcau onto dfros
    if order == 0:
        slope = 1
        intercept = np.median(y) - np.median(x)
    else:
        try:
            print(x, y)
            slope, intercept = np.polyfit(x, y, order)
        except SystemError or numpy.linalg.LinAlgError:
            print("SystemError")
            slope = 1
            intercept = np.median(y) - np.median(x)

    return slope, intercept


def get_rostral_caudal_values(chunk: int, acquisition_df: pd.DataFrame):
    iros = chunk + 0.5
    icau = iros + 0.5

    type_list = np.unique(acquisition_df["type"])

    dfros = acquisition_df.loc[acquisition_df["cluster"].astype(float) == float(iros)]
    dfcau = acquisition_df.loc[acquisition_df["cluster"] == icau]

    x = []
    y = []
    for t in type_list:
        dfros_t = dfros.loc[dfros["type"] == t]
        dfcau_t = dfcau.loc[dfcau["type"] == t]

        if len(dfros_t) > 0 and len(dfcau_t) > 0:
            x.append(dfcau_t["density"].values)
            y.append(dfros_t["density"].values)

    x = np.concatenate(x)
    y = np.concatenate(y)

    assert len(x) == len(y), f"length of x: {len(x)}, length of y: {len(y)}"

    # y = dfros['density'].values #rostral/front
    # x = dfcau['density'].values #caudal/back

    return x, y


def plot_regression(i, x, x1, y, slope, intercept, chunk, next_chunk, n_chunks):
    # plot the linear regression
    plt.subplot(n_chunks, 2, 2 * i + 1)
    plt.scatter(x, y, label="data", color="blue")
    plt.plot(x, slope * x + intercept, label="fit", color="red")
    plt.xlabel("Density (caudal)")
    plt.ylabel("Density (rostral)")
    # add slope and intercept to the plot
    plt.text(
        0.5,
        0.8,
        f"slope: {slope:.2f}\nintercept: {intercept:.2f}",
        transform=plt.gca().transAxes,
    )
    plt.title(f"chunk {chunk} to chunk {next_chunk}")

    plt.subplot(n_chunks, 2, 2 * i + 2)
    plt.scatter([chunk] * len(y), y, label="rostral", color="green")
    plt.scatter([next_chunk] * len(x), x, label="caudal", color="red")
    plt.scatter([next_chunk + 0.1] * len(x1), x1, label="corrected", color="blue")
    plt.xlabel("chunk")
    plt.ylabel("Density")
    plt.legend()


def correct_within_acquisition(
    acquisition_df: pd.DataFrame, plot_png: str, order: int = 1, clobber: bool = False
):
    chunk_min = acquisition_df["chunk"].min()
    chunk_max = acquisition_df["chunk"].max()
    n_chunks = chunk_max - chunk_min + 1

    plt.figure(figsize=(12, 6 * n_chunks // 2))

    for i, chunk in enumerate(np.arange(chunk_min, chunk_max)):
        next_chunk = chunk + 1

        # caudal/posterior == x, rostral/anterior == y

        x, y = get_rostral_caudal_values(chunk, acquisition_df)

        # regress caudal values onto rostral values
        slope, intercept = inter_chunk_regr(x, y, order=order)

        idx = acquisition_df["chunk"] == next_chunk

        # apply the correction to the next chunk
        acquisition_df.loc[idx, "slope"] = slope
        acquisition_df.loc[idx, "intercept"] = intercept

        density_corrected = slope * acquisition_df.loc[idx, "density"] + intercept

        # apply the correction to the next chunk
        acquisition_df.loc[idx, "density"] = density_corrected

        x1 = acquisition_df.loc[idx, "density"]

        plot_regression(i, x, x1, y, slope, intercept, chunk, next_chunk, n_chunks)

    # ensure that the density values are not negative
    density_offset = min(0, acquisition_df["density"].min())

    acquisition_df.loc[:, "density"] = acquisition_df["density"] - density_offset

    # Update the intercept to account for the offset
    acquisition_df.loc[:, "intercept"] = acquisition_df["intercept"] - density_offset

    plt.tight_layout()
    plt.savefig(plot_png)
    plt.close()

    return acquisition_df


def calculate_correction_factors(
    correction_df,
    output_dir: str,
    correction_factor_csv: str,
    n_points: int = 10,
    order: int = 1,
    clobber: bool = False,
):
    plot_dir = f"{output_dir}/plots/"
    os.makedirs(plot_dir, exist_ok=True)

    correction_df = get_ends_of_chunk(correction_df, 0.25)

    out_list = []

    if not os.path.exists(correction_factor_csv) or clobber:
        correction_df["slope"] = 1
        correction_df["intercept"] = 0

        correction_df["density_orig"] = correction_df["density"].copy()

        for acquisition in correction_df["acquisition"].unique():
            acquisition_df = correction_df.loc[
                correction_df["acquisition"] == acquisition
            ]

            plot_png = f"{plot_dir}/{acquisition}_{n_points}_order-{order}_regr.png"

            acquisition_df = correct_within_acquisition(
                acquisition_df, plot_png, order=order, clobber=clobber
            )

            out_list.append(acquisition_df)

        out_df = pd.concat(out_list)
        out_df.to_csv(correction_factor_csv, index=False)
    else:
        out_df = pd.read_csv(correction_factor_csv)

    return out_df


def get_ends_of_chunk(df: pd.DataFrame, point_offset=0.25):
    # identify the first and last n rows of each chunk by "y0"
    df.sort_values(by=["y0"], inplace=True)

    assert (
        point_offset > 0 and point_offset < 0.5
    ), "point_offset must be between 0 and 0.5"

    df["cluster"] = 0.0

    cluster_df_list = []
    for (chunk, _, _), groupdf in df.groupby(["chunk", "acquisition", "type"]):
        # for each chunk, get the first and last n rows
        i = float(chunk)

        n_points = max(int(point_offset * len(groupdf)), 1)

        first_n = groupdf.head(n_points)
        last_n = groupdf.tail(n_points)

        first_n.loc[:, "cluster"] = float(i)
        last_n.loc[:, "cluster"] = float(i) + 0.5

        # add the first and last n rows to the dataframe
        cluster_df_list.append(first_n)
        cluster_df_list.append(last_n)

    cluster_df = pd.concat(cluster_df_list)

    correction_df = (
        cluster_df.groupby(["chunk", "acquisition", "type", "cluster"])[
            "density"
        ]  # FIXME maybe we should use all density values not just median
        .median()
        .reset_index()
    )

    return correction_df


def apply_correction(
    in_path: str,
    seg_path: str,
    slope: float,
    intercept: float,
    out_path: str,
    clobber: bool = False,
):
    """Load "img" then apply slope and intercept correction then save to output_dir"""
    if not os.path.exists(out_path) or clobber:
        auto = nib.load(in_path)
        auto_data = auto.get_fdata()

        # seg_data = nib.load(seg_path).get_fdata()
        # seg_data = resize(seg_data.astype(float), auto_data.shape, order=0)

        # median0 = np.median(auto_data[seg_data > 0])

        auto_data = auto_data * slope + intercept

        # median1 = np.median(auto_data[seg_data > 0])

        # print('Slope:', slope, 'Intercept:', intercept, 'Median0:', median0, 'Median1:', median1,'Median2', median0*slope + intercept)

        nib.Nifti1Image(auto_data, auto.affine).to_filename(out_path)


def correct_batch_effect(
    sect_info_csv: str,
    chunk_info_csv: str,
    output_dir: str,
    n_points: int = 10,
    order: int = 1,
    clobber: bool = False,
):
    """Correct batch effect in the input csv file and save the corrected data to the output directory." """
    out_sect_info_csv = f"{output_dir}/{os.path.splitext(os.path.basename(sect_info_csv))[0]}_batch-corrected.csv"

    if not os.path.exists(out_sect_info_csv) or clobber:
        sect_info_df = pd.read_csv(sect_info_csv)
        chunk_info_df = pd.read_csv(chunk_info_csv)

        nii_dir = f"{output_dir}/nii/"
        os.makedirs(nii_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        df_list = []

        for (sub, hemisphere), curr_chunk_info_df in chunk_info_df.groupby(
            ["sub", "hemisphere"]
        ):
            curr_sect_info_df = sect_info_df.loc[
                (sect_info_df["sub"] == sub)
                & (sect_info_df["hemisphere"] == hemisphere)
            ]

            correction_df_csv = f"{output_dir}/sub-{sub}_hemi-{hemisphere}_correction_{n_points}_order-{order}.csv"

            percentiles_csv = f"{output_dir}/sub-{sub}_hemi-{hemisphere}_percentiles_{n_points}_order-{order}.csv"

            print("Getting percentiles for sub:", sub, "hemi:", hemisphere)
            df = get_dataset_percentiles(
                curr_sect_info_df,
                curr_chunk_info_df,
                percentiles_csv,
                output_dir + "percentiles/",
                clobber=clobber,
            )

            assert "img" in df.columns, f"img not in columns: {df.columns}"

            print("Calculating correction factors for sub:", sub, "hemi:", hemisphere)
            correction_df = calculate_correction_factors(
                df,
                output_dir,
                correction_df_csv,
                n_points=n_points,
                order=order,
                clobber=clobber,
            )

            correction_df["density_corr"] = (
                correction_df["density"] * correction_df["slope"]
                + correction_df["intercept"]
            )

            # merge correction_df into df by chunk and acquisition
            df = df.merge(
                correction_df[
                    ["chunk", "acquisition", "slope", "intercept", "density_corr"]
                ],
                on=["chunk", "acquisition"],
                how="left",
            )

            # define the corrected image path
            df["img_corr"] = df["img"].apply(
                lambda x: f"{nii_dir}/{os.path.basename(x).replace('.nii.gz', '_batch-corr.nii.gz')}"
            )

            # apply the correction to the density
            print("Applying correction to images for sub:", sub, "hemi:", hemisphere)
            Parallel(n_jobs=-1)(
                delayed(apply_correction)(
                    row["img"],
                    row["seg"],
                    row["slope"],
                    row["intercept"],
                    row["img_corr"],
                    clobber=clobber,
                )
                for _, row in df.iterrows()
            )

            # for (chunk, acq), df0 in df.groupby(['chunk','acquisition']):
            #    print(chunk, acq)
            #    print(df0['slope'].values)
            #    print(df0['acquisition'].values)

            print("Plotting correction over chunk for sub:", sub, "hemi:", hemisphere)
            plot_correction_over_chunk(
                df, output_dir + "/qc/", n_points=n_points, order=order, clobber=clobber
            )

            df_list.append(df)

        df = pd.concat(df_list)

        # merge df into sect_info_df for slope and intercept
        sect_info_df = sect_info_df.merge(
            df[
                [
                    "img",
                    "img_corr",
                    "chunk",
                    "acquisition",
                    "slope",
                    "intercept",
                    "y0",
                    "density_corr",
                ]
            ],
            on=["img", "chunk", "acquisition"],
            how="left",
        )
        sect_info_df["img_orig"] = sect_info_df["img"].copy()
        sect_info_df["img"] = sect_info_df["img_corr"]
        sect_info_df.drop(columns=["img_corr"], inplace=True)

        # remove duplicates
        sect_info_df = sect_info_df.drop_duplicates(
            subset=["img", "sub", "hemisphere", "chunk", "acquisition", "y0"],
            keep="last",
        )

        # check if img is nan
        if sect_info_df["img"].isnull().any():
            # drop rows with missing images
            sect_info_df = sect_info_df.dropna(subset=["img"])

        sect_info_df.to_csv(out_sect_info_csv, index=False)

    sect_info_df = pd.read_csv(out_sect_info_csv)

    return out_sect_info_csv


def correct_section_means(sect_info_csv: str, output_dir: str, clobber: bool = False):
    """1. Calculate the averge distance between samples ('y0')
    2. Apply gaussian smoothing to density_corr with sd = average distance
    3. Save the smoothed density_corr to a new csv file
    4. Plot the smoothed density_corr vs unsmoothed
    """
    os.makedirs(output_dir, exist_ok=True)

    out_sect_info_csv = f"{output_dir}/{os.path.splitext(os.path.basename(sect_info_csv))[0]}_smoothed.csv"

    if not os.path.exists(out_sect_info_csv) or clobber:
        sect_info_df = pd.read_csv(sect_info_csv)

        # sort by y0
        sect_info_df.sort_values(by=["y0"], inplace=True)

        # calculate the average distance between samples

        # apply gaussian smoothing to density_corr with sd = average distance
        for acquisition, group in sect_info_df.groupby("acquisition"):
            y_min = group["y0"].min()
            y_max = group["y0"].max()

            avg_distance_fwhm = np.diff(np.unique(group["y0"])).mean()

            print(f"Average distance between samples: {avg_distance_fwhm}")

            avg_distance_sd = avg_distance_fwhm / 2.3548  # convert fwhm to sd

            print(f"Y min: {y_min}, Y max: {y_max}")

            y_space = np.arange(y_min, y_max + 1)

            density_corr = group["density_corr"].values

            # use 1d linear interpolation to create density_corr array over all values in y_space
            density_corr_interp = np.interp(y_space, group["y0"], density_corr)

            # apply gaussian smoothing to density_corr_interp
            density_corr_smooth_all = gaussian_filter(
                density_corr_interp, sigma=avg_distance_sd
            )

            plt.figure(figsize=(12, 6))
            sns.scatterplot(x=group["y0"], y=density_corr, label="density_corr")
            sns.lineplot(x=y_space, y=density_corr_interp, label="density_interp")
            sns.lineplot(
                x=y_space, y=density_corr_smooth_all, label="density_corr_smooth"
            )
            plt.savefig(f"{output_dir}/{acquisition}_density_corr_smooth.png")
            print(f"{output_dir}/{acquisition}_density_corr_smooth.png")

            # create a new dataframe with the smoothed density_corr
            density_corr_smooth = density_corr_smooth_all[group["y0"].values - y_min]

            # add the smoothed density_corr to the dataframe
            sect_info_df.loc[
                sect_info_df["acquisition"] == acquisition, "density_corr_smooth"
            ] = density_corr_smooth

        # save the smoothed density_corr to a new csv file
        sect_info_df.to_csv(out_sect_info_csv, index=False)

        # melt the dataframe to long format for density_corr and density_corr_smooth
        sect_info_df = pd.melt(
            sect_info_df,
            id_vars=["img", "chunk", "acquisition", "y0"],
            value_vars=["density_corr", "density_corr_smooth"],
            var_name="type",
            value_name="density",
        )

        # plot the smoothed density_corr vs unsmoothed
        for acquisition, group in sect_info_df.groupby("acquisition"):
            plt.figure(figsize=(12, 6))
            sns.lineplot(data=group, x="y0", y="density", hue="type")
            plt.title(f"{acquisition} smoothed vs unsmoothed")
            plt.xlabel("Y index")
            plt.ylabel("Receptor Density (Percentiles)")
            plt.legend(title="Type")
            plt.tight_layout()
            plt.savefig(f"{output_dir}/{acquisition}_smoothed_vs_unsmoothed.png")
            plt.close()

    return out_sect_info_csv


def intensity_correction(
    sect_info_csv, chunk_info_csv, output_dir, clobber: bool = False
):
    os.makedirs(output_dir, exist_ok=True)

    batch_sect_info_csv = correct_batch_effect(
        sect_info_csv, chunk_info_csv, output_dir + "/chunk/", clobber=clobber
    )

    # image_sect_info_csv = correct_section_means(
    #    batch_sect_info_csv,
    #    output_dir + '/section/',
    #    clobber=clobber
    #    )
    #
    # apply the correction to the original images

    return batch_sect_info_csv
