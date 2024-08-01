"""Quality control for data set."""
import argparse
import os
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from brainbuilder.utils import ants_nibabel as nib
from brainbuilder.utils import utils
from joblib import Parallel, delayed


def get_min_max(fn: str) -> str:
    """Get min and max values from image file.

    :param fn: image file name
    :return: min, max
    """
    print("qc read ", fn)
    ar = utils.load_image(fn)
    if isinstance(ar, np.ndarray):
        return np.min(ar), np.max(ar)
    else:
        return np.nan, np.nan


def get_min_max_parallel(
    df: pd.DataFrame, column: str, num_cores: int = 1
) -> Tuple[float, float]:
    """Get min and max values from image file in parallel.

    :param df: dataframe
    :param column: column name
    :return: min, max
    """
    min_max = Parallel(n_jobs=num_cores, backend="multiprocessing")(
        delayed(get_min_max)(row[column]) for i, (_, row) in enumerate(df.iterrows())
    )

    min_max = np.array(min_max)
    min_max = min_max[~np.isnan(min_max)]

    return np.min(min_max), np.max(min_max)


def data_set_quality_control(
    sect_info_csv: str,
    base_output_dir: str,
    column: str = "raw",
    num_cores: int = 1,
    clobber: bool = False,
) -> None:
    """Quality control for data set.

    :param sect_info_csv: section information csv file
    :param base_output_dir: base output directory
    :param column: column name
    :param clobber: clobber
    :return: None
    """
    os.makedirs(base_output_dir, exist_ok=True)

    sect_info = pd.read_csv(sect_info_csv, index_col=False)

    # sect_info = sect_info.loc[ (sect_info['sub'] == 11539) & (sect_info['acquisition'] == 'dpmg') ]
    for (sub, hemisphere, acquisition, chunk), df in sect_info.groupby(
        [
            "sub",
            "hemisphere",
            "acquisition",
            "chunk",
        ]
    ):
        output_dir = f"{base_output_dir}/sub-{sub}/hemi-{hemisphere}/{column}/"

        os.makedirs(output_dir, exist_ok=True)

        out_png = f"{output_dir}/sub-{sub}_hemi-{hemisphere}_chunk-{chunk}_acq-{acquisition}_{column}.png"
        n = len(df)

        if utils.check_run_stage([out_png] * n, df[column].values, clobber=clobber):
            n_images = len(df)
            n_cols = np.ceil(np.sqrt(n_images)).astype(int)
            n_rows = np.ceil(n_images / n_cols).astype(int)

            plt.figure(figsize=(n_cols * 2, n_rows * 2))  # , dpi=300)

            vmin, vmax = get_min_max_parallel(df, column, num_cores=num_cores)
            df.sort_values(by=["sample"], inplace=True)

            for i, (_, row) in enumerate(df.iterrows()):
                print(
                    row["sub"],
                    row["hemisphere"],
                    row["chunk"],
                    row["acquisition"],
                    row["sample"],
                )
                plt.subplot(n_rows, n_cols, i + 1)
                img = nib.load(row[column]).get_fdata()
                plt.imshow(img, cmap="nipy_spectral", vmin=vmin, vmax=vmax)
                plt.axis("off")
                title_string = f'{row["chunk"]} {row["acquisition"]} {row["sample"]}'
                plt.title(title_string)

            plt.tight_layout()
            print("\tSaving figure to: ", out_png)
            plt.savefig(out_png)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--sect-info-csv", type=str, dest="sect_info_csv")
    argparser.add_argument("--output-dir", type=str, dest="output_dir")
    argparser.add_argument("--column", type=str, dest="column", default="raw")
    argparser.add_argument(
        "--clobber", action="store_true", default=False, dest="clobber"
    )
    args = argparser.parse_args()

    data_set_quality_control(
        sect_info_csv=args.sect_info_csv,
        output_dir=args.output_dir,
        column=args.column,
        clobber=args.clobber,
    )
