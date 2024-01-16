import argparse
import glob
import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from brainbuilder.interp.batch_correction import calc_batch_correction_params
from brainbuilder.utils.mesh_utils import load_mesh_ext, visualization
from joblib import Parallel, delayed
from sklearn.linear_model import LinearRegression

global ligand


class Simulation:
    def __init__(
        self,
        epoch,
        sphere_coords_filename,
        surf_coords_filename,
        out_dir,
        reference_volume_filename,
        noise_ratio: float = -1,
        chunk_reduction: float = -1,
        clobber: bool = False,
    ):
        """Generate instance of simulation to validate surface-based batch correction

        :param epoch: epoch number
        :param sphere_coords_filename: path to sphere coordinates
        :param surf_coords_filename: path to surface coordinates
        :param out_dir: path to output directory
        :param reference_volume_filename: path to reference volume
        :param noise_ratio: standard deviation of noise
        :param clobber: boolean indicating whether to overwrite existing files
        """
        os.makedirs(out_dir, exist_ok=True)
        print("Epoch", epoch)

        self.epoch = epoch
        self.out_dir = out_dir
        self.n_slabs = 6
        self.n_vtx = load_mesh_ext(surf_coords_filename)[0].shape[0]

        self.slab = np.random.randint(1, self.n_slabs)
        self.sphere_coords_filename = sphere_coords_filename
        self.surf_coords_filename = surf_coords_filename

        self.reference_volume_filename = reference_volume_filename

        self.labels_qc_filename = f"{out_dir}/surface_values_labels_{epoch}.png"
        self.orig_qc_filename = f"{out_dir}/surface_values_orig_{epoch}.png"
        self.corrected_qc_filename = f"{out_dir}/surface_values_corr_{epoch}.png"
        self.noise_qc_filename = f"{out_dir}/surface_values_noise_{epoch}.png"
        self.batch_qc_filename = f"{out_dir}/surface_values_batch_{epoch}.png"

        self.vtx_values_orig_filename = f"{out_dir}/surface_values_orig_{epoch}"
        self.vtx_values_batch_filename = f"{out_dir}/surface_values_batch_{epoch}"
        self.vtx_values_noise_filename = f"{out_dir}/surface_values_noise_{epoch}"
        self.label_values_filename = f"{out_dir}/surface_values_labels_{epoch}"

        self.vtx_values_corrected_filename = (
            f"{out_dir}/surface_values_corrected_{epoch}"
        )

        if noise_ratio < 0:
            noise_ratio = np.random.uniform(0, 1)
        self.noise_ratio = noise_ratio

        if chunk_reduction < 0:
            chunk_reduction = np.random.uniform(0, 0.5)
        self.chunk_reduction = chunk_reduction

        # create batches in the surface
        self.labels = self.create_batches(
            chunk_reduction=chunk_reduction, clobber=clobber
        )

        # generate random surface values
        self.vtx_values, self.slope = self.generate_random_surface_values(
            clobber=clobber
        )

        print("1", self.vtx_values)

        # add noise to the surface values
        self.nsr = self.add_noise(noise_ratio, use_noise=True, clobber=clobber)

        print("2", self.vtx_values)

        # add batch effects to the surface values
        self.vtx_values_batch = self.add_batch_effects(
            use_batch_effects=True, clobber=clobber
        )

        self.correct_batch_effects()

    def add_noise(self, noise_ratio, use_noise=True, clobber=False):
        """Generate random gaussian noise"""
        if use_noise:
            if not os.path.exists(self.vtx_values_noise_filename + ".npz") or clobber:
                print("Add noise")
                orig = np.load(self.vtx_values_orig_filename + ".npz")["data"]

                std = np.std(orig[orig > 0]) * noise_ratio

                noise = np.random.normal(0, std, orig.shape)

                nsr = np.sum(np.abs(noise)) / np.sum(np.abs(orig))

                noisy_orig = orig + noise

                np.savez(self.vtx_values_noise_filename, data=noisy_orig, nsr=nsr)
                visualization(
                    self.surf_coords_filename, noisy_orig, self.noise_qc_filename
                )
            else:
                img = np.load(self.vtx_values_noise_filename + ".npz")
                nsr = img["nsr"]
        else:
            shutil.copy(
                self.vtx_values_orig_filename + ".npz",
                self.vtx_values_noise_filename + ".npz",
            )
            nsr = 0

        return nsr

    def correct_batch_effects(self, clobber=False):
        # 1. calculate batch correction parameters
        print(np.unique(np.load(self.label_values_filename + ".npz")["data"]))
        params, unique_paired_values = calc_batch_correction_params(
            self.label_values_filename,
            self.vtx_values_batch_filename,
            self.sphere_coords_filename,
            self.surf_coords_filename,
            self.out_dir,
            label_start=0,
            label_offset=1,
            clobber=clobber,
        )

        # 2. apply correction
        self.apply_correction(params, unique_paired_values)

        # 3. evaluate correction
        self.r2, self.batch_effect_size = self.evaluate()

        # 4. display stuff
        print("r2 =", self.r2)
        print("batch effect size = ", self.batch_effect_size)
        print("nsr = ", self.nsr)
        print("slope =", self.slope)
        print("chunk reduction =", self.chunk_reduction)

        output_csv = f"{self.out_dir}/results_{self.epoch}.csv"

        df = pd.DataFrame(
            {
                "epoch": [self.epoch],
                "r2": [self.r2],
                "batch_effect_size": [self.batch_effect_size],
                "nsr": [self.nsr],
                "slope": [np.abs(self.slope[1])],
                "chunk_reduction": [self.chunk_reduction],
            }
        )

        print("Writing:", output_csv)
        df.to_csv(output_csv, index=None)

    def apply_correction(self, params, unique_paired_values):
        paired_values = unique_paired_values

        labels_vtx = np.load(self.label_values_filename + ".npz")["data"]
        labels = np.unique(labels_vtx)[1:]

        vtx_values_batch = np.load(self.vtx_values_batch_filename + ".npz")["data"]
        vtx_values_corrected = np.zeros(vtx_values_batch.shape)

        vtx_values = np.load(self.vtx_values_orig_filename + ".npz")["data"]

        for i, row in params.iterrows():
            slab = row["label"]
            offset = row["offset"]

            idx = labels_vtx == slab

            vtx_values_corrected[idx] = vtx_values_batch[idx] + offset

        idx = labels_vtx > 0

        approx_mean = np.mean(vtx_values[idx])

        vtx_values_corrected[idx] = (
            vtx_values_corrected[idx] - np.mean(vtx_values_corrected[idx]) + approx_mean
        )

        visualization(
            self.surf_coords_filename, vtx_values_corrected, self.corrected_qc_filename
        )
        np.savez(self.vtx_values_corrected_filename, data=vtx_values_corrected)

    def evaluate(self):
        labels = np.load(self.label_values_filename + ".npz")["data"]
        idx = labels > 0

        load = lambda fn: np.load(fn + ".npz")["data"][idx].reshape(-1, 1)
        avg_diff = lambda x, y: np.mean(np.abs(x - y))

        batch = load(self.vtx_values_batch_filename)
        noise = load(self.vtx_values_noise_filename)
        corrected = load(self.vtx_values_corrected_filename)
        orig = load(self.vtx_values_orig_filename)

        r2 = LinearRegression().fit(orig, corrected).score(orig, corrected)
        sum_orig = np.mean(np.abs(orig))

        batch_effect_size = avg_diff(batch, orig) / sum_orig

        return r2, batch_effect_size

    def create_batches(self, border_offset=5, chunk_reduction=0.1, clobber=False):
        """Create batches in the surface"""
        if not os.path.exists(self.label_values_filename + ".npz") or clobber:
            # create np array where we will keep track of the labels
            labels = np.zeros(self.n_vtx)

            # load the y coordinates of the surface
            coords = load_mesh_ext(self.surf_coords_filename)[0]
            y = coords[:, 1]

            ymin = np.min(y)
            ymax = np.max(y)

            # percentiles = np.arange(0,101,100/(self.n_slabs))
            percentiles = np.linspace(ymin, ymax, self.n_slabs + 1)

            for i, (ylimc, ylimr) in enumerate(
                zip(percentiles[:-1:], percentiles[1::])
            ):
                d = np.abs(ylimr - ylimc) * chunk_reduction

                # print('\tc', ylimc, 'r', ylimr, 'd', d )
                ylimr = ylimr - d
                ylimc = ylimc + d

                # print('\tcaudal', ylimc, 'rostral', ylimr )
                idx = (y >= ylimc) & (y < ylimr)

                assert (
                    np.sum(idx) > 0
                ), "Error: no vertices between rostral and caudal limits"

                labels[idx] = i + 1

            print("label values", np.unique(labels))

            visualization(self.surf_coords_filename, labels, self.labels_qc_filename)

            np.savez(self.label_values_filename, data=labels)

        labels = np.load(self.label_values_filename + ".npz")["data"]

        return labels

    def add_batch_effects(self, use_batch_effects=True, clobber=False):
        if not os.path.exists(self.vtx_values_batch_filename + ".npz") or clobber:
            if use_batch_effects:
                vtx_values_batch = np.zeros(self.n_vtx)

                scale_factor = np.ones(
                    self.n_slabs
                )  # np.random.normal(0.5,1.5,self.n_slabs)

                print("3", self.vtx_values)
                print(np.sum(self.vtx_values > 0))

                idx = np.abs(self.vtx_values) > 0

                values_min = np.min(self.vtx_values[idx])
                values_max = np.max(self.vtx_values[idx])

                offset_factor = np.random.uniform(values_min, values_max, self.n_slabs)

                for i, slab in enumerate(np.unique(self.labels)[1:]):
                    idx = slab == self.labels

                    print("slab", slab, "idx", np.sum(idx))

                    assert np.sum(idx) > 0, "Error: no vertices in label slab " + slab
                    vtx_values_batch[idx] = (
                        offset_factor[i] + scale_factor[i] * self.vtx_values[idx]
                    )

                np.savez(self.vtx_values_batch_filename, data=vtx_values_batch)
                visualization(
                    self.surf_coords_filename, vtx_values_batch, self.batch_qc_filename
                )

            else:
                shutil.copy(
                    self.vtx_values_orig_filename + ".npz",
                    self.vtx_values_batch_filename + ".npz",
                )

                vtx_values_batch = np.load(self.vtx_values_batch_filename + ".npz")[
                    "data"
                ]
                visualization(
                    self.surf_coords_filename, vtx_values_batch, self.batch_qc_filename
                )

        vtx_values_batch = np.load(self.vtx_values_batch_filename + ".npz")["data"]

        return vtx_values_batch

    def generate_random_surface_values(self, clobber=False):
        if not os.path.exists(self.vtx_values_orig_filename + ".npz") or clobber:
            sphere_coords = load_mesh_ext(self.surf_coords_filename)[0]

            value_min = np.min(sphere_coords, axis=0)
            value_max = np.max(sphere_coords, axis=0)
            value_range = value_max - value_min

            sphere_coords = (sphere_coords - value_min) / value_range

            print(value_range.shape)

            slope = np.random.uniform(-5, 5, 3)
            print("True Slope:", slope, "\n")
            sphere_coords = sphere_coords * slope
            y = sphere_coords[:, 1]
            vtx_values = np.sum(sphere_coords, axis=1)

            vtx_values[self.labels == 0] = 0

            visualization(self.surf_coords_filename, vtx_values, self.orig_qc_filename)
            np.savez(self.vtx_values_orig_filename, data=vtx_values, slope=slope)

        f = np.load(self.vtx_values_orig_filename + ".npz")

        vtx_values = f["data"]

        slope = f["slope"]

        return vtx_values, slope


def plot_results(df, dist_df):
    df = df.sort_values(by=["epoch"]).reset_index(drop=True)
    dist_df = dist_df.sort_values(by=["epoch"]).reset_index(drop=True)
    df["Distance (mm)"] = dist_df["distance"]
    df["distance"] = dist_df["distance"]
    df["Noise-to-Signal Ratio"] = df["nsr"]
    df["Batch Effect Size"] = df["batch_effect_size"]
    df["Sagittal Gradient Size"] = df["slope"]

    columns = [
        "Distance (mm)",
        "Noise-to-Signal Ratio",
        "Batch Effect Size",
        "Sagittal Gradient Size",
    ]

    for x in columns:
        plt.figure(figsize=(10, 10))
        ax = sns.scatterplot(x=x, y="r2", data=df)
        ax.set(xlabel=x, ylabel="r2")
        ax.set_title("r2 vs " + x)
        out_png = f"./batch_correction/r2_vs_{x}.png"
        print("Writing", out_png)
        ax.figure.savefig(out_png)
        ax.figure.clf()

    plt.figure(figsize=(10, 10))
    ax = sns.scatterplot(
        x="Distance (mm)",
        y="Sagittal Gradient Size",
        hue="r2",
        data=df,
        palette="nipy_spectral_r",
    )
    out_png = "./batch_correction/dist_gradient.png"
    ax.figure.savefig(out_png)
    ax.figure.clf()

    plt.figure(figsize=(10, 10))
    sns.histplot(data=df, x="r2", stat="probability", bins=20)
    out_png = "./batch_correction/r2_hist.png"
    plt.savefig(out_png)
    plt.clf()

    df.dropna(inplace=True)

    x = df[columns]
    y = df["r2"]

    import statsmodels.formula.api as sm

    print(df.columns)
    regr = sm.ols("r2 ~  nsr + batch_effect_size * distance * slope", data=df).fit()

    print(regr.summary())


def calc_distance(f, coords):
    tdf = pd.read_csv(f)
    epoch = f.split("/")[-2].split("_")[0]
    tdf["epoch"] = epoch

    i = tdf["curr_idx"].values
    j = tdf["next_idx"].values
    dist = np.sqrt(np.sum((coords[i, :] - coords[j, :]) ** 2, axis=1))
    tdf["distance"] = dist
    return tdf


def calculate_average_distances(paired_values_files, surface_filename):
    coords = load_mesh_ext(surface_filename)[0]

    df = pd.DataFrame()

    results = Parallel(n_jobs=8)(
        delayed(calc_distance)(f, coords) for f in paired_values_files
    )

    df = pd.concat(results)

    df_out = df.groupby(["epoch"])["distance"].mean().reset_index()

    return df_out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument(
        "--clobber",
        "-c",
        dest="clobber",
        default=False,
        action="store_true",
        help="Overwrite existing results",
    )
    args = parser.parse_args()

    n_epochs = 1000

    existing_results = glob.glob("./batch_correction/*/results_*.csv")
    n_existing_results = len(existing_results)

    n_start = n_existing_results + 1 if n_existing_results > 0 else 0

    reference_volume_filename = "data/mr1/mri1_gm_bg_srv.nii.gz"

    surf_coords_fn = "data/mr1/MR1_gray_surface_R_81920.surf.gii"
    sphere_coords_fn = "data/mr1_test_output/4_interp/sub-mr1/hemi-R/acq-synth/surfaces/surf_2mm_0.0.sphere"
    n_jobs = 6
    print(n_start, n_epochs)
    Parallel(n_jobs=n_jobs)(
        delayed(Simulation)(
            epoch,
            sphere_coords_fn,
            surf_coords_fn,
            f"./batch_correction/{epoch}/",
            reference_volume_filename,
            clobber=args.clobber,
        )
        for epoch in range(n_start, n_epochs)
    )

    results_files = glob.glob("./batch_correction/*/results_*.csv")
    paired_values_files = glob.glob("./batch_correction/*/paired_values.csv")

    dist_df_fn = "./batch_correction/dist_df.csv"
    if not os.path.exists(dist_df_fn) or args.clobber:
        dist_df = calculate_average_distances(paired_values_files, surf_coords_fn)
    else:
        dist_df = pd.read_csv(dist_df_fn)

    df = pd.concat([pd.read_csv(f) for f in results_files])

    plot_results(df, dist_df)
