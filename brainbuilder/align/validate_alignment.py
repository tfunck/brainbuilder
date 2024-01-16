import os

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from brainbuilder.utils import utils
from joblib import Parallel, delayed

global ligand_receptor_dict
ligand_receptor_dict = {
    "ampa": "AMPA",
    "kain": "Kainate",
    "mk80": "NMDA",
    "ly34": "mGluR2/3",
    "flum": "GABA$_A$ Benz.",
    "cgp5": "GABA$_B$",
    "musc": "GABA$_A$ Agonist",
    "sr95": "GABA$_A$ Antagonist",
    "pire": r"Muscarinic M$_1$",
    "afdx": r"Muscarinic M$_2$ (antagonist)",
    "damp": r"Muscarinic M$_3$",
    "epib": r"Nicotinic $\alpha_4\beta_2$",
    "oxot": r"Muscarinic M$_2$ (oxot)",
    "praz": r"$\alpha_1$",
    "uk14": r"$\alpha_2$ (agonist)",
    "rx82": r"$\alpha_2$ (antagonist)",
    "dpat": r"5-HT$_{1A}$",
    "keta": r"5HT$_2$",
    "sch2": r"D$_1$",
    "dpmg": "Adenosine 1",
    "cellbody": "Cell Body",
    "myelin": "Myelin",
}


def dice(x, y):
    num = np.sum((x == 1) & (y == 1)) * 2
    den = np.sum(x) + np.sum(y)
    dice_score = num / den
    return dice_score


def modified_dice(x, y):
    num = np.sum((x == 1) & (y == 1))
    den = np.sum(y)
    dice_score = num / den
    return dice_score


def prepare_volume(vol):
    vol = (vol - np.min(vol)) / (vol.max() - vol.min())
    vol = np.round(vol).astype(int)
    return vol


def global_dice(fx_vol, mv_vol, clobber=False):
    fx_vol = prepare_volume(fx_vol)
    mv_vol = prepare_volume(mv_vol)

    dice_score = modified_dice(fx_vol, mv_vol)

    return dice_score, np.sum(mv_vol)


def local_metric(fx_vol, mv_vol, metric, offset=5):
    """Calculate the local metric between two volumes with kernel windows of size <offset>.
    :param: fx_vol: np.ndarray
    :param: mv_vol: np.ndarray
    :param: metric: function
    :param: offset: int
    :return: local_dice_volume: np.ndarray
    """
    fx_vol = prepare_volume(fx_vol)
    mv_vol = prepare_volume(mv_vol)

    fx_min = np.min(fx_vol)
    mv_min = np.min(mv_vol)

    local_dice_section = np.zeros(fx_vol.shape)

    for y in range(fx_vol.shape[0]):
        for x in range(fx_vol.shape[1]):
            x0 = max(0, x - offset)
            x1 = min(fx_vol.shape[1], x + offset)
            y0 = max(0, y - offset)
            y1 = min(fx_vol.shape[0], y + offset)

            # if  mv_vol[y, x] > mv_min: #only consider overlap in moving image
            if mv_vol[y, x] > 0:
                fx_sub = fx_vol[y0:y1, x0:x1]
                mv_sub = mv_vol[y0:y1, x0:x1]

                m = metric(fx_sub, mv_sub)
                m = m if not np.isnan(m) else 0

                local_dice_section[y, x] = m

    return local_dice_section


def qc_low_dice_section(row, slab, y, mri_vol, mv_vol, fx_vol, qc_dir):
    temp_fn = f"{qc_dir}/{slab}_{y}_example.png"
    if not os.path.exists(temp_fn):
        plt.cla()
        plt.clf()

        fig, ax = plt.subplots(2, 1, figsize=(9, 9))  # , sharex=True, sharey=True)
        plt.style.use("dark_background")
        ax[0].set_axis_off()
        ax[0].imshow(mv_vol)

        ax[1].imshow(fx_vol)
        ax[1].set_axis_off()
        plt.tight_layout()

        plt.savefig(temp_fn)
        plt.cla()
        plt.clf()


def get_section_metric(fx_fn, mv_fn, out_png, idx, verbose=False):
    img0 = nib.load(fx_fn)
    fx_vol = prepare_volume(img0.get_fdata())
    mv_vol = prepare_volume(nib.load(mv_fn).get_fdata())

    # section_dice_mean, fx_sum = global_dice(fx_vol, mv_vol)
    section_dice = local_metric(fx_vol, mv_vol, dice, offset=2)
    section_dice_mean = np.mean(section_dice[mv_vol > 0])

    fx_sum = np.sum(fx_vol)

    plt.cla()
    plt.clf()
    plt.title(f"Dice: {section_dice_mean:.3f}")
    plt.subplot(1, 3, 1)
    plt.imshow(fx_vol)
    plt.subplot(1, 3, 2)
    plt.imshow(mv_vol)
    plt.subplot(1, 3, 3)
    dice_vol = np.zeros(fx_vol.shape)
    dice_vol[(fx_vol > 0) & (mv_vol > 0)] = 1
    dice_vol[fx_vol * mv_vol > 0] = 2
    plt.imshow(mv_vol * fx_vol)
    plt.savefig(out_png)

    if verbose:
        print("\tValidation: ", out_png)

    return section_dice_mean, fx_sum, idx


def calculate_volume_accuracy(
    sect_info: pd.DataFrame,
    tfm_dir: str,
    num_cores: int = 0,
    clobber: bool = False,
):
    """Calculate the accuracy of the alignment of histological sections to the structural reference volume"""
    pd.DataFrame({})

    num_cores = int(num_cores) if num_cores > 0 else int(os.cpu_count() / 2)

    to_do = []

    for idx, (i, row) in enumerate(sect_info.iterrows()):
        y = row["sample"]
        base = os.path.basename(row["raw"]).split(".")[0]

        cls_fn = row["2d_align_cls"]
        tfm_dir = os.path.dirname(cls_fn)

        fx_fn = utils.gen_2d_fn(f"{tfm_dir}/y-{y}", "_fx")

        out_png = f"{tfm_dir}/{base}_y-{y}_dice.png"

        to_do.append((cls_fn, fx_fn, out_png, idx))

    parrallel_results = Parallel(n_jobs=num_cores)(
        delayed(get_section_metric)(fx_fn, cls_fn, out_png, idx, verbose=True)
        for cls_fn, fx_fn, out_png, idx in to_do
    )

    sect_info["dice"] = [0] * len(sect_info)

    for dice, total, idx in parrallel_results:
        sect_info["dice"].iloc[idx] = dice

    return sect_info


def output_stats(in_df):
    out_df = pd.DataFrame({})
    for (slab, resolution), temp_df in in_df.groupby(["slab", "resolution"]):
        w = temp_df["weight"].values
        acc = temp_df["align_dice"].values
        # print(w)
        # print(acc)
        w_norm = np.array(w) / np.sum(w)
        total_accuracy = np.sum(w_norm * acc)
        std_accuracy = np.sqrt(np.average((acc - total_accuracy) ** 2, weights=w_norm))
        print("Alignment Accuracy")
        print(np.round(total_accuracy, 3), "+/-", np.round(std_accuracy, 3))
        row = pd.DataFrame(
            {
                "slab": [slab],
                "resolution": [resolution],
                "dice": [total_accuracy],
                "sd": [std_accuracy],
            }
        )
        out_df = pd.concat([out_df, row])

    all_total_accuracy = np.sum(
        in_df["align_dice"] * (in_df["weight"] / in_df["weight"].sum())
    )
    all_std_accuracy = np.sqrt(
        np.average(
            (in_df["align_dice"] - all_total_accuracy) ** 2, weights=in_df["weight"]
        )
    )
    row = pd.DataFrame(
        {
            "slab": ["all"],
            "resolution": [resolution],
            "dice": [all_total_accuracy],
            "sd": [all_std_accuracy],
        }
    )
    out_df = pd.concat([out_df, row])

    return out_df


def validate_section_alignment(
    sect_info: pd.DataFrame, qc_dir: str, clobber: bool = False
):
    """Validate alignment of histological sections to structural reference volume"""
    print("\n\t\tValidate Alignment\n")

    os.makedirs(qc_dir, exist_ok=True)
    out_csv = f"{qc_dir}/alignment_accuracy.csv"

    cls_newer_than_out_csv = False
    if os.path.exists(out_csv):
        cls_newer_than_out_csv = [
            utils.newer_than(fn, out_csv) for fn in sect_info["2d_align_cls"].values
        ]

        if True in cls_newer_than_out_csv or clobber:
            clobber = True

    out_df = calculate_volume_accuracy(sect_info, qc_dir, clobber=clobber)
    out_df.to_csv(out_csv, index=True)

    # plot
    return out_df
