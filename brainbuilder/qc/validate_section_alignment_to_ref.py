"""Validate alignment of histological sections to structural reference volume."""
import os
from typing import Callable, Tuple

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


def dice(x: float, y: float) -> float:
    """Calculate the Dice coefficient between two binary arrays.

    Args:
        x (np.ndarray): Binary array.
        y (np.ndarray): Binary array.

    Returns:
        float: Dice coefficient.
    """
    num = np.sum((x == 1) & (y == 1)) * 2
    den = np.sum(x) + np.sum(y)
    dice_score = num / den
    return dice_score


def modified_dice(x: float, y: float) -> float:
    """Calculate the modified Dice coefficient between two binary arrays.

    Args:
        x (np.ndarray): Binary array.
        y (np.ndarray): Binary array.

    Returns:
        float: Modified Dice coefficient.
    """
    num = np.sum((x == 1) & (y == 1))
    den = np.sum(y)
    dice_score = num / den
    return dice_score


def prepare_volume(vol: np.ndarray) -> np.ndarray:
    """Prepare the volume by normalizing and rounding it.

    :param vol (np.ndarray): Input volume.
    :return np.ndarray: Prepared volume.
    """
    vol = (vol - np.min(vol)) / (vol.max() - vol.min())
    vol = np.round(vol).astype(int)
    return vol


def global_dice(
    fx_vol: np.ndarray, mv_vol: np.ndarray, clobber: bool = False
) -> Tuple[float, int]:
    """Calculate the global dice score between two volumes.

    Args:
        fx_vol (np.ndarray): Fixed volume.
        mv_vol (np.ndarray): Moving volume.
        clobber (bool, optional): Whether to clobber existing results. Defaults to False.

    Returns:
        Tuple[float, int]: Global dice score and sum of moving volume.
    """
    fx_vol = prepare_volume(fx_vol)
    mv_vol = prepare_volume(mv_vol)

    dice_score = modified_dice(fx_vol, mv_vol)

    return dice_score, np.sum(mv_vol)


def local_metric(
    fx_vol: np.ndarray, mv_vol: np.ndarray, metric: Callable, offset: int = 5
) -> np.ndarray:
    """Calculate the local metric between two volumes with kernel windows of size <offset>.

    :param: fx_vol: np.ndarray
    :param: mv_vol: np.ndarray
    :param: metric: function
    :param: offset: int
    :return: local_dice_volume: np.ndarray
    """
    fx_vol = prepare_volume(fx_vol)
    mv_vol = prepare_volume(mv_vol)

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


def qc_low_dice_section(
    row: pd.DataFrame,
    slab: int,
    y: int,
    mri_vol: np.array,
    mv_vol: np.array,
    fx_vol: np.array,
    qc_dir: str,
) -> None:
    """Perform quality control on low dice section.

    :param row: The row.
    :param slab: The slab.
    :param y: The y-coordinate.
    :param mri_vol: The MRI volume.
    :param mv_vol: The moving volume.
    :param fx_vol: The fixed volume.
    :param qc_dir: The directory for quality control.
    :return: None
    """
    # Add your code here

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


def get_section_metric(
    fx_fn: str, mv_fn: str, out_png: str, idx: int, verbose: bool = False
) -> None:
    """Calculate the section metric for given input files and parameters.

    :param fx_fn: The filename of the fixed volume.
    :param mv_fn: The filename of the moving volume.
    :param out_png: The filename for the output PNG.
    :param idx: The index.
    :param verbose: Whether to enable verbose mode.
    :return: None
    """
    # Add your code here
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


def section_to_ref_dice(
    sect_info: pd.DataFrame,
    tfm_dir: str,
    num_cores: int = 0,
    clobber: bool = False,
) -> pd.DataFrame:
    """Calculate the accuracy of the alignment of histological sections to the structural reference volume.

    :param sect_info: pd.DataFrame
    :param tfm_dir: str
    :param num_cores: int
    :param clobber: bool
    :return: pd.DataFrame
    """
    pd.DataFrame({})

    num_cores = int(num_cores) if num_cores > 0 else int(os.cpu_count() / 2)

    to_do = []

    for idx, (i, row) in enumerate(sect_info.iterrows()):
        y = row["sample"]
        base = row["base"]

        cls_fn = row["2d_align_cls"]
        tfm_dir = os.path.dirname(cls_fn)

        fx_fn = utils.gen_2d_fn(f"{tfm_dir}/{base}_y-{y}", "_fx")

        out_png = f"{tfm_dir}/{base}_y-{y}_dice.png"

        to_do.append((cls_fn, fx_fn, out_png, idx))

    parrallel_results = Parallel(n_jobs=num_cores)(
        delayed(get_section_metric)(fx_fn, cls_fn, out_png, idx, verbose=False)
        for cls_fn, fx_fn, out_png, idx in to_do
    )

    sect_info["dice"] = [0] * len(sect_info)

    for dice, total, idx in parrallel_results:
        sect_info["dice"].iloc[idx] = dice

    return sect_info


def output_stats(in_df: pd.DataFrame) -> pd.DataFrame:
    """Output stats for alignment accuracy.

    :param in_df: pd.DataFrame
    :return: pd.DataFrame
    """
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


def validate_section_alignment_to_ref(
    sect_info: pd.DataFrame, qc_dir: str, output_prefix: str = "", clobber: bool = False
) -> pd.DataFrame:
    """Validate alignment of histological sections to structural reference volume.

    :param sect_info: pd.DataFrame
    :param qc_dir: str
    :param clobber: bool
    :return: pd.DataFrame
    """
    print("\n\t\tValidate Alignment to Reference\n")

    out_csv = f"{qc_dir}/{output_prefix}section_to_ref_dice.csv"
    print(out_csv)

    if not os.path.exists(out_csv) or clobber:
        os.makedirs(qc_dir, exist_ok=True)

        cls_newer_than_out_csv = False
        if os.path.exists(out_csv):
            cls_newer_than_out_csv = [
                utils.newer_than(fn, out_csv) for fn in sect_info["2d_align_cls"].values
            ]

            if True in cls_newer_than_out_csv or clobber:
                clobber = True

        out_df = section_to_ref_dice(sect_info, qc_dir, clobber=clobber)
        out_df.to_csv(out_csv, index=True)

    out_df = pd.read_csv(out_csv)

    m = out_df.groupby(["sub", "hemisphere", "chunk"])["dice"].mean()
    s = out_df.groupby(["sub", "hemisphere", "chunk"])["dice"].std()
    print(m)
    print(s)

    # plot
    return out_df
