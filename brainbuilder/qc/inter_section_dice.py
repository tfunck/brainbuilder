"""Calculate the dice score between adjacent sections."""
import os

import brainbuilder.utils.ants_nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from brainbuilder.qc.validate_section_alignment_to_ref import dice_local
from scipy.stats import spearmanr


def inter_section_dice(
    df_csv: str,
    output_dir: str,
    output_prefix: str = "",
    n_ligands: int = None,
    tissue_str: str = "2d_align_cls",
    metric: str = "local_dice",
    plot_dice_score: bool = True,
    clobber: bool = False,
) -> pd.DataFrame:
    """Iterate over row['2d_align_cls'] to calculate the dice between the previous and next section."""
    output_csv = output_dir + f"/{output_prefix}inter_section_dice.csv"

    if not os.path.exists(output_csv) or clobber:
        df = pd.read_csv(df_csv, index_col=False)
        # Sort by chunk_id and section_number
        df = df.sort_values(by=["chunk", "sample"])
        df.reset_index(drop=True, inplace=True)

        df["sect_dice"] = None
        df["dice_prev"] = None
        df["dice_next"] = None
        df["fx_prev"] = None
        df["fx_next"] = None

        if n_ligands is None:
            n_ligands = 20  # df["acquisition"].nunique()

        # Iterate over chunks
        for _, chunk_df in df.groupby("chunk"):
            # Iterate over sections
            for i, (index, row) in enumerate(chunk_df.iterrows()):
                # Load the current section
                curr_img = nib.load(row[tissue_str])
                curr_seg = curr_img.get_fdata()
                curr_sample = row["sample"]

                # print(f'{i/len(chunk_df)*100:.2f}%', end='\r')
                # Load the previous section
                prev_row_filename = None
                next_row_filename = None
                if i > 0:
                    prev_row = chunk_df.iloc[i - 1]
                    prev_sample = prev_row["sample"]

                    if np.abs(prev_sample - curr_sample) > n_ligands:
                        print(f"Skipping {prev_sample} and {curr_sample}")
                        prev_seg = None
                    else:
                        prev_row_filename = prev_row[tissue_str]
                        prev_seg = nib.load(prev_row_filename).get_fdata()
                else:
                    prev_seg = None

                offset = 5  # int(np.rint(5 / curr_img.affine[0, 0]))

                sect_dice = 0
                n = 0
                # Load the next section
                if i < chunk_df.shape[0] - 1:
                    next_row = chunk_df.iloc[i + 1]
                    next_sample = next_row["sample"]

                    if np.abs(next_sample - curr_sample) > n_ligands:
                        print(f"Skipping {next_sample} and {curr_sample}")
                        next_seg = None
                    else:
                        next_row_filename = next_row[tissue_str]
                        next_seg = nib.load(next_row_filename).get_fdata()
                else:
                    next_seg = None

                # Calculate the dice between the current and previous section
                if prev_seg is not None:
                    if metric == "local_dice":
                        dice_prev = dice_local(prev_seg, curr_seg, offset=offset)

                    elif metric == "dice":
                        dice_prev = (
                            np.sum(prev_seg * curr_seg)
                            * 2.0
                            / (np.sum(prev_seg) + np.sum(curr_seg))
                        )
                    elif metric == "GC":
                        # calculate global correlation
                        idx = (prev_seg > 0) & (curr_seg > 0)
                        dice_prev = spearmanr(
                            prev_seg[idx].flatten(), curr_seg[idx].flatten()
                        )[0]
                    else:
                        print("Error: Metric not recognized")
                        exit(1)

                    sect_dice += dice_prev
                    n += 1

                else:
                    dice_prev = None

                # Calculate the dice between the current and next section
                if next_seg is not None:
                    if metric == "local_dice":
                        dice_next = dice_local(next_seg, curr_seg, offset=offset)
                    elif metric == "dice":
                        dice_next = (
                            np.sum(next_seg * curr_seg)
                            * 2.0
                            / (np.sum(next_seg) + np.sum(curr_seg))
                        )
                    elif metric == "GC":
                        # calculate global correlation
                        idx = (next_seg > 0) & (curr_seg > 0)
                        dice_next = spearmanr(
                            next_seg[idx].flatten(), curr_seg[idx].flatten()
                        )[0]
                    else:
                        print("Error: Metric not recognized")
                        exit(1)

                    sect_dice += dice_next
                    n += 1
                else:
                    dice_next = None

                if n > 0:
                    # Add the dice to the dataframe
                    df.loc[df.index == index, "sect_dice"] = sect_dice / n
                    df.loc[df.index == index, "dice_prev"] = dice_prev
                    df.loc[df.index == index, "dice_next"] = dice_next
                    df.loc[df.index == index, "fx_prev"] = prev_row_filename
                    df.loc[df.index == index, "fx_next"] = next_row_filename
        # Drop rows with None values
        df = df.dropna(subset=["sect_dice"])

        df.to_csv(output_csv, index=False)
    else:
        df = pd.read_csv(output_csv)

    if plot_dice_score:
        plot(df, output_dir, output_prefix=output_prefix, clobber=clobber)
        plot_sections(df, output_dir, output_prefix=output_prefix, clobber=clobber)

    return df


def find_target_image(row: pd.Series, find_lower: bool = False) -> str:
    """Find the target image with the lower or higher dice score."""
    dice_prev = row["dice_prev"]
    dice_next = row["dice_next"]

    if dice_prev is None and dice_next is None:
        print("Both dice_prev and dice_next are None")
        return None
    elif dice_prev is None or np.isnan(dice_prev):
        print("dice_prev is None")
        return row["fx_next"]
    elif dice_next is None or np.isnan(dice_next):
        print("dice_next is None")
        return row["fx_prev"]
    elif find_lower:  # want to find section with lower dice
        if dice_prev < dice_next:
            return row["fx_prev"]
        else:
            return row["fx_next"]
    elif not find_lower:  # want to find section with higher dice
        if dice_prev > dice_next:
            return row["fx_prev"]
        else:
            return row["fx_next"]
    else:
        return None


def plot_sections(
    df: pd.DataFrame,
    output_dir: str,
    cols: int = 2,
    output_prefix: str = "",
    clobber: bool = False,
) -> None:
    """Plot the sections with the highest and lowest dice score for each chunk."""
    output_png = f"{output_dir}/{output_prefix}sections.png"
    plt.rcParams.update({"font.size": 10})

    if not os.path.exists(output_png) or clobber:
        df.sort_values(by=["chunk", "sect_dice"], inplace=True)
        df.reset_index(drop=True, inplace=True)

        n = df["chunk"].nunique()

        plt.figure(figsize=(10, 10))
        # Set black background for the plot
        plt.style.use("dark_background")

        for i, (chunk, chunk_df) in enumerate(df.groupby("chunk")):
            # Load n sections with the highest and the lowest dice_sect
            bad_row = chunk_df.loc[chunk_df["sect_dice"] > 0.5].iloc[0]

            avg_row = chunk_df.iloc[chunk_df.shape[0] // 2]
            good_row = chunk_df.iloc[-1]

            bad_seg = nib.load(bad_row["2d_align_cls"]).get_fdata()
            bad_fx_filename = find_target_image(bad_row, find_lower=True)
            bad_fx = nib.load(bad_fx_filename).get_fdata()

            avg_fx_filename = find_target_image(avg_row)
            avg_seg = nib.load(avg_row["2d_align_cls"]).get_fdata()
            avg_fx = nib.load(avg_fx_filename).get_fdata()

            good_fx_filename = find_target_image(good_row)
            good_seg = nib.load(good_fx_filename).get_fdata()
            good_fx = nib.load(find_target_image(good_row)).get_fdata()

            # calculate the edges of the bad_fx and good_fx images with Sobel filter
            from scipy.ndimage import sobel

            bad_fx = np.abs(sobel(bad_fx))
            good_fx = np.abs(sobel(good_fx))
            avg_fx = np.abs(sobel(avg_fx))

            plt.subplot(n, cols, 1 + i * cols)
            plt.imshow(bad_seg, cmap="gray")
            plt.imshow(bad_fx, cmap="hot", alpha=0.6)
            plt.title(
                f'Slab: {bad_row["chunk"]} Dice Score: {np.round(bad_row["sect_dice"],2)}'
            )
            plt.axis("off")

            if cols == 3:
                plt.subplot(n, cols, 2 + i * cols)
                plt.imshow(avg_seg, cmap="gray")
                plt.imshow(avg_fx, cmap="hot", alpha=0.6)
                plt.title(
                    f'Slab: {avg_row["chunk"]} Dice Score: {np.round(avg_row["sect_dice"],2)}'
                )
                plt.axis("off")

            plt.subplot(n, cols, cols + i * cols)
            plt.imshow(good_seg, cmap="gray")
            plt.imshow(good_fx, cmap="hot", alpha=0.6)
            plt.title(
                f'Slab: {good_row["chunk"]} Dice Score: {np.round(good_row["sect_dice"],2)}'
            )
            plt.axis("off")

        plt.tight_layout()
        print(f"{output_dir}/sections.png")
        plt.savefig(output_png, dpi=200)


def plot(
    df: pd.DataFrame, output_dir: str, output_prefix: str = "", clobber: bool = False
) -> pd.DataFrame:
    """Plot the dice score for each section in each chunk."""
    out_png = output_dir + f"/{output_prefix}inter_section_dice.png"

    if not os.path.exists(out_png) or clobber:
        # Use seaborn to create a plot of sect_dice with a separate plot by chunk
        plt.figure(figsize=(20, 10))
        plt.title("Inter-section Dice Score")
        df["Slab"] = df["chunk"]
        print(df)
        g = sns.FacetGrid(df, col="chunk", col_wrap=3)
        # map the above form facetgrid with some attributes
        g.map(sns.scatterplot, "chunk", "sect_dice", alpha=0.25)
        # adding legend
        g.add_legend()
        # Set y label to Dice Score
        plt.ylabel("Dice Score")
        # Set x label to Section Number
        plt.xlabel("Tissue Slab")
        plt.savefig(out_png)
        print(out_png)

    print("Intersection Alignment -- Chunk Average")
    m = df.groupby("chunk")["sect_dice"].mean()
    s = df.groupby("chunk")["sect_dice"].std()
    c = df.groupby("chunk")["sect_dice"].count()
    chunk_average_df = pd.DataFrame(
        {"chunk": m.index, "mean": m.values, "std": s.values, "c": c.values}
    )

    print(chunk_average_df)
    print("Global Mean:", np.sum((m * c) / np.sum(c)))
    print("Global Std:", np.sum((s * c) / np.sum(c)))
    return df


if __name__ == "__main__":
    df_csv = "/data/receptor/human/brainbuilder_human_out/3_multires_align/sect_info_multiresolution_alignment.csv"
    output_dir = "/data/receptor/human/brainbuilder_human_out/qc/"
    df = pd.read_csv(df_csv)
    df = inter_section_dice(df, output_dir)
