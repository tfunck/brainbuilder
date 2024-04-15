"""Quality control for section."""
import argparse

import numpy as np
import pandas as pd


def qc_section(df_csv: str, aquisition: str, target_section: float)->None:
    """Quality control for section.
    
    :param df_csv: dataframe csv file
    :param aquisition: aquisition name
    :param target_section: target section
    :return: None
    """
    df = pd.read_csv(df_csv)
    if "aquisition" in df.columns:
        df = df[df["aquisition"] == aquisition]
    else:
        df = df[df["ligand"] == aquisition]

    def closest_section(target_section:int, section_list:list)->int:
        """Find closest section.
        
        :param target_section: target section
        :param section_list: section list
        :return: index
        """
        return np.argmin(np.abs(section_list - target_section))

    if "slab_order" in df.columns:
        idx = closest_section(target_section, df["slab_order"].values)
    else:
        idx = closest_section(target_section, df["order"].values)

    row = df.iloc[idx]

    for col in row.index:
        print(col, "\t", row[col])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--df_csv", type=str, help="path to the csv file")
    parser.add_argument("--aquisition", type=str, help="aquisition name")
    parser.add_argument("--target_section", type=float, help="target section")
    args = parser.parse_args()

    qc_section(
        df_csv=args.df_csv,
        aquisition=args.aquisition,
        target_section=args.target_section,
    )
