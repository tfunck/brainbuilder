import nibabel as nib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
from joblib import Parallel, delayed, cpu_count
from brainbuilder.utils.utils import load_image

def get_min_max(fn):
    ar = load_image(fn)
    return np.min(ar), np.max(ar)

def get_min_max_parallel(df, column):

    n_jobs = int(cpu_count() )

    min_max = Parallel(n_jobs=n_jobs)(delayed(get_min_max)(row[column]) for i, (_, row) in enumerate(df.iterrows()))

    min_max = np.array(min_max)
    min_max = min_max[~np.isnan(min_max)]

    return np.min(min_max), np.max(min_max)


def section_quality_control(
        sect_info_csv : str,
        output_dir: str,
        column:str='raw',
        clobber=False
        ):
    os.makedirs(output_dir, exist_ok=True)

    sect_info = pd.read_csv(sect_info_csv, index_col=False)

    for (sub, hemisphere, acquisition, chunk), df in sect_info.groupby(['sub','hemisphere','acquisition', 'chunk']):
        print(f'Acquisition: {acquisition}')
        out_png = f'{output_dir}/sub-{sub}_hemi-{hemisphere}_chunk-{chunk}_acq-{acquisition}_{column}.png'
    
        if not os.path.exists(out_png) or clobber :
            n_images = len(df)
            n_cols = np.ceil(np.sqrt(n_images)).astype(int)
            n_rows = np.ceil(n_images / n_cols).astype(int)

            plt.figure(figsize=(n_cols*2, n_rows*2))
            
            vmin, vmax = get_min_max_parallel(df, column)
            print(f'\tMin: {vmin}, Max: {vmax}')
            df.sort_values(by=['sample'], inplace=True)
            print(df['sample'].values)

            for i, (_, row) in enumerate(df.iterrows()):
                plt.subplot(n_rows, n_cols, i+1)
                img = nib.load(row[column]).get_fdata()
                plt.imshow(img, cmap='gray', vmin=vmin, vmax=vmax)
                plt.axis('off')
                plt.title(row['sample'])
            
            plt.tight_layout()
            print('\tSaving figure to: ', out_png)
            plt.savefig(out_png)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--sect-info-csv', type=str, dest='sect_info_csv')
    argparser.add_argument('--output-dir', type=str, dest='output_dir')
    argparser.add_argument('--column', type=str, dest='column', default='raw')
    argparser.add_argument('--clobber', action='store_true', default=False, dest='clobber')
    args = argparser.parse_args()

    section_quality_control(
        sect_info_csv=args.sect_info_csv,
        output_dir=args.output_dir,
        column=args.column,
        clobber=args.clobber

        )
