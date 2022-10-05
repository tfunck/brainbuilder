
from rat.process import reconstruct
from rat.create_data_frame import create_data_frame
from sys import argv
import pandas as pd

if __name__ == '__main__' :
    subject_id = argv[1]
    auto_dir=argv[2]
    template_fn = argv[3]
    scale_factors_json = argv[4]
    out_dir = argv[5]
    default_csv_fn=argv[6]
    mask_dir = f'{auto_dir}/mask_dir/'
    subject_dir = f'{out_dir}/{subject_id}/' 
    crop_dir = f'{subject_dir}/crop/'
    init_dir = f'{subject_dir}/{subject_id}/init_align/'

    csv_fn = f'{subject_dir}/{subject_id}.csv'

    create_data_frame(default_csv_fn, auto_dir, crop_dir, csv_fn)
    brain='L6'
    hemi = 'B' 
    reconstruct(subject_id, auto_dir, template_fn, scale_factors_json, out_dir, csv_fn, brain=brain, hemi=hemi, resolution_list=[0.5,0.4,0.3,0.2,0.1], lowres=0.1)
