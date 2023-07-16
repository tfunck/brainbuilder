from macaque.process import reconstruct
from macaque.process import create_section_dataframe
from sys import argv
import os

if __name__ == '__main__' :
    #mandatory inputs
    subject_id = argv[1]
    auto_dir = argv[2]
    template_fn = argv[3]
    scale_factors_json = argv[5]
    out_dir = argv[6]
    csv_fn = f'{out_dir}/{subject_id}/{subject_id}.csv'

    mask_dir = f'{auto_dir}/mask_dir/'
    subject_dir = f'{out_dir}/{subject_id}/' 
    crop_dir = f'{out_dir}/{subject_id}/crop/'
    init_dir = f'{out_dir}/{subject_id}/init_align/'

    for path in [subject_dir, crop_dir, init_dir] : os.makedirs(path,exist_ok=True)

    df = create_section_dataframe(auto_dir, crop_dir, csv_fn, template_fn)
    
    flip_dict = {'rostral_to_caudal':(0,), 'caudal_to_rostral':(0,1) }
    reconstruct(str(subject_id), auto_dir, template_fn, scale_factors_json, out_dir, csv_fn, pytorch_model='Task501', flip_dict=flip_dict, hemi='L', resolution_list=[5, 4, 3, 2, 1, 0.5], n_depths=10)
