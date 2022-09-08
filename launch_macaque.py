from macaque.process import reconstruct
from macaque.process import create_section_data_frame
from sys import argv

if __name__ == '__main__' :
    subject_id = argv[1]
    template_fn = argv[3]
    out_dir = argv[6]
    mask_dir = f'{auto_dir}/mask_dir/'
    subject_dir = f'{out_dir}/{subject_id}/' 
    crop_dir = f'{out_dir}/{subject_id}/crop/'
    init_dir = f'{out_dir}/{subject_id}/init_align/'

    csv_fn = f'{out_dir}/{subject_id}/{subject_id}.csv'

    create_section_dataframe(auto_dir, crop_dir, csv_fn, template_fn)
    
    flip_dict = {'rostral_to_caudal':(0,), 'caudal_to_rostral':(0,1) }
    reconstruct(subject_id, auto_dir, template_fn, scale_factors_json, out_dir, csv_fn, pytorch_model='Task501', flip_dict=flip_dict)
