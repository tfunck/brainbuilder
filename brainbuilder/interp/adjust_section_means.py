import nibabel as nib
import numpy as np
import glob
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
import re
import os
from scipy.ndimage.filters import gaussian_filter1d


def get_section_means(file_list:list, correction_factors=None):
    """Find and ssort section files"""    
    otsu_mean_list = []
    ar_mean_otsu = []

    if correction_factors is None:
        correction_factors = [1] * len(file_list)
    

    for filename, correction_factor in zip(file_list, correction_factors) :
        img = nib.load(filename)
        ar = img.get_fdata() * correction_factor
        t = threshold_otsu(ar)
        
        ar_mean_otsu = np.median(ar[ar>t]  )
        
        otsu_mean_list.append(ar_mean_otsu)

    return otsu_mean_list

def get_sections(input_dir:str):
    file_list = glob.glob(f'{input_dir}/y-*_rsl.nii.gz')
    section_numbers = [re.sub('_rsl.nii.gz','', os.path.basename(x)).split('-')[1] for x in file_list]
    section_numbers = [int(x) for x in section_numbers]

    file_list = [x for _,x in sorted(zip(section_numbers, file_list))]
    section_numbers = sorted(section_numbers)
    return file_list, section_numbers 

def get_interpolator_function( section_numbers, mean_list):
    """Get interpolator function"""
    from scipy.interpolate import interp1d
    interpolator = interp1d( section_numbers, mean_list,kind='linear')
    return interpolator

def smooth_section_means(section_numbers, interp_f):
    """Smooth section means"""
    x=np.linspace(min(section_numbers), max(section_numbers), len(section_numbers) * 2  )
    y=interp_f(x)

    fwhm = np.mean(np.abs(np.diff(y)))

    sigma = fwhm / 2.355
    print(f'FWHM: {fwhm} sigma: {sigma}')
    y_smooth = gaussian_filter1d(y, sigma=sigma)    
    return x, y_smooth

def fix_sections_means(file_list, mean_list, section_numbers, interp_f):
    output_dict={}

    for filename, mean, section_number in zip(file_list, mean_list, section_numbers):
        new_mean = interp_f(section_number)
        output_dict[section_number] = ( new_mean - mean )

    return output_dict


def calculate_section_adjustment_factors(file_list:list, section_numbers:list, output_dir:str, correction_factors=None)->dict:
    """Adjusts the section means to be more consistent by smoothing the means. 

    :param file_list (list): A list of file paths.
    :param section_numbers (list): A list of section numbers.
    :param output_dir (str): The output directory path.
    :return: A dictionary of adjustment factors.
    """
    os.makedirs(output_dir, exist_ok=True)

    #sort the file list and section numbers
    file_list = [x for _,x in sorted(zip(section_numbers, file_list))]
    section_numbers = sorted(section_numbers)

    mean_list = get_section_means(file_list, correction_factors=correction_factors)

    interp_f = get_interpolator_function(section_numbers, mean_list)   
   
    x, y_smooth  = smooth_section_means(section_numbers, interp_f)

    smooth_interp_f = get_interpolator_function(x, y_smooth)

    adj_factor_dict = fix_sections_means(file_list, mean_list, section_numbers, smooth_interp_f)

    plt.figure(figsize=(12,12))
    plt.scatter(section_numbers, mean_list)
    plt.plot(section_numbers, mean_list, c='r', label='raw mean')
    plt.plot(x, y_smooth, c='g', label='adjusted mean')
    plt.savefig(f'{output_dir}/section_means.png')
    plt.close()

    return adj_factor_dict 


