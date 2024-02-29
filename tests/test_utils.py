"""Test functions for utils"""
import multiprocessing

import ants
import imageio
import numpy as np
import pandas as pd
import pytest
from brainbuilder.utils.utils import (
    AntsParams,
    create_2d_affine,
    estimate_memory_usage,
    get_available_memory,
    get_chunk_direction,
    get_chunk_pixel_size,
    get_maximum_cores,
    get_thicken_width,
    load_image,
    set_cores,
)


@pytest.fixture
def ants_params():
    resolution_list = [0.5, 1.0, 2.0, 4.0]
    resolution = 2.0
    base_itr = 10
    max_resolution = 4.0
    verbose = False

    return AntsParams(resolution_list, resolution, base_itr, max_resolution,  verbose)

def test_ant_params_initialization(ants_params):
    print(ants_params.resolution_list); exit(0)
    assert ants_params.resolution_list == [4.0, 2.0, 1.0,  0.5]
    assert ants_params.max_n == 4
    assert ants_params.cur_n == 2
    assert ants_params.max_itr == 50
    assert ants_params.f_list == [1, 2, 4]
    assert ants_params.max_downsample_factor == 1
    assert ants_params.f_str == "1x2x4vox"
    assert ants_params.s_list == [0.0, 0.0, 0.0]
    assert ants_params.s_str == "0.0x0.0x0.0vox"
    assert ants_params.itr_str == "[4x3x2x1,1e-7,20 ]"

def test_ant_params_print(ants_params, capsys):
    ants_params.print()
    captured = capsys.readouterr()
    assert captured.out == "Iterations:\t 4x3x2x1\nFactors:\t 1x2x4vox\nSmoothing:\t 0.0x0.0x0.0vox\n"

def test_ant_params_gen_itr_str(ants_params):
    assert ants_params.gen_itr_str(50, 10) == "[50,1e-7,20 ]"

def test_ant_params_gen_smoothing_factor_string(ants_params):
    assert ants_params.gen_smoothing_factor_string([0.0, 0.0, 0.0]) == "0.0x0.0x0.0vox"

def test_ant_params_gen_smoothing_factor_list(ants_params):
    assert ants_params.gen_smoothing_factor_list([0.5, 1.0, 2.0, 4.0]) == [0.0, 0.0, 0.0]

def test_ant_params_get_smoothing_params(ants_params):
    assert ants_params.get_smoothing_params([0.5, 1.0, 2.0, 4.0]) == "0.0x0.0x0.0vox"

def test_ant_params_calc_downsample_factor(ants_params):
    assert ants_params.calc_downsample_factor(2.0, 2.0) == "1"

def test_ant_params_gen_downsample_factor_list(ants_params):
    assert ants_params.gen_downsample_factor_list([0.5, 1.0, 2.0, 4.0]) == ["1", "2", "4"]

def test_ant_params_get_downsample_params(ants_params):
    assert ants_params.get_downsample_params([0.5, 1.0, 2.0, 4.0]) == "1x2x4"

def test_get_available_memory():
    # Test the get_available_memory function
    assert isinstance(get_available_memory(), int)



def test_get_chunk_pixel_size():
    # Create a sample chunk info dataframe
    chunk_info = pd.DataFrame({
        "sub": ["subject1", "subject2"],
        "hemisphere": ["left", "right"],
        "chunk": [1, 2],
        "pixel_size_0": [0.1, 0.2],
        "pixel_size_1": [0.3, 0.4],
        "section_thickness": [0.5, 0.6]
    })

    # Test case 1: Valid inputs
    sub = "subject1"
    hemi = "left"
    chunk = 1
    expected_result = (0.1, 0.3, 0.5)
    assert get_chunk_pixel_size(sub, hemi, chunk, chunk_info) == expected_result

    # Test case 2: Invalid inputs
    sub = "subject3"  # Non-existent subject
    hemi = "right"
    chunk = 3  # Non-existent chunk
    expected_result = (None, None, None)  # Expected result when inputs are invalid
    assert get_chunk_pixel_size(sub, hemi, chunk, chunk_info) == expected_result


def test_load_image():
    # Test loading an existing image file
    image_path = "path/to/existing/image.jpg"
    image = load_image(image_path)
    assert isinstance(image, np.ndarray)

    # Test loading a non-existing image file
    non_existing_image_path = "path/to/non_existing/image.jpg"
    image = load_image(non_existing_image_path)
    assert image is None

    # Test loading a NIfTI image file
    nifti_image_path = "path/to/nifti/image.nii"
    image = load_image(nifti_image_path)
    assert isinstance(image, np.ndarray)

    # Test loading an invalid file type
    invalid_file_path = "path/to/invalid/file.txt"
    image = load_image(invalid_file_path)
    assert image is None


def test_get_maximum_cores():
    # Test case 1: Example values
    n_elemnts_list = [100, 200, 300]
    n_bytes_per_element_list = [4, 8, 12]
    max_memory = 0.8
    expected_result = 2  # Assuming available_memory = 1000, estimated_memory = 600
    assert get_maximum_cores(n_elemnts_list, n_bytes_per_element_list, max_memory) == expected_result

    # Test case 2: Empty lists
    n_elemnts_list = []
    n_bytes_per_element_list = []
    max_memory = 0.8
    expected_result = 0
    assert get_maximum_cores(n_elemnts_list, n_bytes_per_element_list, max_memory) == expected_result

    # Test case 3: Zero max_memory
    n_elemnts_list = [100, 200, 300]
    n_bytes_per_element_list = [4, 8, 12]
    max_memory = 0
    expected_result = 0
    assert get_maximum_cores(n_elemnts_list, n_bytes_per_element_list, max_memory) == expected_result

    # Add more test cases as needed

def test_estimate_memory_usage():
    # Test case 1: n_elements = 0, n_bytes_per_element = 4
    assert estimate_memory_usage(0, 4) == 0

    # Test case 2: n_elements = 10, n_bytes_per_element = 8
    assert estimate_memory_usage(10, 8) == 80

    # Test case 3: n_elements = 100, n_bytes_per_element = 2
    assert estimate_memory_usage(100, 2) == 200

    # Add more test cases as needed

def test_create_2d_affine():
    # Test case 1: Example values
    pixel_size_0 = 0.1
    pixel_size_1 = 0.2
    section_thickness = 0.3
    expected_result = np.array([[0.1, 0.0, 0.0, 0.0],
                               [0.0, 0.2, 0.0, 0.0],
                               [0.0, 0.0, 0.3, 0.0],
                               [0.0, 0.0, 0.0, 1.0]])
    assert np.array_equal(create_2d_affine(pixel_size_0, pixel_size_1, section_thickness), expected_result)

    # Test case 2: Zero values
    pixel_size_0 = 0.0
    pixel_size_1 = 0.0
    section_thickness = 0.0
    expected_result = np.eye(4)
    assert np.array_equal(create_2d_affine(pixel_size_0, pixel_size_1, section_thickness), expected_result)

    # Add more test cases as neededimport pandas as pd


def test_get_chunk_direction():
    # Create a sample chunk info dataframe
    chunk_info = pd.DataFrame({
        "sub": ["subject1", "subject2"],
        "hemisphere": ["left", "right"],
        "chunk": [1, 2],
        "direction": ["up", "down"]
    })

    # Test case 1: Valid inputs
    sub = "subject1"
    hemi = "left"
    chunk = 1
    expected_result = "up"
    assert get_chunk_direction(sub, hemi, chunk, chunk_info) == expected_result

    # Test case 2: Invalid inputs
    sub = "subject3"  # Non-existent subject
    hemi = "right"
    chunk = 3  # Non-existent chunk
    expected_result = None  # Expected result when inputs are invalid
    assert get_chunk_direction(sub, hemi, chunk, chunk_info) == expected_result


def test_set_cores():
    # Test case 1: num_cores = 0
    expected_result = multiprocessing.cpu_count()
    assert set_cores(0) == expected_result

    # Test case 2: num_cores > 0
    num_cores = 4
    assert set_cores(num_cores) == num_cores

    # Add more test cases as neededimport numpy as np


def test_get_thicken_width():
    # Test case 1: Default section thickness
    resolution = 1.0
    expected_result = 26
    assert get_thicken_width(resolution) == expected_result

    # Test case 2: Custom section thickness
    resolution = 2.0
    section_thickness = 0.05
    expected_result = 41
    assert get_thicken_width(resolution, section_thickness) == expected_result

    # Add more test cases as needed

def test_get_section_intervals():
    pass

def test_check_transformation_not_empty():
    pass

def test_simple_ants_apply_tfm():
    pass

def test_get_seg_fn():
    pass

def test_gen_2d_fn():
    pass

def test_save_sections():
    pass

def test_get_to_do_list():
    pass
def test_create_2d_sections():
    pass
def test_splitext():
    pass
def test_unbuffered():
    pass
def test_shell():
    pass
def test_gen_new_filename():
    pass
def test_get_values_from_df():
    pass
def test_get_params_from_affine():
    pass
def test_parse_resample_arguments():
    pass
def test_newer_than():
    pass
def test_compare_timestamp_of_files():
    pass
def test_check_run_stage():
    pass
def test_resample_to_resolution():
    pass