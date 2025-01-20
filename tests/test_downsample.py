"""Tests for the downsample module."""
import pytest
from brainbuilder.downsample import downsample_sections
from create_synthetic_data import generate_synthetic_data
import os


def test_downsample_sections(generate_synthetic_data, tmp_path)->bool:
    """Test downsample_sections."""
    data =  generate_synthetic_data

    output_dir = tmp_path 
    output_dir.mkdir(exist_ok=True)

    downsample_sections(
        chunk_info_csv = data[1], 
        sect_info_csv = data[0], 
        resolution = 0.1,
        output_dir = str(output_dir), 
        clobber = True
    )
    assert True

