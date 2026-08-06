# BrainBuilder

[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/tfunck/brainbuilder/blob/main/LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![pages](https://img.shields.io/badge/api-docs-blue)](documentation/html/brainbuilder/index.html)
![stability-stable](https://img.shields.io/badge/stability-stable-green.svg)

BrainBuilder reconstructs 3D volumes from sparse 2D post-mortem brain sections by combining section-wise alignment, 3D registration to a structural reference, and missing-section interpolation.

![BrainBuilder](docs/images/banner.png)

## Table of Contents

- [What It Does](#what-it-does)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Documentation Guide](#documentation-guide)
- [Optional Features](#optional-features)
- [Documentation](#documentation)
- [Reference](#reference)
- [Developer Guide Doc](docs/developer-guide.md)
- [Input Files Doc](docs/input-files.md)
- [Landmark Setup Doc](docs/landmarks.md)
- [Outputs Doc](docs/outputs.md)
- [Troubleshooting Doc](docs/troubleshooting.md)

## What It Does

BrainBuilder performs three main stages:

1. Initial rigid 2D alignment to build a first 3D volume for each chunk.
2. Multi-resolution refinement that alternates 3D chunk-to-reference alignment with 2D section-to-reference alignment.
3. Interpolation of missing sections, with volumetric interpolation as the default and surface-based interpolation available when cortical surfaces are provided.

![Schema](docs/images/schema.png)

## Installation

### Requirements

1. Install ANTs. Use the official binaries and follow the ANTs installation guide:
   https://github.com/ANTsX/ANTs/releases

2. Install Python dependencies:

```bash
python3 -m pip install -r requirements.txt
```

3. Install BrainBuilder:

```bash
python3 -m pip install -e .
```

4. Install the bundled `morphint` package:

```bash
cd morphint
python3 -m pip install -e .
```

## Quick Start

```python
from brainbuilder.reconstruct import reconstruct

reconstruct(
    "hemi_info.csv",
    "chunk_info.csv",
    "sect_info.csv",
    resolution_list=[4, 3, 2, 1],
    output_dir="/path/to/output",
)
```

For a first test run, use conservative settings that are fast and require the fewest inputs:

```python
reconstruct(
    "hemi_info.csv",
    "chunk_info.csv",
    "sect_info.csv",
    resolution_list=[4, 2],
    output_dir="/path/to/output",
    interp_method="volumetric",
    num_cores=1,
    base_lin_itr_2d=25,
    base_nl_itr_2d=10,
    base_lin_itr_3d=25,
    base_nl_itr_3d=10,
)
```

Key runtime controls include:

- `resolution_list`: multi-resolution schedule.
- `interp_method`: `volumetric` or `surface`.
- `use_syn`: enable 2D nonlinear alignment.
- `use_3d_syn_cc`: enable 3D SyN with CC.
- `base_lin_itr_2d`, `base_nl_itr_2d`, `base_lin_itr_3d`, `base_nl_itr_3d`: iteration budgets for alignment.

## Documentation Guide

Core user documentation has been split into focused pages:

- Developer architecture and section-column lifecycle: [docs/developer-guide.md](docs/developer-guide.md)
- Input files, required columns, and CSV examples: [docs/input-files.md](docs/input-files.md)
- Landmark setup and label mapping rules: [docs/landmarks.md](docs/landmarks.md)
- Output directories and key result files: [docs/outputs.md](docs/outputs.md)
- Common failures and recovery steps: [docs/troubleshooting.md](docs/troubleshooting.md)

## Optional Features

### Surface-Based Interpolation

Set `interp_method="surface"` to use cortical-surface interpolation. This requires `gm_surf` and `wm_surf` in `hemi_info.csv`.

### Landmark-Guided Alignment

Landmark-assisted chunk localization is optional. For detailed setup and constraints, see [docs/landmarks.md](docs/landmarks.md).

## Documentation

- API docs: [documentation/html/brainbuilder/index.html](documentation/html/brainbuilder/index.html)
- morphint package: [morphint/README.md](morphint/README.md)

## Reference

Funck, T., Wagstyl, K., Lepage, C. et al. Brainbuilder: a software pipeline for 3D reconstruction of cortical maps from multi-modal 2D data sets. Commun Biol 8, 1015 (2025). https://doi.org/10.1038/s42003-025-08267-6
