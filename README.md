# julich-receptor-atlas

### Purpose

3D reconstruction of 2D histological sections.

Manuscript with full description: https://www.biorxiv.org/content/10.1101/2022.11.18.517039v1

## Useage

### Inputs 

subject_id: subject name

auto_dir: autoradiography directory

template_fn: reference template volume for reconstruction

scale factors .json : json file that specifies slicing direction and section pixel size in each tissue slab

out_dir : output directory

section info .csv : .csv file that contains the fields specified below



singularity exec   ~/projects/julich-receptor-atlas/receptor.simg bash -c "python3.8 ~/projects/julich-receptor-atlas/launch_macaque.py rh_11530_mf macaque/rh_11530_mf/ templates/MEBRAINS_segmentation_NEW_gm_bg_left.nii.gz macaque/rh_11530_mf_points_lpi.txt macaque/rh_11530_mf/scale_factors.json ${out_dir}" 

### Scale Factors .json

### Section Info .csv

#### Required Fields:

raw : path to raw section file

brain : id of brain 

hemisphere : L or R hemisphere

slab : tissue slab from which section was acquired

ligand: ligand or method used to produce section

order : section order number within slab

rotate : rotation to apply to raw image

#### Optional :

conversion_factor : conversion factor to apply to convert from radioactivity concentration to receptor binding, Default = 1.

repeat :  

repeat count :

binding : type of binding, specific (S) or unspecific (UB)


#### Generated during reconstruction
slab_order :

global_order :

volume_order : 

seg_fn : 

crop : path to cropped section file

aligned : path to aligned section file
