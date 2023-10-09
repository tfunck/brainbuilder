# BrainBuilder

### Purpose
BrainBuilder is a software packagefor reconstructing 3-dimensional cortical maps from data sets of 2-dimensional post-mortem serial brain sections processed for the quantification of multiple different biological components.

Manuscript with full description: https://www.biorxiv.org/content/10.1101/2022.11.18.517039v1

## Useage

```python3
from brainbuilder.reconstruct import reconstruct

reconstruct('hemi_info.csv', 'chunk_info.csv', 'sect_info.csv', resolution_list=[4,3,2,1], '/output/dir/')
```

### Inputs 

### Hemisphere Info

#### Mandatory fields: 
**sub** : subject id

**hemisphere** : hemisphere

**struct_ref_vol** : path to structural reference volume

**gm_surf** : path to gray matter surface file (.surf.gii or .pial)

**wm_surf** : path to white matter surface file (.surf.gii or .pial)



### Chunk Info
#### Mandatory fields
**sub** : subject id

**hemisphere** : hemisphere

**chunk** : id for the tissue chunk (aka slab) 

**pixel_size_0** : pixel size in the x dimension (mm)

**pixel_size_1** : pixel size in the z dimension (mm)

**section_thickness** : thickness of the section (mm)

**direction** : the direction of sectioning for the slab

#### Optional Info
**caudal_limit** : float, the spatial coordinate of the caudal limit of the tissue on the reference template
**rostral_limit** : float, the spatial coordinate of the rostral limit of the tissue on the reference template

### Section Info

#### Mandatory Fields

**sub** : subject id

**hemisphere** : hemisphere

**chunk** : id for the tissue chunk (aka slab) 

**sample** : integer sample number that indicates the order of the section within the data set

**acquisition** : the kind of acquisition (e.g., autoradiography ligand or histological staining)

**raw** : path to the raw section (.nii.gz)

#### Optional Fields

**conversion_factor** : conversion factor to apply to convert pixel intensitites into biological parameter 

