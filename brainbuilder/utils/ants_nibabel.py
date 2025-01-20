"""Module for reading and writing nifti files using ANTsPy."""
import os

import ants
import nibabel as nb
import numpy as np

ras = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])
lpi = np.array([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]])


class Nifti1Image:
    """Class for Nifti1Image object with ANTSPy interface."""
    def __init__(
        self, dataobj:np.ndarray,
        affine:np.ndarray,
        direction:list=[], 
        direction_order:list="ras", 
        dtype:int=None
    )->None :
        """Constructor for Nifti1Image class.
        
        :param dataobj: Data object
        :param affine: Affine matrix
        :param direction: Direction matrix, defaults to []
        :param direction_order: Direction order, defaults to "ras"
        :param dtype: Data type, defaults to None
        :return : None
        """
        if type(dtype) != type(None):
            if dtype == np.uint8:
                dataobj = (
                    255 * (dataobj - dataobj.min()) / (dataobj.max() - dataobj.min())
                ).astype(np.uint8)
        self.affine = affine
        self.dataobj = dataobj
        self.shape = dataobj.shape
        ndim = len(self.shape)
        if direction_order == "ras" and len(direction) == 0:
            direction = ras
        elif direction_order == "lpi" and len(direction) == 0:
            direction = lpi
        elif len(direction) != 0:
            direction=direction
        else:
            print(
                "Error: <direction_order> not supported, specify <direction> directly"
            )
            exit(0)
        self.direction = list(np.array(direction)[0:ndim, 0:ndim])

    def to_filename(self, filename:str)->None:
        """Write the image to a nifti file.

        :param filename: Filename of the nifti file
        :return: None
        """
        write_nifti(self.dataobj, self.affine, filename, direction=self.direction)

    def get_fdata(self)->np.ndarray:
        """Get the data object."""
        return self.dataobj

    def get_data(self)->np.ndarray:
        """Get the data object."""
        return self.dataobj


def safe_image_read(fn:str)->ants.core.ants_image.ANTsImage:
    """Read a nifti file using ANTsPy.
    
    :param fn: Filename of the nifti file
    :return: ANTsPy image object
    """
    if os.path.islink(fn) :
        fn = os.readlink(fn)

    try:
        img = ants.image_read(fn)
    except RuntimeError:
        print("Error: cannot load file", fn)
        exit(1)

    return img


def read_affine_antspy(fn:str)->np.ndarray:
    """Read affine matrix from a nifti file using ANTsPy.
    
    :param fn: Filename of the nifti file
    :return: Affine matrix
    """
    img = safe_image_read(fn)
    spacing = img.spacing
    origin = img.origin
    # print('spacing', spacing)
    # print('origin',origin)

    affine = np.eye(4)

    for i, (s, o) in enumerate(zip(spacing, origin)):
        affine[i, i] = s
        affine[i, 3] = o
    orientation = img.orientation
    if len(img.shape) == 3 and img.shape[-1] != 1:
        if orientation != "RAS":
            print(f"Warning: file has {orientation}, not RAS. {fn}")

    return affine


def read_affine(fn:str, use_antspy:bool=True)->np.ndarray:
    """Read affine matrix from a nifti file.
    
    :param fn: Filename of the nifti file
    :param use_antspy: Whether to use ANTsPy to read the affine matrix, defaults to True
    :return: Affine matrix
    """
    if use_antspy:
        affine = read_affine_antspy(fn)
    else:
        affine = nb.load(fn).affine
        # print('reading', affine)

    return affine


def load(fn:str)->Nifti1Image:
    """Load a nifti file.

    :param fn: Filename of the nifti file
    :return: Nifti1Image object
    """
    affine = read_affine(fn)
    img = ants.image_read(fn)
    vol = img.numpy()
    direction = img.direction
    # direction_order = img.direction_order
    nii_obj = Nifti1Image(vol, affine, direction=direction)

    return nii_obj


def write_nifti(vol:np.ndarray, affine:np.ndarray, out_fn:str, direction:list=[])->None:
    """Write a nifti file.

    :param vol: Volume to write
    :param affine: Affine matrix
    :param out_fn: Filename of the nifti file
    :param direction: Direction matrix, defaults to []
    :return: None
    """
    ndim = len(vol.shape)
    idx0 = list(range(0, ndim))
    idx1 = [3] * ndim

    origin = list(affine[idx0, idx1])
    spacing = list(affine[idx0, idx0])

    if direction == []:
        if len(spacing) == 2:
            direction = [[1.0, 0.0], [0.0, 1.0]]
        else:
            # Force to write in RAS coordinates
            direction = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]]
    ants_image = ants.from_numpy(
        vol, origin=origin, spacing=spacing, direction=direction
    )
    assert True not in np.isnan(affine.ravel()), "Bad affine matrix. NaN detected"
    ants.image_write(ants_image, out_fn)
