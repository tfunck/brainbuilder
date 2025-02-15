"""Functions for loading and writing mesh files."""
import os

import h5py as h5
import nibabel as nb
import numpy as np
import pandas as pd

global freesurfer_ext
freesurfer_ext = [
    ".orig",
    ".pial",
    ".white",
    ".sphere",
    ".inflated",
    ".curv",
    ".sulc",
    ".thickness",
    ".annot",
    ".label",
]

def load_values(in_fn:str, data_str:str="data")->np.ndarray:
    """Load values from a file.
    
    :param in_fn: Filename of the file
    :param data_str: Data string to load, defaults to "data"
    :return: Values from the file
    """
    if os.path.exists(in_fn + ".npz"):
        values = np.load(in_fn + ".npz")[data_str]
        return values

    ext = os.path.splitext(in_fn)[1]
    if ext == ".npz":
        values = np.load(in_fn)[data_str]
    else:
        values = pd.read_csv(in_fn, header=None, index_col=None).values
    return values


def load_mesh_ext(in_fn:str, faces_fn:str="", correct_offset:bool=False)->np.ndarray:
    """Load a mesh file with the correct function based on the file extension.
    
    :param in_fn: Filename of the mesh
    :param faces_fn: Filename of the faces file, defaults to 
    :param correct_offset: Whether to correct the offset of the mesh, defaults to False
    :return: Coordinates and faces of the mesh
    """
    ext = os.path.splitext(in_fn)[1]
    faces = None
    volume_info = None
    if ext in [".pial", ".white", ".gii", ".sphere", ".inflated"]:
        coords, faces, volume_info = load_mesh(in_fn, correct_offset=correct_offset)
    elif ext == ".npz":
        coords = np.load(in_fn)["points"]
    else:
        coords = h5.File(in_fn)["data"][:]
        if os.path.splitext(faces_fn)[1] == ".h5":
            faces_h5 = h5.File(faces_fn, "r")
            faces = faces_h5["data"][:]
    return coords, faces

def write_gifti(surf_mesh:str, coords:np.ndarray, faces:np.ndarray)->None:
    """Write a mesh to a GIFTI file.
    
    :param surf_mesh: Filename of the mesh
    :param coords: Coordinates of the mesh
    :param faces: Faces of the mesh
    :return: None
    """
    coord_array = nb.gifti.GiftiDataArray(
        data=coords.astype(np.float32), intent=nb.nifti1.intent_codes["NIFTI_INTENT_POINTSET"]
    )
    face_array = nb.gifti.GiftiDataArray(
        data=faces, intent=nb.nifti1.intent_codes["NIFTI_INTENT_TRIANGLE"]
    )
    gii = nb.gifti.GiftiImage(darrays=[coord_array, face_array])

    nb.save(gii, surf_mesh)
    return None

def load_mesh(surf_mesh:str, correct_offset:bool=False)->np.ndarray:
    """Load a mesh file.
    
    :param surf_mesh: Filename of the mesh
    :param correct_offset: Whether to correct the offset of the mesh, defaults to False
    :return: Coordinates and faces of the mesh
    """
    volume_info = None
    if isinstance(surf_mesh, str):
        if True in [surf_mesh.endswith(x) for x in freesurfer_ext]:
            coords, faces, volume_info = nb.freesurfer.io.read_geometry(
                surf_mesh, read_metadata=True
            )

            if correct_offset and isinstance(volume_info, type(None)):
                try:
                    origin = volume_info["cras"]

                    xras = np.array(volume_info["xras"])
                    yras = np.array(volume_info["yras"])
                    zras = np.array(volume_info["zras"])

                    def get_sign(ar:np.ndarray)->np.ndarray:
                        """Get the sign of the array.
                        
                        :param ar: Array to get the sign of
                        :return: Sign of the array
                        """
                        return np.sign(ar[np.argmax(np.abs(ar))])

                    xdir = get_sign(xras)
                    ydir = get_sign(yras)
                    zdir = get_sign(zras)
                    xdir = ydir = zdir = 1
                    print("Adding origin to surface", origin)
                    print("\tDirections:", xdir, ydir, zdir)
                    coords[:, 0] = coords[:, 0] + xdir * origin[0]
                    coords[:, 1] = coords[:, 1] + ydir * origin[1]
                    coords[:, 2] = coords[:, 2] + zdir * origin[2]
                except KeyError:
                    pass

        elif surf_mesh.endswith("gii"):
            coords_class, faces_class = (
                nb.load(surf_mesh).get_arrays_from_intent(
                    nb.nifti1.intent_codes["NIFTI_INTENT_POINTSET"]
                )[0],
                nb.load(surf_mesh).get_arrays_from_intent(
                    nb.nifti1.intent_codes["NIFTI_INTENT_TRIANGLE"]
                )[0],
            )
            coords = coords_class.data
            faces = faces_class.data
            print(correct_offset, volume_info)
            if correct_offset:
                try:
                    string_list = ["VolGeomC_R", "VolGeomC_A", "VolGeomC_S"]
                    origin = [
                        float(coords_class.meta[string]) for string in string_list
                    ]

                    def get_sign(ar:np.ndarray)->np.ndarray:
                        """Get the sign of the array.
                        
                        :param ar: Array to get the sign of
                        :return: Sign of the array
                        """
                        return np.sign(ar[np.argmax(np.abs(ar))])

                    # xdir = get_sign(xras), ydir = get_sign(yras), zdir = get_sign(zras)
                    # xdir = ydir = -1 ; zdir = 1
                    xdir = ydir = zdir = 1
                    print("Adding origin to surface", origin)
                    print("\tDirections:", xdir, ydir, zdir)
                    coords[:, 0] = coords[:, 0] + xdir * origin[0]
                    coords[:, 1] = coords[:, 1] + ydir * origin[1]
                    coords[:, 2] = coords[:, 2] + zdir * origin[2]

                except KeyError:
                    pass
        elif isinstance(surf_mesh, dict):
            if "faces" in surf_mesh and "coords" in surf_mesh:
                coords, faces = surf_mesh["coords"], surf_mesh["faces"]
            else:
                raise ValueError(
                    "If surf_mesh is given as a dictionary it must "
                    'contain items with keys "coords" and "faces"'
                )
        else:
            raise ValueError(
                "surf_mesh must be a either filename or a dictionary "
                'containing items with keys "coords" and "faces"'
            )

    return coords, faces, volume_info


# function to load mesh geometry
def load_mesh_geometry(surf_mesh:np.ndarray)->dict:
    """Returns coords, numbers of neighbours per vertex, and indices of neighbours.

    :param surf_mesh: Filename of the mesh
    :return: Coordinates, number of neighbours per vertex, and indices of neighbours
    """
    coords, faces, _ = load_mesh(surf_mesh)
    neighbours, counts = get_neighbours(faces)
    return {"coords": coords, "neighbour_count": counts, "neighbours": neighbours}


def get_neighbours(triangles:np.ndarray)->list:
    """Get neighbours from triangles.
    
    :param triangles: Triangles of the mesh
    :return: Neighbours of the mesh
    """
    n_vert = np.max(triangles) + 1
    neighbours = [[] for i in range(n_vert)]
    counts = []
    for tri in triangles:
        neighbours[tri[0]].extend([tri[1], tri[2]])
        neighbours[tri[2]].extend([tri[0], tri[1]])
        neighbours[tri[1]].extend([tri[2], tri[0]])
    # Get unique neighbours
    for k in range(len(neighbours)):
        neighbours[k] = f7(neighbours[k])
        counts.append(len(neighbours[k]))
    return neighbours, counts


def f7(seq:list)->list:
    """Returns uniques but in order to retain neighbour triangle relationship.

    :param seq: Sequence to get uniques from
    :return: Uniques of the sequence
    """
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]





