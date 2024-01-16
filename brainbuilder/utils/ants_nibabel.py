import ants
import numpy as np

ras = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])
lpi = np.array([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]])


class Nifti1Image:
    def __init__(
        self, dataobj, affine, direction=[], direction_order="ras", dtype=None
    ):
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
            pass
        else:
            print(
                "Error: <direction_order> not supported, specify <direction> directly"
            )
            exit(0)

        self.direction = list(np.array(direction)[0:ndim, 0:ndim])

    def to_filename(self, filename):
        write_nifti(self.dataobj, self.affine, filename, direction=self.direction)

    def get_fdata(self):
        return self.dataobj

    def get_data(self):
        return self.dataobj


def safe_image_read(fn):
    img = ants.image_read(fn)
    try:
        img = ants.image_read(fn)
    except RuntimeError:
        print("Error: cannot load file", fn)
        exit(1)

    return img


def read_affine_antspy(fn):
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


def read_affine(fn, use_antspy=True):
    if use_antspy:
        affine = read_affine_antspy(fn)
    else:
        affine = nb.load(fn).affine
        # print('reading', affine)

    return affine


def load(fn):
    affine = read_affine(fn)
    img = ants.image_read(fn)
    vol = img.numpy()
    direction = img.direction
    # direction_order = img.direction_order
    nii_obj = Nifti1Image(vol, affine, direction=direction)

    return nii_obj


def write_nifti(vol, affine, out_fn, direction=[]):
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
