import os

import ants
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import nibabel as nb
import brainbuilder.utils.ants_nibabel as nib

matplotlib.use("Agg")
import multiprocessing

import h5py as h5
import numpy as np

from scipy.ndimage import gaussian_filter

from brainbuilder.utils.mesh_io import load_mesh, save_mesh
from brainbuilder.utils.utils import shell



def get_edges_from_faces(faces):
    # for convenience create vector for each set of faces
    f_i = faces[:, 0]
    f_j = faces[:, 1]
    f_k = faces[:, 2]

    # combine node pairs together to form edges
    f_ij = np.column_stack([f_i, f_j])
    f_jk = np.column_stack([f_j, f_k])
    f_ki = np.column_stack([f_k, f_i])

    # concatenate the edges into one big array
    edges_all = np.concatenate([f_ij, f_jk, f_ki], axis=0).astype(np.uint32)

    # there are a lot of redundant edges that we can remove
    # first sort the edges within rows because the ordering of the nodes doesn't matter
    edges_all_sorted_0 = np.sort(edges_all, axis=1)
    # create a vector to keep track of vertex number
    edges_all_range = np.arange(edges_all.shape[0]).astype(int)
    #
    edges_all_sorted = np.column_stack([edges_all_sorted_0, edges_all_range])

    # sort the rows so that duplicate edges are adjacent to one another
    edges_range_sorted = pd.DataFrame(edges_all_sorted).sort_values([0, 1]).values
    edges_sorted = edges_range_sorted[:, 0:2]

    # convert sorted indices to indices that correspond to face numbers
    # DEBUG commented out following line because it isnt' used:
    # sorted_indices = edges_range_sorted[:,2] % faces.shape[0]

    # the edges are reshuffled once by sorting them by row and then by extracting unique edges
    # we need to keep track of these transformations so that we can relate the shuffled edges to the
    # faces they belong to.
    edges, edges_idx, counts = np.unique(
        edges_sorted, axis=0, return_index=True, return_counts=True
    )
    edges = edges.astype(np.uint32)

    assert np.sum(counts != 2) == 0, "Error: more than two faces per edge {}".format(
        edges_sorted[edges_idx[counts != 2]]
    )
    # edge_range = np.arange(edges_all.shape[0]).astype(int) % faces.shape[0]
    return edges


def get_surf_from_dict(d):
    keys = d.keys()
    if "upsample_h5" in keys:
        surf_fn = d["upsample_h5"]
    elif "depth_rsl_fn" in keys:
        surf_fn = d["depth_rsl_fn"]
    elif "surf" in keys:
        surf_fn = d["surf"]
    else:
        assert False, f"Error: could not find surface in keys, {keys}"
    return surf_fn


def volume_to_mesh(coords, vol, starts, steps, dimensions):
    '''
    Interpolate volume values to mesh vertices
    
    '''

    x = np.rint((coords[:, 0] - starts[0]) / steps[0]).astype(int)
    y = np.rint((coords[:, 1] - starts[1]) / steps[1]).astype(int)
    z = np.rint((coords[:, 2] - starts[2]) / steps[2]).astype(int)

    np.min(x)
    np.min(z)

    xmax = np.max(x)
    np.max(y)
    zmax = np.max(z)

    if zmax >= vol.shape[2]:
        print(
            f"\n\nWARNING: z index {zmax} is greater than dimension {vol.shape[2]}\n\n"
        )
    if xmax >= vol.shape[0]:
        print(
            f"\n\nWARNING: x index {xmax} is greater than dimension {vol.shape[0]}\n\n"
        )

    idx = (
        (x >= 0)
        & (y >= 0)
        & (z >= 0)
        & (x < dimensions[0])
        & (y < dimensions[1])
        & (z < dimensions[2])
    )

    # get nearest neighbour voxel intensities at x and z coordinate locations
    values = vol[x[idx], y[idx], z[idx]]

    return values, idx


def mesh_to_volume(
    coords,
    vertex_values,
    dimensions,
    starts,
    steps,
    origin=[0, 0, 0],
    interp_vol=None,
    n_vol=None,
    validate=True,
):
    """
    About
        Interpolate mesh values into a volume
    Arguments
        coords
        vertex_values
        dimensions
        starts
        steps
        interp_vol
        n_vol
    Return
        interp_vol
        n_vol
    """
    if type(vertex_values) != np.ndarray or type(n_vol) != np.ndarray:
        interp_vol = np.zeros(dimensions)
        n_vol = np.zeros_like(interp_vol)

    x = np.rint((coords[:, 0] - starts[0]) / steps[0]).astype(int)
    y = np.rint((coords[:, 1] - starts[1]) / steps[1]).astype(int)
    z = np.rint((coords[:, 2] - starts[2]) / steps[2]).astype(int)

    idx = (
        (x >= 0)
        & (y >= 0)
        & (z >= 0)
        & (x < dimensions[0])
        & (y < dimensions[1])
        & (z < dimensions[2])
    )

    # perc_mesh_in_volume = np.sum(~idx)/idx.shape[0]
    # assert perc_mesh_in_volume < 0.1, f'Error: significant portion ({perc_mesh_in_volume}) of mesh outside of volume '

    if validate:
        assert np.sum(idx) > 0, "Assert: no voxels found inside mesh_to_volume"
    x = x[idx]
    y = y[idx]
    z = z[idx]

    vertex_values = vertex_values[idx]

    for i, (xc, yc, zc) in enumerate(zip(x, y, z)):
        interp_vol[xc, yc, zc] += vertex_values[i]

        n_vol[xc, yc, zc] += 1

    return interp_vol, n_vol


def multi_mesh_to_volume(
    profiles,
    surfaces,
    depth_list,
    dimensions,
    starts,
    steps,
    resolution,
    ref_fn=None,
):
    interp_vol = np.zeros(dimensions)
    n_vol = np.zeros_like(interp_vol)

    for ii in range(profiles.shape[1]):
        
        surf_fn = surfaces[depth_list[ii]]['depth_rsl_fn']
        
        if "npz" in os.path.splitext(surf_fn)[-1]:
            pass
        else:
            pass

        points = load_mesh_ext(surf_fn)[0] 

        assert (
            points.shape[0] == profiles.shape[0]
        ), f"Error mismatch in number of points ({points.shape[0]}, {profiles.shape[0]}) between {surf_fn} and vertex values file"
        
        print("\tSURF", surf_fn)
        print(np.sum(np.abs(profiles[:, ii])))

        interp_vol, n_vol = mesh_to_volume(
            points,
            profiles[:, ii],
            dimensions,
            starts,
            steps,
            interp_vol=interp_vol,
            n_vol=n_vol,
        )

    interp_vol[n_vol > 0] = interp_vol[n_vol > 0] / n_vol[n_vol > 0]

    assert np.sum(np.abs(interp_vol)) != 0, "Error: interpolated volume is empty"
    return interp_vol




def unique_points(points, scale=1000000000):
    # rpoints = np.rint(points * scale).astype(np.int64)
    upoints, unique_index, unique_inverse = np.unique(
        points.astype(np.float128).round(decimals=3),
        axis=0,
        return_index=True,
        return_inverse=True,
    )

    return points[unique_index, :], unique_index, unique_inverse


def upsample_over_faces(
    surf_fn,
    resolution,
    out_fn,
    face_mask=None,
    profiles_vtr=None,
    chunk_start=None,
    chunk_end=None,
    ref_faces=None,
):
    print(surf_fn)
    coords, faces = load_mesh_ext(surf_fn)

    if type(faces) == type(None):
        if type(ref_faces) != type(None):
            del faces
            faces = ref_faces
        else:
            print("Error: ref faces not defined")

    if type(face_mask) != np.ndarray:
        face_mask = np.ones(faces.shape[0]).astype(bool)

    if ".surf.gii" in out_fn:
        pass

    # Choice 1: truncate vertices by volume boundaries OR by valid y sections where histological
    # sections have been acquired
    if type(face_mask) == None:
        if chunk_start == None:
            chunk_start = min(coords[:, 1])
        if chunk_end == None:
            chunk_end = max(coords[:, 1])
        # find the vertices that are inside of the chunk
        valid_idx = np.where(
            (coords[:, 1] >= chunk_start) & (coords[:, 1] <= chunk_end)
        )[0]
        # create a temporary array for the coords where the exluded vertices are equal NaN
        # this is necessary because we only want to upsample a subset of the entire mesh
        new_coords = np.zeros_like(coords)
        new_coords[:] = np.NaN
        new_coords[valid_idx, :] = coords[valid_idx]
        face_coords = face_coords[valid_faces_idx, :]
        face_mask = np.where(~np.isnan(np.sum(face_coords, axis=(1, 2))))[0]
    else:
        faces[face_mask]
        face_coords = coords[faces]
    # del coords

    # Choice 2 : if values are provided, interpolate these over the face, otherwise create 0-array
    if type(profiles_vtr) != type(None):
        face_vertex_values = profiles_vtr[faces]
        face_vertex_values = face_vertex_values[face_mask, :]
    else:
        face_vertex_values = np.zeros([face_coords.shape[0], 3])

    # resolution is divided by 2 to oversample the surface and guarantee two points per voxel.
    # this is useful because when the mesh is warped into chunk space the vertices can be pulled apart
    # and may not sample the volume with sufficient density to have 1 vertex per voxel
    points, values, new_points_gen = calculate_upsampled_points(
        faces, face_coords, face_vertex_values, resolution / 2
    )

    assert points.shape[1] == 3, "Error: shape of points is incorrect " + points.shape
    points, unique_index, unique_reverse = unique_points(points)

    np.savez(out_fn, points=points, values=values)

    new_points_gen = [new_points_gen[i] for i in unique_index]

    # for i in range(points.shape[0]):
    #    new_points_gen[i].idx = i
    #    print(points[i])
    #    print( new_points_gen[i].generate_point(coords) )
    #    print()

    print("\t\tSaved", out_fn)
    assert (
        len(new_points_gen) == points.shape[0]
    ), f"Error: the amount of points does not equal the amount of point generators {len(new_points_gen)} vs {points.shape[0]}"

    return points, values, new_points_gen


def get_faces_from_neighbours(ngh):
    face_dict = {}
    print("\tCreate Faces")
    for i in range(len(ngh.keys())):
        if i % 1000:
            print(f"2. {100*i/ngh.shape[0]} %", end="\r")
        for ngh0 in ngh[i]:
            for ngh1 in ngh[ngh0]:
                print(i, ngh0, ngh1)
                if ngh1 in ngh[i]:
                    face = [i, ngh0, ngh1]
                    face.sort()
                    face_str = sorted_str(face)
                    try:
                        face_dict[face_str]
                    except KeyError:
                        face_dict[face_str] = face

    n_faces = len(face_dict.keys())

    faces = np.zeros(n_faces, 3)
    for i, f in enumerate(faces.values()):
        faces[i] = f

    return faces


def get_triangle_vectors(points):
    v0 = points[1, :] - points[0, :]
    v1 = points[2, :] - points[0, :]
    return v0, v1


def volume_to_surface(coords, volume_fn, values_fn="", use_ants_image_reader=True, gauss_sd=0):
    if use_ants_image_reader :
        nibabel_ = nib
    else :
        nibabel_ = nb 
    
    img = nibabel_.load(volume_fn)
    vol = img.get_fdata()

    if gauss_sd > 0:
        print('\tGaussian Smoothing, sd:', gauss_sd)
        vol = gaussian_filter(vol, gauss_sd)

    starts = img.affine[[0, 1, 2], 3]

    #FIXME not sure why this is abs. Commented for now because it caused errors
    #step = np.abs(img.affine[[0, 1, 2], [0, 1, 2]])
    step = img.affine[[0, 1, 2], [0, 1, 2]]
    dimensions = vol.shape

    interp_vol, _ = mesh_to_volume(
        coords,
        np.ones(coords.shape[0]),
        dimensions,
        starts,
        img.affine[[0, 1, 2], [0, 1, 2]],
    )

    nibabel_.Nifti1Image(
        interp_vol.astype(np.float32), nibabel_.load(volume_fn).affine
    ).to_filename("tmp.nii.gz")

    coords_idx = np.rint((coords - starts) / step).astype(int)

    idx0 = (coords_idx[:, 0] >= 0) & (coords_idx[:, 0] < dimensions[0])
    idx1 = (coords_idx[:, 1] >= 0) & (coords_idx[:, 1] < dimensions[1])
    idx2 = (coords_idx[:, 2] >= 0) & (coords_idx[:, 2] < dimensions[2])
    idx_range = np.arange(coords_idx.shape[0]).astype(int)

    idx = idx_range[idx0 & idx1 & idx2]

    coords_idx = coords_idx[idx0 & idx1 & idx2]

    values = vol[coords_idx[:, 0], coords_idx[:, 1], coords_idx[:, 2]]

    if values_fn != "":
        pd.DataFrame(values).to_filename(values_fn, index=False, header=False)

    return values, idx


def mult_vector(v0, v1, x, y, p):
    v0 = v0.astype(np.float128)
    v1 = v1.astype(np.float128)
    x = x.astype(np.float128)
    y = y.astype(np.float128)
    p = p.astype(np.float128)

    mult = lambda a, b: np.multiply(
        np.repeat(a.reshape(a.shape[0], 1), b.shape, axis=1), b
    ).T
    w0 = mult(v0, x).astype(np.float128)
    w1 = mult(v1, y).astype(np.float128)
    # add the two vector components to create points within triangle
    p0 = p + w0 + w1
    return p0


def interpolate_face(points, values, resolution, output=None, new_points_only=False):
    # calculate vector on triangle face
    v0, v1 = get_triangle_vectors(points.astype(np.float128))

    # calculate the magnitude of the vector and divide by the resolution to get number of
    # points along edge
    calc_n = lambda v: np.ceil(np.sqrt(np.sum(np.power(v, 2))) / resolution).astype(int)
    mag_0 = calc_n(v0)
    mag_1 = calc_n(v1)

    n0 = max(
        2, mag_0
    )  # want at least the start and end points of the edge between two vertices
    n1 = max(2, mag_1)

    # calculate the spacing from 0 to 100% of the edge
    l0 = np.linspace(0, 1, n0).astype(np.float128)
    l1 = np.linspace(0, 1, n1).astype(np.float128)

    # create a percentage grid for the edges
    xx, yy = np.meshgrid(l1, l0)

    # create flattened grids for x, y , and z coordinates
    x = xx.ravel()
    y = yy.ravel()
    z = 1 - np.add(x, y)

    valid_idx = x + y <= 1.0  # eliminate points that are outside the triangle
    x = x[valid_idx]
    y = y[valid_idx]
    z = z[valid_idx]

    # multiply edge by the percentage grid so that we scale the vector
    p0 = mult_vector(v0, v1, x, y, points[0, :].astype(np.float128))

    interp_values = values[0] * x + values[1] * y + values[2] * z
    """
    if new_points_only : 
        filter_arr = np.ones(p0.shape[0]).astype(bool)
        dif = lambda x,y : np.abs(x-y)<0.0001
        ex0= np.where( (dif(p0,points[0])).all(axis=1) )[0][0]
        ex1= np.where( (dif(p0,points[1])).all(axis=1) )[0][0]
        ex2 = np.where((dif(p0,points[2])).all(axis=1) )[0][0]
        filter_arr[ ex0 ] = filter_arr[ex1] = filter_arr[ex2] = False

        p0 = p0[filter_arr]
        interp_values = interp_values[filter_arr]
    """
    return p0, interp_values, x, y


class NewPointGenerator:
    def __init__(self, idx, face, x, y):
        self.idx = idx
        self.face = face
        self.x = x.astype(np.float128)
        self.y = y.astype(np.float128)

    def generate_point(self, points):
        cur_points = points[self.face].astype(np.float128)

        v0, v1 = get_triangle_vectors(cur_points)

        # new_point = mult_vector(v0, v1, self.x, self.y, points[0,:])
        comp0 = v0.astype(np.float128) * self.x.astype(np.float128)
        comp1 = v1.astype(np.float128) * self.y.astype(np.float128)
        # print('vector components', comp0, comp1, cur_points[0,:])
        new_point = comp0 + comp1 + cur_points[0, :]

        return new_point


def calculate_upsampled_points(
    faces, face_coords, face_vertex_values, resolution, new_points_only=False
):
    points = np.zeros([face_coords.shape[0] * 5, 3], dtype=np.float128)
    values = np.zeros([face_coords.shape[0] * 5])
    n_points = 0
    new_points_gen = {}

    for f in range(face_coords.shape[0]):
        if f % 1000 == 0:
            print(f"\t\tUpsampling Faces: {100.*f/face_coords.shape[0]:.3}", end="\r")

        p0, v0, x, y = interpolate_face(
            face_coords[f],
            face_vertex_values[f],
            resolution,
            new_points_only=new_points_only,
        )

        if n_points + p0.shape[0] >= points.shape[0]:
            points = np.concatenate(
                [points, np.zeros([face_coords.shape[0], 3]).astype(np.float128)],
                axis=0,
            )
            values = np.concatenate([values, np.zeros(face_coords.shape[0])], axis=0)

        new_indices = n_points + np.arange(p0.shape[0]).astype(int)
        cur_faces = faces[f]

        for i, idx in enumerate(new_indices):
            new_points_gen[idx] = NewPointGenerator(idx, cur_faces, x[i], y[i])

        points[n_points : (n_points + p0.shape[0])] = p0
        values[n_points : (n_points + v0.shape[0])] = v0
        n_points += p0.shape[0]

    points = points[0:n_points]
    values = values[0:n_points]

    return points, values, new_points_gen


def transform_surface_to_chunks(
    chunk_info,
    depth,
    out_dir,
    surf_fn,
    ref_gii_fn=None,
    faces_fn=None,
    ext=".surf.gii",
) -> dict:

    surf_chunk_dict = {}

    for (chunk), chunk_df in chunk_info.groupby(["chunk"]):
        thickened_fn = chunk_df["nl_2d_vol_fn"].values[0]
        nl_3d_tfm_fn = chunk_df["nl_3d_tfm_fn"].values[0]
        surf_chunk_space_fn = f"{out_dir}/chunk-{chunk[0]}_{os.path.basename(surf_fn)}"

        print("\tFROM:", surf_fn)
        print("\tTO:", surf_chunk_space_fn)
        print("\tWITH:", nl_3d_tfm_fn)

        surf_chunk_dict[chunk] = surf_chunk_space_fn

        if not os.path.exists(surf_chunk_space_fn):
            apply_ants_transform_to_gii(
                surf_fn,
                [nl_3d_tfm_fn],
                surf_chunk_space_fn,
                0,
                ref_gii_fn=ref_gii_fn,
                faces_fn=faces_fn,
                ref_vol_fn=thickened_fn,
            )

    return surf_chunk_dict

def load_values(in_fn,data_str='data'):
    #TODO: move to mesh_io.py

    if os.path.exists(in_fn+'.npz') :
        values = np.load(in_fn+'.npz')[data_str]
        return values

    ext = os.path.splitext(in_fn)[1]
    if  ext == '.npz' :
        values = np.load(in_fn)[data_str]
    else :
        values = pd.read_csv(in_fn, header=None, index_col=None).values
    return values

def load_mesh_ext(in_fn, faces_fn="", correct_offset=False):
    # TODO: move to mesh_io.py
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


def visualization(surf_coords_filename, values, output_filename):
    def get_valid_idx(c,r):
        cmean=np.mean(c)
        cr = np.std(c)/r
        idx = (c > cmean-cr) & (c < cmean+cr)
        return idx

    if len(values.shape) > 1 :
        values=values.reshape(-1,)

    print(surf_coords_filename) 
    surf_coords = load_mesh_ext(surf_coords_filename)[0]

    x = surf_coords[:,0]
    y = surf_coords[:,1]
    z = surf_coords[:,2]
    x_idx = get_valid_idx(x,5)
    z_idx = get_valid_idx(z,5)
    sns.set(rc={'axes.facecolor':'black', 'figure.facecolor':'black'})

    plt.figure(figsize=(22,12))
    plt.subplot(1,2,1)
    #sns.set_style("dark")
    ax1 = sns.scatterplot(x=y[x_idx], y=z[x_idx], hue=values[x_idx], palette='nipy_spectral',alpha=0.2)

    sns.despine(left=True, bottom=True)
    plt.subplot(1,2,2)
    ax2 = sns.scatterplot(x=y[z_idx], y=x[z_idx], hue=values[z_idx], palette='nipy_spectral',alpha=0.2)
    sns.despine(left=True, bottom=True)
    
    for ax in [ax1,ax2]:

        legend=ax.get_legend()
        if not isinstance(legend, type(None)):
            legend.remove()

        ax.grid(False)

    print('\tWriting', output_filename)
    plt.savefig(output_filename)
    plt.clf()
    plt.cla()


def apply_ants_transform_to_gii(
    in_gii_fn,
    tfm_list,
    out_gii_fn,
    invert,
    ref_gii_fn=None,
    faces_fn=None,
    ref_vol_fn=None,
):
    print("transforming", in_gii_fn)
    print("to", out_gii_fn)

    origin = [0, 0, 0]
    if type(ref_gii_fn) == type(None):
        ref_gii_fn = in_gii_fn

    if os.path.splitext(ref_gii_fn)[1] in [".pial", ".white"]:
        _, _, volume_info = load_mesh(ref_gii_fn)
        # origin = volume_info['cras']
    else:
        volume_info = ref_gii_fn

    coords, faces = load_mesh_ext(in_gii_fn)
    ants.read_transform(tfm_list[0])
    # if np.sum(tfm.fixed_parameters) != 0 :
    #    print( '/MR1/' in os.path.dirname(in_gii_fn))
    #    if '/MR1/' in os.path.dirname(in_gii_fn):
    #        flipx=flipy=-1
    #        flipz=1
    #        flip_label='MR1'
    #    else :

    flipx = flipy = -1  # HUMAN/gifti
    flipz = 1  # HUMAN/gifti

    signx = signy = signz = 1

    open(in_gii_fn, "r")

    out_path, out_ext = os.path.splitext(out_gii_fn)

    # the for loops are here because it makes it easier to trouble shoot to check how the vertices need to be flipped to be correctly transformed by ants
    # for origin in [ [0,0,0], true_origin] :
    #    for flipx in [1,-1] :
    #        for flipy in [1,-1]:
    #            for flipz in [1,-1]:
    #                for signx in [1,-1] :
    #                    for signy in [1,-1]:
    #                        for signz in [1,-1]:
    # flip_label=f'params_{np.sum(np.abs(origin))}_{flipx}{flipy}{flipz}_{signx}{signy}{signz}'
    flip_label = ""
    coord_fn = out_path + f"{flip_label}_ants_reformat.csv"
    temp_out_fn = out_path + f"{flip_label}_ants_reformat_warped.csv"

    coords = np.concatenate([coords, np.zeros([coords.shape[0], 2])], axis=1)
    coords[:, 0] = flipx * (coords[:, 0] + signx * origin[0])  # GIFTI

    coords[:, 1] = flipy * (coords[:, 1] + signy * origin[1])  # GIFTI

    coords[:, 2] = flipz * (coords[:, 2] + signz * origin[2])  # GIFTI

    df = pd.DataFrame(coords, columns=["x", "y", "z", "t", "label"])
    df.to_csv(coord_fn, columns=["x", "y", "z", "t", "label"], header=True, index=False)

    shell(
        f"antsApplyTransformsToPoints -d 3 -i {coord_fn} -t [{tfm_list[0]},{invert}]  -o {temp_out_fn}",
        verbose=True,
    )
    df = pd.read_csv(temp_out_fn, index_col=False)
    df["x"] = flipx * (df["x"] - origin[0])
    df["y"] = flipy * (df["y"] + origin[1])
    df["z"] = flipz * (df["z"] - origin[2])

    new_coords = df[["x", "y", "z"]].values

    out_basename, out_ext = os.path.splitext(out_gii_fn)

    nii_fn = out_path + flip_label + ".nii.gz"
    if ref_vol_fn != None:
        img = nb.load(ref_vol_fn)
        steps = img.affine[[0, 1, 2], [0, 1, 2]]
        starts = img.affine[[0, 1, 2], 3]
        dimensions = img.shape
        print("\t\tReference volume:", ref_vol_fn)
        interp_vol, n = mesh_to_volume(
            new_coords,
            np.ones(new_coords.shape[0]),
            dimensions,
            starts,
            steps,
            validate=False,
        )

        if np.sum(interp_vol) > 0:
            interp_vol[n > 0] = interp_vol[n > 0] / n[n > 0]
            print("\tWriting surface to volume file:", nii_fn)
            nib.Nifti1Image(interp_vol, nib.load(ref_vol_fn).affine, direction_order='lpi').to_filename(nii_fn)

    if out_ext == ".h5":
        f_h5 = h5.File(out_gii_fn, "w")
        f_h5.create_dataset("data", data=new_coords)
        f_h5.close()
        save_mesh(out_path + ".surf.gii", new_coords, faces, volume_info=volume_info)
    elif out_ext == ".npz":
        assert new_coords.shape[1] == 3, (
            "Error: shape of points is incorrect " + new_coords.shape
        )
        np.savez(out_basename, points=new_coords)
    else:
        print("\tWriting Transformed Surface:", out_gii_fn, faces.shape)
        save_mesh(out_gii_fn, new_coords, faces, volume_info=volume_info)

    # obj_fn = out_path +  '.obj'
    # save_obj(obj_fn,coords, faces)
