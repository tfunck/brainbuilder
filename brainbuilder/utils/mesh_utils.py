import os
import re

import ants
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from joblib import Parallel, delayed, cpu_count
from sklearn.metrics import pairwise_distances

import brainbuilder.utils.ants_nibabel as nib
import brainbuilder.utils.utils as utils

matplotlib.use("Agg")
import multiprocessing

import h5py as h5
import nibabel
import numpy as np

from brainbuilder.utils.mesh_io import load_mesh, save_mesh
from brainbuilder.utils.utils import shell


def interleve(x,step):
    '''

    '''
    return np.concatenate([ x[i:x.shape[0]+1:step] for i in range(step) ] )

def magnitude(V):
    D = np.power(V, 2)
    if  len(D.shape)==1:
        D=np.sum(D)
    else :
        D=np.sum(D, axis=1)
    return np.sqrt(D)

def difference(V):
    D = np.power(V, 2)
    if  len(D.shape)==1:
        D=np.sum(D)
    else :
        D=np.sum(D, axis=1)
    return  D

def pair_vectors(c0,c1):
    X = np.repeat(c0, c1.shape[0], axis=0)
    # create array with repeated columns of c0
    Y = interleve(np.repeat(c1, c0.shape[0], axis=0), c0.shape[0])
    return X,Y

def diff(theta0,theta1,r):
    # integrate_theta0^theta1 r d(theta)
    # -> r(theta1 - theta0)
    return 

def get_average_magnitude(p0,p1):

    r = np.mean([r0,r1])
    return r
    
def spherical_distance(v0,v1):
    #caclulate radius from points
    #points may not be perfectly spherical so we take an average

    r0 = magnitude(v0[:,0:2])  #radius of points in p0 on azithumal XY plane
    r1 = magnitude(v1[:,0:2])  #radius of points in p1 on azithumal XY plane  

    p0 = magnitude(v0) #magnitude of points in p0 in 3d
    p1 = magnitude(v1) #magnitude of points in p0 in 3d

    r = np.mean(np.array([r0,r1]))
    p = np.mean(np.array([p0,p1]))

    phi0 = np.arccos(v0[:,2]/p0)
    phi1 = np.arccos(v1[:,2]/p1)

    # x = ρ sinφ cosθ
    # y = ρ sinφ sinθ
    # z = ρ cosφ --> cosφ = z/ρ --> φ = arccos(z/ρ)
    # θ = arccos(z/p)

    theta0 = np.arccos(v0[:,0]/r0)
    theta1 = np.arccos(v1[:,0]/r1)

    theta0[ np.isnan(theta0) ] = 0
    theta1[ np.isnan(theta1) ] = 0

    X=np.column_stack([phi0,theta0])
    Y=np.column_stack([phi1,theta1])
    dist = pairwise_distances(X,Y,metric='haversine') * p 
    return dist


def pairwise_coord_distances(c0:np.ndarray, c1:np.ndarray,use_l2=True) -> np.ndarray:
    '''
    calculate the pairwise distance between
        c0: 3d coordinates
        c1: 3d coordinates
        
    '''
    c0 = c0.astype(np.float16)
    c1 = c1.astype(np.float16)
    try :
        X,Y = pair_vectors(c0,c1)

        if method == 'l2' :
            D = magnitude(X-Y)
        elif method == 'diff' :
            D = difference(X-Y)
        elif method == 'spherical':
            D = spherical_distance(X-Y)



        D = D.reshape(c0.shape[0],c1.shape[0])
    except MemoryError or np.core._exceptions.MemoryError:
        print('Warning: OOM. Defaulting to slower pairwise distance calculator', end='...\n')
        D=np.zeros([c0.shape[0],c1.shape[0]], dtype=np.float16)
        for i in range(c0.shape[0]) : #iterate over rows of co
            if i%100==0: print(f'Completion: {np.round(100*i/c0.shape[0],1)}',end='\r')
            #calculate magnitude of point in row i of c0 versus all points in c1
            D[i] = magnitude(c1-c0[i])

    return D

def smooth_surface(coords, values, sigma, sigma_n=5):

    #precaculate the denominator and sigma squared for the normal distribution
    den = 1/(sigma*np.sqrt(2*np.pi))
    sigma_sq = np.power(sigma, 2)
    
    #define the normal distribution
    normal_pdf = lambda dist : den * np.exp(-0.5 *  dist / sigma_sq)

    #calculate the pairwise distance between all coordinates
    #dist = pairwise_coord_distances(coords, coords, use_l2=False) 
    dist = spherical_distance(coords, coords)
    
    wghts = normal_pdf(dist)
    del dist

    # normalize the weights
    wghts = wghts / np.sum(wghts, axis=1)[..., np.newaxis]

    wghts_sum = np.sum(wghts, axis=1)

    assert np.sum( wghts_sum - 1 ) < 0.0001, f'Error: weights do not sum to 1 {wghts_sum}'

    wghts[np.isnan(wghts)] = 0

    # repeat the values for each coordinate so that it can be multiplied by weights matrix
    #values = np.ones([coords.shape[0],1])
    # multiply the values by the weights matrix and sum along rows
    smoothed_values = np.matmul(wghts,values)
    del wghts
    
    # replace any NaN values with the original values
    idx = np.isnan(smoothed_values)
    smoothed_values[idx] = values[idx]
    del idx


    return smoothed_values.reshape(-1,)

def local_smooth_surf(
    coords,
    values,
    sigma,
    radius,
    xparams,
    yparams,
    zparams,
    x,
    y,
    z,
):

    xw = coords[:,0]
    yw = coords[:,1]
    zw = coords[:,2]

    xmin, xmax, xstep = xparams
    ymin, ymax, ystep = yparams
    zmin, zmax, zstep = zparams

    # .     |           |
    # x0    |           | 
    # .     x           x+step
    x0, x1 = x - radius, x + xstep + radius
    y0, y1 = y - radius, y + ystep + radius
    z0, z1 = z - radius, z + zstep + radius

    core_idx = np.where( (x <= xw) & (xw < x+xstep) &
                         (y <= yw) & (yw < y+ystep) &
                         (z <= zw) & (zw < z+zstep) 
                        )[0]

    search_idx = np.where( (x0 <= xw) & (xw < x1) &
                           (y0 <= yw) & (yw < y1) &
                           (z0 <= zw) & (zw < z1)
                      )[0]

    # identify which of the local coordiantes are within the core
    local_coords = coords[search_idx,:]

    local_core_idx = np.where((x <= local_coords[:,0]) & (local_coords[:,0] < x+xstep) &
                              (y <= local_coords[:,1]) & (local_coords[:,1] < y+ystep) &
                              (z <= local_coords[:,2]) & (local_coords[:,2] < z+zstep)
                              )[0]
    if core_idx.shape[0] > 0 :

        local_values = values[search_idx].reshape(-1,1)

        smoothed_local_values = smooth_surface(local_coords, local_values, sigma)
    else : 
        smoothed_local_values = values[core_idx]

    
    return core_idx, smoothed_local_values[local_core_idx]

def smooth_surface_by_parts(coords, values, sigma, n_sigma=3, step=10):

    smoothed_values = np.zeros_like(values)

    assert step>0, f'Error: step must be greater than 0, {step}'

    radius = n_sigma*sigma

    xw = coords[:,0]
    yw = coords[:,1]
    zw = coords[:,2]

    xmin, xmax = np.min(xw), np.max(xw)
    ymin, ymax = np.min(yw), np.max(yw)
    zmin, zmax = np.min(zw), np.max(zw)

    xstep = (xmax-xmin)/step
    ystep = (ymax-ymin)/step   
    zstep = (zmax-zmin)/step

    x_range = np.arange(xmin, xmax, xstep)
    y_range = np.arange(ymin, ymax, ystep)
    z_range = np.arange(zmin, zmax, zstep)

    n_dist = 0 
    n = 1
    to_do=[]
    for i, x in enumerate(x_range):
        for j, y in enumerate(y_range):
            for k, z in enumerate(z_range):
                to_do.append(
                        (
                            (xmin,xmax,xstep),
                            (ymin,ymax,ystep),
                            (zmin,zmax,zstep),
                            x,
                            y,
                            z,
                        )
                    )
                x1, y1, z1 = x + xstep, y + ystep, z + zstep
                idx = ( (coords[:,0] > x) & (coords[:,0] < x1) &\
                        (coords[:,1] > y) & (coords[:,1] < y1) &\
                        (coords[:,2] > z) & (coords[:,2] < z1) 
                       )
                n_dist += np.sum(idx) ** 2
                n += 1
                
    n_elems = len(to_do)
    n_vtx = (xstep*ystep*zstep)

    avg_n_dist = n_dist / n

    element_list = avg_n_dist * 2  + [values.shape[0] , coords.shape[0]*3]
    print(avg_n_dist * values.dtype.itemsize / 1024 / 1024 / 1024 )
    element_size_list = [ values.dtype.itemsize ] + [values.dtype.itemsize, coords.dtype.itemsize] 

    n_cores = utils.get_maximum_cores(element_list, element_size_list)

    results = Parallel(n_jobs=n_cores)(delayed(local_smooth_surf)(coords,values,sigma,radius,xparam,yparam,zparam,x,y,z) for xparam,yparam,zparam,x,y,z in to_do) 

    for core_idx, smoothed_local_values in results:
        smoothed_values[core_idx] =  smoothed_local_values
    
    return smoothed_values

def smooth_surface_profiles(profiles_fn, surf_depth_mni_dict, sigma, clobber=False):

    smoothed_profiles_fn = profiles_fn + '_smoothed'

    if not os.path.exists(smoothed_profiles_fn) or clobber:
        print('\tSmoothing surface profiles')

        profiles = np.load(profiles_fn+'.npz')['data']
        smoothed_profiles = np.zeros_like(profiles)

        for i, (depth, surf_dict) in enumerate(surf_depth_mni_dict.items()):
            print('\t\tDepth:', depth)
            surf_fn = surf_dict['sphere_rsl_fn']
            coords = load_mesh_ext(surf_fn)[0]
            profile = profiles[:, i]

            smoothed_profile = smooth_surface_by_parts(coords, profile, sigma)
            #plt.scatter(profile, smoothed_profile); plt.savefig(f'/tmp/tmp_{i}.png')
            #plt.clf(); plt.cla()

            smoothed_profiles[:, i] = smoothed_profile

        np.savez(smoothed_profiles_fn, data=smoothed_profiles)

    return smoothed_profiles_fn

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


def volume_to_mesh(
        coords: np.ndarray,
        vol: np.ndarray,
        starts: np.ndarray,
        steps: np.ndarray,
        dimensions: np.ndarray,
    )->np.ndarray:
    '''
    Interpolate volume values to mesh vertices
    
    '''

    x = np.rint((coords[:, 0] - starts[0]) / steps[0]).astype(int)
    y = np.rint((coords[:, 1] - starts[1]) / steps[1]).astype(int)
    z = np.rint((coords[:, 2] - starts[2]) / steps[2]).astype(int)

    xmax = np.max(x)
    zmax = np.max(z)

    if zmax >= vol.shape[2]:
        print(
            f"\nWARNING: z index {zmax} is greater than dimension {vol.shape[2]}\n"
        )
    if xmax >= vol.shape[0]:
        print(
            f"\nWARNING: x index {xmax} is greater than dimension {vol.shape[0]}\n"
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

def write_mesh_to_volume(profiles, surfaces, volume_fn, output_fn, resolution, clobber=False):
   
    if not os.path.exists(output_fn) or clobber:
        img = nibabel.load(volume_fn)
        starts = img.affine[0:3, 3]
        steps = np.diag(img.affine)[0:3]
        dimensions = img.shape

        vol = multi_mesh_to_volume(
            profiles,
            surfaces,
            dimensions,
            starts,
            steps,
            resolution,
        )

        print(f"\tWriting mesh to volume {output_fn}")
        nib.Nifti1Image(vol, nib.load(volume_fn).affine,direction_order='lpi').to_filename(output_fn)
    else :
        vol = nib.load(output_fn).get_fdata()

    return vol

def mesh_to_volume(
    coords,
    vertex_values,
    dimensions,
    starts,
    steps,
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
    dimensions,
    starts,
    steps,
    resolution,
):
    interp_vol = np.zeros(dimensions)
    n_vol = np.zeros_like(interp_vol)

    for ii in range(profiles.shape[1]):
        
        surf_fn = surfaces[ii]
        
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


def volume_to_surface(coords, volume_fn, values_fn=""):
    print(volume_fn)
    img = nibabel.load(volume_fn)
    vol = img.get_fdata()

    starts = img.affine[[0, 1, 2], 3]
    step = np.abs(img.affine[[0, 1, 2], [0, 1, 2]])
    dimensions = vol.shape

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
    #sns.set(rc={'axes.facecolor':'black', 'figure.facecolor':'black'})

    plt.figure(figsize=(22,12))
    plt.subplot(1,2,1)
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
    plt.close()


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
        img = nibabel.load(ref_vol_fn)
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
