"""Prepare surfaces for surface-based interpolation."""
import os
import re
from typing import Optional, Tuple

import nibabel as nb_surf
import numpy as np
import pandas as pd

import brainbuilder.utils.ants_nibabel as nib
from brainbuilder.utils import utils
from brainbuilder.utils.mesh_io import load_mesh, load_mesh_ext, write_gifti
from brainbuilder.utils.mesh_utils import (
    apply_ants_transform_to_gii,
    mesh_to_volume,
    upsample_over_faces,
)
from brainbuilder.utils.utils import shell


def transform_surface_to_chunks(
    chunk_info: pd.DataFrame,
    depth: float,
    out_dir: str,
    surf_fn: str,
    ref_gii_fn: Optional[str] = None,
    faces_fn: Optional[str] = None,
    ext: str = ".surf.gii",
) -> dict:
    """Transform a surface to the histological space of each chunk.
    
    :param chunk_info: pd.DataFrame, chunk information
    :param depth: float, cortical depth
    :param out_dir: str, path to output directory
    :param surf_fn: str, path to surface file
    :param ref_gii_fn: str, path to reference surface file
    :param faces_fn: str, path to faces file
    :param ext: str, extension of surface file
    :return dict: surf_chunk_dict
    """
    surf_chunk_dict = {}

    for (sub, hemisphere, chunk), chunk_df in chunk_info.groupby(
        [
            "sub",
            "hemisphere",
            "chunk",
        ]
    ):
        thickened_fn = chunk_df["nl_2d_vol_fn"].values[0]
        nl_3d_tfm_fn = chunk_df["nl_3d_tfm_fn"].values[0]
        surf_chunk_space_fn = f"{out_dir}/sub-{sub}_hemi-{hemisphere}_chunk-{chunk}_{os.path.basename(surf_fn)}"

        surf_chunk_dict[chunk] = surf_chunk_space_fn

        if not os.path.exists(surf_chunk_space_fn) or utils.newer_than(
            nl_3d_tfm_fn, surf_chunk_space_fn
        ):
            print("\tFROM:", surf_fn)
            print("\tTO:", surf_chunk_space_fn)
            print("\tWITH:", nl_3d_tfm_fn)
            print("\tREF:", thickened_fn)

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


def prepare_surfaces(
    chunk_info: pd.DataFrame,
    ref_vol_fn: str,
    gm_surf_fn: str,
    wm_surf_fn: str,
    depth_list: list,
    output_dir: str,
    resolution: float,
    verbose: bool = False,
    clobber: bool = False,
)-> tuple:
    """Prepare surfaces for surface-based interpolation.

    :param chunk_info_csv: path to the chunk info csv
    :param ref_vol_fn: path to the structural reference volume
    :param gm_surf_fn: path to the gray matter surface
    :param wm_surf_fn: path to the white matter surface
    :param depth_list: list of cortical depths
    :param output_dir: output directory
    :param resolution: resolution of the surfaces
    :param clobber: bool
    :return dict: surf_depth_mni_dict, surf_depth_chunk_dict
    """
    os.makedirs(output_dir, exist_ok=True)

    print("\tGenerating surfaces across cortical depth.")
    surf_depth_mni_dict = generate_cortical_depth_surfaces(
        ref_vol_fn,
        depth_list,
        resolution,
        wm_surf_fn,
        gm_surf_fn,
        output_dir,
    )

    print("\tInflating surfaces.")
    surf_depth_mni_dict = inflate_surfaces(
        surf_depth_mni_dict, output_dir, resolution, depth_list, clobber=clobber
    )

    print("\tUpsampling surfaces.")
    surf_depth_mni_dict = upsample_surfaces(
        surf_depth_mni_dict,
        output_dir,
        gm_surf_fn,
        resolution,
        depth_list,
        ref_vol_fn,
        clobber=clobber,
    )

    # For each chunk, transform the mesh surface to the receptor space
    print("\tTransforming surfaces to chunk space.")
    surf_depth_chunk_dict = transfrom_depth_surf_to_chunk_space(
        chunk_info, surf_depth_mni_dict, output_dir
    )

    return surf_depth_mni_dict, surf_depth_chunk_dict


def transfrom_depth_surf_to_chunk_space(
    chunk_info: pd.DataFrame, surf_depth_mni_dict: dict, surf_rsl_dir: str
)-> dict:
    """For each chunk, transform the mesh surface to the histological space.

    :param chunk_info: pd.DataFrame, chunk information
    :param surf_depth_mni_dict: dict, keys are cortical depths, values are dicts containing surfaces in reference space
    :param surf_rsl_dir: str, path to output directory
    :return surf_depth_chunk_dict : dict, keys are cortical depths, values are dicts containing surfaces in histological space of each chunk
    """
    surf_depth_chunk_dict = {}

    for depth, depth_dict in surf_depth_mni_dict.items():
        faces_fn = None

        ref_gii_fn = depth_dict["depth_surf_fn"]

        surf_fn = depth_dict["depth_rsl_fn"]

        surf_chunk_dict = transform_surface_to_chunks(
            chunk_info,
            depth,
            surf_rsl_dir,
            surf_fn,
            faces_fn=faces_fn,
            ref_gii_fn=ref_gii_fn,
        )

        surf_depth_chunk_dict[float(depth)] = surf_chunk_dict

    return surf_depth_chunk_dict


def get_surf_base(surf_fn: str) -> str:
    """Get the basename of a surface file.

    :param surf_fn: str, path to surface file
    :return str: basename of surface file
    """
    surf_base = os.path.basename(surf_fn)
    for string in [".pial", ".smoothwm", ".gz", ".gii", ".surf"]:
        surf_base = re.sub(string, "", surf_base)

    return surf_base


def generate_cortical_depth_surfaces(
    ref_vol_fn: str,
    depth_list: list,
    resolution: float,
    wm_surf_fn: str,
    gm_surf_fn: str,
    output_dir: str,
) -> dict:
    """Generate cortical depth surfaces.

    :param ref_vol_fn: str, path to reference volume file
    :param depth_list: list, cortical depths
    :param resolution: float, maximum resolution of the reconstruction
    :param wm_surf_fn: str, path to white matter surface file
    :param gm_surf_fn: str, path to gray matter surface file
    :param output_dir: str, path to output directory
    :return dict: surf_depth_mni_dict
    """
    surf_depth_mni_dict = {}

    img = nb_surf.load(ref_vol_fn)
    dimensions = img.shape
    steps = img.affine[[0, 1, 2], [0, 1, 2]]
    starts = img.affine[[0, 1, 2], [3, 3, 3]]

    gm_coords, gm_faces, gm_info = load_mesh(gm_surf_fn, correct_offset=True)
    wm_coords, wm_faces, wm_info = load_mesh(wm_surf_fn, correct_offset=True)

    d_coords = wm_coords - gm_coords

    del wm_coords

    surf_base = get_surf_base(gm_surf_fn)

    for depth in depth_list:
        depth_surf_fn = "{}/{}_{}mm_{}.surf.gii".format(
            output_dir,
            surf_base,
            resolution,
            depth,
        )
        depth_vol_fn = "{}/{}_{}mm_{}.nii.gz".format(
            output_dir, surf_base, resolution, depth
        )

        surf_depth_mni_dict[depth] = {"depth_surf_fn": depth_surf_fn}

        coords = gm_coords + depth * d_coords

        if not os.path.exists(depth_surf_fn):
            write_gifti(depth_surf_fn, coords, gm_faces)

            # Create a volume of the cortical depth surface for QC purposes
            interp_vol, _ = mesh_to_volume(
                coords, np.ones(coords.shape[0]), dimensions, starts, steps
            )
            print("\tWriting", depth_vol_fn)
            nib.Nifti1Image(
                interp_vol, nib.load(ref_vol_fn).affine, direction_order="lpi"
            ).to_filename(depth_vol_fn)

    return surf_depth_mni_dict


def inflate_surfaces(
    surf_depth_mni_dict: dict,
    output_dir: str,
    resolution: float,
    depth_list: list,
    clobber: bool = False,
) -> dict:
    """Upsampling of meshes at various depths across cortex produces meshes with different n vertices.

     To create a set of meshes across the surfaces across the cortex that have the same number of
     vertices, we first upsample and inflate the wm mesh.
     Then, for each mesh across the cortex we resample that mesh so that it has the same polygons
     and hence the same number of vertices as the wm mesh.
     Each mesh across the cortical depth is inflated (from low resolution, not the upsampled version)
     and then resampled so that it has the high resolution number of vertices.

    :param surf_depth_mni_dict: dict, keys are cortical depths, values are dicts containing surfaces in stereotaxic space
    :param output_dir: str, path to output directory
    :param resolution: float, maximum resolution of the reconstruction
    :param depth_list: list, cortical depths
    :param clobber: bool, whether to overwrite existing files
    :return dict: surf_depth_mni_dict
    """
    for depth in depth_list:
        depth += 0.0
        depth_surf_fn = surf_depth_mni_dict[float(depth)]["depth_surf_fn"]

        base_surf = re.sub(".surf.gii", "", os.path.basename(depth_surf_fn))

        inflate_fn = "{}/{}_{}mm_{}.inflate".format(
            output_dir, base_surf, resolution, depth
        )
        sphere_fn = "{}/{}_{}mm_{}.sphere".format(
            output_dir, base_surf, resolution, depth
        )
        sphere_rsl_fn = "{}/{}_{}mm_{}_sphere_rsl.npz".format(
            output_dir, base_surf, resolution, depth
        )
        surf_depth_mni_dict[depth]["sphere_rsl_fn"] = sphere_rsl_fn
        surf_depth_mni_dict[depth]["sphere_fn"] = sphere_fn
        surf_depth_mni_dict[depth]["inflate_fn"] = inflate_fn

        if not os.path.exists(sphere_fn) or clobber:
            print("\tInflate to sphere")
            shell(f"~/freesurfer/bin/mris_inflate -n 100  {depth_surf_fn} {inflate_fn}")
            print(f"\t\t{inflate_fn}")
            shell(f"~/freesurfer/bin/mris_sphere -q  {inflate_fn} {sphere_fn}")
            print(f"\t\t{sphere_fn}")

    return surf_depth_mni_dict


def generate_face_and_coord_mask(
        edge_mask_idx: np.ndarray,
        faces: np.ndarray,
        coords: np.ndarray
        ) -> Tuple[np.ndarray, np.ndarray]:
    """Generate face and coordinate masks.

    :param edge_mask_idx: np.ndarray, indices of the edge mask
    :param faces: np.ndarray, faces of the mesh
    :param coords: np.ndarray, coordinates of the mesh
    :return np.ndarray: face_mask
    :return np.ndarray: coord_mask
    """
    mask_idx_unique = np.unique(edge_mask_idx)
    n = max(faces.shape[0], coords.shape[0])

    face_mask = np.zeros(faces.shape[0]).astype(np.bool)
    coord_mask = np.zeros(coords.shape[0]).astype(np.bool)

    for i in range(n):
        if i < faces.shape[0]:
            f0, f1, f2 = faces[i]
            if f0 in mask_idx_unique:
                face_mask[i] = True
            if f1 in mask_idx_unique:
                face_mask[i] = True
            if f2 in mask_idx_unique:
                face_mask[i] = True
        if i < coords.shape[0]:
            if i in mask_idx_unique:
                coord_mask[i] = True
    return face_mask, coord_mask


def resample_points(surf_fn:str, new_points_gen: list)-> tuple:
    """Resample points on a mesh surface using the new points generator class.

    Basically the new points generator class generates new points on the surface of the mesh based on
    the surface triangle around the point.
    
    :param surf_fn: str, path to surface file
    :param new_points_gen: list, new points generator class
    :return np.ndarray: new_points
    :return np.ndarray: points
    """
    points, faces = load_mesh_ext(surf_fn)
    n = len(new_points_gen)

    new_points = np.zeros([n, 3])

    for i, gen in enumerate(new_points_gen):
        new_points[i] = gen.generate_point(points)

    # new_points, unique_index, unique_reverse = unique_points(new_points)
    new_points += np.random.normal(0, 0.0001, new_points.shape)
    return new_points, points


def upsample_surfaces(
    surf_depth_mni_dict: dict,
    output_dir: str,
    gm_surf_fn: str,
    resolution: float,
    depth_list: list,
    ref_vol_fn: str,
    clobber: bool = False,
) -> dict:
    """Upsample surfaces across cortical depth.

    :param surf_depth_mni_dict: dict, keys are cortical depths, values are dicts containing surfaces in stereotaxic space
    :param output_dir: str, path to output directory
    :param gm_surf_fn: str, path to gray matter surface
    :param resolution: float, maximum resolution of the reconstruction
    :param depth_list: list, cortical depths
    :param ref_vol_fn: str, path to structural reference volume
    :param clobber: bool, whether to overwrite existing files
    :return dict: surf_depth_mni_dict
    """
    surf_base = get_surf_base(gm_surf_fn)

    "{}/{}_{}mm_{}_rsl.obj".format(output_dir, surf_base, resolution, 0)
    upsample_0_fn = "{}/surf_{}mm_{}_rsl.surf.gii".format(
        output_dir, resolution, depth_list[0]
    )
    upsample_1_fn = "{}/{}_{}mm_{}_rsl.surf.gii".format(
        output_dir, surf_base, resolution, depth_list[-1]
    )
    input_list = []
    output_list = []

    for depth in depth_list:
        depth_rsl_gii = "{}/{}_{}mm_{}_rsl.surf.gii".format(
            output_dir, surf_base, resolution, depth
        )
        depth_rsl_fn = "{}/{}_{}mm_{}_rsl.npz".format(
            output_dir, surf_base, resolution, depth
        )

        surf_depth_mni_dict[depth]["depth_rsl_fn"] = depth_rsl_fn
        surf_depth_mni_dict[depth]["depth_rsl_gii"] = depth_rsl_gii

        depth_surf_fn = surf_depth_mni_dict[depth]["depth_surf_fn"]
        sphere_fn = surf_depth_mni_dict[depth]["sphere_fn"]
        sphere_rsl_fn = surf_depth_mni_dict[depth]["sphere_rsl_fn"]

        input_list += [depth_surf_fn, sphere_fn]
        output_list += [depth_rsl_fn, sphere_rsl_fn]

    surf_depth_mni_dict[depth_list[0]]["depth_rsl_gii"] = upsample_0_fn
    surf_depth_mni_dict[depth_list[-1]]["depth_rsl_gii"] = upsample_1_fn

    # DEBUG put the next few lines incase want to try face upsampling instead of full surf upsampling
    ref_gii_fn = surf_depth_mni_dict[0]["depth_surf_fn"]
    surf_depth_mni_dict[0]["depth_rsl_gii"]
    ref_rsl_npy_fn = re.sub(".surf.gii", "", surf_depth_mni_dict[0]["depth_rsl_gii"])
    re.sub(".nii.gz", "_ngh", surf_depth_mni_dict[0]["depth_rsl_gii"])
    coords, faces, volume_info = load_mesh(ref_gii_fn)

    if False in [os.path.exists(fn) for fn in output_list + [ref_rsl_npy_fn + ".npz"]]:
        points, _, new_points_gen = upsample_over_faces(
            ref_gii_fn, resolution, ref_rsl_npy_fn
        )

        img = nb_surf.load(ref_vol_fn)
        steps = img.affine[[0, 1, 2], [0, 1, 2]]
        starts = img.affine[[0, 1, 2], 3]
        dimensions = img.shape
        interp_vol, _ = mesh_to_volume(
            points, np.ones(points.shape[0]), dimensions, starts, steps
        )
        nib.Nifti1Image(
            interp_vol, nib.load(ref_vol_fn).affine, direction_order="lpi"
        ).to_filename(f"{output_dir}/surf_{resolution}mm_{depth}_rsl.nii.gz")

        print("Upsampled points", points.shape, len(new_points_gen))

        ref_points = np.load(ref_rsl_npy_fn + ".npz")["points"]
        n_points = ref_points.shape[0]

        for in_fn, out_fn in zip(input_list, output_list):
            points, old_points = resample_points(in_fn, new_points_gen)
            # full = np.arange(points.shape[0]).astype(int)

            assert (
                n_points == points.shape[0]
            ), f"Error: mismatch in number of points in mesh between {n_points} and {points.shape[0]}"

            nii_fn = os.path.splitext(out_fn)[0] + ".nii.gz"
            nib.Nifti1Image(
                interp_vol, nib.load(ref_vol_fn).affine, direction_order="lpi"
            ).to_filename(nii_fn)
            print("\tWriting (nifti):", nii_fn)
            assert points.shape[1], (
                "Error: shape of points is incorrect " + points.shape
            )
            np.savez(out_fn, points=points)
            print("\tWriting (npz)", out_fn)
    return surf_depth_mni_dict
