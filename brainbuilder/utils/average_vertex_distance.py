import os
from re import sub

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mesh_io import load_mesh_geometry
from utils import shell


def calculate_average_vertex_distance(mesh_fn, local_dist_fn, max_n=None):
    mesh = load_mesh_geometry(mesh_fn)
    coords = np.array(mesh["coords"])

    n_vtx = coords.shape[0]
    idx = np.arange(n_vtx).astype(int)

    if max_n == None:
        max_n = n_vtx
    else:
        np.random.shuffle(idx)

    open(local_dist_fn, "w+")
    ar = np.ones(n_vtx) * -1

    for i, coord_i in enumerate(idx):
        if i % 10000 == 0:
            print("\t{}".format(100 * i / max_n), end="\r")
        coord = coords[coord_i]
        ngh_list = mesh["neighbours"][coord_i]

        local_d = 0
        for ngh in ngh_list:
            coord_ngh = coords[ngh]

            d = np.sum(np.abs(coord - coord_ngh))
            local_d += d

        local_avg_dist = local_d / len(ngh_list)  # mesh['neighbour_count'][coord_i]
        ar[coord_i] = local_avg_dist

        if i > max_n:
            break
    df = pd.DataFrame(ar)
    df.to_csv(local_dist_fn, index=False, header=False)
    return df


def subdivide_polygons(fn, out_fn, n):
    if not os.path.exists("/tmp/flip.xfm"):
        shell("param2xfm -clob -scales -1 1 1 /tmp/flip.xfm")
    shell(f"transform_objects {fn} /tmp/flip.xfm /tmp/surf_flipped.obj")
    shell(f"subdivide_polygons /tmp/surf_flipped.obj /tmp/surf_flipped_hires.obj {n}")
    shell(f"transform_objects /tmp/surf_flipped_hires.obj /tmp/flip.xfm {out_fn}")


# base_mesh_fn='../Juelich-Receptor-Atlas/civet/mri1/surfaces/mri1_mid_surface_rsl_right_81920.obj'
base_str = "../Juelich-Receptor-Atlas/civet/mri1/surfaces/mri1_mid_surface_right_{}.obj"
base_mesh_fn = base_str.format(81920)

n_list = 81920 * np.power(2, np.arange(0, 4))
mean_list = []
max_list = []
for n in n_list:
    fn = base_str.format(n)
    print(n, fn)
    if not os.path.exists(fn):
        print("\tsubdividing polygons")
        subdivide_polygons(base_mesh_fn, fn, n)

    local_dist_fn = sub(".obj", "_avg_distance.txt", fn)

    if not os.path.exists(local_dist_fn):
        dist_df = calculate_average_vertex_distance(fn, local_dist_fn)

    dist_df = pd.read_csv(local_dist_fn, names=["dist"])
    mean_dist = dist_df["dist"].mean()
    std_dist = dist_df["dist"].std()
    min_dist = dist_df["dist"].min()
    max_dist = dist_df["dist"].max()
    n_vtx = dist_df.shape[0]
    print(f"\t{n_vtx}: mean {mean_dist} stdev {std_dist} min {min_dist} max {max_dist}")
    mean_list.append(mean_dist)
    max_list.append(max_dist)
    shell(f"depth_potential -area_simple {fn}", verbose=True)
print(mean_list)

plt.plot(n_list, mean_list, label="Mean Distance (mm)")
plt.plot(n_list, max_list, label="Max Distance (mm)", c="r")
plt.legend()
plt.savefig("average_vertex_distance.png")
