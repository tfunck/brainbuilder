import argparse

import numpy as np
from brainbuilder.interp.surfinterp import interpolate_over_surface
from brainbuilder.utils.mesh_utils import load_mesh_ext, spherical_distance


def get_surf_values(cortex_fn, sphere_fn, order=1):
    coords = load_mesh_ext(cortex_fn)[0]

    y0, y1 = np.percentile(coords[:, 1], [2, 98])

    surface_val = np.zeros(coords.shape[0])

    surface_val[(coords[:, 1] < y0)] = 1
    surface_val[(coords[:, 1] > y1)] = 2

    interp_values = interpolate_over_surface(
        sphere_fn, surface_val, threshold=0, order=order
    )

    return interp_values, coords


"""
def test_interp_cortex(cortex_fn, cortex_rsl_fn, sphere_fn, sphere_rsl_fn, order=1):

    interp_values, coords = get_surf_values(cortex_fn, sphere_fn, order=order)
    interp_rsl_values, coords_rsl = get_surf_values(cortex_rsl_fn, sphere_rsl_fn, order=order)

    plt.figure(figsize=(10,10))
    plt.scatter(coords[:,1], interp_values, alpha=0.01, c='b')
    plt.scatter(coords_rsl[:,1], interp_rsl_values, alpha=0.01, c='r')
    plt.savefig(f'test_interp_cortex_scatter_{order}.png')
    plt.close()

    

    visualization(cortex_fn, interp_values, f'test_interp_cortex_{order}.png')
"""


def test_volume_to_mesh():
    from brainbuilder.utils.mesh_utils import volume_to_mesh

    n_coords = 3000

    dim = 100

    vol = np.zeros((dim, dim, dim))

    coords = np.random.uniform(0, dim - 1, (3, n_coords))

    print("Coords", np.min(coords), np.max(coords))
    chunk_values, chunk_n = volume_to_mesh(coords, vol, [0, 0, 0], [1, 1, 1], [dim] * 3)

    chunk_values[chunk_n > 0] = chunk_values[chunk_n > 0] / chunk_n[chunk_n > 0]

    for val in chunk_values:
        assert val in vol, f"Error: {val} not in {vol}"
    print("Done")


def test_sphereical_distance():
    def get_x(r, a):
        return r * np.cos(a)

    def get_y(r, a):
        return r * np.sin(a)

    def grid_coords(r, a):
        return get_x(r, a), get_y(r, a)

    p = 1

    n = 4

    theta = [0, 0]  # np.linspace(0, 2*np.pi, n) #azimuthal angle
    phi = [0, np.pi]  # np.linspace(0, 2*np.pi, n) #polar angle

    print("true theta", theta)
    print("true phi", phi)

    x = p * np.sin(phi) * np.cos(theta)
    y = p * np.sin(phi) * np.sin(theta)
    z = p * np.cos(phi)

    points = np.vstack([x, y, z]).T

    dist = spherical_distance(points, points)

    print(dist)
    exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    cortex_fn = "/data/receptor/human/brainbuilder_human_out//4_interp/surfaces/surf_3mm_1.0.surf.gii"
    cortex_rsl_fn = "/data/receptor/human/brainbuilder_human_out//4_interp/surfaces/surf_3mm_1.0_rsl.npz"
    sphere_fn = "/data/receptor/human/brainbuilder_human_out//4_interp/surfaces/surf_3mm_1.0.sphere"
    sphere_rsl_fn = "/data/receptor/human/brainbuilder_human_out//4_interp/surfaces/surf_3mm_1.0_sphere_rsl.npz"

    # test_interp_cortex(cortex_fn, cortex_rsl_fn, sphere_fn, sphere_rsl_fn, order=1)
    # test_interp_cortex(cortex_fn, cortex_rsl_fn, sphere_fn, sphere_rsl_fn, order=0)

    test_volume_to_mesh()
    test_sphereical_distance()
