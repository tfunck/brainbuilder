import numpy as np


class SlabReconstructionData:
    def __init__(
        self, brain, hemi, chunks, ligands, depths, volume_dir, surface_dir, resolution
    ):
        self.brain = brain
        self.hemi = hemi
        self.volume_dir = volume_dir
        self.surface_dir = surface_dir
        self.resolution = resolution
        self.ligands = ligands

        self.depths = np.sort(depths.astype(float))
        self.n_depths = len(self.depths)
        self.chunks = np.sort(np.array(chunks).astype(int))

        self.middle_depth = self.depths[np.rint(len(self.depths) / 2).astype(int)]

        self.surfaces = self.create_surface_dict()
        self.stx_surfaces = self.create_stx_surface_dict()
        self.spheres = self.create_spheres_dict()
        self.volumes = self.create_volume_dict()
        self.cls = self.create_volume_dict(tissue_type="_cls")
        self.mid_surfaces_dict = self.set_mid_surfaces()

        self.values_raw = self.create_values_raw_dict()
        self.values_interp = self.create_values_interp_dict()

    def set_current_ligand(self, ligand):
        if self.volumes_all == None:
            self.volumes_all = self.volumes
        self.volumes = self.volumes[ligand]

    def create_values_raw_dict(self, tissue_type=""):
        file_dict = {}
        for ligand in self.ligands:
            file_dict[ligand] = {}
            for depth in self.depths:
                file_dict[ligand][
                    depth
                ] = f"{self.volume_dir}/{self.brain}_{self.hemi}_{ligand}_{self.resolution}mm{tissue_type}_l{self.n_depths}_{depth}_raw.csv"
        return file_dict

    def create_values_interp_dict(self, tissue_type=""):
        file_dict = {}
        for ligand in self.ligands:
            file_dict[
                ligand
            ] = f"{self.volume_dir}/{self.brain}_{self.hemi}_{ligand}_{self.resolution}mm{tissue_type}_l{self.n_depths}.h5"
        return file_dict

    def gen_sphere_filename(self, depth):
        return f"{self.surface_dir}/surf_{self.resolution}mm_{depth}_sphere_rsl.npz"

    def gen_stx_surface_filename(self, depth):
        return f"{self.surface_dir}/surf_{self.resolution}mm_{depth}_rsl.npz"

    def gen_surface_filename(self, chunk, depth):
        return (
            f"{self.surface_dir}/chunk-{chunk}_surf_{self.resolution}mm_{depth}_rsl.npz"
        )

    def gen_volume_filename(self, chunk, ligand, tissue_type=""):
        return f"{self.volume_dir}thickened{tissue_type}_{int(chunk)}_{ligand}_{self.resolution}_l{self.n_depths}.nii.gz"

    def create_volume_dict(self, tissue_type=""):
        files = {}
        for ligand in self.ligands:
            files[ligand] = {}
            for chunk in self.chunks:
                files[ligand][chunk] = self.gen_volume_filename(
                    chunk, ligand, tissue_type=tissue_type
                )

        return files

    def create_spheres_dict(self):
        files = {}
        for chunk in self.chunks:
            files[chunk] = {}
            for depth in self.depths:
                files[chunk][depth] = self.gen_sphere_filename(depth)
        return files

    def create_stx_surface_dict(self):
        files = {}
        for depth in self.depths:
            files[depth] = self.gen_stx_surface_filename(depth)
        return files

    def create_surface_dict(self):
        files = {}
        for chunk in self.chunks:
            files[chunk] = {}
            for depth in self.depths:
                files[chunk][depth] = self.gen_surface_filename(chunk, depth)
        return files

    def set_mid_surfaces(self):
        mid_surfaces_dict = {}

        for chunk in self.chunks:
            mid_surfaces_dict[chunk] = self.surfaces[chunk][self.middle_depth]

        return mid_surfaces_dict

    def get_surfaces_across_chunks(self, depth):
        return [self.surfaces[s][depth] for s in self.chunks]
