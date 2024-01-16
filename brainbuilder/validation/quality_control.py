def quality_control():
    ###
    ### 6. Quality Control
    ###
    df_list = []
    for ligand, final_ligand_fn in final_ligand_dict.items():
        df_ligand = hemi_df.loc[hemi_df["ligand"] == ligand]
        max_resolution = resolution_list[-1]

        depth = args.depth_list[int((len(args.depth_list) + 2) / 2)]

    for ligand, final_ligand_fn in final_ligand_dict.items():
        ligand_csv_path = f"{interp_dir}/*{ligand}_{max_resolution}mm_l{args.n_depths+2}*{depth}_raw.csv"
        ligand_csv_list = glob(ligand_csv_path)
        if len(ligand_csv_list) > 0:
            ligand_csv = ligand_csv_list[0]
        else:
            print("Errpr: could not find ligand_csv", ligand_csv_path)
            exit(1)

        sphere_mesh_fn = glob(
            f"{interp_dir}/surfaces/surf_{max_resolution}mm_{depth}_sphere_rsl.npz"
        )[0]
        cortex_mesh_fn = glob(
            f"{interp_dir}/surfaces/surf_{max_resolution}mm_{depth}_rsl.npz"
        )[0]

        sphere_mesh_orig_fn = glob(
            f"{interp_dir}/surfaces/surf_{max_resolution}mm_0.0.surf.gii"
        )[0]
        cortex_mesh_orig_fn = glob(
            f"{interp_dir}/surfaces/surf_{max_resolution}mm_0.0.sphere"
        )[0]
        inflation_ratio = calculate_dist_of_sphere_inflation(
            cortex_mesh_orig_fn, sphere_mesh_orig_fn
        )

        float(max_resolution) * inflation_ratio
        tdf = validate_interpolation(
            ligand_csv,
            sphere_mesh_fn,
            cortex_mesh_fn,
            args.qc_dir,
            max_resolution,
            ligand=ligand,
            n_samples=10000,
            clobber=False,
        )
        df_list.append(tdf)
    df = pd.concat(df_list)

    out_r2_fn = f"{args.qc_dir}/interpolation_validation_r2.png"

    plot_r2(df, out_r2_fn)


from utils.mesh_utils import get_edges_from_faces, load_mesh_ext


def calculate_dist_of_sphere_inflation(cortex_fn, sphere_fn):
    coord, faces = load_mesh_ext(cortex_fn)
    coord_sphere, faces_sphere = load_mesh_ext(sphere_fn)
    edges = get_edges_from_faces(faces)

    d0 = np.sqrt(np.sum(np.power(coord[edges[:, 0]] - coord[edges[:, 1]], 2), axis=1))
    d1 = np.sqrt(
        np.sum(
            np.power(coord_sphere[edges[:, 0]] - coord_sphere[edges[:, 1]], 2), axis=1
        )
    )

    edge_ratios = d1 / d0

    ratio_average = np.mean(edge_ratios)
    np.std(edge_ratios)
    return ratio_average
