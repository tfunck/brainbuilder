import json


def get_filenames(section_info, out_dir: str, resolution_list: list) -> dict:
    """Create a dictionary with all of the filenames that will be used in the reconstruction

    param: section_info: dataframe with all of the section information
    param: out_dir: directory where the output will be saved
    param: resolution_list: list of resolutions that will be used in the reconstruction
    return files: return dictionary with all filenames
    """
    files = {}
    for brain, bdf in section_info.groupby(["brain"]):
        files[brain] = {}
        for hemi, hdf in bdf.groupby(["hemisphere"]):
            files[brain][hemi] = {}
            for chunk, sdf in hdf.groupby(["chunk"]):
                files[brain][hemi][chunk] = {}

                for resolution_itr, resolution in enumerate(resolution_list):
                    cdict = {}  # current dictionary
                    cdict["brain"] = brain
                    cdict["hemi"] = hemi
                    cdict["chunk"] = chunk

                    # Directories
                    cdict[
                        "cur_out_dir"
                    ] = f"{out_dir}/{brain}_{hemi}_{chunk}/{resolution}mm/"
                    cdict["seg_dir"] = "{}/2_segment/".format(cdict["cur_out_dir"])
                    cdict["srv_dir"] = f"{out_dir}/{brain}_{hemi}_{chunk}/srv/"
                    cdict["align_to_mri_dir"] = "{}/3_align_chunk_to_mri/".format(
                        cdict["cur_out_dir"]
                    )
                    cdict["nl_2d_dir"] = "{}/4_nonlinear_2d".format(
                        cdict["cur_out_dir"]
                    )

                    # Filenames
                    cdict["seg_rsl_fn"] = "{}/{}_{}_{}_seg_{}mm.nii.gz".format(
                        cdict["seg_dir"], brain, hemi, chunk, resolution
                    )
                    cdict["srv_rsl_fn"] = (
                        cdict["srv_dir"]
                        + f"/{brain}_{hemi}_{chunk}_mri_gm_{resolution}mm.nii.gz"
                    )
                    cdict["srv_crop_rsl_fn"] = (
                        cdict["srv_dir"]
                        + f"/{brain}_{hemi}_{chunk}_mri_gm_crop_{resolution}mm.nii.gz"
                    )

                    # if resolution_itr <= max_3d_itr  :
                    cdict[
                        "rec_3d_rsl_fn"
                    ] = "{}/{}_{}_{}_{}mm_rec_space-mri.nii.gz".format(
                        cdict["align_to_mri_dir"], brain, hemi, chunk, resolution
                    )
                    cdict[
                        "srv_3d_rsl_fn"
                    ] = "{}/{}_{}_{}_{}mm_mri_gm_space-rec.nii.gz".format(
                        cdict["align_to_mri_dir"], brain, hemi, chunk, resolution
                    )
                    cdict["align_to_mri_dir"]
                    cdict[
                        "manual_alignment_points"
                    ] = f"{manual_dir}/3d/{brain}_{hemi}_{chunk}_points.txt"
                    cdict[
                        "manual_alignment_affine"
                    ] = f"{manual_dir}/3d/{brain}_{hemi}_{chunk}_manual_affine.mat"
                    cdict[
                        "nl_3d_tfm_fn"
                    ] = f'{cdict["align_to_mri_dir"]}/{brain}_{hemi}_{chunk}_rec_to_mri_{resolution}mm_SyN_CC_Composite.h5'
                    cdict[
                        "nl_3d_tfm_inv_fn"
                    ] = f'{cdict["align_to_mri_dir"]}/{brain}_{hemi}_{chunk}_rec_to_mri_{resolution}mm_SyN_CC_InverseComposite.h5'

                    cdict["nl_2d_vol_fn"] = "{}/{}_{}_{}_nl_2d_{}mm.nii.gz".format(
                        cdict["nl_2d_dir"], brain, hemi, chunk, resolution
                    )
                    cdict[
                        "nl_2d_vol_cls_fn"
                    ] = "{}/{}_{}_{}_nl_2d_cls_{}mm.nii.gz".format(
                        cdict["nl_2d_dir"], brain, hemi, chunk, resolution
                    )
                    cdict["chunk_info_fn"] = "{}/{}_{}_{}_{}_chunk_info.csv".format(
                        cdict["cur_out_dir"], brain, hemi, chunk, resolution
                    )
                    cdict[
                        "srv_space_rec_fn"
                    ] = "{}/{}_{}_{}_srv_space-rec_{}mm.nii.gz".format(
                        cdict["nl_2d_dir"], brain, hemi, chunk, resolution
                    )
                    cdict[
                        "srv_iso_space_rec_fn"
                    ] = "{}/{}_{}_{}_srv_space-rec_{}mm_iso.nii.gz".format(
                        cdict["nl_2d_dir"], brain, hemi, chunk, resolution
                    )
                    # if resolution_itr == max_3d_itr :
                    #    max_3d_cdict=cdict
                    files[brain][hemi][chunk][resolution] = cdict

    json.dump(files, open(files_json, "w+"))
    return files
