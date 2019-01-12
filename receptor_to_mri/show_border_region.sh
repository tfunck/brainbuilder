

mincmath -clobber -add MR1_R_slab_1_receptor_rsl_lin.mnc MR1_R_slab_2_receptor_rsl_lin.mnc MR1_R_slab_1+2.mnc
mincmath -clobber -add MR1_R_slab_2_receptor_rsl_lin.mnc MR1_R_slab_3_receptor_rsl_lin.mnc MR1_R_slab_2+3.mnc
mincmath -clobber -add MR1_R_slab_3_receptor_rsl_lin.mnc MR1_R_slab_4_receptor_rsl_lin.mnc MR1_R_slab_3+4.mnc
mincmath -clobber -add MR1_R_slab_4_receptor_rsl_lin.mnc MR1_R_slab_5_receptor_rsl_lin.mnc MR1_R_slab_4+5.mnc
mincmath -clobber -add MR1_R_slab_5_receptor_rsl_lin.mnc MR1_R_slab_6_receptor_rsl_lin.mnc MR1_R_slab_5+6.mnc

mincresample -clobber -transformation  mni_to_virtual_slab_1.xfm  -tfm_input_sampling   MR1_R_slab_1+2.mnc MR1_R_slab_1+2_space-slab-1.mnc
mincresample -clobber -transformation  mni_to_virtual_slab_2.xfm  -tfm_input_sampling   MR1_R_slab_1+2.mnc MR1_R_slab_1+2_space-slab-2.mnc

mincresample -clobber -transformation  mni_to_virtual_slab_2.xfm  -tfm_input_sampling   MR1_R_slab_2+3.mnc MR1_R_slab_2+3_space-slab-2.mnc
mincresample -clobber -transformation  mni_to_virtual_slab_3.xfm  -tfm_input_sampling   MR1_R_slab_2+3.mnc MR1_R_slab_2+3_space-slab-3.mnc

mincresample -clobber -transformation  mni_to_virtual_slab_3.xfm  -tfm_input_sampling   MR1_R_slab_3+4.mnc MR1_R_slab_3+4_space-slab-3.mnc
mincresample -clobber -transformation  mni_to_virtual_slab_4.xfm  -tfm_input_sampling   MR1_R_slab_3+4.mnc MR1_R_slab_3+4_space-slab-4.mnc

mincresample -clobber -transformation  mni_to_virtual_slab_4.xfm  -tfm_input_sampling   MR1_R_slab_4+5.mnc MR1_R_slab_4+5_space-slab-4.mnc
mincresample -clobber -transformation  mni_to_virtual_slab_5.xfm  -tfm_input_sampling   MR1_R_slab_4+5.mnc MR1_R_slab_4+5_space-slab-5.mnc

mincresample -clobber -transformation  mni_to_virtual_slab_5.xfm  -tfm_input_sampling   MR1_R_slab_5+6.mnc MR1_R_slab_5+6_space-slab-5.mnc
mincresample -clobber -transformation  mni_to_virtual_slab_6.xfm  -tfm_input_sampling   MR1_R_slab_5+6.mnc MR1_R_slab_5+6_space-slab-6.mnc
