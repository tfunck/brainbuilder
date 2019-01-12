   
file_check(){
    if [[ ! -f $1 ]] ; then
        echo Error: did not create $1
        exit 1
    fi
}

downsample() {
        vol=$1
        s=$2
        fwhm_1=$3
        clobber_1=$4
        base=`echo $vol | sed 's/.mnc.gz//'`
        blur=`echo $vol | sed 's/.mnc.gz/_blur.mnc.gz/'`
        blur_rsl=`echo $vol | sed 's/.mnc.gz/_blur_rsl.mnc.gz/'`

        if [[ ! -f $blur || $clobber_1 == 1 ]]; then
            echo
            echo Smoothing $base.mnc.gz with $fwhm_1 FWHM 
            echo
            mincblur -clobber -fwhm $fwhm_1 $vol $base
            gzip -f ${base}"_blur.mnc"
            file_check $blur
            
        fi
        
        if [[ ! -f $blur_rsl || $clobber_1 == 1 ]]; then
            echo
            echo Resampling $blur to step size $s
            echo
            z=`mincinfo -dimlength zspace $vol`
            y=`mincinfo -dimlength yspace $vol`
            x=`mincinfo -dimlength xspace $vol`
            zstep=`mincinfo -attvalue zspace:step $vol`
            ystep=`mincinfo -attvalue yspace:step $vol`
            xstep=`mincinfo -attvalue xspace:step $vol`
            zstart=`mincinfo -attvalue zspace:start $vol`
            ystart=`mincinfo -attvalue yspace:start $vol`
            xstart=`mincinfo -attvalue xspace:start $vol`

            zmax=`echo "(($z * $zstep) - $zstart ) / $s " | bc`
            ymax=`echo "(($y * $ystep) - $ystart ) / $s " | bc`
            xmax=`echo "(($x * $xstep) - $xstart ) / $s " | bc`

            echo Start: $zstart $ystart $xstart
            echo Step:  $zstep $ystep $xstep
            echo Max:   $z $y $x
            echo New Max:  $zmax $ymax $xmax
            mincresample -clobber -step $step -nelements $zmax $ymax $xmax  $blur $blur_rsl
            file_check $blur_rsl
        fi
}

base_mask="input/mask/mr1_brain_mask_r_space-slab-1_no_ventricles.mnc.gz"
base_mri="input/mri/mr1_mri_T1_rsl_rotate_for_slab_1R_58_to_24.mnc.gz"
output_tag="output/receptor_volume_tag.mnc.gz"
output_lin="output/receptor_volume_lin.mnc.gz"
output_nl="output/receptor_volume_nl.mnc.gz"

######################
# Set up directories #
######################
mkdir -p input output 
cd input
mkdir -p tag mri receptor mask
cd ../output
mkdir -p transforms receptor blur rsl masks truncate
cd ..

##################
# Set parameters #
##################
slab_list="1 2 3 4 5 6"
#slab_list="5"
fwhm=1
s=1
step="$s $s $s"
clobber=$1
clobber=${clobber:-0}
echo $clobber

#################
# Run alignment #
#################

for i in $slab_list ; do
    ##########
    # Inputs #
    ##########
    rec_vol="input/receptor/MR1_R_slab_${i}_receptor.mnc.gz" 
    tag_fn="input/tag/slab_${i}.tag"
    mri_vol_0="input/mri/"mr1_mri_T1_rsl_rotate_for_slab_${i}R_*[0-9].mnc.gz

    ###########
    # Outputs #
    ###########
    #Brain masks
    mri_vol=`ls output/truncate/mr1_mri_T1_rsl_rotate_for_slab_${i}R_*[0-9]_truncated.mnc.gz`
    rec_mask="output/masks/MR1_R_slab_${i}_receptor_brain_mask.mnc.gz"
    mri_mask="output/truncate/mr1_brain_mask_slab_${i}.mnc.gz"

    xfm_tag_fn="output/transforms/tag_lin_${i}.xfm"
    xfm_mri_to_virtual_slab="output/transforms/mri_to_virtual_slab_${i}.xfm"
    xfm_virtual_slab_to_mri="output/transforms/virtual_slab_to_mri_${i}.xfm"

    tfm_final="output/transforms/tfm_mri_to_rec_slab-${i}_final.xfm"
    #Blurred and Downsampled Files
    rec_base="MR1_R_slab_${i}_receptor"
    rec_blur="output/blur/MR1_R_slab_${i}_receptor_blur.mnc.gz"
    rec_blur_rsl="output/rsl/MR1_R_slab_${i}_receptor_blur_rsl.mnc.gz"

    mri_base=`echo $mri_vol | sed 's/.mnc.gz//'`
    mri_blur=`echo $mri_vol | sed 's/.mnc.gz/_blur.mnc.gz/'`
    mri_blur_rsl=`echo $mri_vol | sed 's/.mnc.gz/_blur_rsl.mnc.gz/'`

    #Final receptor volumes
    rec_rsl_mri="output/receptor/"`basename $rec_vol | sed 's/.mnc.gz/_space-mri.mnc.gz/'`
    rec_rsl_slab="output/receptor/"`basename $rec_vol | sed 's/.mnc.gz/_space-slab.mnc.gz/'`

    ##########################
    # Align slab to base MRI #
    ##########################
    
    if [[ ! -f $xfm_tag_fn || $clobber == 1 ]]; then
        tagtoxfm -clobber -lsq6 $tag_fn $xfm_tag_fn
    fi

    if [[ ! -f $xfm_mri_to_virtual_slab || ! -f $xfm_mri_to_virtual_slab || $clobber == 1 ]] ; then
      minctracc -clobber -lsq6 -est_translation $base_mri $mri_vol_0 $xfm_mri_to_virtual_slab
      xfminvert -clobber $xfm_mri_to_virtual_slab $xfm_virtual_slab_to_mri
    fi
        if [[ ! -f $tfm_final || $clobber == 1 ]]; then
        xfmconcat -clobber  $xfm_mri_to_virtual_slab  $xfm_tag_fn $tfm_final
    fi

    ################
    # Truncate MRI #
    ################
    if [[ ! -f $mri_vol || $clobber == 1 ]]; then
       start_end_string=`echo $mri_vol_0 | cut -d '.' -f1 | awk '{split($0,ar,"_"); print ar[11] " " ar[9]  }' `
       python3 truncate.py $start_end_string $mri_vol_0 $mri_vol
    fi

    #####################
    # Create brain mask #
    #####################
    if [[ ! -f $rec_mask || $clobber == 1 ]] ; then
        mincmath -clobber -ge -const 0.1 $rec_blur_rsl $rec_mask
    fi
  
    if [[ ! -f $mri_mask || $clobber == 1 ]]; then
       start_end_string=`echo $base_mask | cut -d '.' -f1 | awk '{split($0,ar,"_"); print ar[11] " " ar[9]  }' `
       python3 truncate.py $start_end_string $base_mask $mri_mask
    fi
    
    if [[ ! -f $rec_rsl_slab || $clobber == 1 ]]; then
        mincresample -clobber -transformation  $xfm_tag_fn -invert_transform -like $mri_vol $rec_blur_rsl $rec_rsl_slab
    fi
    #clobber=1
    if [[ ! -f $rec_rsl_mri || $clobber == 1 ]]; then
        mincresample -clobber -transformation  $tfm_final -invert_transform -like $base_mri $rec_blur $rec_rsl_mri
    fi
    #clobber=0

    sum_list="$rec_rsl_mri $sum_list"
done

if [[ ! -f $output_tag || $clobber == 1 ]]; then
    minccalc -clobber -expr "sum(A)" $sum_list $output_tag
fi

exit 0





w1=1
sim1=0.10
stiff=0.962962

tag_to_lin="output/transforms/tfm_tag_to_lin.xfm"
lin_to_nl="output/transforms/tfm_lin_to_nl.xfm"
tag_blur="output/blur/receptor_volume_tag_blur.mnc.gz"
tag_mask="output/masks/receptor_volume_tag_mask.mnc.gz"

if [[ ! -f $tag_blur || $clobber == 1 ]]; then
    base=`echo $tag_blur | sed 's/_blur.mnc.gz//'`
    base_0=`echo $tag_blur | sed 's/.gz//'`
    mincblur -clobber -fwhm 4 $output_tag $base
    gzip -f $base_0 
fi

if [[ ! -f $tag_mask || $clobber == 1 ]]; then
    mincmath -clobber -ge -const 0.1 $output_tag $tag_mask
fi
base="-clobber -nmi -est_center -est_translation -est_scales"

clobber=1
if [[ ! -f $tag_to_lin || $clobber == 1 ]] ; then
   minctracc $base -lsq12 -step 4 4 4 -source_mask $base_mask -model_mask $tag_mask $base_mri $output_tag $tag_to_lin
   #minctracc $base -transformation temp_1.xfm -lsq9 -step 2 2 2 -source_mask $base_mask -model_mask $tag_mask $base_mri $tag_blur temp_2.xfm #$tag_to_lin
   #minctracc $base -transformation temp_2.xfm  -lsq12 -step 2 2 2 -source_mask $base_mask -model_mask $tag_mask $base_mri $tag_blur $tag_to_lin
fi

#if [[ ! -f $lin_to_nl || $clobber == 1 ]]; then
#    minctracc $base -nonlinear -iterations 20 -stiffness 0.96 -step 2 2 2 -weight 1 -similarity_cost_ratio .2 -transformation $tag_to_lin -source_mask $base_mask -model_mask $tag_mask $base_mri $tag_blur $lin_to_nl
#fi

if [[ ! -f $output_lin || $clobber == 1 ]]; then
    mincresample -clobber -transformation $tag_to_lin -invert_transform -like $base_mri $output_tag $output_lin
fi
#if [[ ! -f $output_nl || $clobber == 1 ]]; then
#    mincresample -clobber -transformation $lin_to_nl -invert_transform -like $base_mri $output_tag $output_nl
#fi
register $output_lin $base_mri 
#register $output_nl $base_mri 
exit 0
for i in $slab_list ; do
    ##########
    # Inputs #
    ##########
    #rec_vol="input/receptor/MR1_R_slab_${i}_receptor.mnc.gz" 
    rec_vol=$output_nl
    tag_fn="slab_${i}.tag"
    mri_vol_0="input/mri/"mr1_mri_T1_rsl_rotate_for_slab_${i}R_*[0-9].mnc.gz

    ###########
    # Outputs #
    ###########
    #Brain masks
    mri_vol=`ls output/truncate/mr1_mri_T1_rsl_rotate_for_slab_${i}R_*[0-9]_truncated.mnc.gz`
    rec_mask="output/masks/MR1_R_slab_${i}_receptor_brain_mask.mnc.gz"
    mri_mask="output/truncate/mr1_brain_mask_slab_${i}.mnc.gz"
   
    xfm_tag_fn="output/transforms/tag_lin_${i}.xfm"
    xfm_mri_to_virtual_slab="output/transforms/mri_to_virtual_slab_${i}.xfm"
    xfm_virtual_slab_to_mri="output/transforms/virtual_slab_to_mri_${i}.xfm"

    #Transforms
    tfm_1="output/transforms/tfm_mri_to_rec_slab-${i}_iter-1.xfm"
    #tfm_2="output/transforms/tfm_mri_to_rec_slab-${i}_iter-2.xfm"
    #tfm_3="output/transforms/tfm_mri_to_rec_slab-${i}_iter-3.xfm"
    #tfm_4="output/transforms/tfm_mri_to_rec_slab-${i}_iter-4.xfm"

    #Blurred and Downsampled Files
    rec_base="MR1_R_slab_${i}_receptor"
    rec_blur="output/blur/MR1_R_slab_${i}_receptor_blur.mnc.gz"
    rec_blur_rsl="output/rsl/MR1_R_slab_${i}_receptor_blur_rsl.mnc.gz"

    mri_base=`echo $mri_vol | sed 's/.mnc.gz//'`
    mri_blur=`echo $mri_vol | sed 's/.mnc.gz/_blur.mnc.gz/'`
    mri_blur_rsl=`echo $mri_vol | sed 's/.mnc.gz/_blur_rsl.mnc.gz/'`

    #Final receptor volumes
    rec_rsl_mri="output/receptor/"`basename $rec_vol | sed 's/.mnc.gz/_space-mri.mnc.gz/'`
    rec_rsl_slab="output/receptor/"`basename $rec_vol | sed 's/.mnc.gz/_space-slab.mnc.gz/'`
    
    #######################
    # Iterative alignment #
    #######################  
    if [[ ! -f $tfm_1 || $clobber == 1 ]]; then
        minctracc $base   -transformation $xfm_tag_fn -tol 0.0001 -lsq3 -simplex 1 -step 1 1 1 -source_mask $mri_mask -model_mask $rec_mask $mri_vol $rec_blur_rsl $tfm_1 
    fi
    

    #mincresample -clobber -transformation $xfm_mri_to_virtual_slab -invert_transform -like $mri_vol $mri_vol_0 test.mnc
    #register test.mnc $base_mri

done

