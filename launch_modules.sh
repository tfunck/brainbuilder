for brain in "MR1" "MR2" "MR3"; do
    for hemi in "R" "L"; do
        for slab in "1" "2" "3" "4" "5" "6"; do
            python3 module.1.py --hemi $hemi --brain $brain --slab $slab --csv autoradiograph_info.csv -o /data1/users/tfunck/output -i data -m line_detection_model_lin/model_spi-2_mask-False_rot-True_loss-0.017.hdf5  -l line_detection_model_lin/data.h5  $1
            break
        done
    done
done

ssh tfunck@login.bic.mni.mcgill.ca 'rm -r temp/qc temp/train_qc'
rsync -Praz /data1/users/tfunck/output/qc tfunck@login.bic.mni.mcgill.ca:temp/
