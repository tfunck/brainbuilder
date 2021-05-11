
output=${1-"/project/def-aevans/tfunck/output/"}
#for fn in `ls ${output}/*_*_*/*mm//4_nonlinear_2d/tfm/batch*sh` ; do 
echo $output

for fn in `find ${output} -name "batch*sh"` ; do 
    y=`basename $fn | sed 's/batch_//' | sed 's/.sh//'`
    out_fn="`dirname $fn`/y-${y}.nii.gz"
    if [[ ! -f $out_fn ]]; then
        echo $out_fn
        sed -i  's/-c \[1000x800x600x400x200\]/-c 1000x800x600x400x200/g' $fn
        sh $fn 
    fi
done
