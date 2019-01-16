ii1=$1
ii2=$2
slab=$3
itr=$4
label=$5
acc_fn=$6
acc_all_fn=$7
clobber=$8

if [[ ! -f $acc_fn || $clobber == 1 ]]; then
    if [[ "${ii1##*.}" == "gz" ]]; then
        gunzip -kf $ii1
        vol1=${ii1%.*}
    else 
        vol1=$ii1
    fi

    if [[ "${ii2##*.}" == "gz" ]]; then
        gunzip -kf $ii1
        vol2=${ii2%.*}
    else 
        vol2=$ii2
    fi

    printf "\t\tCalculate $s Categorical Accuracy\n" "$label"
    acc=`python acc.py $vol1 $vol2`
    printf "%d,%d,%s,%f\n" "${slab}" "$itr" "$label" "$acc" > $acc_fn
    if [[ "$vol1" != "$ii1" ]]; then
        rm $vol1
    fi
    if [[ "$vol2" != "$ii2" ]]; then
        rm $vol2
    fi

fi

cat $acc_fn >> $acc_all_fn

