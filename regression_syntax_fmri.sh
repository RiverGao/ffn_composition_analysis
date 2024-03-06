subj_start=$1
subj_end=$2

for j in `seq $subj_start $subj_end`
do
    for tree_type in bu td lc
    do
        echo "$tree_type $j"
        python regression_syntax_fmri.py $tree_type $j > logs/syntax-fmri/log_${tree_type}_subj$j.txt
    done
done
