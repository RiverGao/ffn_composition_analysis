subj_start=$1
subj_end=$2

for j in `seq $subj_start $subj_end`
do
    for model_version in base_shuffled chat_shuffled
    do
        echo "$model_version $j"
        python regression_model_fmri.py $model_version $j > logs/model-fmri/log_${model_version}_subj$j.txt
    done
done
