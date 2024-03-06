subj_start=$1
subj_end=$2

for j in `seq $subj_start $subj_end`
do
    echo "subject $j"
    python regression_intersubject.py $j > logs/intersubject/log_subj$j.txt
done
