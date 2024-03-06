cuda=$1
model_version=$2

for i in {1..9}
do
	echo "Section $i"
	CUDA_VISIBLE_DEVICES=$cuda python composition_scores.py $model_version $i
done
