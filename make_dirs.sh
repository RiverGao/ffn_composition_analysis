for i in {1..9}
do
	for dir in act_out mlp_out scores_composition sortings_composition
	do
		mkdir -p $dir/lpp_en/section$i
	done
done

