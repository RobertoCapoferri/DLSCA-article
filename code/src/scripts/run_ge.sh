for target in SBOX_OUT HW_SO
do
for mode in fixed random
do
CUDA_VISIBLE_DEVICES=0 python3 ./ge_plots.py $HOME/results/$target/1d/${mode}_key &
CUDA_VISIBLE_DEVICES=1 python3 ./ge_plots.py $HOME/results/$target/2d/${mode}_key &
wait
done
done
