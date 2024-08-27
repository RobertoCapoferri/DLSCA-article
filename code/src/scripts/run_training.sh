for target in SBOX_OUT HW_SO
do
for mode in fixed random
do
for ptx in 0 1
do
CUDA_VISIBLE_DEVICES=0 python3 ./training.py $ptx $mode $target D1 &
CUDA_VISIBLE_DEVICES=1 python3 ./training.py $ptx $mode $target D1 D2 &
wait
done
done
done