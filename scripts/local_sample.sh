#!/bin/bash 

model_f=$1 

gt_file=$model_f
gt_dir="$(dirname "${gt_file}")"
echo $gt_dir
bin=0
if [[ $gt_file == *"mnist"* ]]; then
    data="mnist"
    tar="datasets/images/mnist_test.npy" 
    bin=1
elif [[ $gt_file == *"cifar"* ]]; then
    data="cifar"
    tar='datasets/images/cifar_test.pkl_stats_cached.pkl'
elif [[ $gt_file == *"celeba"* ]]; then
    data="celeba"
    tar='datasets/images/celeba_valid.pkl_stats_cached.pkl' 
elif [[ $gt_file == *"omni"* ]]; then
    data="omni"
    tar="datasets/images/omni_test.npy" 
    bin=1
else 
    echo "unknow data for $gt_file" 
    exit
fi
echo "eval data: $data"

ns=10000
srun -p $PARTI --mem=8G --gres=gpu:1 -J $model_f \
    python train_vae.py --comet 0 --eval_only 1 --eval_fid_only 1 --resume $1  batch_size 50 && \
echo $gt_dir
for entry in `find $gt_dir -maxdepth 1 -name "*10k_sample*npy"`
do
    #echo "$entry"
    sample=$entry 
    T=${sample//.npy/.pkl}
    if [[ -f "$T" ]]; then 
        T=''
    else 
	    if [ ! -r $sample ]; then 
		    echo "invalid link $sample"
		    continue 
	    fi 
        echo "eval the npy file $sample "
        srun -p $PARTI --mem=8G --gres=gpu:1 -J $sample \
            python tool/pytorch-fid/fid_score.py \
            --binarized $bin \
            --batch_size 400 --gpu 1 --path $sample $tar 
    fi
done
