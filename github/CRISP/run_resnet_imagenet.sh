# !/bin/bash

pruning_ratio_t=(0.4 0.3 0.3 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2)
finetune_epochs_t=(10 12 12 16 16 16 16 16 16 16 16 16 16 16)
num_cur_cls=(1 2 3 4 5 6 7 8 9 10 20 50 100 1000)
max_parallel_processes=4  # Set the maximum number of parallel processes

run_background_jobs() {
    local current_jobs=0
    for run_id in {1..5}; do
        for iter in {1..10}; do
            python main.py \
                --gpus 0 \
                --arch resnet50 \
                --conv_type BlockL1Conv \
                --num_classes 1000 \
                --pr_target ${pruning_ratio_t[$((iter - 1))]} \
                --run "$run_id" \
                --first-layer-type NMConv \
                --num_epochs 50 \
                --lr_policy cos \
                --lr 0.01 \
                --N 2 \
                --M 4 \
                --dataset imagenet \
                --finetune_epochs ${finetune_epochs_t[$((iter - 1))]} \
                --momentum 0.9 \
                --weight_decay 5e-4 \
                --train_batch_size 64 \
                --eval_batch_size 32 \
                --num_pref_classes ${num_cur_cls[$((iter - 1))]} \
                --save_dir "./final_experiments/imagenet/resnet/crisp/num_pref_classes_${num_cur_cls[$((iter - 1))]}" 
        done
    done
    # # Wait for all background processes to complete
    # wait
}

#run the function
run_background_jobs