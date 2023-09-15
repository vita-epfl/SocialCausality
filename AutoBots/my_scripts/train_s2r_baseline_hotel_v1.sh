#!/bin/bash
#SBATCH --nodes 1
#SBATCH --cpus-per-task 20
#SBATCH --ntasks 1
#SBATCH --account=vita
#SBATCH --mem 90G
#SBATCH --time 72:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

source ~/miniconda3/bin/activate AutoBots
for weight in 1.0 0.25 0.5 0.75; do
    for low_data in 0.05 0.1 0.2 0.3 0.5 0.8 1.0; do
        python train.py \
            --exp-id s2r_baseline_v1_hotel_${low_data}_pw_${weight} \
            --seed 1 \
            --dataset s2r_baseline \
            --dataset-path ./datasets/eth-ucy/processed_6/hotel/ \
            --model-type Autobot-Ego \
            --num-modes 1 \
            --hidden-size 128 \
            --num-encoder-layers 1 \
            --num-decoder-layers 1 \
            --dropout 0.1 \
            --entropy-weight 40.0 \
            --kl-weight 20.0 \
            --use-FDEADE-aux-loss True \
            --tx-hidden-size 384 \
            --batch-size 64 \
            --num-epochs $(echo "150/$low_data" | bc) \
            --learning-rate 0.00075 \
            --learning-rate-sched $(echo "10/$low_data" | bc) $(echo "20/$low_data" | bc) $(echo "30/$low_data" | bc) $(echo "40/$low_data" | bc) $(echo "50/$low_data" | bc) \
            --save-every $(echo "10/$low_data" | bc) \
            --val-every $(echo "10/$low_data" | bc) \
            --low_data $low_data\
            --save_step_start 10000\
            --save_step_end 20000\
            --save_every_ckp 10\
            --dataset-path-real ./datasets/eth-ucy/processed_6/hotel/\
            --dataset-path-synth /scratch/izar/luan/synth_v1_breaked/ \
            --contrastive-weight 1000 \
            --save-dir /scratch/izar/luan/ \
            --batch-size-cl 64 \
            --sim-pred-weight $weight
        
        # evaluate
        python evaluate_last_ckps.py --dataset-path ./datasets/eth-ucy/processed_6/hotel/ \
            --models-path /scratch/izar/luan/results/s2r_baseline/Autobot_ego_NScene:-1_regType:None_s2r_baseline_v1_hotel_${low_data}_pw_${weight}_s1/ \
            --batch-size 64 --save_step_start 10000 --save_step_end 20000 --output_tag s2r_baseline_v1_hotel_${low_data}_pw_${weight}
    done
done