#!/bin/bash
#SBATCH --cpus-per-task 20
#SBATCH --mem 90G
#SBATCH --time 20:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --qos=gpu

module load gcc/11.3.0 python/3.10.4
source venv/bin/activate

((scenes=$2))


#for curv in 0.5 1 1.5 2; do
#  for size in 1000 2000 5000 10000 20000 50000 100000; do
#    for weight in 500 1000 2000 3000; do
#      sbatch run_train.sh consistency "$size" "$weight" "$curv"
#    done
#    for weight in 50 200 500 1000; do
#      sbatch run_train.sh contrastive "$size" "$weight" "$curv"
#    done
#  done
#done


if [ "$1" == "normal" ]
then
  echo "training on $2 scenes normally on $3 curvature cut-off"
  python train.py \
        --exp-id v2_"$3"_bl \
        --seed 1 \
        --dataset synth \
        --save-dir /scratch/izar/arahimi/autobots_causality \
        --dataset-path /work/vita/ahmad_rh/synth_v2_"$3" \
        --model-type Autobot-Ego \
        --num-modes 1 \
        --hidden-size 128 \
        --num-encoder-layers 2 \
        --num-decoder-layers 2 \
        --dropout 0.1 \
        --entropy-weight 40.0 \
        --kl-weight 20.0 \
        --use-FDEADE-aux-loss True \
        --tx-hidden-size 384 \
        --batch-size 16 \
        --num-epochs $((150*100000/$scenes)) \
        --learning-rate 0.00075 \
        --learning-rate-sched $((10*100000/$scenes)) $((20*100000/$scenes)) $((30*100000/$scenes)) $((40*100000/$scenes)) $((50*100000/$scenes)) \
        --save-every $((10*100000/$scenes)) \
        --val-every $((10*100000/$scenes)) \
        --evaluate_causal \
        --train_data_size "$scenes"
#        --weight-path /scratch/izar/arahimi/autobots_causality/results/synth/Autobot_ego_NScene:"$scenes"_regType:None_v2_bl_t_s1/models_$((140*100000/$scenes)).pth \
#        --start-epoch $((140*100000/$scenes))
fi

if [ "$1" == "contrastive" ]
then
  echo "training on $2 scenes with contrastive reg with weight $3"
  python train.py \
        --exp-id v2_"$4"_ft \
        --seed 1 \
        --dataset synth \
        --save-dir /scratch/izar/arahimi/autobots_causality \
        --dataset-path /work/vita/ahmad_rh/synth_v2_"$4" \
        --model-type Autobot-Ego \
        --num-modes 1 \
        --hidden-size 128 \
        --num-encoder-layers 2 \
        --num-decoder-layers 2 \
        --dropout 0.1 \
        --entropy-weight 40.0 \
        --kl-weight 20.0 \
        --use-FDEADE-aux-loss True \
        --tx-hidden-size 384 \
        --batch-size 16 \
        --num-epochs $((150*100000/$scenes)) \
        --learning-rate 0.00075 \
        --learning-rate-sched $((10*100000/$scenes)) $((20*100000/$scenes)) $((30*100000/$scenes)) $((40*100000/$scenes)) $((50*100000/$scenes)) \
        --save-every $((1*100000/$scenes)) \
        --val-every $((1*100000/$scenes)) \
        --evaluate_causal \
        --train_data_size "$scenes" \
        --reg-type contrastive \
        --contrastive-weight "$3" \
        --weight-path /scratch/izar/arahimi/autobots_causality/results/synth/Autobot_ego_NScene:"$scenes"_regType:None_v2_"$4"_bl_s1/models_$((140*100000/$scenes)).pth \
        --start-epoch $((140*100000/$scenes))
fi

if [ "$1" == "consistency" ]
then
  echo "training on $2 scenes with consistency reg with weight $3"
  python train.py \
        --exp-id v2_"$4"_ft \
        --seed 1 \
        --dataset synth \
        --save-dir /scratch/izar/arahimi/autobots_causality \
        --dataset-path /work/vita/ahmad_rh/synth_v2_"$4" \
        --model-type Autobot-Ego \
        --num-modes 1 \
        --hidden-size 128 \
        --num-encoder-layers 2 \
        --num-decoder-layers 2 \
        --dropout 0.1 \
        --entropy-weight 40.0 \
        --kl-weight 20.0 \
        --use-FDEADE-aux-loss True \
        --tx-hidden-size 384 \
        --batch-size 16 \
        --num-epochs $((150*100000/$scenes)) \
        --learning-rate 0.00075 \
        --learning-rate-sched $((10*100000/$scenes)) $((20*100000/$scenes)) $((30*100000/$scenes)) $((40*100000/$scenes)) $((50*100000/$scenes)) \
        --save-every $((1*100000/$scenes)) \
        --val-every $((1*100000/$scenes)) \
        --evaluate_causal \
        --train_data_size "$scenes" \
        --reg-type consistency \
        --consistency-weight "$3" \
        --weight-path /scratch/izar/arahimi/autobots_causality/results/synth/Autobot_ego_NScene:"$scenes"_regType:None_v2_"$4"_bl_s1/models_$((140*100000/$scenes)).pth \
        --start-epoch $((140*100000/$scenes))
fi