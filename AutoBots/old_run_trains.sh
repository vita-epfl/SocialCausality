#!/bin/bash
#SBATCH --nodes 1
#SBATCH --cpus-per-task 10
#SBATCH --ntasks 1
#SBATCH --account=vita
#SBATCH --mem 50G
#SBATCH --time 4:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

module load gcc/11.3.0 python/3.10.4
source venv/bin/activate

((scenes=$2))

if [ "$1" == "normal" ]
then
  echo "training on $2 scenes normally"
  python train.py \
        --exp-id ft \
        --seed 1 \
        --dataset synth \
        --save-dir /scratch/izar/arahimi/autobots_causality \
        --dataset-path data \
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
        --num-epochs $((150*75000/$scenes)) \
        --learning-rate 0.00075 \
        --learning-rate-sched $((10*75000/$scenes)) $((20*75000/$scenes)) $((30*75000/$scenes)) $((40*75000/$scenes)) $((50*75000/$scenes)) \
        --save-every $((1*75000/$scenes)) \
        --val-every $((1*75000/$scenes)) \
        --evaluate_causal \
        --train_data_size "$scenes" \
        --weight-path /scratch/izar/arahimi/autobots_causality/results/synth/Autobot_ego_NScene:"$scenes"_regType:None_reproduce_s1/models_$((140*75000/$scenes)).pth \
        --start-epoch $((140*75000/$scenes))
fi

if [ "$1" == "contrastive" ]
then
  echo "training on $2 scenes with contrastive reg with weight $3"
  python train.py \
        --exp-id ft \
        --seed 1 \
        --dataset synth \
        --save-dir /scratch/izar/arahimi/autobots_causality \
        --dataset-path data \
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
        --num-epochs $((150*75000/$scenes)) \
        --learning-rate 0.00075 \
        --learning-rate-sched $((10*75000/$scenes)) $((20*75000/$scenes)) $((30*75000/$scenes)) $((40*75000/$scenes)) $((50*75000/$scenes)) \
        --save-every $((1*75000/$scenes)) \
        --val-every $((1*75000/$scenes)) \
        --evaluate_causal \
        --train_data_size "$scenes" \
        --reg-type contrastive \
        --contrastive-weight "$3" \
        --weight-path /scratch/izar/arahimi/autobots_causality/results/synth/Autobot_ego_NScene:"$scenes"_regType:None_reproduce_s1/models_$((140*75000/$scenes)).pth \
        --start-epoch $((140*75000/$scenes))
fi

if [ "$1" == "consistency" ]
then
  echo "training on $2 scenes with consistency reg with weight $3"
  python train.py \
        --exp-id rep_ft \
        --seed 1 \
        --dataset synth \
        --save-dir /scratch/izar/arahimi/autobots_causality \
        --dataset-path data \
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
        --num-epochs $((150*75000/$scenes)) \
        --learning-rate 0.00075 \
        --learning-rate-sched $((10*75000/$scenes)) $((20*75000/$scenes)) $((30*75000/$scenes)) $((40*75000/$scenes)) $((50*75000/$scenes)) \
        --save-every $((1*75000/$scenes)) \
        --val-every $((1*75000/$scenes)) \
        --evaluate_causal \
        --train_data_size "$scenes" \
        --reg-type consistency \
        --consistency-weight $3 \
        --weight-path /scratch/izar/arahimi/autobots_causality/results/synth/Autobot_ego_NScene:"$scenes"_regType:None_reproduce_s1/models_$((140*75000/$scenes)).pth \
        --start-epoch $((140*75000/$scenes))
fi