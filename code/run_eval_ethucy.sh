#!/bin/bash
#SBATCH --cpus-per-task 10
#SBATCH --mem 90G
#SBATCH --time 1:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --qos=gpu

module load gcc/11.3.0 python/3.10.4
source venv/bin/activate

python evaluate.py --models-path /scratch/izar/arahimi/autobots_causality/results/synth/Autobot_ego_NScene:100000_regType:None_v2_"$1"_bl_s1/models_149.pth --dataset synth --dataset-path /work/vita/ahmad_rh/synth_v2_"$1"/ --evaluate_causal

python evaluate.py --models-path /scratch/izar/arahimi/autobots_causality/results/synth/Autobot_ego_NScene:100000_regType:None_v2_"$1"_bl_s1/models_149.pth --dataset trajnet++ --dataset-path /work/vita/ahmad_rh/eth_ucy/eth
python evaluate.py --models-path /scratch/izar/arahimi/autobots_causality/results/synth/Autobot_ego_NScene:100000_regType:None_v2_"$1"_bl_s1/models_149.pth --dataset trajnet++ --dataset-path /work/vita/ahmad_rh/eth_ucy/hotel
python evaluate.py --models-path /scratch/izar/arahimi/autobots_causality/results/synth/Autobot_ego_NScene:100000_regType:None_v2_"$1"_bl_s1/models_149.pth --dataset trajnet++ --dataset-path /work/vita/ahmad_rh/eth_ucy/univ
python evaluate.py --models-path /scratch/izar/arahimi/autobots_causality/results/synth/Autobot_ego_NScene:100000_regType:None_v2_"$1"_bl_s1/models_149.pth --dataset trajnet++ --dataset-path /work/vita/ahmad_rh/eth_ucy/zara1
python evaluate.py --models-path /scratch/izar/arahimi/autobots_causality/results/synth/Autobot_ego_NScene:100000_regType:None_v2_"$1"_bl_s1/models_149.pth --dataset trajnet++ --dataset-path /work/vita/ahmad_rh/eth_ucy/zara2