# AutoBots Implementation

This repository adds the official implementation of our contrastive and consistency causality-based regularizers 
to the [AutoBots](https://arxiv.org/abs/2104.00563) model.

### Getting Started

1. Create a python 3.7 environment. 
2. Run `pip install -r requirements.txt`

That should be it!

### Prepare the dataset

Please follow the instructions in the [Synthetic Dataset](../SynthDataset/README.md) to download or generate the synthetic dataset.
You should have the following directory structure:

```
AutoBots
└── data
   ├── train
   │   ├── scene_0.pkl
   │   ├── scene_1.pkl
   │   ├── ...
   │   └── scene_n.pkl
   ├── val
   │   ├── scene_0.pkl
   │   ├── scene_1.pkl
   │   ├── ...
   │   └── scene_v.pkl
   └── test
       ├── scene_0.pkl
       ├── scene_1.pkl
       ├── ...
       └── sceme_t.pkl
```
### Training the AutoBot model

To train the AutoBot model on the synthetic data, without any regularization, run the following command:

```shell
(($scenes=1000))
python train.py \
      --exp-id synth-normal-scenes:"$scenes" \
      --seed 1 \
      --dataset synth \
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
      --num-epochs ((150*75000/$scenes)) \
      --learning-rate 0.00075 \
      --learning-rate-sched $((10*75000/$scenes)) $((20*75000/$scenes)) $((30*75000/$scenes)) $((40*75000/$scenes)) $((50*75000/$scenes)) \
      --save-every $((10*75000/$scenes)) \
      --val-every $((10*75000/$scenes)) \
      --evaluate_causal \
      --train_data_size "$scenes"
```

Note that you can train on different number of scenarios by changing the number 1000 in `(($scenes=1000))`.

To add the contrastive regularizer, add the following arguments:

```shell
      --reg-type contrastive \
      --contrastive-weight 20000 
```

And to add the consistency regularizer, add the following arguments:

```shell
      --reg-type consistency \
      --consistency-weight 10 
```

Finally, to have more stable training, we trained the baseline model for `10*75000/$scenes` epochs
and then added the regularizers for the rest of the training. To do so, add the following arguments:

```shell
      --weight-path results/synth/Autobot_ego_C1_H128_E2_D2_TXH384_NH16_EW40_KLW20_NormLoss_synth-normal-scenes:"$scenes"_s1/models_$((10*75000/$scenes)).pth \
      --start-epoch $((10*75000/$scenes)) 
```

In the above commands, we added the term `75000/$scenes` to the default parameters to have the same number of iterations
when training on different sizes of data, for fair comparison.
## Reference

If you use this repository, please cite our work:

```
@article{liu2023social,
  title={Social Causality: Towards Causally-aware Neural Representations of Multi-agent Interactions},
  author={Liu, Yeujiang and Rahimi, Ahmad and Rajic, Frano and Luan, Po-Chien and Alahi, Alexandre},
  journal={arXiv preprint arXiv:2106.01901},
  year={2023}
}
```


