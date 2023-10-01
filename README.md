## Causally-Aware Representations of Multi-Agent Interactions

<p align="center">
  <img src="docs/background.png" width="600">
</p>

> TL;DR: we investigate causal representation learning in the multi-agent context, from computational formalism to controlled simulations to real-world practice.
> 1. we cast doubt on the notion of non-causal robustness in the previous benchmark, revealing that recent representations tend to underestimate indirect causal effects
> 2. we introduce a causal regularization approach, including a contrastive and a ranking variant, which leads to higher causal awareness and out-of-distribution robustness
> 3. we propose a sim-to-real causal transfer framework, which enables causally-aware representation learning in practical settings even without real-world annotations

<p align="center">
  <img src="docs/method.png" width="800">
</p>

## Getting Started

To install requirements:
```
pip install -r requirements.txt
```

To download dataset:
```
?TODO?
```

To train the [AutoBots](https://openreview.net/forum?id=Dup_dDqkZC5) baseline:
```
?TODO?
```

To run the [data augmentation](https://arxiv.org/abs/2207.03586) baseline:
```
?TODO?
```

To run the contrastive regularization:
```
?TODO?
```

To run the ranking regularization:
```
?TODO?
```

To evaluate on OOD sets:
```
?TODO?
```

## Main Results

Comparison of different methods in terms of causal awareness:
<p align="left">
  <img src="docs/ace.png" height="200">
</p>

Comparison of different methods in terms of out-of-distribution robustness:
<p align="left">
  <img src="docs/ood.png" height="200">
</p>

Comparison of different transfer methods from simulation to the ETH-UCY dataset:
<p align="left">
  <img src="docs/transfer.png" height="200">
</p>

## Citation

If you find this work useful in your research, please consider cite:

```
@article{socialcausality2023,
  title={What If You Were Not There? Learning Causally-Aware Representations of Multi-Agent Interactions},
  journal={openreview},
  year={2023}
}
```
