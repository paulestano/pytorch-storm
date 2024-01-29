# pytorch-storm
PyTorch implementation of a first-order stochastic trust-region method ([**STORM**](https://pubsonline.informs.org/doi/abs/10.1287/ijoo.2019.0016)).

STORM1 exploits a specific optimizer, `storm1.py`, and an ad-hoc scheduling of the learning rate.

## To train ResNet-18 on CIFAR-10 using STORM1
```
cd pytorch-storm1
python train.py
```
## Results
| Method      | Test Acc. (%) |
| ----------- | ----------- |
| SGD + CosineAnnealingLR   | 93.22        |
| STORM1 (ours)   | 93.68        |
