# pytorch-storm1
PyTorch implementation of a first-order stochastic trust-region method ([**STORM1**](https://pubsonline.informs.org/doi/abs/10.1287/ijoo.2019.0016)).

## To train ResNet-18 on CIFAR-10 using STORM1
```
cd pytorch-storm1
python train.py
```
## Results
| Method      | Test Acc. (%) |
| ----------- | ----------- |
| SGD       | 93..02       |
| STORM1 (ours)   | 93.55        |
