# ASPNet

An implementation of ASPNet, proposed in **ATTENTION-GUIDED SECOND-ORDER POOLING CONVOLUTIONAL NETWORKS** by Shannan Chen, Qiule Sun, Cunhua Li, Jianxin Zhang, Qiang Zhang

Now ASPNet (20, 34, 56) are implemented and the code is modified based on [SENet](https://github.com/moskomule/senet.pytorch)

* `python cifar.py` runs resnet20_ASP with Cifar10 and Cifar100 dataset.

## Pre-requirements

* Python>=3.6
* PyTorch>=1.0
* torchvision>=0.4

### For training

To run `cifar.py` you need

* `pip install git+https://github.com/moskomule/homura`
* `pip install miniargs`

## Result

### ASPNet20/32 Cifar10/100

```
python cifar.py [--baseline]
```

|                       | ResNet20         | SE-ResNet20    | ASPNet20        | ASPNet32        |
|:-------------         | :-------------   | :------------- | :-------------  | :-------------  |
|test accuracy(cifar10) |  92.3%           | 92.6%          | 93.8%           | 94.6%           |
|test accuracy(cifar100)|  68.2%           | 69.1%          | 73.1%           | 74.2%           |

### ASPNet18/34 ImageNet


|                        | SE-ResNet18      | ASPNet18       | SE-ResNet34     | ASPNet34       |
|:-------------          | :-------------   | :------------- | :-------------  | :------------- |
|test accuracy(top1)     | 70.59%           | 72.16%         | 73.69%          | 74.83%         |

*The code will be published after the paper was accepted * .