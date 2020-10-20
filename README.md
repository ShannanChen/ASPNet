# ASPNet

An implementation of ASPNet, proposed in **ATTENTION-GUIDED SECOND-ORDER POOLING CONVOLUTIONAL NETWORKS** by Shannan Chen, Qiule Sun, Cunhua Li, Jianxin Zhang, Qiang Zhang

Now ASPNet (20, 34, 56) are implemented and the code is modified based on [SENet](https://github.com/moskomule/senet.pytorch)

* `python cifar.py` runs resnet20_ASP with Cifar10 and Cifar100 dataset.

## Pre-requirements

* Python>=3.7
* PyTorch>=1.1
* torchvision>=0.4

### For training

To run `cifar.py` you need

* `pip install git+https://github.com/moskomule/homura`

## Results

###  Cifar10/100

```
python cifar.py [--baseline]
```

|                        | ResNet20         | SENet20        | ASPNet20        | ResNet32        | SENet32         | ASPNet32        |
|:-------------          | :-------------   | :------------- | :-------------  | :-------------  | :-------------  | :-------------  |
|test accuracy (cifar10) |  92.3%           | 92.6%          | 93.8%           | 93.2%           | 93.3%           | 94.6%           |
|test accuracy (cifar100)|  68.2%           | 69.1%          | 73.1%           | 69.7%           | 71.6%           | 74.2%           |

###  ImageNet


|                         | SENet18          | ASPNet18       | SENet34         | ASPNet34       |
|:-------------           | :-------------   | :------------- | :-------------  | :------------- |
|test accuracy (top1)     | 70.59%           | 72.16%         | 73.69%          | 74.83%         |

## Contact

If you have any questions or suggestions, please contact me (Shannan_Chen@163.com).