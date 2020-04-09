# Image classification on MNIST and CIFAR10
Image Classification on MNIST handwritten digit recognition and CIFAR-10

## Authors
Prasheel Renkuntla
 
## Description
This project demonstrates image classification applied on MNIST and CIFAR10 dataset.
For both datasets, the Neural Network architecture is build on top of different existing architectures like LeNet5, ResNet.

## Dependencies
* Python 3.7
* TensorFlow v2.0.0
* Matplotlib
* sklearn
* scipy
* time
* sys
* os
* seaborn
* If on PC, tensorflow-gpu would be faster.

## Run
On Google Colab (with GPU, Runtime -> Change RunTimeType -> GPU)
Colab files can be directly loaded

To run Classification task on MNIST Data-
    Without GPU, Average Runtime is 10 mins approx.

For ready code-
```
python3.7 ready_code_mnist.py
```

For Architecture like LeNet-
```
python3.7 my_way_mnist_LeNet5.py
```

For my Architecture-
```
python3.7 mnist_classification_my_arch.py
```

To run Classification task on CIFAR-10 Data- (GPU Preferred!)
(similar names .ipynb also provided)

For ResNet 20-
```
python3.7 cifar10_resnet20_25.py
```
For ResNet 32-
```
python3.7 cifar10_resnet32_200.py
```

For ResNet 56-
```
python3.7 cifar10_resnet56_200.py
```