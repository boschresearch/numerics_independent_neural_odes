 
# PyTorch Implementation of: ResNet After All? Neural ODEs and Their Numerical Solution
This is the accompanying code for [ResNet After all? Neural ODEs and Their Numerical Solution](https://openreview.net/forum?id=HxzSxSxLOJZ).
The experiments based on this library are fully supported to run on single/multiple gpus. 
By default, the device is set to cpu. 
All our experiments where run on a single GPU.

## Purpose of the project

This software is a research prototype, solely developed for and published as part of the publication cited above. 
It will neither be maintained nor monitored in any way.

*To further increase the reproducibility of our paper, we will add the scripts for the experiments in the near future.*

## Requirements 

All necessary packages are listed in requirements.txt

## Data

Here we list the resources for the specific data sets:

MNIST: [link](http://yann.lecun.com/exdb/mnist/) 

cifar10: [link](https://www.cs.toronto.edu/~kriz/cifar.html)

Concentric Sphere: Can be generated using the code given in 
[augmented_ode](https://github.com/EmilienDupont/augmented-neural-odes).

## Training

By running  train.py the model automatically runs on the Concentric Sphere dataset.

To adjust parameters and which parameters are available please check options/default_config.yaml.
All parameters which need to be specified are given the the supplementary material to the paper.


## License

Numerics Independent Neural ODEs is open-sourced under the AGPL-3.0 license. See the LICENSE file for details.

