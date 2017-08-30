# VIGAN

This is a PyTorch implementation of the paper "[VIGAN: Missing View Imputation with Generative Adversarial Networks](https://arxiv.org/abs/1708.06724)". Please cite the paper in your publications if you find the source code useful to your research.

## Installation

Install pytorch and torchvision. 

## VIGAN model

VIGAN is the model for imputing missing views based on generative adversarial networks which combines cross-domain relations given unpaired data with multi-view relations given paired data.


<p align="center">
  <img width="300" height="200" src="https://github.com/chaoshangcs/VIGAN/blob/master/imgs/img1.png" width="425"/>
  <img width="300" height="200" src="https://github.com/chaoshangcs/VIGAN/blob/master/imgs/img2.png" width="425"/>
</p>

### Dataset

Benchmark dataset: MNIST (http://deeplearning.net/data/mnist/mnist.pkl.gz).

Train the network to learn to generate digit images and the corresponding edges images of the digits images, inspired by [CoGAN](https://github.com/chaoshangcs/CoGAN_PyTorch). We provide a python code to automaticlly download data and generate training data in the form of pairs of ( digit image, edges image ).

### Train the model
When you train the model, you can tune the parameters in "options" folder.

    python train.py

### Test the model
    python test.py

## Acknowledgments

Code is inspired by [CycleGAN](https://github.com/chaoshangcs/pytorch-CycleGAN-and-pix2pix) and [CoGAN](https://github.com/chaoshangcs/CoGAN_PyTorch).
