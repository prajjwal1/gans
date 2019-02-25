# Generative Adversarial Networks
This repository contains implementation of various architectures of Generative Models.

## Implemented architectures
- GANs
- Wasserstein GANs
- WGAN with gradient penalty

## Prerequisites
- Pytorch >= 1.0
- [Tensorop](https://github.com/prajjwal1/tensorop)
- Numpy

Usage of GPU is highly recommended. 

## Datasets used
- [LSUN](http://lsun.cs.princeton.edu/) 
- [MNIST](http://yann.lecun.com/exdb/mnist/)
- [Celeb A](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)

## Cloning the Repository
```
$ git clone https://github.com/prajjwal1/gans
```
## Examples
- [Wasserstein GANs](https://github.com/prajjwal1/tensorop/blob/master/nbs/WGANs.ipynb)

### Training
To train the model:
```
$ cd WGAN
$ python wgan_gp.py #For WGAN GP
$ python wgan.py # For WGAN
```

## Note
This repository is under constant development. Will be updated regularly. 


## References
- [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434) by Soumith et al.
- [Generative Adversarial Networks](https://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf) by Ian Goodfellow et al.
- [Wasserstein GAN](https://arxiv.org/abs/1701.07875) by Soumith et al.
- [Improved training of Wasserstein GANs](https://arxiv.org/abs/1704.00028) by Arjovsky et al.
- [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593) by Zhu et al.