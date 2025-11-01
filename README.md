# Generative Adversarial Networks (GAN) model from Scratch

## Overview
### This projects explores the field of Deep Learning Neural Networks, specifically in the world of GANs. The primary objective of this project is to build a Deep Convolutional GAN (DCGAN) from scratch in PyTorch, following the architecture laid out in the following:
### 2014 Research Paper : **Generative Adversarial Nets** by *Ian Goodfellow*.
### 2015 Research Paper : *UNSUPERVISED REPRESENTATION LEARNING WITH DEEP CONVOLUTIONAL GENERATIVE ADVERSARIAL NETWORKS* by Alec Radford & Luke Metz

### The implementation successfull generates Handwritten Digitals trained from the popular MNIST dataset. User can train the model on any dataset, and it should generate the images.


## Key Concepts behind GAN
### GAN works on the principle in which two compete against each other, in a zero-sum game.
### 1. The Generator(G): 
- Task: To generate realistic dataspace
- Input: A random noise vector z
- Output: A fake dataspace (eg., a fake image)
- Objective: To produce images so realistic that it can fool the Discriminator to classify them as real

### 2. The Discriminator(D): 
- Task: To distinguish between real and fake data
- Input: A fake image
- Output: Probability (value btw 0 and 1) that the input image is "Real"
- Objective: To correctly identify the real images as real and fake images as fake

During training, the generator is constantly trying to outsmart the discriminator by generating better and better fakes, while the discriminator is working to become a better detective and correctly classify the images. 


## Architecture: DCGAN

### Generator
#### It's job is to upsample the latent space vector z into a full-sized image.
#### Architecture Flow: 

##### Input: z (a 100-dim vector) is projected and reshaped to (1024, 4, 4).
##### CONV 1: (1024, 4, 4)  -> (512, 8, 8)
##### CONV 2: (512, 8, 8)   -> (256, 16, 16)
##### CONV 3: (256, 16, 16) -> (128, 32, 32)
##### CONV 4: (128, 32, 32) -> (nc, 64, 64)
##### Output: nn.Tanh()

Note: nc refers to the number of image channels (3-RGB, 1-B&W)


### Discriminator 

#### The Discriminator's job is to downsample an image into a single probability. It's a standard Convolutional Neural Network (CNN).

#### Architecture Flow:

##### Input: Image (nc, 64, 64)
##### CONV 1: (nc, 64, 64)   -> (128, 32, 32)
##### CONV 2: (128, 32, 32) -> (256, 16, 16)
##### CONV 3: (256, 16, 16) -> (512, 8, 8)
##### CONV 4: (512, 8, 8)   -> (1024, 4, 4)
##### Output: (1024, 4, 4) -> (1, 1, 1) -> nn.Sigmoid()


## Colab Link: https://colab.research.google.com/drive/1Oc47WQUD4YSISgI5OOzb995leTcLwqyl?usp=sharing