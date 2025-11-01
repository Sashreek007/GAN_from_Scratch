# Generative Adversarial Networks (GAN) Model from Scratch

## Overview

This project explores the field of Deep Learning Neural Networks, specifically focusing on Generative Adversarial Networks (GANs). The primary objective is to build a Deep Convolutional GAN (DCGAN) from scratch using PyTorch, following the architectures described in foundational research papers.

### Research Papers

- **2014**: *Generative Adversarial Nets* by Ian Goodfellow et al.
- **2015**: *Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks* by Alec Radford and Luke Metz

The implementation successfully generates handwritten digits trained on the popular MNIST dataset. The model is designed to be flexible and can be trained on custom datasets to generate domain-specific images.

## Key Concepts Behind GANs

GANs operate on an adversarial training principle where two neural networks compete against each other in a zero-sum game framework.

### 1. The Generator (G)

- **Task**: Generate realistic synthetic data
- **Input**: Random noise vector `z` sampled from a latent space
- **Output**: Synthetic data samples (e.g., generated images)
- **Objective**: Produce samples realistic enough to fool the Discriminator into classifying them as real

### 2. The Discriminator (D)

- **Task**: Distinguish between real and synthetic data
- **Input**: Either real data from the training set or fake data from the Generator
- **Output**: Probability value between 0 and 1 indicating whether the input is real
- **Objective**: Correctly classify real samples as real (output ≈ 1) and fake samples as fake (output ≈ 0)

During training, the Generator continuously improves at creating realistic samples to deceive the Discriminator, while the Discriminator becomes increasingly adept at identifying synthetic data. This adversarial process drives both networks toward optimal performance.

## Architecture: DCGAN

### Generator Network

The Generator upsamples a low-dimensional latent vector into a full-resolution image using transposed convolutions.

#### Architecture Flow

```
Input: z (100-dimensional latent vector)
  ↓ Project and reshape to (1024, 4, 4)
  
ConvTranspose2d Layer 1: (1024, 4, 4)   → (512, 8, 8)
ConvTranspose2d Layer 2: (512, 8, 8)    → (256, 16, 16)
ConvTranspose2d Layer 3: (256, 16, 16)  → (128, 32, 32)
ConvTranspose2d Layer 4: (128, 32, 32)  → (nc, 64, 64)
  
Output: Tanh activation → Image (nc, 64, 64)
```

**Note**: `nc` refers to the number of image channels (1 for grayscale, 3 for RGB)

### Discriminator Network

The Discriminator downsamples an input image into a single probability score using standard convolutional layers.

#### Architecture Flow

```
Input: Image (nc, 64, 64)
  
Conv2d Layer 1: (nc, 64, 64)   → (128, 32, 32)
Conv2d Layer 2: (128, 32, 32)  → (256, 16, 16)
Conv2d Layer 3: (256, 16, 16)  → (512, 8, 8)
Conv2d Layer 4: (512, 8, 8)    → (1024, 4, 4)
  
Flatten and Dense: (1024, 4, 4) → (1, 1, 1)
  
Output: Sigmoid activation → Probability [0, 1]
```

## Training Process

The training follows the standard GAN training procedure with alternating updates:

1. **Train Discriminator**: Update D to maximize its ability to distinguish real from fake samples
2. **Train Generator**: Update G to maximize D's error (minimize D's ability to detect fakes)

This minimax game continues until convergence, where the Generator produces highly realistic samples.

## Requirements

```
torch>=1.9.0
torchvision>=0.10.0
numpy>=1.19.0
matplotlib>=3.3.0
```

## Usage

```python
# Train the DCGAN model
python train.py --dataset mnist --epochs 100 --batch_size 64

# Generate samples
python generate.py --model_path checkpoints/generator.pth --num_samples 16
```

## Results

The model successfully generates handwritten digits that closely resemble the MNIST dataset after sufficient training epochs.

## References

1. Goodfellow, I., et al. (2014). "Generative Adversarial Nets." *Advances in Neural Information Processing Systems*.
2. Radford, A., Metz, L., & Chintala, S. (2015). "Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks." *arXiv preprint arXiv:1511.06434*.



## Colab Link: https://colab.research.google.com/drive/1Oc47WQUD4YSISgI5OOzb995leTcLwqyl?usp=sharing
