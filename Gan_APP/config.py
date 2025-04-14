# config.py

# Noise vector dimension for the GAN
noise_dim = 100

# Output image size (flattened 28x28 for MNIST)
image_dim = 28 * 28

# Set device to CPU
import torch
device = torch.device("cpu")
