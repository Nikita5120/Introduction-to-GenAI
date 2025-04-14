import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

# Generator and Discriminator from generator.py or define here
from generator import Generator

# Basic Discriminator (for training only)
class Discriminator(nn.Module):
    def __init__(self, input_dim=784):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.model(x)

# Config
batch_size = 64
epochs = 20
noise_dim = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
mnist = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
dataloader = DataLoader(mnist, batch_size=batch_size, shuffle=True)

# Models
generator = Generator(noise_dim=noise_dim).to(device)
discriminator = Discriminator().to(device)

# Loss and optimizers
loss_fn = nn.BCELoss()
opt_g = torch.optim.Adam(generator.parameters(), lr=0.0002)
opt_d = torch.optim.Adam(discriminator.parameters(), lr=0.0002)

# Training loop
for epoch in range(epochs):
    for real, _ in dataloader:
        real = real.view(-1, 784).to(device)
        batch_size = real.size(0)

        # Train Discriminator
        noise = torch.randn(batch_size, noise_dim).to(device)
        fake = generator(noise)
        d_real = discriminator(real)
        d_fake = discriminator(fake.detach())

        d_loss = loss_fn(d_real, torch.ones_like(d_real)) + \
                 loss_fn(d_fake, torch.zeros_like(d_fake))

        opt_d.zero_grad()
        d_loss.backward()
        opt_d.step()

        # Train Generator
        output = discriminator(fake)
        g_loss = loss_fn(output, torch.ones_like(output))

        opt_g.zero_grad()
        g_loss.backward()
        opt_g.step()

    print(f"Epoch {epoch+1}/{epochs} - D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")

# Save generator
os.makedirs("models", exist_ok=True)
torch.save(generator.state_dict(), "models/generator.pth")
print("✅ Generator saved to models/generator.pth")
