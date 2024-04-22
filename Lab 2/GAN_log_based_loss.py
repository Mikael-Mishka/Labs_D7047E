import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Assuming the Generator and Discriminator classes are defined as before

class Generator(nn.Module):
    def __init__(self, input_size, hidden_dim, output_size):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim * 2, output_size),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)

class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_dim):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# Hyperparameters setup
batch_size = 128
learning_rate = 0.0002
epochs = 20000
input_size = 100  # Size of the latent vector
hidden_dim = 256
output_size = 28 * 28  # Assuming MNIST (28x28 images)
image_size = 28

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# Initialize the Generator and Discriminator
generator = Generator(input_size, hidden_dim, output_size).to(device)
discriminator = Discriminator(output_size, hidden_dim).to(device)

# Optimizers
g_optimizer = optim.Adam(generator.parameters(), lr=learning_rate)
d_optimizer = optim.Adam(discriminator.parameters(), lr=learning_rate)

# Training loop
for epoch in range(epochs):
    for i, (images, _) in enumerate(train_loader):
        # Prepare real images
        batch_size=len(images)
        images = images.reshape(batch_size, -1).to(device)

        # Generate fake images
        z = torch.randn(batch_size, input_size).to(device)
        fake_images = generator(z)

        # Calculate discriminator loss with updated loss formula
        D_real = discriminator(images)
        D_fake = discriminator(fake_images)
        D_loss = -(torch.mean(torch.log(D_real)) + torch.mean(torch.log(1. - D_fake)))

        # Backprop and optimize for discriminator
        d_optimizer.zero_grad()
        D_loss.backward()
        d_optimizer.step()

        # Generate fake images for generator update
        z = torch.randn(batch_size, input_size).to(device)
        fake_images = generator(z)
        D_fake = discriminator(fake_images)

        # Calculate generator loss with updated loss formula
        G_loss = -torch.mean(torch.log(D_fake))

        # Backprop and optimize for generator
        g_optimizer.zero_grad()
        G_loss.backward()
        g_optimizer.step()

        if (i + 1) % 100 == 0:
            print(
                f'Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{len(train_loader)}], D_loss: {D_loss.item()}, G_loss: {G_loss.item()}')