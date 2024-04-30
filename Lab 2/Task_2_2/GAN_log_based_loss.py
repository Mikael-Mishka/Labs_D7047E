import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

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


# Hyperparameters
batch_size = 128
learning_rate = 0.0002
epochs = 20_000
input_size = 784  # Size of the latent vector to generate the images #Network parameters
hidden_dim = 256 #Network parameter
output_size = 28*28  # This should match the size of the images in the dataset (e.g., 784 for MNIST) #Network parameters
image_size = 28
iterations = 0

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#augmentation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


#Ladda datan som vi vill använda på nätverket - Load the data we want to use on the network
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)


#We create a obejct of each type:

generator = Generator(input_size, hidden_dim, output_size).to(device)
discriminator = Discriminator(output_size, hidden_dim).to(device)

# Optimizers
g_optimizer = optim.Adam(generator.parameters(), lr=learning_rate)
d_optimizer = optim.Adam(discriminator.parameters(), lr=learning_rate)

# Loss function
criterion = nn.BCELoss()

break_epoch_loop = False
upper_limit = 100_000

# Metric 'g-loss', 'd-loss', 'D(x)', 'D(G(z))'
g_losses = []
d_losses = []
D_x = []
D_G_z = []

for epoch in range(epochs):

    if break_epoch_loop:
        break

    for i, (images, _) in enumerate(train_loader):
        if iterations >= upper_limit:
            break_epoch_loop = True
            break
        batch_size=len(images)


        images = images.reshape(batch_size, -1).to(device)

        # Create labels
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        # Train Discriminator
        outputs = discriminator(images)
        d_loss_real = criterion(outputs, real_labels)
        real_score = outputs

        # Generate fake images
        z = torch.randn(batch_size, input_size).to(device)
        fake_images = generator(z)
        outputs = discriminator(fake_images)
        d_loss_fake = criterion(outputs, fake_labels)
        fake_score = outputs

        # Backprop and optimize
        d_loss = d_loss_real + d_loss_fake
        d_optimizer.zero_grad()
        g_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        # Train Generator
        z = torch.randn(batch_size, input_size).to(device)
        fake_images = generator(z)
        outputs = discriminator(fake_images)
        g_loss = criterion(outputs, real_labels)

        d_optimizer.zero_grad()
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

        iterations += 1

        if (i + 1) % 100 == 0:
            print(
                f'Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{len(train_loader)}], d_loss: {d_loss.item()}, g_loss: {g_loss.item()}, D(x): {real_score.mean().item()}, D(G(z)): {fake_score.mean().item()}')


        g_losses.append(g_loss.item())
        d_losses.append(d_loss.item())
        D_x.append(real_score.mean().item())
        D_G_z.append(fake_score.mean().item())


# Plot the losses
import matplotlib.pyplot as plt

x_points = [i for i in range(iterations)]

print(len(x_points), len(g_losses), len(d_losses), len(D_x), len(D_G_z))

plt.plot(x_points, g_losses, label='g-loss')
plt.plot(x_points, d_losses, label='d-loss')
plt.plot(x_points, D_x, label='D(x)')
plt.plot(x_points, D_G_z, label='D(G(z))')
plt.legend()
plt.show()