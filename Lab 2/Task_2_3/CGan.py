import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from typing_extensions import TypedDict

class Generator(nn.Module):
    def __init__(self, input_size, hidden_dim, output_size):
        super(Generator, self).__init__()

        num_classes = 10

        self.label_embed = nn.Embedding(num_classes, num_classes)

        self.net = nn.Sequential(
            nn.Linear(input_size + num_classes, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim * 2, output_size),
            nn.Tanh()
        )

    def forward(self, z, labels):
        embed = self.label_embed(labels).squeeze(1)
        #print(embed.shape, z.shape, embed.type(), z.type())
        x = torch.cat([z, embed], 1)
        return self.net(x)

class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_dim):
        super(Discriminator, self).__init__()

        num_classes = 10
        self.label_embed = nn.Embedding(num_classes, num_classes)

        self.net = nn.Sequential(
            nn.Linear(input_size + num_classes, hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x, labels): # x = [1, 2, 3] embed = [8.2, 4.3] -> [1, 2, 3, 8.2, 4.3]
        embed = self.label_embed(labels).squeeze(1)
        #print(embed.shape, x.shape, embed.type(), x.type())
        x = torch.cat([x, embed], 1)
        return self.net(x)



class GeneratorArgs(TypedDict):
    images: torch.Tensor
    labels: torch.Tensor
    optimizer: torch.optim.Optimizer
    criterion: torch.nn.Module
    device: torch.device


def train_D(kwargs: GeneratorArgs, generator: Generator, discriminator: Discriminator) -> torch.Tensor:
    # Assign all TypedDict values to variables
    images, labels, optimizer, criterion, device = kwargs['images'], kwargs['labels'], kwargs['optimizer'], kwargs['criterion'], kwargs['device']

    # Get discriminator Sigmoid output
    real_validity = discriminator(images, labels)

    # Real labels because we have ouput based on the real images
    real_labels = torch.ones(images.size(0), 1).to(device)

    # Calculate loss
    real_d_loss = criterion(real_validity, real_labels)

    # Noise input
    z = torch.randn(images.size(0), images.size(1)).to(device)

    # Fake labels
    fake_labels = torch.zeros(images.size(0), 1).to(device)

    # Generate fake images
    fake_images = generator(z, labels)

    # Get discriminator Sigmoid output
    fake_validity = discriminator(fake_images, labels)

    # Calculate loss
    fake_d_loss = criterion(fake_validity, fake_labels)

    # Total loss averaged loss
    d_loss = (real_d_loss + fake_d_loss) / 2

    # Backpropagation
    d_loss.backward()

    # Update weights
    optimizer.step()

    return d_loss


def train_G(kwargs: GeneratorArgs, generator: Generator, discriminator: Discriminator) -> torch.Tensor:

    images, labels, optimizer, criterion, device = kwargs['images'], kwargs['labels'], kwargs['optimizer'], kwargs['criterion'], kwargs['device']

    # Noise input
    #   - assumes batch-first shape (b, ...)
    z = torch.randn(images.size(0), images.size(1)).to(device)

    # Real labels
    real_labels = torch.ones(images.size(0), 1).to(device)

    artificial_images = generator(z, labels)
    d_fake = discriminator(artificial_images, labels)
    g_loss = criterion(d_fake, real_labels)

    g_loss.backward()
    optimizer.step()

    return g_loss


def main():
    # Hyperparameters
    batch_size = 128
    learning_rate = 0.0002
    epochs = 100
    input_size = 784  # Size of the latent vector to generate the images #Network parameters
    hidden_dim = 256  # Network parameter
    output_size = 28 * 28  # This should match the size of the images in the dataset (e.g., 784 for MNIST) #Network parameters
    image_size = 28

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # augmentation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Ladda datan som vi vill använda på nätverket
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    # We create a obejct of each type:

    generator = Generator(input_size, hidden_dim, output_size).to(device)
    discriminator = Discriminator(output_size, hidden_dim).to(device)

    # Optimizers
    g_optimizer = optim.Adam(generator.parameters(), lr=learning_rate)
    d_optimizer = optim.Adam(discriminator.parameters(), lr=learning_rate)

    # Loss function
    criterion = nn.BCELoss()

    # Create the TypedDict for the generator
    generator_kwargs: GeneratorArgs = {
        'images': None,
        'labels': None,
        'optimizer': g_optimizer,
        'criterion': criterion,
        'device': device
    }

    # Create the TypedDict for the discriminator
    discriminator_kwargs: GeneratorArgs = {
        'images': None,
        'labels': None,
        'optimizer': d_optimizer,
        'criterion': criterion,
        'device': device
    }

    # Update this over time
    best_generator_loss = 999.0
    best_model = None

    # TRAINING FROM HERE

    for epoch in range(epochs):

        for i, (images, labels) in enumerate(train_loader):

            batch_size = images.size(0)

            # Send them to the device
            images = images.view(batch_size, -1).to(device)
            labels = labels.view(batch_size, -1).to(device)

            # Zero grads
            g_optimizer.zero_grad()
            d_optimizer.zero_grad()

            # Populate the TypedDicts
            generator_kwargs['images'] = images
            generator_kwargs['labels'] = labels
            discriminator_kwargs['images'] = images
            discriminator_kwargs['labels'] = labels

            # Train Discriminator
            d_loss = train_D(discriminator_kwargs, generator, discriminator)

            # Train Generator
            g_loss = train_G(generator_kwargs, generator, discriminator)

            # real_score = D(x), fake_score = D(G(z))
            real_score = discriminator(images, labels)
            z = torch.randn(batch_size, input_size).to(device)
            # fake_score = D(G(z, labels), labels)
            fake_score = discriminator(generator(z, labels), labels)

            g_loss_val = g_loss.item()

            if best_generator_loss > g_loss_val:
                best_generator_loss = g_loss_val
                best_model = generator.state_dict()


            if (i + 1) % 100 == 0:
                print(
                    f'Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{len(train_loader)}], d_loss: {d_loss.item()}, g_loss: {g_loss.item()}, D(x): {real_score.mean().item()}, D(G(z)): {fake_score.mean().item()}')


    # Try to create a an 4 MNIST image from generator G(z, label) Assume one sample
    z = torch.randn(1, input_size).to(device)
    label3 = torch.tensor([3]).to(device)
    label0 = torch.tensor([0]).to(device)

    # Create new generator to load the best model
    generator.load_state_dict(best_model)

    print(generator)
    generated_image_3 = generator(z, label3).reshape(image_size, image_size).cpu().detach().numpy()
    generated_image_0 = generator(z, label0).reshape(image_size, image_size).cpu().detach().numpy()

    import matplotlib.pyplot as plt
    plt.imshow(generated_image_3, cmap='gray')
    plt.show()
    plt.imshow(generated_image_0, cmap='gray')
    plt.show()


if __name__ == "__main__":
    main()