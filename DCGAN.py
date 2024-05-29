import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader


# Define the Generator network
class Generator(nn.Module):
    def __init__(self, latent_dim=100, image_channels=1, hidden_dim=64):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, hidden_dim * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_dim * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(hidden_dim * 4, hidden_dim * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(hidden_dim * 2, hidden_dim, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(True),
            nn.ConvTranspose2d(hidden_dim, image_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)


# Define the Discriminator network
class Discriminator(nn.Module):
    def __init__(self, image_channels=1, hidden_dim=64):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(image_channels, hidden_dim, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_dim * 2, hidden_dim * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_dim * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_dim * 4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)


# Function to train the GAN
def train_gan(gan, dataset, epochs, device, target_label=None):
    # Set up optimizers
    generator_optimizer = torch.optim.Adam(gan.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    discriminator_optimizer = torch.optim.Adam(gan.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    loss_function = nn.BCELoss()

    # DataLoader
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    # Move models to device
    gan.generator.to(device)
    gan.discriminator.to(device)

    # Training loop
    for epoch in range(epochs):
        for batch_idx, (real_images, _) in enumerate(dataloader):
            # Train Discriminator
            real_images = real_images.to(device)
            discriminator_optimizer.zero_grad()
            real_labels = torch.ones(real_images.size(0), 1).to(device)
            fake_labels = torch.zeros(real_images.size(0), 1).to(device)

            real_outputs = gan.discriminator(real_images)
            real_loss = loss_function(real_outputs, real_labels)
            real_loss.backward()

            latent_vector = torch.randn(real_images.size(0), gan.latent_dim, 1, 1).to(device)
            fake_images = gan.generator(latent_vector)
            fake_outputs = gan.discriminator(fake_images.detach())
            fake_loss = loss_function(fake_outputs, fake_labels)
            fake_loss.backward()

            discriminator_optimizer.step()

            # Train Generator
            generator_optimizer.zero_grad()
            fake_outputs = gan.discriminator(fake_images)
            generator_loss = loss_function(fake_outputs, real_labels)
            generator_loss.backward()
            generator_optimizer.step()

            if batch_idx % 100 == 0:
                print(f"Epoch [{epoch}/{epochs}], Batch {batch_idx}/{len(dataloader)}, "
                      f"Generator Loss: {generator_loss.item():.4f}, "
                      f"Discriminator Loss: {(real_loss + fake_loss).item():.4f}")


# Define transform to normalize data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# # Load FashionMNIST dataset
# train_dataset = datasets.FashionMNIST(root='./data', train=True, transform=transform, download=True)
#
# # Define GAN
# latent_dim = 100
# image_channels = 1
# hidden_dim = 64
# gan = Generator(latent_dim=latent_dim, image_channels=image_channels, hidden_dim=hidden_dim), Discriminator(image_channels=image_channels, hidden_dim=hidden_dim)
#
# # Train GAN
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# train_gan(gan, train_dataset, epochs=10, device=device)
