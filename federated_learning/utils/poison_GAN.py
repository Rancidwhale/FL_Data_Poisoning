import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from DCGAN import Generator, Discriminator, train_gan

def create_poisoned_fashion_mnist(target_label):
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.FashionMNIST(root='./data', train=True, transform=transform, download=True)

    poisoned_data = []
    for image, label in dataset:
        if label == target_label:
            poisoned_data.append((image, target_label))

    return poisoned_data


# Poison data using GAN
def poison_data_with_GAN(logger, distributed_train_dataset, num_workers, poisoned_workers, target_label):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Instantiate GAN models
    latent_dim = 100
    image_channels = 1
    hidden_dim = 64
    gan = Generator(latent_dim=latent_dim, image_channels=image_channels, hidden_dim=hidden_dim).to(device)
    discriminator = Discriminator(image_channels=image_channels, hidden_dim=hidden_dim).to(device)

    # Pre-training (Optional)
    # train_gan(gan, distributed_train_dataset, epochs=5, device=device)
    logger.info("Poisoning data for workers: {}".format(str(poisoned_workers)))
    # Poisoning Training
    poisoned_data = create_poisoned_fashion_mnist(target_label)
    train_gan(gan, poisoned_data, epochs=5, device=device, target_label=target_label)

    # Flip labels for poisoned workers
    for worker_idx in range(num_workers):
        if worker_idx in poisoned_workers:
            # Flip labels in distributed_train_dataset for poisoned workers
            distributed_train_dataset[worker_idx][1] = torch.where(distributed_train_dataset[worker_idx][1] == 1,9,
                                                                   distributed_train_dataset[worker_idx][1])

    return distributed_train_dataset
