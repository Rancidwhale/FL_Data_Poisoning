import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import os
from loguru import logger

generator_input_size = 100


# Define GAN architecture
class Generator(nn.Module):
    def __init__(self, input_size, output_size):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, output_size),
            nn.Sigmoid()  # Sigmoid to scale pixel values between 0 and 1
        )

    def forward(self, x):
        return self.model(x)


# Define function for poisoning data using GAN
def poison_data_with_GAN(logger, distributed_train_dataset_numpy, num_workers, poisoned_workers, target_label):
    # Define input and output sizes based on the GAN architecture
    input_size = 100  # Example: Dimensionality of the input noise vector
    output_size = 28 * 28  # Example: Dimensionality of Fashion MNIST images

    # Instantiate the Generator class with input and output sizes
    generator = Generator(input_size, output_size)
    poisoned_samples = []
    with torch.no_grad():
        for i in range(num_workers):
            # Generate fake sample using GAN
            noise = torch.randn(1, generator_input_size)
            fake_sample = generator(noise)
            # Convert fake sample from PyTorch tensor to numpy array
            fake_sample_numpy = fake_sample.detach().numpy()
            # Modify label
            fake_label = target_label
            # Add fake sample and label to poisoned_samples
            poisoned_samples.append((fake_sample_numpy, fake_label))

    # Inject poisoned samples into distributed_train_dataset
    for worker_index, samples in distributed_train_dataset_numpy.items():
        if worker_index in poisoned_workers:
            distributed_train_dataset_numpy[worker_index].extend(poisoned_samples)

    return distributed_train_dataset_numpy



# Main function
def main():
    # Set random seed for reproducibility
    np.random.seed(0)
    torch.manual_seed(0)

    # Define parameters
    batch_size = 64
    num_epochs_GAN = 100
    lr_GAN = 0.0002
    num_poisoned_samples = 1000
    num_workers = 50
    target_label = 9  # New label for poisoned data

    # Initialize logger
    logger.add("poison_GAN.log")

    # Load original dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    original_dataset = datasets.FashionMNIST(root='./data', train=True, transform=transform, download=True)
    original_data_loader = DataLoader(original_dataset, batch_size=batch_size, shuffle=True)

    # Initialize GAN and train
    generator_input_size = 100
    generator_output_size = 28 * 28
    generator = Generator(generator_input_size, generator_output_size)
    train_GAN(generator, original_data_loader, num_epochs_GAN, lr_GAN)

    # Convert original_data_loader to numpy format
    distributed_train_dataset_numpy = convert_distributed_data_into_numpy(distributed_train_dataset)

    # Poison data using GAN
    poisoned_data_numpy = poison_data_with_GAN(logger, distributed_train_dataset_numpy, num_workers, poisoned_workers,
                                               target_label)

    # Convert poisoned_data_numpy back to dictionary format (if needed)
    # Now, you can use poisoned_data_numpy for training federated learning model


if __name__ == "__main__":
    main()
