import os
import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import scipy
import numpy as np
NUM_WORKERS = os.cpu_count()

class ShiftDataset(Dataset):
    def __init__(self, shifting=True, rotate_degs=None, roll_pixels=None, dataset=None, transform=None):
        super(ShiftDataset, self).__init__()
        self.shifting = shifting
        self.rotate_degs = rotate_degs
        self.roll_pixels = roll_pixels
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index):
        image, label = self.dataset[index]
        if self.transform:
            image = self.transform(image)
        # Apply augmentations if training
        if self.shifting:
            if self.rotate_degs:
                image = scipy.ndimage.rotate(image.numpy(), self.rotate_degs, axes=(1, 2), reshape=False)
                image = torch.from_numpy(image)
            if self.roll_pixels:
                image = np.roll(image.numpy(), self.roll_pixels, axis=1)
                image = torch.from_numpy(image)
        # return {"data": image, "label": label}
        return image, label

    def __len__(self):
        return len(self.dataset)

def load_dataset(dataset_class, train=False, transform=None):
    return dataset_class(root='./data', train=train, download=True, transform=transform)

def get_transform(mean, std):
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

# Get train and test MNIST datasets
def get_train_test_mnist():
    transform = get_transform((0.1307,), (0.3081,))
    train_mnist = load_dataset(torchvision.datasets.MNIST, train=True, transform=transform)
    test_mnist = load_dataset(torchvision.datasets.MNIST, train=False, transform=transform)
    return train_mnist, test_mnist

# Get shifted MNIST with rotation and roll augmentations
def get_shifted_mnist(rotate_degs=2, roll_pixels=2):
    transform = get_transform((0.1307,), (0.3081,))
    mnist_data = load_dataset(torchvision.datasets.MNIST, train=False, transform=None)
    return ShiftDataset(shifting=True, rotate_degs=rotate_degs, roll_pixels=roll_pixels, dataset=mnist_data, transform=transform)

# Get FashionMNIST as OOD dataset with same MNIST transformation
def get_ood_mnist():
    transform = get_transform((0.1307,), (0.3081,))
    return load_dataset(torchvision.datasets.FashionMNIST, train=False, transform=transform)

def get_all_test_dataloaders(batch_size=1024, rotate_degs=2, roll_pixels=4):
    _, test_data = get_train_test_mnist()
    shift_data = get_shifted_mnist(rotate_degs=rotate_degs, roll_pixels=roll_pixels)
    ood_data = get_ood_mnist()
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, drop_last=True)
    shift_loader = DataLoader(shift_data , batch_size=batch_size, shuffle=False, drop_last=True)
    ood_loader = DataLoader(ood_data, batch_size=batch_size, shuffle=False, drop_last=True)
    return test_loader, shift_loader, ood_loader

# if __name__ == "__main__":
#     rotate_degs_list = [k for k in range(15, 181, 15)]
#     roll_pixels_list = [k for k in range(2, 28, 2)]
#     # Get shifted MNIST with rotation and roll augmentations
#     dataset = get_shifted_mnist(rotate_degs=rotate_degs_list[0], roll_pixels=roll_pixels_list[2])
#     print(dataset[8]["data"].shape)
#     # plot the data
#     import matplotlib.pyplot as plt
#     img = dataset[8]["data"].squeeze()
#     plt.imshow(img)
#     plt.tight_layout()
#     plt.show()
