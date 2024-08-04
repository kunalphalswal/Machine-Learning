import torch
from torch import nn
from torch.utils.data import DataLoader #used to iterate through data and use it in our code.
from torchvision import datasets # we will be using vision datasets. pytorch has domain specific datasets like text, vision, audio
from torchvision.transforms import ToTensor

# Dataset stores the samples and their corresponding labels, and DataLoader wraps an iterable around the Dataset.
# Every TorchVision Dataset includes two arguments: transform and target_transform to modify the samples and labels respectively.

#what does toTensor mean?
# Download training data from open datasets.
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)


