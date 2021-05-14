"""Provides functions for image input and output.


"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import datasets, transforms


def imshow(image, ax=None, title=None, normalize=True):
    """Display an image on screen from a PyTorch Tensor.

    Code adapted from: https://ryanwingate.com/intro-to-machine-learning/deep-learning-with-pytorch/loading-image-data-into-pytorch/
    """
    if ax is None:
        fig, ax = plt.subplots()
    image = image.numpy().transpose((1, 2, 0))

    if normalize:
        mean = np.array([0.5, 0.5, 0.5])
        std = np.array([0.5, 0.5, 0.5])
        image = std * image + mean
        image = np.clip(image, 0, 1)

    ax.imshow(image)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='both', length=0)
    ax.set_xticklabels('')
    ax.set_yticklabels('')

    return ax


def read_input(path, size=256):

    transform = transforms.Compose([transforms.Resize(size),
                                    transforms.CenterCrop(size),
                                    transforms.ToTensor()])

    dataset = datasets.ImageFolder(path, transform=transform)

    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=1,
                                             shuffle=False)
    return dataloader
