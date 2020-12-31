import torch
import numpy as np
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import os


def load_imagenet(n_ex, size=224, batch_size):
    IMAGENET_SL = size
    IMAGENET_PATH = 'data/ILSVRC2012_img_val'
    imagenet = ImageFolder(IMAGE_PATH,
                          transforms.Compose([
                            transforms.Resize(IMAGENET_SL),
                            transforms.CenterCrop(IMAGENET_SL),
                            transforms.ToTensor()
                          ]))
    torch.manual_seed(0)

    imagenet_loader = DataLoader(imagenet, batch_size=batch_size, num_workers=0)
    return imagenet_loader


datasets_dict = {
    'imagenet': load_imagenet,
}
bs_dict = {
    'imagenet': 100
}