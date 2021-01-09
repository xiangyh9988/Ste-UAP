import torch
import numpy as np
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import os


def load_imagenet(n_ex, size=224):
    IMAGENET_SL = size
    # IMAGENET_PATH = 'data/ILSVRC2012_img_val'
    IMAGENET_PATH = 'D:/code/adversarial-examples/circle-attack/data/ILSVRC2012_img_val'
    imagenet = ImageFolder(IMAGENET_PATH,
                          transforms.Compose([
                            transforms.Resize(IMAGENET_SL),
                            transforms.CenterCrop(IMAGENET_SL),
                            transforms.ToTensor()
                          ]))
    torch.manual_seed(0)

    imagenet_loader = DataLoader(imagenet, batch_size=n_ex, shuffle=True, num_workers=0)
    x_test, y_test = next(iter(imagenet_loader))
    return np.array(x_test, dtype=np.float32), np.array(y_test)


datasets_dict = {
    'imagenet': load_imagenet,
}
bs_dict = {
    'imagenet': 100
}