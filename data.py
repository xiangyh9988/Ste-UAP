import torch
import numpy as np
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import os
from PIL import Image

from transform import *
import pdb

class Steganography(object):
    def __init__(self, secret: np.ndarray):
        self.secret = secret
    
    def __call__(self, img):
        img_ = np.array(img).copy()
        secret = cv2.resize(self.secret, (img_.shape[1], img_.shape[0]))
        stego = DWT_SVD(img_, secret)
        return Image.fromarray(stego.astype('uint8'))

def load_imagenet(n_ex, size, batch_size, mode='clean', secret=None):
    IMAGENET_SL = size
    # IMAGENET_PATH = 'data/ILSVRC2012_img_val'
    IMAGENET_PATH = 'D:/code/adversarial-examples/circle-attack/data/ILSVRC2012_img_val'
    if mode == 'clean':
        imagenet = ImageFolder(IMAGENET_PATH,
                            transforms.Compose([
                                transforms.Resize(IMAGENET_SL),
                                transforms.CenterCrop(IMAGENET_SL),
                                transforms.ToTensor()
                            ]))
    elif mode == 'noise':
        imagenet = ImageFolder(IMAGENET_PATH,
                            transforms.Compose([
                                transforms.Resize(IMAGENET_SL),
                                transforms.CenterCrop(IMAGENET_SL),
                                Steganography(secret),
                                transforms.Resize(IMAGENET_SL),
                                transforms.ToTensor()
                            ]))
    torch.manual_seed(0)

    imagenet_loader = DataLoader(imagenet, batch_size=batch_size, shuffle=True, num_workers=0)
    x_test, y_test = next(iter(imagenet_loader))
    # return np.array(x_test, dtype=np.float32), np.array(y_test)
    return imagenet_loader


datasets_dict = {
    'imagenet': load_imagenet,
}
bs_dict = {
    'imagenet': 100
}