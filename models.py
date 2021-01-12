import torch
# import tensorflow as tf
import numpy as np
import math
import kornia

from torchvision import models as torch_models
import torch.nn as nn
import pdb

from transform import *
from utils import imutils
global idx
idx = 0

class Model:
    def __init__(self, batch_size, gpu_memory):
        self.batch_size = batch_size
        self.gpu_memory = gpu_memory
    
    def predict(self, x):
        raise NotImplementedError("use ModelTF or ModelPT")

class ModelPT(Model):
    """
    Wrapper class around PyTorch models.
    (Ref. Square-Attack https://github.com/max-andr/square-attack/blob/master/models.py)

    In order to incorporate a new model, one has to ensure that self.model is a callable object that returns logits,
    and that the preprocessing of the inputs is done correctly (e.g. subtracting the mean and dividing over the
    standard deviation).
    """
    def __init__(self, model_name, batch_size, gpu_memory):
        super().__init__(batch_size, gpu_memory)
        self.model = model_class_dict[model_name](pretrained=True).cuda()
        self.batch_size = batch_size
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda()
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda()

        self.model.eval()
    
    def add_noise(self, x_batch: torch.Tensor, secret: np.ndarray):
        global idx
        x_batch_np = x_batch.cpu().numpy()
        x_batch_np = np.uint8(x_batch_np*255)
        x_adv_np = np.zeros_like(x_batch_np, dtype=np.float32)
        for i in range(len(x_batch_np)):
            # pdb.set_trace()
            stego = DWT_SVD(x_batch_np[i].transpose(1, 2, 0), secret)
            l2 = imutils.get_norm(x_batch_np[i].transpose(1, 2, 0), stego, 2)
            linf = imutils.get_norm(x_batch_np[i].transpose(1, 2, 0), stego, 'inf')
            # cv2.imwrite(f'adv_examples_h/stego/stego-{idx}-{l2}-{linf}.jpg', stego[:, :, ::-1])
            # cv2.imwrite(f'adv_examples_h/clean/clean-{idx}-{l2}-{linf}.jpg', x_batch_np[i].transpose(1, 2, 0)[:, :, ::-1])
            # cv2.imwrite(f'adv_examples/stego/stego-{idx}-{l2}-{linf}.jpg', stego[:, :, ::-1])
            # cv2.imwrite(f'adv_examples/clean/clean-{idx}-{l2}-{linf}.jpg', x_batch_np[i].transpose(1, 2, 0)[:, :, ::-1])
            # cv2.imwrite(f'reconstruct/stego/stego-{idx}-{l2}-{linf}.jpg', stego[:, :, ::-1])
            # cv2.imwrite(f'reconstruct/clean/clean-{idx}-{l2}-{linf}.jpg', x_batch_np[i].transpose(1, 2, 0)[:, :, ::-1])
            idx += 1
            stego = kornia.image_to_tensor(stego) / 255
            x_adv_np[i] = stego
        return torch.as_tensor(x_adv_np)

    def predict(self, dataloader, n_ex, secret):
        total = 0
        clean_corr, adv_corr = 0, 0
        logits_clean_list = []
        logits_adv_list = []
        with torch.no_grad():
            for _, (x_test, y_test) in enumerate(dataloader):
                y_test = y_test.cpu().numpy()
                x_clean = x_test.cuda()                
                x_adv = self.add_noise(x_test, secret).cuda()

                x_clean = ((x_clean - self.mean) / self.std)
                x_adv = ((x_adv - self.mean) / self.std)

                logits_clean = self.model(x_clean).cpu().numpy()
                logits_adv = self.model(x_adv).cpu().numpy()
                logits_clean_list.append(logits_clean)
                logits_adv_list.append(logits_adv)

                clean_corr += np.sum(logits_clean.argmax(1) == y_test)
                adv_corr += np.sum(logits_adv.argmax(1) == y_test)
                total += x_test.shape[0]
                if total >= n_ex:
                    break
        
        print('clean accuracy: {:.2%}'.format(clean_corr/total))
        print('noise accuracy: {:.2%}'.format(adv_corr/total))
        pdb.set_trace()
        labels_clean = np.vstack(logits_clean_list).argmax(1)
        labels_adv = np.vstack(logits_adv_list).argmax(1)
        fooling_rate = np.mean(labels_clean != labels_adv)
        print('fooling rate: {:.2%}'.format(fooling_rate))

    

model_class_dict = {
    'pt_vgg': torch_models.vgg16_bn,
    'pt_resnet': torch_models.resnet50,
    'pt_inception': torch_models.inception_v3,
    'pt_densenet': torch_models.densenet121,
    'pt_mobilenet': torch_models.mobilenet_v2,
}

all_model_names = list(model_class_dict.keys())