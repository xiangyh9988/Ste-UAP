import torch
# import tensorflow as tf
import numpy as np
import math
import kornia

from torchvision import models as torch_models
import torch.nn as nn


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
        self.mean = np.reshape([0.485, 0.456, 0.406], [1, 3, 1, 1]).astype(np.float32)
        self.std = np.reshape([0.229, 0.224, 0.225], [1, 3, 1, 1]).astype(np.float32)

        self.model.eval()
    
    def predict(self, x: torch.Tensor):
        x = (x - self.mean) / self.std
        x = x.astype(np.float32)
        n_batches = math.ceil(x.shape[0]/self.batch_size)
        logits_list = []
        with torch.no_grad():
            for i in range(n_batches):
                x_batch = x[i*self.batch_size:(i+1)*self.batch_size]
                x_batch_torch = torch.as_tensor(x_batch, device=torch.device('cuda'))
                logits = self.model(x_batch_torch).cpu().numpy()
                logits_list.append(logits)
        logits = np.vstack(logits_list)
        return logits

    

model_class_dict = {
    'pt_vgg': torch_models.vgg16_bn,
    'pt_resnet': torch_models.resnet50,
    'pt_inception': torch_models.inception_v3,
    'pt_densenet': torch_models.densenet121,
}

all_model_names = list(model_class_dict.keys())