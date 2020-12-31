import torch
import tensorflow as tf
import numpy as np
import math
import kornia

from torchvision import models as torch_models
import torch.nn as nn


class Model(nn.Module):

    def __init__(self, model_name, batch_size):
        super(Model, self).__init__()
        self.model = model_class_dict[model_name](pretrained=True).cuda()
        self.batch_size = batch_size
        self.mean = torch.tensor([0.485, 0.456, 0.406])
        self.std = torch.tensor([0.229, 0.224, 0.225])

        self.model.eval()
    
    def predict(self, x):
        ''' x np.ndarray image '''
        x: torch.Tensor = kornia.image_to_tensor(x)[:, :, :, ::-1]
        n_batches = math.ceil(x.shape[0]/self.batch_size)
        logits_list = []
        with torch.no_grad():
            for i in range(n_batches):
                x_batch = x[i*self.batch_size:(i+1)*self.batch_size]
                logits = self.model(x_batch).cpu().numpy()
                logits_list.append(logits)
        logits = np.vstack(logits_list)
        return logits

    

model_class_dict = {
    'pt_vgg': torch_models.vgg16_bn,
    'pt_resnet': torch_models.resnet50,
    'pt_inception': torch_models.inception_v3,
    'pt_densenet': torch_models.densenet121,
}