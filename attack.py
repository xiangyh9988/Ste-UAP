import torch
import kornia
import cv2
import os
import numpy as np
from datetime import datetime
import argparse

import data
import models
from datetime import datetime
from utils import imutils, log
from transform import *
np.set_printoptions(precision=5, suppress=True)

import pdb


def generate_stego(x_test: np.ndarray, secret: np.ndarray, corr_classified):
    ''' Generate stego image as adversarial examples
        host - BGR BxCxHxW 
        secret - BGR HxWxC
    '''
    print('start generating adversarial examples...')
    x_test_adv = np.zeros_like(x_test)
    pdb.set_trace()
    for i in range(x_test_adv.shape[0]):
        if not corr_classified[i]:
            continue
        x_test_adv[i] = (DWT_SVD(np.uint8(x_test[i].transpose(1, 2, 0)*255), secret).transpose(2, 0, 1)/255).astype(np.float32)
    print('generated.')
    print('='*30)
    return x_test_adv


if __name__ == '__main__':
    ''' assign arguments '''
    parser = argparse.ArgumentParser(description="Steganographic UAP attack.")
    # threat model
    parser.add_argument('--model', type=str, default='pt_resnet', choices=models.all_model_names, help='Threat model name.')
    # path to store output
    parser.add_argument('--out_folder', type=str, default='logs', help='Log folder to store all output.')
    # GPU number
    parser.add_argument('--gpu', type=str, default='0', help='GPU number. Multiple GPUs are possible for PyTorch models.')
    # number of testing examples
    parser.add_argument('--n_ex', type=int, default=10000, help='Number of testing examples to test on.')
    args = parser.parse_args()

    ''' prepration '''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    # set dataset
    # '...mnist...'   --> mnist
    # '...cifar10...' --> cifar10
    # others          --> imagenet
    dataset = 'mnist' if 'mnist' in args.model else 'cifar10' if 'cifar10' in args.model else 'imagenet'
    # get secret image and and convert it to BGR
    secret_img = cv2.imread(f'images/secret_{dataset}.JPEG')
    secret_img = cv2.cvtColor(secret_img, cv2.COLOR_BGR2RGB)
    # timestamp (string) format: yyyy-MM-DD HH-MM-SS
    # e.g. 2021-01-08 14:32:29
    timestamp = str(datetime.now())[:-7]

    # hyperparameters string, the saved output file name
    hps_str = f"{timestamp}-model={args.model}-dataset={dataset}-n_ex={args.n_ex}".replace(' ', '-').replace(':', '-')
    # batch size from data.py
    batch_size = data.bs_dict[dataset]
    # model type
    model_type = 'pt' if 'pt_' in args.model else 'tf'
    # number of classes (1000 for imagenet, 10 for mnist and cifar10)
    n_cls = 1000 if dataset=='imagenet' else 10
    # gpu memory for tf model (percentage)
    gpu_memory = 0.5 if dataset=='mnist' and args.n_ex > 1000 else 0.15 if dataset=='mnist' else 0.99
    # path to save log
    log_path = f'{args.out_folder}/{hps_str}.log'
    # logger
    log = log.Logger(log_path)
    log.print(f'All hps: {hps_str}')
    
    print('start loading dataset...')
    # inception 299x299, others 224x224
    if args.model != 'pt_inception':
        clean_loader = data.datasets_dict[dataset](args.n_ex, 224, batch_size)
        # noise_loader = data.datasets_dict[dataset](args.n_ex, 224, batch_size, mode='noise', secret=secret_img)
        secret_img = cv2.resize(secret_img, (224, 224))
    else:
        clean_loader = data.datasets_dict[dataset](args.n_ex, 299, batch_size)
        # noise_loader = data.datasets_dict[dataset](args.n_ex, 299, batch_size, mode='noise', secret=secret_img)
        secret_img = cv2.resize(secret_img, (299, 299))
    print('dataset loaded.')
    print('='*30)
    # x_test, y_test = x_test[:args.n_ex], y_test[:args.n_ex]
    # get threat model
    models_class_dict = {'tf': None, 'pt': models.ModelPT}
    model = models_class_dict[model_type](args.model, batch_size, gpu_memory)

    ''' clean accuracy '''
    print('start predicting (clean & noise)...')
    model.predict(clean_loader, args.n_ex, secret_img)