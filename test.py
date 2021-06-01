'''
Training script for ImageNet
Copyright (c) Wei YANG, 2017
'''
from __future__ import print_function

import argparse
import os
import shutil
import time
import random

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import models
from math import cos, pi

from celeba import CelebA
from utils import Bar, Logger, AverageMeter, accuracy, accuracy1, mkdir_p, savefig
from tensorboardX import SummaryWriter

import torch
import torch.utils.data as data

from PIL import Image
import os
import os.path

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

transform1 = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
])


def accuracy1(output, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        print(output)
        m = torch.nn.Softmax(dim=1)
        output = m(output)
        print(output)



def main():
    print()
    model = models.__dict__["mobilenetv2"](pretrained=True)





    img_path = "/home/chenxin/disk1/github/face-attribute-prediction/CelebA/img_align_celeba_png/negative_0007.jpg"
    img_PIL = Image.open(img_path).convert('RGB')
    img_PIL_Tensor = transform1(img_PIL)
    # img_PIL_Tensor = torch.unsqueeze(img_PIL_Tensor, dim=0)

    img_PIL_Tensor = torch.stack((img_PIL_Tensor, img_PIL_Tensor), 0)
    output = model(img_PIL_Tensor)

    for j in range(len(output)):
        accuracy1(output[j])

    # print(output)
    # m = torch.nn.Softmax(dim=1)
    #
    # for i in range(2):
    #     f = m(output[i])
    #     print(f)



if __name__ == '__main__':
    main()
