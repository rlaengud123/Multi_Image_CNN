import os
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from Densenet import *


class DenseNet121(nn.Module):

    def __init__(self, isTrained):
        super(DenseNet121, self).__init__()

        self.densenet121 = densenet121(pretrained=isTrained)

    def forward(self, x):
        x = self.densenet121(x)
        return x


class DenseNet161(nn.Module):

    def __init__(self, isTrained):
        super(DenseNet161, self).__init__()

        self.densenet161 = densenet161(pretrained=isTrained)

    def forward(self, x):
        x = self.densenet161(x)
        return x


class DenseNet169(nn.Module):

    def __init__(self, isTrained):
        super(DenseNet169, self).__init__()

        self.densenet169 = densenet169(pretrained=isTrained)


    def forward(self, x):
        x = self.densenet169(x)
        return x


class DenseNet201(nn.Module):

    def __init__(self, isTrained):
        super(DenseNet201, self).__init__()

        self.densenet201 = densenet201(pretrained=isTrained)


    def forward(self, x):
        x = self.densenet201(x)
        return x