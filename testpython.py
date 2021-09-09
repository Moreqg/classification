# from efficientnet_pytorch import EfficientNet



import torch.nn as nn
import torch
from mydataloader.dataset import MyDataSet
from torch.utils.data import DataLoader
import numpy as np
import json


#将倒数第二层的特征图，融合，输出。
bo_fullname = r'E:\QinGang\classification\models\efficientnet-b0\data\13best.pt'
b3_fullname = r'E:\QinGang\classification\models\efficientnet-b3\data\14best.pt'
b5_fullname = r'E:\QinGang\classification\models\efficientnet-b5\data\14best.pt'
resnet50_fullname = r'E:\QinGang\classification\models\resnet50\data\60best.pt'
train_dir = r'E:\QinGang\data\train_data'
from libs.tta import tta
from torchvision import models
if __name__ == "__main__":
    model = models.resnet18()
    for name, param in model.named_parameters():
        print(name, param.shape, param.requires_grad)

    for param in model.parameters():
        print(param.shape, param.requires_grad)