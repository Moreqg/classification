from efficientnet_pytorch import EfficientNet

def get_efficientnet_b7():
    model = EfficientNet.from_pretrained("efficientnet-b7", num_classes=3)
    # model._fc = nn.Linear(1280, 256, bias=False)
    return model

def get_efficientnet_b5():
    model = EfficientNet.from_pretrained("efficientnet-b5", num_classes=3)
    # model._fc = nn.Linear(1280, 256, bias=False)
    return model

def get_efficientnet_b3():
    model = EfficientNet.from_pretrained("efficientnet-b3", num_classes=3)
    # model._fc = nn.Linear(1280, 256, bias=False)
    return model

def get_efficientnet_b0():
    model = EfficientNet.from_pretrained("efficientnet-b0", num_classes=3)
    # model._fc = nn.Linear(1280, 256, bias=False)
    return model

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
if __name__ == "__main__":
    train_dataset = MyDataSet(train_dir, data_type='train', k_data_num=6)
    train_dataloader = DataLoader(train_dataset, shuffle=False, batch_size=1)
    result_list = []  #(result: , label: )
    for i,(image, label) in enumerate(train_dataloader):
        print(i)
        with torch.no_grad():
            image = image.cuda()
            model1 = torch.load(bo_fullname)
            model1.eval()
            model1._fc = nn.Sequential()
            result1 = tta(model1, image).squeeze()
            # result1 = model1(image).squeeze()  #最后加tta

            model2 = torch.load(b3_fullname)
            model2.eval()
            model2._fc = nn.Sequential()
            # result2 = model2(image).squeeze()
            result2 = tta(model2, image).squeeze()
            # print(result2.shape)
            model3 = torch.load(b5_fullname)
            model3.eval()
            model3._fc = nn.Sequential()
            # result3 = model3(image).squeeze()
            result3 = tta(model3, image).squeeze()
            # print(result3.shape)
            model4 = torch.load(resnet50_fullname)
            model4.eval()
            model4.fc = nn.Sequential()
            # result4 = model4(image).squeeze()
            result4 = tta(model4, image).squeeze()
            # print(result4.shape)
            result = torch.cat([result1, result2, result3, result4], dim=0)  # cuda.torch

            # print(result.shape)
            # print(label.item())
            result = np.array(result.cpu()).tolist()  # (6912)
            # print(result)
            # print(label.item())
            sample = {'result': result, 'label':label.item()}
            result_list.append(sample)
    fp = open(r'E:\QinGang\classification\results\result_models_fuse\full_b0_b3_b5_resnet\result.json', 'a')
    for json_content in result_list:
        string = json.dumps(json_content)
        fp.write(string)
        fp.write('\n')
    fp.close()