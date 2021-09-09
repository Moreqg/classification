from torchvision import models
import torch.nn as nn
import torch

def get_resnet34():
    model = models.resnet34(pretrained=True)

    fc_inputs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(fc_inputs, 256),
        nn.ReLU(),
        nn.Linear(256, 3),
        nn.LogSoftmax(dim=1),
    )

    return model

def ResNet():

    resnet50 = models.resnet50(pretrained=True)
    for param in resnet50.parameters():
        # print("myresnet:9".format(param))
        param.requires_grad = False
    fc_inputs = resnet50.fc.in_features
    resnet50.fc = nn.Sequential(
        nn.Linear(fc_inputs, 256),
        nn.ReLU(),
        nn.Linear(256, 4),
        nn.LogSoftmax(dim=1),
    )
    # resnet50.fc = nn.Sequential(
    #     nn.Linear(fc_inputs, 256)
    # )
    return resnet50


