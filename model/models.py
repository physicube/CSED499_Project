import torch.nn as nn
from torchvision.models import vgg16, resnet50

from model.vgg_face_dag import vgg_face_dag

pretrained_path = "model/pretrained/"

def VGG_Face_PubFig(deep_layer_fe=True):
    model = vgg_face_dag(weights_path=pretrained_path + "vgg_face_dag.pth")

    # freeze layers
    for param in model.parameters():
        param.requires_grad = False
    model.fc8 = nn.Linear(4096, 83)
    model.cuda()

    return model

def VGG16_GTSRB(deep_layer_fe=True):
    model = vgg16(pretrained=True)

    # freeze layers
    for param in model.parameters():
        param.requires_grad = False
    model.classifier[6] = nn.Linear(4096, 43)
    model.cuda()

    return model

def ResNet50_VGGFlower(deep_layer_fe=True):
    model = resnet50(pretrained=True)

    # freeze layers
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(2048, 102)
    model.cuda()
    
    return model
