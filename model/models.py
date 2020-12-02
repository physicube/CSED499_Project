import torch.nn as nn
import torch
from advertorch.utils import NormalizeByChannelMeanStd as normalize
from torchvision.models import vgg16, resnet50

from model.vgg_face_dag import vgg_face_dag

pretrained_path = "model/pretrained/"
saved_weight_path = 'saved/'

def VGG_Face_PubFig(saved=False, deep_layer_fe=True):
    model = vgg_face_dag(weights_path=pretrained_path + "vgg_face_dag.pth")

    # add transform layer
    mean, stddev = ((0.5454, 0.4380, 0.3741), (0.2536, 0.2286, 0.2218))
    norm_layer = normalize(mean, stddev)
    
    if deep_layer_fe:
        norm_layer.requires_grad = False
        # freeze layers    
        for param in model.parameters():
            param.requires_grad = False
        model.fc8 = nn.Linear(4096, 83)

    if saved:
        model_path = saved_weight_path + 'Vgg_face_dag_weight.pt'
        model.load_state_dict(torch.load(model_path))

    model = nn.Sequential(norm_layer, model)
    model.cuda()

    return model

def VGG16_GTSRB(deep_layer_fe=True):
    model = vgg16(pretrained=True)

    # add transform layer
    mean, stddev = ((0.3403, 0.3121, 0.3214), (0.2724, 0.2608, 0.2669))
    norm_layer = normalize(mean, stddev)

    if deep_layer_fe:
        norm_layer.requires_grad = False
        # freeze layers
        for param in model.parameters():
            param.requires_grad = False
        model.classifier[6] = nn.Linear(4096, 43)

    model = nn.Sequential(norm_layer, model)
    model.cuda()

    return model

def ResNet50_VGGFlower(deep_layer_fe=True):
    model = resnet50(pretrained=True)

     # add transform layer
    mean, stddev = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    norm_layer = normalize(mean, stddev)

    if deep_layer_fe:
        norm_layer.requires_grad = False
        # freeze layers
        for param in model.parameters():
            param.requires_grad = False
        model.fc = nn.Linear(2048, 102)

    model = nn.Sequential(norm_layer, model)
    model.cuda()

    return model
