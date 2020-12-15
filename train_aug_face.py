from numpy.lib.shape_base import dsplit
import torch
import advertorch
import ctypes
import copy
from torch import equal
from torch.autograd import Variable
import torch.nn as nn
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from torchvision.utils import save_image

from data_loader.data_loader import VGGFaceAdvDataLoader
from model.models import VGG_Face_PubFig
from model.loss import CrossEntropyLoss
from utils.metric import DSSIM
from model.optimizer import Adadelta, SGD
from utils.transform import calc_normalize
from utils.config import setup


ctypes.cdll.LoadLibrary('caffe2_nvrtc.dll') 
data_dir = 'data/pubfig65_adv2/train/'
batch_size = 32
model_path = 'saved/Vgg_face_dag_weight.pt'

internal_val = 0

def get_internal_representation(model, img, k=-1):
    model(img)
    res = internal_val
    return res


def hook(model, inputs):
    global internal_val

    internal_val = inputs[0]

get_sample = lambda x: list(iter(x))[0]

if __name__=="__main__":
    setup()
    target_loader = VGGFaceAdvDataLoader(data_dir + 'target/', batch_size, is_train=True, shuffle=False)
    attack_loader = VGGFaceAdvDataLoader(data_dir + 'attack/', batch_size, is_train=True, shuffle=False)
    

     # lambda
    budget = 0.01
    lamb = 1 # temporary
    d_tar = 10
    clip_min = 0.0
    clip_max = 1.0
    lr = 1.0
    num_epoch = 100

    loss_fn = CrossEntropyLoss()

    model = VGG_Face_PubFig(saved=True)
    for param in model.parameters():
        param.requires_grad = True

    model.train()
    model[1].fc8.register_forward_pre_hook(hook)
    optimizer = SGD(model.parameters(), lr=lr)

    for batch_idx, (targets, attacks) in enumerate(zip(target_loader, attack_loader)):
        target_imgs, target_labels = targets
        attack_imgs, attack_labels = attacks
        target_imgs, target_labels = target_imgs.cuda(), target_labels.cuda()
        attack_imgs, attack_labels = attack_imgs.cuda(), attack_labels.cuda()

        for epoch in range(num_epoch):
            print('Epoch {}/{}'.format(epoch + 1, num_epoch))
            print('-' * 10)
            with torch.no_grad():
                sk_source = get_internal_representation(model, attack_imgs)
                sk_attack = get_internal_representation(model, target_imgs)

            optimizer.zero_grad()
            outputs = model(target_imgs)
            loss_ce = loss_fn(outputs, target_labels)

            dist = torch.dist(sk_source, sk_attack)
            loss_term = d_tar - dist
        
            loss = loss_ce + lamb * loss_term
            loss.backward()

            print(loss_ce, loss_term)
            print('loss :', loss)

    torch.save(model.state_dict(), 'saved/VGGFace_PubFig65_aug.pt')
