from model import loss
import torch
import advertorch
import ctypes
import copy
from torch import equal
from torch import optim
from torch.autograd import Variable
import torch.nn as nn
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

from data_loader.data_loader import BaseDataLoader, VGGFaceDataLoader
from model.models import VGG_Face_PubFig
from utils.metric import DSSIM
from model.optimizer import Adadelta, SGD
from utils.transform import calc_normalize

internal_val = 0

def get_internal_representation(model, img, k=-1):
    model(img)
    tk_source = internal_val

    return tk_source


def hook(model, inputs):
    global internal_val
    internal_val = inputs[0]



class AdvarsarialLoss(nn.Module):
    def __init__(self, attack_model, source, target, lamb=1, budget=0.3):
        super(AdvarsarialLoss, self).__init__()

        self.model = attack_model
        self.source = source
        self.target = target
        self.tk_target = get_internal_representation(self.model, self.target)
        self.lamb = lamb
        self.budget = budget

    def forward(self, img, tk_img):
        term_internal = torch.dist(tk_img, self.tk_target)

        dist_perturb = DSSIM(img, self.source)
        
        term_perturb = dist_perturb - self.budget
        term_perturb = term_perturb ** 2 if term_perturb > 0 else 0

        res = term_internal + self.lamb * term_perturb
        return res

class AdversarialAttack:
    def __init__(
        self,
        model: nn.Module,
        data_loader: BaseDataLoader,
        optimizer: nn.Module,
        loss_fn : nn.Module,
        num_epochs : int,
        budget : float,
        lamb : float
        ):
        self.model = model
        self.data_loader = data_loader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.num_epochs = num_epochs
        self.budget = budget
        self.lamb = lamb


    def train(self):
        source_img, source_label = self.data_loader.get_random_batch()
        target_img, target_label = self.data_loader.get_random_batch()

        assert (len(source_img) == len(target_img))
        
        perturb_img = Variable(source_img.cuda(), requires_grad=True)
        source_img = source_img.cuda()
        target_img = target_img.cuda()
        loss_fn = AdvarsarialLoss(self.model, source_img, target_img, self.lamb, self.budget)

        for epoch in range(self.num_epochs):
            tk_perturb = get_internal_representation(self.model, perturb_img)
            self.optimizer.zero_grad()
            loss = loss_fn(perturb_img, tk_perturb)

            print('epoch {} : loss {}'.format(epoch, loss))
            loss.backward()
            self.optimizer.step()
