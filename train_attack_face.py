import torch
import advertorch
import ctypes
import copy
from torch import equal
from torch.autograd import Variable
import torch.nn as nn
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

from data_loader.data_loader import VGGFaceDataLoader
from model.models import VGG_Face_PubFig
from utils.metric import DSSIM
from model.optimizer import Adadelta, SGD
from utils.transform import calc_normalize


ctypes.cdll.LoadLibrary('caffe2_nvrtc.dll') 
data_dir = 'data/pubfig65/'
batch_size = 32
model_path = 'saved/Vgg_face_dag_weight.pt'

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



if __name__=="__main__":
    data_loader = VGGFaceDataLoader(data_dir, batch_size, is_train=False)

    source_img, source_label = data_loader.get_random_sample()
    target_img, target_label = data_loader.get_random_sample()
    
    assert (not torch.eq(source_label, target_label))
    result = transforms.ToPILImage()(source_img)
    result.show()
    result = transforms.ToPILImage()(target_img)
    result.show()

    perturb_img = Variable(source_img.unsqueeze(0).cuda(), requires_grad=True)
    source_img = source_img.unsqueeze(0).cuda()
    target_img = target_img.unsqueeze(0).cuda()
    
    # lambda
    budget = 0.003
    lamb = 10 # temporary

    model = VGG_Face_PubFig(saved=True)

    for param in model.parameters():
        param.requires_grad = False

    model.eval()
    model[1].fc8.register_forward_pre_hook(hook)
    optimizer = SGD([perturb_img], lr=10.0)
    loss_fn = AdvarsarialLoss(model, source_img, target_img, lamb, budget)

    num_epochs = 10
    for epoch in range(num_epochs):
        tk_perturb = get_internal_representation(model, perturb_img)
        optimizer.zero_grad()
        loss = loss_fn(perturb_img, tk_perturb)

        print('epoch {} : loss {}'.format(epoch, loss))
        loss.backward()
        optimizer.step()
    
    result = transforms.ToPILImage()(perturb_img.squeeze(0).cpu())
    result.show()

    outputs = model(perturb_img)
    _, preds = torch.max(outputs.data, 1)
    print()
    print('source label :', source_label)
    print('target label :',target_label)
    print('preds :', preds)

