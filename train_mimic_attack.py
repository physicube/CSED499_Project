from numpy.lib.utils import source
import torch
import advertorch
import ctypes
import copy
from torch import rand
from torch import autograd
from torch.autograd import Variable
import torch.nn as nn
from torchvision import datasets, transforms

from data_loader.data_loader import GTRSBDataLoader
from model.models import VGG16_GTSRB
from utils.metric import DSSIM
from model.optimizer import Adadelta
import matplotlib.pyplot as plt

ctypes.cdll.LoadLibrary('caffe2_nvrtc.dll') 
data_dir = 'data'
batch_size = 32
model_path = 'saved/VGG_weight.pt'

internal_val = 0

def get_internal_representation(model, img, k=-1):
    model(img)
    tk_source = internal_val

    return tk_source

class AdvarsarialLoss(nn.Module):
    def __init__(self, attack_model, img_target, lamb=1, budget=0.3):
        super(AdvarsarialLoss, self).__init__()

        self.model = attack_model
        self.target = img_target
        self.tk_target = get_internal_representation(self.model, self.target)
        self.lamb = lamb
        self.budget = budget

    def forward(self, img_pert, img_src):
        tk_perturb = get_internal_representation(self.model, img_pert)
        term_internal = torch.dist(tk_perturb, self.tk_target)

        dist_perturb = DSSIM(img_pert, img_src)
        
        term_perturb = dist_perturb - self.budget
        term_perturb = term_perturb ** 2 if term_perturb > 0 else 0

        res = term_internal + self.lamb * term_perturb
        print(res.grad)
        return res


def hook(model, inputs):
    global internal_val
    internal_val = inputs[0]

if __name__=="__main__":
    data_loader = GTRSBDataLoader(data_dir, batch_size, is_train=True)
    it_data = next(iter(data_loader))

    source_img = it_data[0][0]
    source_label = it_data[1][0]
    target_img = it_data[0][1]
    target_label = it_data[1][1]
    

    result = transforms.ToPILImage()(source_img)
    result.show()
    result = transforms.ToPILImage()(target_img)
    result.show()

    perturb_img = Variable(source_img.unsqueeze(0).cuda(), requires_grad=True)
    source_img = source_img.unsqueeze(0).cuda()
    target_img = target_img.unsqueeze(0).cuda()
    

    budget = 0.003
    lamb = 1

    model = VGG16_GTSRB()
    model.load_state_dict(torch.load(model_path))
    
    for param in model.parameters():
        param.requires_grad = False

    model.eval()
    model.classifier[6].register_forward_pre_hook(hook)
    optimizer = Adadelta([perturb_img], lr=1.0)

    tk_source = get_internal_representation(model, source_img)
    tk_target = get_internal_representation(model, target_img)

    num_epochs = 2000
    for epoch in range(num_epochs):
        tk_perturb = get_internal_representation(model, perturb_img)
        optimizer.zero_grad()
        term_internal = torch.dist(tk_perturb, tk_target)

        dist_perturb = DSSIM(perturb_img, source_img)
        
        term_perturb = dist_perturb - budget
        term_perturb = term_perturb ** 2 if term_perturb > 0 else 0

        loss = term_internal + lamb * term_perturb
        print('epoch {} : loss {}'.format(epoch, loss))
        loss.backward()
        optimizer.step()

    result = transforms.ToPILImage()(perturb_img.squeeze(0).cpu())
    result.show()

    outputs = model(perturb_img)
    _, preds = torch.max(outputs.data, 1)
    print('source label :', source_label)
    print('target label :',target_label)
    print('preds :', preds)

