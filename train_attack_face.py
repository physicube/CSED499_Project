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

from data_loader.data_loader import VGGFaceDataLoader
from model.models import VGG_Face_PubFig
from utils.metric import DSSIM
from model.optimizer import Adadelta, SGD
from utils.transform import calc_normalize
from utils.config import setup



ctypes.cdll.LoadLibrary('caffe2_nvrtc.dll') 
data_dir = 'data/pubfig65/'
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

class AdvarsarialLoss(nn.Module):
    def __init__(self, model, source, target, lamb=1, budget=0.3):
        super(AdvarsarialLoss, self).__init__()

        self.model = model
        self.source = source
        self.target = target
        #self.tk_target = get_internal_representation(model, target).clone()
        self.lamb = lamb
        self.budget = budget

    def forward(self, img, tk_img, tk_target):
        term_internal = torch.dist(tk_img, tk_target)

        dist_perturb = DSSIM(img, self.source)
        
        term_perturb = dist_perturb - self.budget
        term_perturb = term_perturb ** 2 if term_perturb > 0 else 0

        res = term_internal + self.lamb * term_perturb
        return res




if __name__=="__main__":
    data_loader = VGGFaceDataLoader(data_dir, batch_size, is_train=True)

     # lambda
    budget = 0.01
    lamb = 10000 # temporary
    clip_min = 0.0
    clip_max = 1.0
    lr = 1.0
    
    model = VGG_Face_PubFig(saved=True)
    for param in model.parameters():
        param.requires_grad = False

    model.eval()
    model[1].fc8.register_forward_pre_hook(hook)


    total = 0
    correct = 0
    for batch_idx, (inputs, labels) in enumerate(data_loader):
        
        print('batch idx :', batch_idx)
        print()

        #source_imgs, source_labels = data_loader.get_random_batch()
        target_imgs, target_labels = inputs, labels
        source_imgs, source_labels = data_loader.get_random_batch()
        
        #if len(source_labels) != batch_size: continue
            
        while len(source_imgs) != len(target_imgs):
            source_imgs, source_labels = data_loader.get_random_batch()

        source_imgs = source_imgs.cuda()
        target_imgs = target_imgs.cuda()
        perturb_imgs = Variable(source_imgs, requires_grad=True).cuda()

        optimizer = SGD([perturb_imgs], lr=lr)
        with torch.no_grad():
            loss_fn = AdvarsarialLoss(model, source_imgs, target_imgs, lamb, budget)
            tk_target = get_internal_representation(model, target_imgs)

        
        num_epochs = 1000
        loss = 0
        for epoch in range(num_epochs):
            tk_perturb = get_internal_representation(model, perturb_imgs)
            optimizer.zero_grad()
            loss = loss_fn(perturb_imgs, tk_perturb, tk_target)

            loss.backward()

            perturb_imgs.data = perturb_imgs.data.clamp(clip_min, clip_max)
            optimizer.step()


        with torch.no_grad():
            outputs = model(perturb_imgs)
            _, preds = torch.max(outputs.data, 1)

        
        total += len(source_labels)
        correct += torch.sum(preds.cpu() == target_labels.data)

        print('batch {} : loss {}'.format(batch_idx, loss))
        print()


        # save img
        for i in range(len(perturb_imgs)):
            filename = '{}_{}_{}.png'.format(batch_idx, source_labels[i], target_labels[i])
            save_image(perturb_imgs[i], 'data/PubFig65_adv2/train/attack/' + filename)
            save_image(target_imgs[i], 'data/PubFig65_adv2/train/target/' + filename)

        
    print('total prediction rate {}/{}'.format(correct, total))
    print()

        
