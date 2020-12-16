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

from data_loader.data_loader import VGGFaceAdvDataLoader, VGGFaceDataLoader
from model.models import VGG_Face_PubFig
from model.loss import CrossEntropyLoss
from model.optimizer import Adadelta, SGD
from utils.config import setup


ctypes.cdll.LoadLibrary('caffe2_nvrtc.dll') 
data_dir = 'data/Pubfig65_adv2/'
batch_size = 32
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
internal_val = 0

def get_internal_representation(model, img):
    model(img)
    res = internal_val
    return res


def hook(model, inputs):
    global internal_val

    internal_val = inputs[0]

def val_model(model, data_loader):
    total = 0
    correct = 0

    model.eval()
    with torch.no_grad():
        for batch_idx, (imgs, labels) in enumerate(data_loader):
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, preds = torch.max(outputs.data, 1)

            total += len(labels)
            correct += torch.sum(preds == labels.data)

    print('{} Acc: {:.4f}%'.format("val",  correct * 100.0 / total))
    print('correct : {}, total : {}'.format(correct, total))
    print()

if __name__=="__main__":
    setup()
    target_loader = VGGFaceAdvDataLoader(data_dir + 'train/target/', batch_size, is_train=True, shuffle=False)
    attack_loader = VGGFaceAdvDataLoader(data_dir + 'train/attack/', batch_size, is_train=True, shuffle=False)
    
    test_loader = VGGFaceAdvDataLoader(data_dir + 'test/attack/', batch_size, is_train=False)
    orig_loader = VGGFaceDataLoader('data/PubFig65/', batch_size, is_train=False)
    # lambda
    lamb = 0.001 # temporary
    d_tar = 30.0
    lr = 0.01
    num_epoch = 100

    loss_fn = CrossEntropyLoss()

    model = VGG_Face_PubFig(saved=True)
    
    for param in model.parameters():
        param.requires_grad = True

    model.train()
    model[1].fc8.register_forward_pre_hook(hook)
    optimizer = SGD(model.parameters(), lr=lr)
    
    print('attack validation')
    #val_model(model, test_loader)
    print('''val Acc: 96.3077%\ncorrect : 626, total : 650\n''')
    


    print('original prediction rate')
    #val_model(model, orig_loader)
    print('''val Acc: 98.6154%\ncorrect : 641, total : 650\n''')
    for epoch in range(num_epoch):
        print('Epoch {}/{}'.format(epoch + 1, num_epoch))
        print('-' * 30)
        running_loss = 0.0

        # train
        for batch_idx, (targets, attacks) in enumerate(zip(target_loader, attack_loader)):
            target_imgs, target_labels = targets
            attack_imgs, attack_labels = attacks
            target_imgs, target_labels = target_imgs.to(device), target_labels.to(device)
            attack_imgs, attack_labels = attack_imgs.to(device), attack_labels.to(device)

            with torch.no_grad():
                sk_source = get_internal_representation(model, attack_imgs)
                sk_attack = get_internal_representation(model, target_imgs)

            model.train()
            optimizer.zero_grad()
            model.zero_grad()

            outputs = model(target_imgs)
            loss_ce = loss_fn(outputs, target_labels)

            dist = torch.dist(sk_source, sk_attack)
            loss_term = d_tar - dist
        
            loss = loss_ce + lamb * loss_term
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * target_imgs.size(0)

            
            print(loss_ce.data, (lamb * loss_term).data, loss.data)
        print('train loss :', running_loss)

        # validate
        print('attack validation')
        val_model(model, test_loader)

        print('original prediction rate')
        val_model(model, orig_loader)
        '''
        if epoch == 1:
            exit(0)
        '''

    torch.save(model.state_dict(), 'saved/VGGFace_PubFig65_aug.pt')
