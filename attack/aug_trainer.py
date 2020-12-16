import torch
from torch import device
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
import time
import copy

class AugTrainer:
    def __init__(self,
        model: nn.Module,
        loss_fn: nn.Module,
        optimizer: nn.Module,
        target_loader: DataLoader,
        attack_loader: DataLoader,
        val_loader: DataLoader,
        num_epoch: int = 100
        ):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer

        self.target_loader = target_loader
        self.attack_loader = attack_loader
        self.val_loader = val_loader
        self.num_epoch = num_epoch

    def _train_epoch(self, epoch):
        print('Epoch {}/{}'.format(epoch + 1, self.num_epoch))
        print('-' * 10)

        for batch_idx, (targets, attacks) in enumerate(zip(self.target_loader, self.attack_loader)):
            target_imgs, target_labels = targets
            attack_imgs, attack_labels = attacks
            target_imgs, target_labels = target_imgs.cuda(), target_labels.cuda()
            attack_imgs, attack_labels = attack_imgs.cuda(), attack_labels.cuda()

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

            print(loss_ce.data, loss_term.data)
            print('batch :', batch_idx,  'loss :', loss.data)