import torch
from torch import device
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
import time
import copy

class Trainer:
    """
    Trainer class
    """
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: nn.Module,
        train_loader: DataLoader,
        lr_scheduler: nn.Module = None,
        val_loader: DataLoader = None,
        num_epoch: int = 100
        ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        # dataloaders
        self.train_loader = train_loader
        self.val_loader = val_loader

        # config
        self.num_epoch = num_epoch
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # metadata
        self.best_acc = 0.0
        self.best_model_weights = None

    def _train_epoch(self, epoch):
        running_loss = 0.0
        running_corrects = 0

        self.model.train()
        for batch_idx, (inputs, labels) in enumerate(self.train_loader):
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            _, preds = torch.max(outputs.data, 1)
            loss = self.criterion(outputs, labels)

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        
        epoch_loss = running_loss / self.train_loader.dataset_size
        epoch_acc = running_corrects.double() / self.train_loader.dataset_size
        print('{} Loss: {:.4f} Acc: {:.4f}'.format("train", epoch_loss, epoch_acc))


    def _val_epoch(self, epoch):
        running_loss = 0.0
        running_corrects = 0

        self.model.eval()
        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(self.val_loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
        
        epoch_loss = running_loss / self.val_loader.dataset_size
        epoch_acc = running_corrects.double() / self.val_loader.dataset_size
        print('{} Loss: {:.4f} Acc: {:.4f}'.format("val", epoch_loss, epoch_acc))

        if epoch_acc > self.best_acc:
            self.best_acc = epoch_acc
            self.best_model_weights = copy.deepcopy(self.model.state_dict())

    def train(self):
        start_time = time.time()

        for epoch in range(self.num_epoch):
            print('Epoch {}/{}'.format(epoch, self.num_epoch - 1))
            print('-' * 10)

            self._train_epoch(epoch)

            if self.val_loader:
                self._val_epoch(epoch)
                
            print()
            
        time_elapsed = time.time() - start_time
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(self.best_acc))

        final_weights = self.best_model_weights if self.val_loader else self.model.state_dict()
        model_name = self.model.__class__.__name__
        torch.save(final_weights, 'saved/{}_weight.pt'.format(model_name))

        # load best val accuracy
        self.model.load_state_dict(final_weights)
        return self.model

