import torch
import torch.nn as nn
from torch.optim import lr_scheduler, Adadelta
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import ctypes
import time
import copy

from vgg_face_dag import vgg_face_dag
from datasets import PubFig83

dataroot = 'pubfig83/'
num_features = 83
batch_size = 32
image_size = 224
num_workers = 4

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ctypes.cdll.LoadLibrary('caffe2_nvrtc.dll') # to fix print error of pytorch 

# dataset
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

train_dataset = PubFig83(is_train=True, data_path=dataroot, transform=transform)
val_dataset = PubFig83(is_test=True, data_path=dataroot, transform=transform)

class_names = train_dataset.classes
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, 
                                        shuffle=True, num_workers=num_workers, pin_memory=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, 
                                        shuffle=True, num_workers=num_workers, pin_memory=True)

dataloaders = {'train': train_dataloader, 'val': val_dataloader}
dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}


def initialize_model() -> nn.Module:
    model = vgg_face_dag('vgg_face_dag.pth')

    # freeze layers
    for param in model.parameters():
        param.requires_grad = False
        
    model.fc8 = nn.Linear(4096, num_features)
    model.cuda()

    return model


def train_model(model: nn.Module, criterion, optimizer, scheduler, num_epochs=200) -> None:
    start_time = time.time()

    best_model_weights = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        # forward propagation

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            
            for batch_idx, (inputs, labels) in enumerate(dataloaders[phase]):
                inputs, labels = inputs.to(device), labels.to(device)
                
                optimizer.zero_grad()

                # forward
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs.data, 1)
                    loss = criterion(outputs, labels)

                    # backward
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # copy best weights
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_weights = copy.deepcopy(model.state_dict())
        print()

    time_elapsed = time.time() - start_time
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    torch.save(best_model_weights, 'model_state_dict.pt')

    model.load_state_dict(best_model_weights)
    return model

if __name__ == '__main__':
    model = initialize_model()
    optimizer = torch.optim.Adadelta(model.parameters(), lr=1.0)
    loss_fn = torch.nn.CrossEntropyLoss()

    train_model(model, loss_fn, optimizer, '')