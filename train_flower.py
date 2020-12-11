from model.models import ResNet50_VGGFlower
from model.loss import CrossEntropyLoss
from model.optimizer import SGD
from trainer.trainer import Trainer
from data_loader.data_loader import VGGFlowerDataLoader
import ctypes


ctypes.cdll.LoadLibrary('caffe2_nvrtc.dll') 
data_dir = 'data/flower_data/'
batch_size = 50
num_workers = 4

if __name__=="__main__":
    model = ResNet50_VGGFlower()
    print(model)
    optimizer = SGD(model.parameters(), lr=0.01)
    loss_fn = CrossEntropyLoss()
    
    train_loader = VGGFlowerDataLoader(data_dir, batch_size, is_train=True)
    val_loader = VGGFlowerDataLoader(data_dir, batch_size, is_train=False)
    print(train_loader.dataset_size, val_loader.dataset_size)
    
    trainer = Trainer(model, loss_fn, optimizer, train_loader, val_loader=val_loader, num_epoch=150)

    trainer.train()
    trainer.save_best_weight('ResNet50_VGGFlower')