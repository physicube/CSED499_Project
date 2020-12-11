from model.models import VGG16_GTSRB
from model.loss import CrossEntropyLoss
from model.optimizer import Adadelta
from trainer.trainer import Trainer
from data_loader.data_loader import GTRSBDataLoader
import ctypes

ctypes.cdll.LoadLibrary('caffe2_nvrtc.dll') 
data_dir = 'data'
batch_size = 32
num_workers = 4

if __name__=="__main__":
    model = VGG16_GTSRB()
    print(model.name)

    optimizer = Adadelta(model.parameters(), lr=1.0)
    loss_fn = CrossEntropyLoss()
    
    train_loader = GTRSBDataLoader(data_dir, batch_size, is_train=True)
    val_loader = GTRSBDataLoader(data_dir, batch_size, is_train=False)

    trainer = Trainer(model, loss_fn, optimizer, train_loader, val_loader=val_loader, num_epoch=50)

    trainer.train()
    trainer.save_best_weight('VGG16_GTRSB')