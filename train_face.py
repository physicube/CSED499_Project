from model.models import VGG_Face_PubFig
from model.loss import CrossEntropyLoss
from model.optimizer import Adadelta
from trainer.trainer import Trainer
from data_loader.data_loader import VGGFaceDataLoader
import ctypes


ctypes.cdll.LoadLibrary('caffe2_nvrtc.dll') 
data_dir = 'data/pubfig65/'
batch_size = 32
num_workers = 4

if __name__=="__main__":
    model = VGG_Face_PubFig()
    print(model)
    optimizer = Adadelta(model.parameters(), lr=1.0)
    loss_fn = CrossEntropyLoss()
    
    train_loader = VGGFaceDataLoader(data_dir, batch_size, is_train=True)
    val_loader = VGGFaceDataLoader(data_dir, batch_size, is_train=False)

    trainer = Trainer(model, loss_fn, optimizer, train_loader, val_loader=val_loader, num_epoch=200)

    trainer.train()