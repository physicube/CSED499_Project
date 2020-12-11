from model.models import VGG_Face_PubFig
from model.loss import CrossEntropyLoss
from model.optimizer import Adadelta
from trainer.trainer import Trainer
from data_loader.data_loader import VGGFaceDataLoader
from utils.config import fix_random_seed
import ctypes


ctypes.cdll.LoadLibrary('caffe2_nvrtc.dll') # for windows
data_dir = 'data/pubfig65/'
batch_size = 32
num_workers = 4
random_seed = 1337

if __name__=="__main__":
    fix_random_seed(random_seed)
    model = VGG_Face_PubFig()

    optimizer = Adadelta(model.parameters(), lr=1.0)
    loss_fn = CrossEntropyLoss()
    
    train_loader = VGGFaceDataLoader(data_dir, batch_size, is_train=True)
    val_loader = VGGFaceDataLoader(data_dir, batch_size, is_train=False)

    trainer = Trainer(model, loss_fn, optimizer, train_loader, val_loader=val_loader, num_epoch=200)

    trainer.train()
    trainer.save_best_weight('VGGFace_PubFig65_deep')