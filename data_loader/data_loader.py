from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from data_loader.gtsrb import GTSRB
from data_loader.pubfig65 import PubFig65
from random import sample, randint

class BaseDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, shuffle, num_workers, pin_memory=True):
        super().__init__(dataset, batch_size=batch_size, 
                                            shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)

    def get_random_batch(self):
        return sample(list(iter(self)), 1)[0]


    def get_random_sample(self):
        (imgs, labels) = self.get_random_batch()
        idx = randint(0, len(imgs) - 1)

        return (imgs[idx], labels[idx])

class VGGFaceDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle=True, num_workers=4, is_train=True):
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])

        self.data_dir = data_dir
        if is_train:
            self.dataset = PubFig65(data_path=self.data_dir, is_train=True, transform=transform)
        else:
            self.dataset = PubFig65(data_path=self.data_dir, is_test=True, transform=transform)
        self.dataset_size = len(self.dataset)
        self.is_train = is_train

        super().__init__(self.dataset, batch_size=batch_size, 
                                            shuffle=shuffle, num_workers=num_workers, pin_memory=True)


class GTRSBDataLoader(BaseDataLoader):
    """
    GTSRB data loader with custom validation set
    """
    def __init__(self, data_dir, batch_size, shuffle=True, num_workers=4, is_train=True):
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        self.data_dir = data_dir
        self.dataset = GTSRB(root_dir=self.data_dir, train=is_train, transform=self.transform)
        self.dataset_size = len(self.dataset)
        self.is_train = is_train

        super().__init__(self.dataset, batch_size=batch_size, 
                                            shuffle=shuffle, num_workers=num_workers, pin_memory=True)
class VGGFlowerDataLoader(BaseDataLoader):
    """
    GTSRB data loader with custom validation set
    """
    def __init__(self, data_dir, batch_size, shuffle=True, num_workers=4, is_train=True):
        if is_train:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(30),
                transforms.ToTensor()
            ])
            self.data_dir = data_dir + 'train/'
        else:
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor()
            ])
            self.data_dir = data_dir + 'valid/'
        self.dataset = datasets.ImageFolder(self.data_dir, transform=self.transform)
        self.dataset_size = len(self.dataset)
        self.is_train = is_train

        super().__init__(self.dataset, batch_size=batch_size, 
                                            shuffle=shuffle, num_workers=num_workers, pin_memory=True)                                  