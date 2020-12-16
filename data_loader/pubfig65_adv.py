import numpy as np
from torch.utils.data import Dataset
from os import path
from PIL import Image

class PubFig65Adv(Dataset):    
    data_path = path.join("/","hdd_data","pub","pubfig83","dataset/")

    def __init__(self, is_train=False, is_test=False, is_val=False, transform=None, 
                 on_memory=True, data_path=None):
        if (is_train and is_test) or (is_train and is_val) or (is_val and is_test):
            raise ValueError
        if not(is_train or is_test or is_val):
            raise ValueError
        if data_path:
            self.data_path = data_path

        self.is_train = is_train
        self.is_test = is_test
        self.is_val = is_val
        self.transform = transform
        self.on_memory = on_memory
        
        self._load_class()
        if on_memory:
            self._load_memory()
        
    def _load_class(self):
        from os import listdir
        self.classes = listdir(self.data_path)
        sorted_classes = list(map(int, self.classes))
        sorted_classes.sort()

        self.classes = list(map(str, sorted_classes))
        
    def _load_memory(self):
        from os import listdir
        start, end = 0, None

        
        self.images = []
        self.labels = []
        for identity in self.classes:
            files = listdir(self.data_path+identity)
            files = files[start:end]
            label = self.classes.index(identity)
            for filename in files:
                item_path = path.join(self.data_path, identity, filename)
                with Image.open(item_path) as item:
                    self.images.append(np.array(item))
                    self.labels.append(label)
            
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, label) where target is index of the target class.
        """
        image = self.images[index]
        image = Image.fromarray(image)
        # self.idx ranges 1-8189, labels ranges 0-8188
        label = self.labels[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.labels)
