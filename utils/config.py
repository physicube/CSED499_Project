import torch
import random, numpy, ctypes

random_seed = 1337

def os_setup():
    ctypes.cdll.LoadLibrary('caffe2_nvrtc.dll') 


def _fix_random_seed(seed: int):
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

def setup():
    _fix_random_seed(random_seed)