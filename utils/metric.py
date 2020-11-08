from utils.pytorch_ssim import SSIM
import torch.nn as nn

def DSSIM(x, y):
    ssim_loss = SSIM()

    return (1.0 - ssim_loss(x, y)) / 2.0
