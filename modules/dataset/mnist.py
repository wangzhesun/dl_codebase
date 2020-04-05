import torch
from torchvision import datasets, transforms

def get_train_set(cfg):
    return datasets.MNIST('../datasets', train = True, download = True,
                       transform=transforms.Compose([
                           transforms.ToTensor()
                       ]))

def get_val_set(cfg):
    pass

def get_test_set(cfg):
    return datasets.MNIST('../datasets', train = False, download = True,
                        transform=transforms.Compose([
                           transforms.ToTensor()
                       ]))