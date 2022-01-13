# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 13:46:07 2022

@author: Rahul Batish
"""

from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import torch
import torch.nn as nn
from torchvision.utils import make_grid
from torchvision.utils import save_image
from IPython.display import Image
import matplotlib.pyplot as plt
import numpy as np
import random
import torchvision
import torch.optim as optim
from torchvision import datasets, transforms

################################ image show  ##################################

def imshow(image, ax=None, title=None, normalize=True):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    image = image.numpy().transpose((1, 2, 0))

    if normalize:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean
        image = np.clip(image, 0, 1)

    ax.imshow(image)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='both', length=0)
    ax.set_xticklabels('')
    ax.set_yticklabels('')

    return ax
    
def returnloader(DATA_DIR = './celeb',insize = 256,outsize=8,batch_size=32):  
    transform = transforms.Compose([transforms.Resize(outsize),
                                     transforms.ToTensor()])
    dataset = datasets.ImageFolder(DATA_DIR, transform=transform) # TODO: create the ImageFolder
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32) # TODO: use the ImageFolder dataset to create the DataLoader
    return dataloader

# Run this to test your data loader
dataloader=returnloader()
images, labels = next(iter(dataloader))
for images , labels in dataloader:
    for image in images:
        print(image.shape)
        # image=maxpooler(image,256,4)
        print(image.shape)
        imshow(image, normalize=False)
        
    break

#################################### model ###################################








