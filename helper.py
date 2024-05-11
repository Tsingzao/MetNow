import torch
from torch.utils import data
from torchvision import transforms
import numpy as np
import os
import cv2
import glob
import netCDF4 as nc
import json
import h5py
import random


class MetData(data.Dataset):
    def __init__(self, split='test', transform=None):
        super(MetData, self).__init__()
        self.data = None
        self.split = split
        self.transform = transform
        self.get_files()

    def __getitem__(self, item):
        img = self.data[item].transpose((1,2,0))/70.
        if self.transform:
            img = self.transform(img)
        return img.float()

    def __len__(self):
        return len(self.data)

    def get_files(self):
        with h5py.File('./data/data.h5', 'r') as h:
            data = h['data'][:]
        self.data = data

