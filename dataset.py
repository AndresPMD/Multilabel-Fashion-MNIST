#!/usr/bin/python
# encoding: utf-8

import os
import random
import torch
import numpy as np
import time
from torch.utils.data import Dataset

import cPickle

from PIL import Image

from image import *
from utils import *


# --------------------------------- *** FASHION MNIST FOR MULTILABEL DATASET *** --------------------

class fashionDataset(Dataset):

    def __init__(self, root, num_classes=None, shape=None, shuffle=True, transform=None, target_transform=None, train=False, seen=0, batch_size=32, num_workers=10):
       with open(root, 'r') as file:
           self.lines = file.readlines()

       if shuffle:
           random.shuffle(self.lines)

       self.nSamples  = len(self.lines)
       self.transform = transform
       self.target_transform = target_transform
       self.train = train
       self.shape = shape
       self.seen = seen
       self.batch_size = batch_size
       self.num_workers = num_workers
       self.num_classes = num_classes

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        imgpath = self.lines[index].rstrip()

        if self.train:
            jitter = 0.1
            hue = 0.1
            saturation = 1.5
            exposure = 1.5
            # GET THE DATA AUGMENTED IMAGE AND A VECTOR LABEL WITH GROUND TRUTH

            img, label= load_data_train_fashion(imgpath, self.num_classes, self.shape, jitter, hue, saturation, exposure)
            label = torch.from_numpy(label)
            label = label.type(torch.FloatTensor)

        else:

            img,label = load_data_test_fashion(imgpath, self.num_classes, self.shape)
            label = torch.from_numpy(label)
            label = label.type(torch.FloatTensor)

        # TRANSFORM IMAGE TO TORCH TENSOR
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        self.seen = self.seen + self.num_workers

        #t2 = int(round(time.time() * 1000))
        #print("Time to BATCHER: %.4f ms" % (t2 - t1))

        return img, label

class fashionDataset_LSTM(Dataset):

    def __init__(self, root, num_classes=None, shape=None, shuffle=True, transform=None, target_transform=None, train=False, seen=0, batch_size=32, num_workers=10):
       with open(root, 'r') as file:
           self.lines = file.readlines()

       if shuffle:
           random.shuffle(self.lines)

       self.nSamples = len(self.lines)
       self.transform = transform
       self.target_transform = target_transform
       self.train = train
       self.shape = shape
       self.seen = seen
       self.batch_size = batch_size
       self.num_workers = num_workers
       self.num_classes = num_classes

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        imgpath = self.lines[index].rstrip()

        if self.train:
            jitter = 0.1
            hue = 0.1
            saturation = 1.5
            exposure = 1.5
            # GET THE DATA AUGMENTED IMAGE AND A VECTOR LABEL WITH GROUND TRUTH

            img, label= load_data_train_fashion(imgpath, self.num_classes, self.shape, jitter, hue, saturation, exposure)
            label = torch.from_numpy(label)
            label = label.type(torch.FloatTensor)

        else:

            img,label = load_data_test_fashion(imgpath, self.num_classes, self.shape)
            label = torch.from_numpy(label)
            label = label.type(torch.FloatTensor)

        # TRANSFORM IMAGE TO TORCH TENSOR
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        self.seen = self.seen + self.num_workers

        #t2 = int(round(time.time() * 1000))
        #print("Time to BATCHER: %.4f ms" % (t2 - t1))

        return img, label