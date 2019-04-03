#!/usr/bin/python
# encoding: utf-8
import random
import os
from PIL import Image, ImageFile
import numpy as np
from utils import *
import logging
import time


def scale_image_channel(im, c, v):
    cs = list(im.split())
    cs[c] = cs[c].point(lambda i: i * v)
    out = Image.merge(im.mode, tuple(cs))
    return out

def distort_image(im, hue, sat, val):
    im = im.convert('HSV')
    cs = list(im.split())
    cs[1] = cs[1].point(lambda i: i * sat)
    cs[2] = cs[2].point(lambda i: i * val)
    
    def change_hue(x):
        x += hue*255
        if x > 255:
            x -= 255
        if x < 0:
            x += 255
        return x
    cs[0] = cs[0].point(change_hue)
    im = Image.merge(im.mode, tuple(cs))

    im = im.convert('RGB')
    #constrain_image(im)
    return im

def rand_scale(s):
    scale = random.uniform(1, s)
    if(random.randint(1,10000)%2): 
        return scale
    return 1./scale

def random_distort_image(im, hue, saturation, exposure):
    dhue = random.uniform(-hue, hue)
    dsat = rand_scale(saturation)
    dexp = rand_scale(exposure)
    res = distort_image(im, dhue, dsat, dexp)
    return res

def data_augmentation(img, shape, jitter, hue, saturation, exposure):
    oh = img.height  
    ow = img.width
    
    dw =int(ow*jitter)
    dh =int(oh*jitter)

    pleft  = random.randint(-dw, dw)
    pright = random.randint(-dw, dw)
    ptop   = random.randint(-dh, dh)
    pbot   = random.randint(-dh, dh)

    swidth =  ow - pleft - pright
    sheight = oh - ptop - pbot

    sx = float(swidth)  / ow
    sy = float(sheight) / oh
    

    cropped = img.crop( (pleft, ptop, pleft + swidth - 1, ptop + sheight - 1))

    dx = (float(pleft)/ow)/sx
    dy = (float(ptop) /oh)/sy

    sized = cropped.resize(shape)

    flip = random.randint(1,1000)%2
    if flip:
       sized = sized.transpose(Image.FLIP_LEFT_RIGHT)
    flip = 0
    img = random_distort_image(sized, hue, saturation, exposure)

    return img, flip, dx,dy,sx,sy 

def fill_truth_detection(labpath):

    if os.path.getsize(labpath):
        text_file = open(labpath, 'r')
        lines = text_file.read()
        label = np.array(list(lines.replace('[', '').replace(']', '').replace('-1.', '0').replace('0.', '0').replace('1.', '1').replace(" ", "").replace(
                '\n', '')), dtype=np.float)
        label = np.reshape(label, (-1))
        return label
    else:
        print("Label Path Error..!")


def load_data_detection(imgpath, shape, jitter, hue, saturation, exposure):
    labpath = imgpath.replace('JPEGImages', 'split1/train').replace('.jpg', '.txt').replace('.png','.txt')
    ## Data augmentation
    img = Image.open(imgpath).convert('RGB')
    img,flip,dx,dy,sx,sy = data_augmentation(img, shape, jitter, hue, saturation, exposure)
    label = fill_truth_detection(labpath)
    # To debug
    # getclass_from_label(label)
    return img,label


def load_data_test(imgpath, shape):
    labpath = imgpath.replace('JPEGImages', 'split1/test').replace('.jpg', '.txt').replace('.png','.txt')
    img = Image.open(imgpath).convert('RGB')
    img = img.resize(shape)
    label = fill_truth_detection(labpath)

    return img,label


### FUNCTIONS FOR PRODUCT DATASET ###

def load_data_train_fashion(imgpath, num_classes, shape, jitter, hue, saturation, exposure):

    labpath = imgpath.replace('images', 'labels').replace('jpg','txt')
    ## Data augmentation
    img = Image.open(imgpath).convert('RGB')
    img,flip,dx,dy,sx,sy = data_augmentation(img, shape, jitter, hue, saturation, exposure)

    label = fill_multilabel(labpath)
    return img,label


def load_data_test_fashion(imgpath, num_classes, shape):

    labpath = imgpath.replace('images', 'labels').replace('jpg','txt')
    img = Image.open(imgpath).convert('RGB')
    img = img.resize(shape)
    label = fill_multilabel(labpath)
    return img,label


def fill_multilabel(labpath):
    if os.path.getsize(labpath):
        text_file = open(labpath, 'r')
        line = text_file.read()
        label_vector = np.asarray(line.strip().replace('[', '').replace(']', '').replace(',', '').split()).astype(int).tolist()
        multilabel = np.zeros(10)
        for position in label_vector:
            multilabel[position] = 1
        return multilabel
    else:
        print("Label Path Error..!")


