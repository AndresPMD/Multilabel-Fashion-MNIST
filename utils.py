import sys
import os
import time
import pickle

import math
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from torch.autograd import Variable

import matplotlib.pyplot as plt

import struct # get_image_size
import imghdr # get_image_size

import tensorflow as tf

def sigmoid(x):
    return 1.0/(math.exp(-x)+1.)

def softmax(x):
    x = torch.exp(x - torch.max(x))
    x = x/x.sum()
    return x


def bbox_iou(box1, box2, x1y1x2y2=True):
    if x1y1x2y2:
        mx = min(box1[0], box2[0])
        Mx = max(box1[2], box2[2])
        my = min(box1[1], box2[1])
        My = max(box1[3], box2[3])
        w1 = box1[2] - box1[0]
        h1 = box1[3] - box1[1]
        w2 = box2[2] - box2[0]
        h2 = box2[3] - box2[1]
    else:
        mx = min(box1[0]-box1[2]/2.0, box2[0]-box2[2]/2.0)
        Mx = max(box1[0]+box1[2]/2.0, box2[0]+box2[2]/2.0)
        my = min(box1[1]-box1[3]/2.0, box2[1]-box2[3]/2.0)
        My = max(box1[1]+box1[3]/2.0, box2[1]+box2[3]/2.0)
        w1 = box1[2]
        h1 = box1[3]
        w2 = box2[2]
        h2 = box2[3]
    uw = Mx - mx
    uh = My - my
    cw = w1 + w2 - uw
    ch = h1 + h2 - uh
    carea = 0
    if cw <= 0 or ch <= 0:
        return 0.0

    area1 = w1 * h1
    area2 = w2 * h2
    carea = cw * ch
    uarea = area1 + area2 - carea
    return carea/uarea

def bbox_ious(boxes1, boxes2, x1y1x2y2=True):
    if x1y1x2y2:
        mx = torch.min(boxes1[0], boxes2[0])
        Mx = torch.max(boxes1[2], boxes2[2])
        my = torch.min(boxes1[1], boxes2[1])
        My = torch.max(boxes1[3], boxes2[3])
        w1 = boxes1[2] - boxes1[0]
        h1 = boxes1[3] - boxes1[1]
        w2 = boxes2[2] - boxes2[0]
        h2 = boxes2[3] - boxes2[1]
    else:
        mx = torch.min(boxes1[0]-boxes1[2]/2.0, boxes2[0]-boxes2[2]/2.0) # XMIN OF PREDICTION AND GROUND TRUTH
        Mx = torch.max(boxes1[0]+boxes1[2]/2.0, boxes2[0]+boxes2[2]/2.0) # XMAX OF PREDICTION AND GROUND TRUTH
        my = torch.min(boxes1[1]-boxes1[3]/2.0, boxes2[1]-boxes2[3]/2.0) # SAME WITH YMIN AND YMAX:
        My = torch.max(boxes1[1]+boxes1[3]/2.0, boxes2[1]+boxes2[3]/2.0)
        w1 = boxes1[2]
        h1 = boxes1[3]
        w2 = boxes2[2]
        h2 = boxes2[3]
    uw = Mx - mx
    uh = My - my
    cw = w1 + w2 - uw
    ch = h1 + h2 - uh
    mask = ((cw <= 0) + (ch <= 0) > 0)
    area1 = w1 * h1
    area2 = w2 * h2
    carea = cw * ch
    carea[mask] = 0
    uarea = area1 + area2 - carea
    return carea/uarea

def nms(boxes, nms_thresh):
    if len(boxes) == 0:
        return boxes
    # RECEIVES BOXES LIST, WITH DIMENSION LIST OF BOXES ( X * Y *W * H * PHOC)
    det_confs = torch.zeros(len(boxes))
    for i in range(len(boxes)): # 1 - CONFIDENCE TO SORT ?
        det_confs[i] = 1-boxes[i][4]                

    _,sortIds = torch.sort(det_confs)
    out_boxes = []
    for i in range(len(boxes)):
        box_i = boxes[sortIds[i]] # BOXES WITH BETTER CONFIDENCES FIRST
        if box_i[4] > 0: # IF CONFIDENCE BIGGER THAN  0 APPEND
            out_boxes.append(box_i)
            for j in range(i+1, len(boxes)): # COMPARE IOU, IF IT IS BIGGER THAN THE NMS_THRESHOLD MAKE THE LESS CONFIDENT 0
                box_j = boxes[sortIds[j]]
                if bbox_iou(box_i, box_j, x1y1x2y2=False) > nms_thresh:
                    #print(box_i, box_j, bbox_iou(box_i, box_j, x1y1x2y2=False))
                    box_j[4] = 0

    return out_boxes

def sort_phocs(boxes, nms_thresh):
    if len(boxes) == 0:
        return boxes
    # RECEIVES BOXES LIST, WITH DIMENSION LIST OF BOXES ( X * Y *W * H * PHOC)
    det_confs = torch.zeros(len(boxes))
    for i in range(len(boxes)): # 1 - CONFIDENCE TO SORT
        det_confs[i] = 1-boxes[i][4]

    _,sortIds = torch.sort(det_confs)
    out_boxes = []

    for i in range(len(boxes)):
        if i <= 30:
            box_i = boxes[sortIds[i]]
            if box_i[4] > 0:
                out_boxes.append(box_i)
        else:
            break

    return out_boxes

    '''
        box_i = boxes[sortIds[i]] # BOXES WITH BETTER CONFIDENCES FIRST
        if box_i[4] > 0: # IF CONFIDENCE BIGGER THAN  0 APPEND
            out_boxes.append(box_i)
            for j in range(i+1, len(boxes)): # COMPARE IOU, IF IT IS BIGGER THAN THE NMS_THRESHOLD MAKE THE LESS CONFIDENT 0
                box_j = boxes[sortIds[j]]
                if bbox_iou(box_i, box_j, x1y1x2y2=False) > nms_thresh:
                    #print(box_i, box_j, bbox_iou(box_i, box_j, x1y1x2y2=False))
                    box_j[4] = 0
    '''


def convert2cpu(gpu_matrix):
    return torch.FloatTensor(gpu_matrix.size()).copy_(gpu_matrix)

def convert2cpu_long(gpu_matrix):
    return torch.LongTensor(gpu_matrix.size()).copy_(gpu_matrix)



def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# To debug
def getclass_from_label(label):
    max_index = np.argmax(label)
    classes = ['Bakery','Barber', 'Bistro', 'Bookstore', 'Cafe', 'ComputerStore','CountryStore', 'Dinner',
               'DiscountHouse', 'DryCleaner','Funeral', 'Hotspot', 'MassageCenter', 'MedicalCenter', 'PackingStore',
               'PawnShop','PetShop', 'Pharmacy', 'Pizzeria', 'RepairShop', 'Restaurant', 'School', 'SteakHouse',
               'Tavern', 'TeaHouse', 'Theatre', 'Tobacco', 'Motel']
    classname = classes[max_index]
    print (classname)

def add_summary_value(writer, key, value, iteration, printGraphic=0):
    summary = tf.Summary(value=[tf.Summary.Value(tag=key, simple_value=value)])
    writer.add_summary(summary, iteration)

    if printGraphic:
        writer.flush()

def load_phoc_features(result_path, max_phocs):
    '''
    Reads the PHOC results and returns a tensor
    :param result_path, max_phocs
    :return: numpy array
    '''
    text_file = result_path
    phoc_full_string = ''
    max_phocs = max_phocs
    phoc_matrix = np.zeros((max_phocs, 604))
    with open(text_file) as f:
        phoc_list = list()
        for line in f:
            if (line[0]) == '[':
                phoc_full_string = ''
            if (line[-2]) == ']':  # or [-2] because of \n - > Means last line
                phoc_string = line.replace("\n", "").replace("[", "").replace("]", "")
                phoc_full_string = phoc_full_string + phoc_string
                # ORIGINAL CODE: phoc_vector = np.array(list(phoc_full_string))
                phoc_vector = np.fromstring(phoc_full_string, dtype=float, count=-1, sep=' ')

                # print("NAIVE EMB SIZE IS: ", np.shape(phoc_vector))
                phoc_list.append(phoc_vector)
                continue

            phoc_string = line.replace("\n", "").replace("[", "").replace("]", "")
            phoc_full_string = phoc_full_string + phoc_string

    for i in range (len(phoc_list)):
        if i >= max_phocs:
            break
        phoc_matrix[i][:] = phoc_list[i]

    return phoc_matrix

def load_features(result_path, max_words):
    '''
    Reads the results and returns a tensor according to the textual embedding
    :param result_path, max_words
    :return: numpy array
    '''
    text_file = result_path
    embedded_full_string = ''
    max_words = max_words
    embedded_matrix = np.zeros((max_words, 300))
    with open(text_file) as f:
        embedded_list = list()
        for line in f:
            if (line[0]) == '[':
                embedded_full_string = ''
            if (line[-2]) == ']':  # or [-2] because of \n - > Means last line
                embedded_string = line.replace("\n", "").replace("[", "").replace("]", "")
                embedded_full_string = embedded_full_string + embedded_string
                embedded_vector = np.fromstring(embedded_full_string, dtype=float, count=-1, sep=' ')

                embedded_list.append(embedded_vector)
                continue

            embedded_string = line.replace("\n", "").replace("[", "").replace("]", "")
            embedded_full_string = embedded_full_string + embedded_string

    for i in range (len(embedded_list)):
        if i >= max_words:
            break
        embedded_matrix[i][:] = embedded_list[i]

    return embedded_matrix

def load_vector(result_path):
    text_file = open(result_path,'r')
    lines = text_file.readlines()
    vector_list = list()
    for line in (lines):
        vector_list.append(line.strip())

    text_file.close()
    vector = np.array(vector_list, dtype=np.float32)
    return vector

def load_visual_features(result_path):
    '''
    Reads the results and returns a tensor according to the textual embedding
    :param result_path, max_words
    :return: numpy array
    '''
    text_file = result_path
    embedded_full_string = ''

    embedded_matrix = np.zeros((1, 2048))
    with open(text_file) as f:
        embedded_list = list()
        for line in f:
            if (line[0]) == '[':
                embedded_full_string = ''
            if (line[-2]) == ']':  # or [-2] because of \n - > Means last line
                embedded_string = line.replace("\n", "").replace("[", "").replace("]", "")
                embedded_full_string = embedded_full_string + embedded_string
                embedded_vector = np.fromstring(embedded_full_string, dtype=float, count=-1, sep=' ')

                embedded_list.append(embedded_vector)
                continue

            embedded_string = line.replace("\n", "").replace("[", "").replace("]", "")
            embedded_full_string = embedded_full_string + embedded_string

    for i in range (len(embedded_list)):
        embedded_matrix[0][i*512:(i+1)*512] = embedded_list[i]
    return embedded_matrix
