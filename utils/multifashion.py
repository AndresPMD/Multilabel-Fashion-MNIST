import sys
sys.path.insert(0, "/SSD/fashion-mnist/utils/")
from mnist_reader import *
from tqdm import tqdm
from matplotlib.pyplot import imshow
import PIL
import numpy as np
from PIL import Image
import os


images_path = '/SSD/fashion-mnist/images/'
labels_path = '/SSD/fashion-mnist/labels/'

img, label = load_mnist("/SSD/fashion-mnist/data/", 'train')
test_img, test_label = load_mnist("/SSD/fashion-mnist/data/", 't10k')

img = np.concatenate((img,test_img))
label = np.concatenate((label,test_label))

# FORM W = 28+28 AND H = 28+28 IMAGES AND TAKE ITS LABELS FOR MULTICLASS. (56x56)

for i in range (np.shape(img)[0]/4):
    #new_label = [label[4*i], label[4*i+1], label[4*i+2], label[4*i+3]]
    #text_label = open(labels_path+str(i)+'.txt','w')
    #text_label.write(str(new_label))
    #text_label.close()

    a = np.reshape(img[4*i], (28,28))
    b = np.reshape(img[4*i+1], (28, 28))
    comb0 = np.concatenate((a,b), axis = 1)
    c = np.reshape(img[4*i+2], (28, 28))
    d = np.reshape(img[4*i+3], (28, 28))
    comb1 = np.concatenate((c, d), axis=1)
    comb = np.concatenate((comb0,comb1), axis=0)
    new_img = Image.fromarray(comb)
    new_img.save(images_path+str(i)+'.jpg')



