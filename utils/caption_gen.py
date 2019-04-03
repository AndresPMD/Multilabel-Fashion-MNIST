import os
import numpy as np
from tqdm import tqdm

classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
label_path = '/SSD/fashion-mnist/labels/'
labels = os.listdir(label_path)

captions_path = '/SSD/fashion-mnist/captions/'


for label_file in tqdm(labels):
    file = open(label_path+label_file)
    file_label = file.readline().strip().replace('[','').replace(']','')
    label_vector = (np.fromstring(file_label, dtype = int, sep = ','))

    caption_file = open(captions_path+label_file,'wb')
    full_caption = ['<start>']
    for i in range (0, np.shape(label_vector)[0]):
        pos = label_vector[i]
        full_caption.append(classes[pos])
    full_caption.append('<end>')
    caption_file.write(str(full_caption))
    caption_file.close()










