from __future__ import print_function

import torch.optim as optim
import torchvision
from torch.optim import Adam
from torchvision import transforms
import dataset
from utils import *
from models import *
import time
import os
from PIL import Image
from sklearn.metrics import average_precision_score, f1_score


threshold = 0.5
# Parameters
trainlist = '/SSD/fashion-mnist/trainlist.txt'
testlist = '/SSD/fashion-mnist/testlist.txt'

# Debug:
backupdir = '/SSD/fashion-mnist/backup_lstm/'
# Normal run
#backupdir = '/SSD/fashion-mnist/backup/'


load_pretrained_model = False
model_weight_file = 'weights0007.pt'

gpus          = '0'  # e.g. 0,1,2,3
ngpus         = len(gpus.split(','))
num_workers   = 4

batch_size    = 256
learning_rate = 0.00001#0.1
momentum  = 0.9
decay = 0.0005

nsamples = sum(1 for line in open(trainlist))
max_epochs    = 15
max_batches   = (nsamples)/batch_size * max_epochs
num_classes = sum(1 for line in open('/SSD/fashion-mnist/classes.txt'))
use_cuda      = True
seed          = int(time.time())
save_interval = 1  # epoches
validation_interval = 1 # epoches

init_width = 224
init_height = 224

if not os.path.exists(backupdir):
    os.mkdir(backupdir)
###############
print('--- MODEL PARAMETERS ---')
print("Num classes: ", num_classes)
print("Batch size: ", batch_size)
print("Starting learning rate: ", learning_rate)
###############
torch.manual_seed(seed)
if use_cuda:
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus
    torch.cuda.manual_seed(seed)

# CREATE THE MODEL
print('Creating Model...')
resnet152 = torchvision.models.resnet152(pretrained=True)

model = nn.Sequential(*list(resnet152.children())[:-1])
for param in model.parameters():
    param.requires_grad = False

# Baseline
model_net = Simple_LSTM(num_classes)

for param in model_net.parameters():
    param.requires_grad = True

if use_cuda:
    model = model.cuda()
    model_net = model_net.cuda()

# LOSS CRITERIA
criterion = nn.BCEWithLogitsLoss()
#criterion = nn.MSELoss()

# OPTIMIZER
#optimizer = optim.SGD(model_net.parameters(), lr=learning_rate, momentum=momentum, dampening=0, weight_decay=decay*batch_size)

optimizer = optim.Adam(params=filter(lambda p: p.requires_grad, model_net.parameters()), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=decay*batch_size)

# LOAD WEIGHTS and VARIABLE seen
seen = 0
processed_batches = seen/batch_size

# TEST LOADER
test_loader = torch.utils.data.DataLoader(dataset.fashionDataset_LSTM(testlist, num_classes=num_classes, shape=(init_width, init_height),
                                                                 shuffle=False, transform=transforms.Compose([transforms.ToTensor(), ]),
                                                                 train=False, seen=seen, batch_size=batch_size,
                                                                 num_workers=num_workers),
                                          batch_size=batch_size, shuffle=False)

def adjust_learning_rate(optimizer, processed_batches):
    """Sets the learning rate to the initial LR decayed by 10 every fixed epochs"""
    # CODE A POLICY OF DECAYING LEARNING RATE ?
    lr = float(learning_rate/10)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def viewer(data, index):
    # Allows to see data images for debugging
    j = ((data.data.cpu().numpy()[index]) * 255).astype(int)
    j = j.astype('uint8')
    img = Image.fromarray(j[0])
    img.show()
    return

def train(epoch):
    global processed_batches
    global learning_rate
    global seen

    model.train()
    model_net.train()

    train_loader = torch.utils.data.DataLoader(dataset.fashionDataset_LSTM(trainlist, num_classes, shape=(init_width, init_height),
                                                                      shuffle=True, transform=transforms.Compose([transforms.ToTensor(),]),
                                                                      train=True, seen=seen, batch_size=batch_size, num_workers=num_workers),
                                               batch_size=batch_size, shuffle=False)

    # ADJUST LEARNING RATE PER EPOCH
    if epoch == 2 or epoch == 4:
        learning_rate = adjust_learning_rate(optimizer, processed_batches)
        print('Current Learning rate is: %f' % (optimizer.param_groups[0]['lr']))


    # Begging training...
    t1 = int(round(time.time() * 1000))
    for batch_idx, (data, labels) in enumerate (train_loader):
        #t3 =  int(round(time.time() * 1000))
        sample_size = np.shape(data)[0]
        if use_cuda:
            data = data.cuda()
            labels = labels.cuda()
        data = Variable(data)
        labels = Variable(labels)

        optimizer.zero_grad()

        resnet_features = model(data)
        output = model_net(resnet_features.view(sample_size, -1))

        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        processed_batches += 1
        seen = batch_size * processed_batches

        #if (seen % 2048 == 0):
        #    test()

        t2 = int(round(time.time() * 1000))
        if batch_idx % 10 == 0:
            print('[Epoch: %d, Data Points Seen: %5d] Loss is: %.4f  Time/Epoch: %4.f ms' % (epoch, seen, loss.data[0], float(t2-t1)))
        #print("Dataloader time is: ", t3-t1)
        t1 = int(round(time.time() * 1000))

    # Set the saving periodicity
    if (epoch + 1) % save_interval == 0:
        print('---------- SAVING MODEL ----------')
        save_name = 'weights000' + str(epoch) + '.pt'
        torch.save(model_net.state_dict(), backupdir + save_name)
        print('Model Saved as:', save_name)


def test():

    average_precision_vector = []
    average_f1score_vector = []
    test_batches = 0

    model.eval()
    model_net.eval()
    print('---------- TEST SET ---------')
    print('Calculating performance on test set...')

    for batch_idx, (data, labels) in enumerate (test_loader):
        sample_size = np.shape(data)[0]

        if use_cuda:
            data = data.cuda()
            labels = labels.cuda()
        data = Variable(data, volatile=True)
        resnet_features = model(data)
        output = model_net(resnet_features.view(sample_size, -1))
        nnSigmoid = nn.Sigmoid()
        output = nnSigmoid(output)
        prediction = output.data.cpu().numpy()
        prediction = np.round(prediction)

        for i in range(sample_size):
            f1score = f1_score(labels.cpu().numpy()[i], prediction[i], average='micro')
            average_f1score_vector.append(f1score)
            average_precision = average_precision_score(labels.cpu().numpy()[i], prediction[i])
            average_precision_vector.append(average_precision)

        #if batch_idx % 10 == 0:
            #print('Processing batch: ', batch_idx)
        test_batches += 1

    print('mAP of the network on the test set: %.2f %%' % (np.mean(average_precision_vector) * 100))
    print('F1 SCORE of the network on the test set: %.2f %%' % (np.mean(average_f1score_vector) * 100))

###### MAIN CODE #####


if load_pretrained_model:
    print("Loading Pre-trained model...")
    model_net.load_state_dict(torch.load(backupdir+model_weight_file))
    model_net.eval()
    print("Weights Loaded successfully!")

#test()
print('Beggining Training...')
print ('Current Learning rate is: %f'% (learning_rate))
for epoch in range(max_epochs):
    print ('Current EPOCH is: ', epoch)
    train(epoch)
    test()
    print('---------- EPOCH COMPLETE ----------')
print('------------------- TRAINING COMPLETE -------------------')


