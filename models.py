import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence



# ResNet output feature size is 2048
class SimpleNet(nn.Module):
    '''
    Simple FC layer to Resnet's output
    '''
    def __init__(self, num_classes):
        super(SimpleNet, self).__init__()
        self.num_classes = num_classes
        self.bn1 = nn.BatchNorm1d(2048)
        self.fc1 = nn.Linear(2048, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, visual_features):
        x = F.relu(self.fc1(self.bn1(visual_features)))
        # DROPOUT
        x = (F.dropout(self.fc2(self.bn2(x)), p = 0.2, training = self.training))
        return x

class Simple_LSTM(nn.Module):
    '''
    CNN + LSTM
    '''

    def __init__(self, num_classes):
        """Set the hyper-parameters and build the layers."""
        """"
        vocab_size should have the size of the number of classes
        """

        super(Simple_LSTM, self).__init__()
        self.num_classes = num_classes
        self.lstm = nn.LSTM(
            input_size=2048,
            hidden_size=128,
            num_layers=1,
            batch_first=True)
        self.linear = nn.Linear(128, num_classes)

    def forward(self, visual_features):

        r_out, (_, _) = self.lstm(visual_features.view(visual_features.shape[0], 1, -1))
        r_out2 = self.linear(r_out[:, -1, :])
        return r_out2



