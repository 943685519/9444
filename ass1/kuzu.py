# kuzu.py
# COMP9444, CSE, UNSW

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F

class NetLin(nn.Module):
    # linear function followed by log_softmax
    def __init__(self):
        super(NetLin, self).__init__()
        # INSERT CODE HERE
        self.Linear = nn.Linear(in_features=28*28,out_features=10,bias=True)
        self.LogSoftmax =nn.LogSoftmax()


    def forward(self, x):
        output = x.view(x.shape[0],-1)
        output = self.Linear(output)
        output = self.LogSoftmax(output)
        return output # CHANGE CODE HERE

class NetFull(nn.Module):
    # two fully connected tanh layers followed by log softmax
    def __init__(self):
        super(NetFull, self).__init__()
        # INSERT CODE HERE
        self.L1 = nn.Linear(in_features=28*28,out_features=200,bias=True)
        self.L2 = nn.Linear(in_features=200,out_features=10,bias=True)
        #self.L3 = nn.Linear(in_features=100, out_features=10, bias=True)
        self.tanh = nn.Tanh()
        self.LogSoftmax =nn.LogSoftmax()

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        output1 = self.L1(x)
        output2 = self.tanh(output1)
        output3 = self.L2(output2)
        output4 = self.LogSoftmax(output3)
        return output4
         # CHANGE CODE HERE

class NetConv(nn.Module):
    # two convolutional layers and one fully connected layer,
    # all using relu, followed by log_softmax
    def __init__(self):
        super(NetConv, self).__init__()
        # INSERT CODE HERE
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.Linear1 = nn.Linear(in_features=12544, out_features=512, bias=True)
        self.Linear2 = nn.Linear(in_features=512, out_features=10, bias=True)
        self.maxpooling = nn.MaxPool2d(kernel_size=2)
        self.tanh = nn.Tanh()
        self.LogSoftmax = nn.LogSoftmax()
    def forward(self, x):
        output1 = self.conv1(x)
        output2 = F.relu(output1)
        #output2 = F.tanh(output1)
        output3 = self.conv2(output2)
        output4 = F.relu(output3)
        #output4 = F.tanh(output3)
        output5 = self.maxpooling(output4)

        X1 = output5.view(output5.shape[0], -1)

        output6 = self.Linear1(X1)
        output7 = F.relu(output6)
        #output7 = F.tanh(output6)
        output8 = self.Linear2(output7)
        output9 = self.LogSoftmax(output8)

        return output9
         # CHANGE CODE HERE
