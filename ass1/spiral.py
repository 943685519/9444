# spiral.py
# COMP9444, CSE, UNSW

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class PolarNet(torch.nn.Module):
    def __init__(self, num_hid):
        super(PolarNet, self).__init__()
        # INSERT CODE HERE
        self.Linear1 = nn.Linear(in_features=2,out_features=num_hid,bias=True)
        self.Linear2 = nn.Linear(in_features=num_hid,out_features=1,bias=True)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()


    def forward(self, input):
        x = input[:, 0]
        y = input[:, 1]
        r = torch.sqrt(x**2+y**2).reshape(-1, 1)
        a = torch.atan2(y, x).reshape(-1, 1)
        output1 = torch.cat((r,a),-1)
        output2 = self.Linear1(output1)
        self.hidden_layer_1 = self.tanh(output2)
        output4 = self.Linear2(self.hidden_layer_1)
        output5 = self.sigmoid(output4)
        return output5

class RawNet(torch.nn.Module):
    def __init__(self, num_hid):
        super(RawNet, self).__init__()
        self.Linear1 = nn.Linear(in_features=2,out_features=num_hid,bias=True)
        self.Linear2 = nn.Linear(in_features=num_hid,out_features=num_hid,bias=True)
        self.Linear3 = nn.Linear(in_features=num_hid,out_features=1,bias=True)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        # INSERT CODE HERE

    def forward(self, input):
        output1 = self.Linear1(input)
        self.hidden_layer_1 = self.tanh(output1)
        output3 = self.Linear2(self.hidden_layer_1)
        self.hidden_layer_2 = self.tanh(output3)
        output5 = self.Linear3(self.hidden_layer_2)
        output6 =self.sigmoid(output5)

        return output6

def graph_hidden(net, layer, node):
    xrange = torch.arange(start=-7, end=7.1, step=0.01, dtype=torch.float32)
    yrange = torch.arange(start=-6.6, end=6.7, step=0.01, dtype=torch.float32)
    xcoord = xrange.repeat(yrange.size()[0])
    ycoord = torch.repeat_interleave(yrange, xrange.size()[0], dim=0)
    grid = torch.cat((xcoord.unsqueeze(1), ycoord.unsqueeze(1)), 1)
    with torch.no_grad():  # suppress updating of gradients
        net.eval()  # toggle batch norm, dropout
        net(grid)
        #net.train()  # toggle batch norm, dropout back again
        if layer == 1:
            output = (net.hidden_layer_1[:, node]>= 0).float()
        else:
            output = (net.hidden_layer_2[:, node]>= 0).float()

        plt.clf()
        plt.pcolormesh(xrange, yrange, output.cpu().view(yrange.size()[0], xrange.size()[0]), cmap='Wistia')
    # INSERT CODE HERE
