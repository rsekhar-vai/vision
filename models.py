import torch
import torch.nn as nn


class SimpleConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,16,kernel_size=3,padding=1)
        self.act1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16,8,kernel_size=3,padding=1)
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(8 * 8 * 8, 32)
        self.act3 = nn.ReLU()
        self.fc2 = nn.Linear(32,10)

    def forward(self,x):
        out1 = self.pool1(self.act1(self.conv1(x)))
        out2 = self.pool2(self.act2(self.conv2(out1)))
        out3 = out2.flatten(1,-1)
        out4 = self.act3(self.fc1(out3))
        out = self.fc2(out4)
        return out