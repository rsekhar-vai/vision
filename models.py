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

class ConvWithBnorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,16,kernel_size=3,padding=1)
        self.bnorm1 = nn.BatchNorm2d(16)
        self.act1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16,8,kernel_size=3,padding=1)
        self.bnorm2 = nn.BatchNorm2d(8)
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(8 * 8 * 8, 32)
        self.act3 = nn.ReLU()
        self.fc2 = nn.Linear(32,10)

    def forward(self,x):
        out1 = self.pool1(self.act1(self.bnorm1(self.conv1(x))))
        out2 = self.pool2(self.act2(self.bnorm2(self.conv2(out1))))
        out3 = out2.flatten(1,-1)
        out4 = self.act3(self.fc1(out3))
        out = self.fc2(out4)
        return out


class ConvWithDropout(nn.Module):
    def __init__(self,p=0.4):
        super().__init__()
        self.conv1 = nn.Conv2d(3,16,kernel_size=3,padding=1)
        #self.bnorm1 = nn.BatchNorm2d(16)
        self.act1 = nn.ReLU()
        self.dropout1 = nn.Dropout2d(p)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16,8,kernel_size=3,padding=1)
        #self.bnorm2 = nn.BatchNorm2d(8)
        self.act2 = nn.ReLU()
        self.dropout2 = nn.Dropout2d(p)

        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(8 * 8 * 8, 32)
        self.act3 = nn.ReLU()
        self.fc2 = nn.Linear(32,10)

    def forward(self,x):
        out = self.pool1(self.act1(self.conv1(x)))
        out = self.dropout1(out)
        out = self.pool2(self.act2(self.conv2(out)))
        out = self.dropout2(out)
        out = out.flatten(1,-1)
        out = self.act3(self.fc1(out))
        out = self.fc2(out)
        return out

class ConvWithBnormDout(nn.Module):
    def __init__(self,p=0.4):
        super().__init__()
        self.conv1 = nn.Conv2d(3,16,kernel_size=3,padding=1)
        self.bnorm1 = nn.BatchNorm2d(16)
        self.act1 = nn.ReLU()
        self.dropout1 = nn.Dropout2d(p)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16,8,kernel_size=3,padding=1)
        self.bnorm2 = nn.BatchNorm2d(8)
        self.act2 = nn.ReLU()
        self.dropout2 = nn.Dropout2d(p)

        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(8 * 8 * 8, 32)
        self.act3 = nn.ReLU()
        self.fc2 = nn.Linear(32,10)

    def forward(self,x):
        out = self.pool1(self.act1(self.bnorm1(self.conv1(x))))
        out = self.dropout1(out)
        out = self.pool2(self.act2(self.bnorm2(self.conv2(out))))
        out = self.dropout2(out)
        out = out.flatten(1,-1)
        out = self.act3(self.fc1(out))
        out = self.fc2(out)
        return out