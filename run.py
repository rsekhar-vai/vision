import torch
import torch.nn as nn
import torchvision
import numpy as np
from utils.functions import fit_model
from models import *

data_path = "data/"
data_train = torchvision.datasets.CIFAR10(data_path,train=True,download=True,
                                          transform=torchvision.transforms.ToTensor())
data_test = torchvision.datasets.CIFAR10(data_path,train=False,download=True,
                                         transform=torchvision.transforms.ToTensor())



model = ConvWithBnorm()
results = fit_model(data_train,data_test,model)

model =  SimpleConv()
results = fit_model(data_train,data_test,model)


model = nn.Sequential(nn.Flatten(1,-1),
                      nn.Linear(3072,200),
                      nn.ReLU(),
                      nn.Linear(200,150),
                      nn.ReLU(),
                      nn.Linear(150,10))

fit_model(data_train,data_test,model)
