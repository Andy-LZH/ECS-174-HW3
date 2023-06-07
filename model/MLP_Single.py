import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP_Single(nn.Module):
    def __init__(self):
        super(MLP_Single, self).__init__()
        self.fc1 = nn.Linear(32*32*3, 128) # input layer
        self.fc2 = nn.Linear(128, 10) # output layer

    def forward(self, x):
        x = torch.flatten(x, 1) # flatten input image to 1D vector
        x = self.fc1(x)
        x = F.relu(x) # activation function
        x = self.fc2(x)
        return x