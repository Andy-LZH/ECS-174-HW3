import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP_NoRelu(nn.Module):
    def __init__(self):
        super(MLP_NoRelu, self).__init__()
        self.fc1 = nn.Linear(32*32*3, 140) # input layer
        self.fc2 = nn.Linear(140, 84) 
        self.fc3 = nn.Linear(84, 10) # output layer

    def forward(self, x):
        x = torch.flatten(x, 1) # flatten input image to 1D vector
        x = self.fc1(x)
        # x = F.relu(x) # activation function
        x = self.fc2(x)
        # x = F.relu(x)
        x = self.fc3(x)
        output = F.log_softmax(x, dim=1)
        return output