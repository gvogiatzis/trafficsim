import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPnet(nn.Module):
    def __init__(self, *sizes):
        self.num_actions = sizes[-1]
        super(MLPnet, self).__init__()
        self.layers = nn.ModuleList()
        for s,s_ in zip(sizes[:-1],sizes[1:]):
            self.layers.append(nn.Linear(s,s_))
    def forward(self, x):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        x = self.layers[-1](x)
        return x