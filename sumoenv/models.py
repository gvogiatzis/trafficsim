import torch
import torch.nn as nn
import torch.nn.functional as F


class TrafficControlerNet(nn.Module):
    def __init__(self,*input):
        super().__init__()
        self.constructor_input = input

class MLPnet(TrafficControlerNet):
    def __init__(self, *sizes):
        super().__init__(*sizes)
        self.num_actions = sizes[-1]
        self.layers = nn.ModuleList()
        for s,s_ in zip(sizes[:-1],sizes[1:]):
            self.layers.append(nn.Linear(s,s_))
    def forward(self, x):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        x = self.layers[-1](x)
        return x
    
class MLPnet2(TrafficControlerNet):
    def __init__(self, *sizes):
        super().__init__(*sizes)
        self.num_actions = sizes[-1]
        self.layers = nn.ModuleList()
        for s,s_ in zip(sizes[:-1],sizes[1:]):
            self.layers.append(nn.Linear(s,s_))
    def forward(self, x):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        x = self.layers[-1](x)
        return x

def saveModel(model, fname):
    cl = model.__class__
    cons_input = model.constructor_input
    weights = model.state_dict()
    torch.save({'class':cl, 'cons-input':cons_input, 'weights':weights}, fname)

def loadModel(fname):
    d = torch.load(fname)
    cl = d['class']    
    cons_input = d['cons-input']
    weights = d['weights']

    model = cl(*cons_input)
    model.load_state_dict(weights)
    return model