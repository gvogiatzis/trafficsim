import torch
import torch.nn as nn
import torch.nn.functional as F


class TrafficControlerNet(nn.Module):
    def __init__(self,*input):
        super().__init__()
        self.constructor_input = input
    
    def to_dict(self):
        cl = self.__class__
        cons_input = self.constructor_input
        weights = self.state_dict()
        return {'class':cl, 'cons-input':cons_input, 'weights':weights}
    


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

def saveModel(model: TrafficControlerNet, fname: str):
    torch.save(model.to_dict(), fname)

def loadModel_from_dict(model_dict:dict):
    cl = model_dict['class']    
    cons_input = model_dict['cons-input']
    weights = model_dict['weights']

    model = cl(*cons_input)
    model.load_state_dict(weights)
    return model


def loadModel(fname: str):
    d = torch.load(fname)
    cl = d['class']    
    cons_input = d['cons-input']
    weights = d['weights']

    model = cl(*cons_input)
    model.load_state_dict(weights)
    return model