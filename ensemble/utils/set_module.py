import importlib
import torch

def set_optimizer(models, **kwargs):

    opt1 = torch.optim.Adam(models[0].parameters(), lr=0.001)
    opt2 = torch.optim.Adam(models[1].parameters(), lr=0.001)
    
    optimizer = [opt1,opt2]

    return optimizer


