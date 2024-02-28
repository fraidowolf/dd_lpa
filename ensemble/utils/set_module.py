import importlib
import torch

global lr1
global lr2
lr1 = 0.001
lr2 = 0.001

def set_optimizer(models, **kwargs):

    opt1 = torch.optim.Adam(models[0].parameters(), lr=lr1)
    opt2 = torch.optim.Adam(models[1].parameters(), lr=lr2)
    
    optimizer = [opt1,opt2]

    return optimizer


