import numpy as np
import torch.nn as nn
import torch

def median(x,ensemble):
    if(isinstance(ensemble[0],nn.ModuleList)):
        y_ = ensemble[0][1](x)
        y = torch.empty(y_.shape).unsqueeze(0)
        y = torch.repeat_interleave(y,len(ensemble),0)
        
        for i,e in enumerate(ensemble):
            y[i] = e[1](x)

        return torch.median(y,0).values
    return 0


def estimate_xy(x,y,ensemble):
    if(isinstance(ensemble[0],nn.ModuleList)):

        x_est = torch.empty(x.shape).unsqueeze(0)
        x_est = torch.repeat_interleave(x_est,len(ensemble),0)
        
        y_ = ensemble[0][1](x)
        y_est = torch.empty(y_.shape).unsqueeze(0)
        y_est = torch.repeat_interleave(y_est,len(ensemble),0)
        
        xy = torch.cat((x,y),1)

        for i,e in enumerate(ensemble):
            x_ = e[0](xy)
            x_est[i] = x_
            y_est[i] = e[1](x_)

        return torch.median(x_est,0).values,torch.median(y_est,0).values
    return 0


def quantile(x,ensemble, q=0.95):
    if(isinstance(ensemble[0],nn.ModuleList)):
        y_ = ensemble[0][1](x)
        y = torch.empty(y_.shape).unsqueeze(0)
        y = torch.repeat_interleave(y,len(ensemble),0)
        
        for i,e in enumerate(ensemble):
            y[i] = e[1](x)

        return np.quantile(y.detach().numpy(),q,axis=0), np.quantile(y.detach().numpy(),1-q,axis=0)
    return 0
